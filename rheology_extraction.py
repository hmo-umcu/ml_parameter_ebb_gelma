"""
Rheological feature extractor for logarithmic amplitude sweep files.

Reads all .xls / .xlsx / .csv amplitude sweep files from an input directory,
extracts quantitative rheological features algorithmically (no manual inspection),
saves per-file plots and feature tables, then collects all features into a
single summary table.

Usage:
    python rheology_extractor.py --input_dir <path> --output_dir <path>

Output layout:
    <output_dir>/
        rheological_features_all.csv          <- summary table (all files)
        <filename_stem>/
            <filename_stem>_plot.png          <- G', G'', tan(delta) vs strain
            <filename_stem>_features.csv      <- features for this file

Feature definitions
--------------------
G_prime_LVE_Pa, G_dprime_LVE_Pa, tan_delta_LVE
    Mean values over the LVE window (strain < LVE_STRAIN_LIMIT = 10%).
    Fixed strain ceiling avoids false extension of the LVE window due to
    strain hardening (G' rising slowly before yielding), which is common in GelMA.

G_prime_peak_Pa, G_prime_peak_strain_pct
    Maximum G' value and the strain at which it occurs.
    Accounts for strain hardening before yield.

yield_strain_pct
    Strain at which G' drops to 90% of its peak value, searched only
    in the post-peak region. Interpolated between bracketing data points.
    NaN if yield is not reached within the sweep range.

yield_stress_Pa
    Approximation: G'_LVE * (yield_strain / 100).
    Uses resting stiffness (LVE G') as the modulus reference.

tan_delta_at_10pct_strain, tan_delta_at_100pct_strain
    Linearly interpolated tan(delta) at two fixed reference strains.
    Always defined as long as the sweep covers those strains.
    Serve as guaranteed scalar proxies for viscoelastic character,
    replacing crossover strain which may not always be observable.

crossover_strain_pct, crossover_modulus_Pa
    Conditional: extracted by interpolation when G' and G'' cross within
    the measured range. NaN if crossover is not observed. Always check
    for NaN before using these features.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
LVE_STRAIN_LIMIT    = 10.0   # % — upper strain ceiling for LVE window
YIELD_DROP_FRACTION = 0.10   # G' must drop by this fraction of peak to count as yield
SUPPORTED           = {".xls", ".xlsx", ".csv"}

# ── colours and font sizes ────────────────────────────────────────────────────
C_GP     = "#185FA5"   # blue   — G' solid line
C_GPP    = "#185FA5"   # blue   — G'' dashed line
C_LVE    = "#EF9F27"   # amber  — LVE region shading
C_YIELD  = "#D85A30"   # coral  — yield onset line
C_PEAK   = "#7F77DD"   # purple — G' peak line
C_CROSS  = "#BA7517"   # dark amber — G'/G'' crossover line
C_TAN    = "#1D9E75"   # teal   — tan(delta) line
C_REF10  = "#555555"   # dark grey — 10% reference marker
C_REF100 = "#222222"   # darker grey — 100% reference marker

FONT_LABEL  = 16
FONT_TICK   = 14
FONT_LEGEND = 14
FONT_TITLE  = 16
LEGEND_KW   = {"handlelength": 2.8, "handleheight": 1.4, "handletextpad": 0.9}


# ── file loading ──────────────────────────────────────────────────────────────

def load_amplitude_sweep(filepath: Path) -> pd.DataFrame:
    """
    Load a TA Instruments amplitude sweep export (.xls, .xlsx, .csv).
    Returns a clean DataFrame sorted by strain with columns:
        step_time_s, G_prime_Pa, G_dprime_Pa, frequency_Hz, strain_pct, tan_delta
    """
    suffix = filepath.suffix.lower()

    if suffix == ".xls":
        raw = pd.read_excel(filepath, sheet_name=None, engine="xlrd", header=None)
    elif suffix == ".xlsx":
        raw = pd.read_excel(filepath, sheet_name=None, engine="openpyxl", header=None)
    elif suffix == ".csv":
        raw = {"sheet": pd.read_csv(filepath, header=None)}
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # find the amplitude sweep sheet by checking first cell
    target = None
    for sheet in raw.values():
        if not sheet.empty and "amplitude" in str(sheet.iloc[0, 0]).lower():
            target = sheet
            break
    if target is None:
        target = list(raw.values())[0]

    # rows: 0 = sheet title, 1 = column headers, 2 = units, 3+ = data
    target.columns = [
        "step_time_s", "G_prime_Pa", "G_dprime_Pa",
        "frequency_Hz", "strain_pct", "tan_delta"
    ]
    df = target.iloc[3:].reset_index(drop=True).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (df
            .dropna(subset=["strain_pct", "G_prime_Pa", "G_dprime_Pa"])
            .sort_values("strain_pct")
            .reset_index(drop=True))


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, filename_stem: str) -> dict:
    """
    Extract all rheological features algorithmically.
    Returns a dict of scalar values; NaN where a feature cannot be determined.
    """
    features = {"sample": filename_stem}
    strains = df["strain_pct"].values
    gp      = df["G_prime_Pa"].values
    gpp     = df["G_dprime_Pa"].values
    tan_d   = gpp / gp

    # ── LVE window ────────────────────────────────────────────────────────────
    lve = df[df["strain_pct"] < LVE_STRAIN_LIMIT]
    if lve.empty:
        features["G_prime_LVE_Pa"]  = np.nan
        features["G_dprime_LVE_Pa"] = np.nan
        features["tan_delta_LVE"]   = np.nan
    else:
        features["G_prime_LVE_Pa"]  = lve["G_prime_Pa"].mean()
        features["G_dprime_LVE_Pa"] = lve["G_dprime_Pa"].mean()
        features["tan_delta_LVE"]   = (lve["G_dprime_Pa"] / lve["G_prime_Pa"]).mean()

    # ── G' peak ───────────────────────────────────────────────────────────────
    peak_idx = int(np.argmax(gp))
    g_peak   = gp[peak_idx]
    features["G_prime_peak_Pa"]         = g_peak
    features["G_prime_peak_strain_pct"] = strains[peak_idx]

    # ── yield onset: G' drops to (1 - YIELD_DROP_FRACTION) of peak ───────────
    yield_threshold = g_peak * (1.0 - YIELD_DROP_FRACTION)
    post = np.where(
        (np.arange(len(gp)) >= peak_idx) & (gp <= yield_threshold)
    )[0]

    if len(post) > 0:
        i = post[0]
        if i > 0 and gp[i] != gp[i - 1]:
            frac = (yield_threshold - gp[i - 1]) / (gp[i] - gp[i - 1])
            features["yield_strain_pct"] = strains[i - 1] + frac * (strains[i] - strains[i - 1])
        else:
            features["yield_strain_pct"] = strains[i]
    else:
        features["yield_strain_pct"] = np.nan

    # ── yield stress approximation ────────────────────────────────────────────
    g_lve = features["G_prime_LVE_Pa"]
    ys    = features["yield_strain_pct"]
    features["yield_stress_Pa"] = (
        g_lve * ys / 100.0
        if not (np.isnan(g_lve) or np.isnan(ys))
        else np.nan
    )

    # ── tan(delta) at fixed reference strains ─────────────────────────────────
    f_tan = interp1d(strains, tan_d, kind="linear",
                     bounds_error=False, fill_value=np.nan)
    for ref in (10.0, 100.0):
        key = f"tan_delta_at_{int(ref)}pct_strain"
        features[key] = (
            float(f_tan(ref))
            if strains.min() <= ref <= strains.max()
            else np.nan
        )

    # ── G'/G'' crossover (conditional) ───────────────────────────────────────
    diff         = gp - gpp
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        i       = sign_changes[0]
        s0, d0  = strains[i],     diff[i]
        s1, d1  = strains[i + 1], diff[i + 1]
        frac    = -d0 / (d1 - d0) if d1 != d0 else 0.0
        co_s    = s0 + frac * (s1 - s0)
        co_m    = float(interp1d([s0, s1], [gp[i], gp[i + 1]])(co_s))
        features["crossover_strain_pct"] = co_s
        features["crossover_modulus_Pa"]  = co_m
    else:
        features["crossover_strain_pct"] = np.nan
        features["crossover_modulus_Pa"]  = np.nan

    return features


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sweep(df: pd.DataFrame, features: dict, out_path: Path) -> None:
    """
    Two-panel figure:
        top    — G' and G'' vs strain (log-log)
        bottom — tan(delta) vs strain (log-lin)
    LVE region shaded in amber in both panels.
    Key features annotated with reference lines.
    """
    strains = df["strain_pct"].values
    gp      = df["G_prime_Pa"].values
    gpp     = df["G_dprime_Pa"].values
    tan_d   = gpp / gp

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(features["sample"], fontsize=13, y=0.99)

    # ── panel 1: G' and G'' ──────────────────────────────────────────────────
    ax1.loglog(strains, gp,  color=C_GP,  lw=2.5,
               label="G'  (storage modulus)")
    ax1.loglog(strains, gpp, color=C_GPP, lw=2.0, ls="--", alpha=0.75,
               label="G''  (loss modulus)")

    ax1.axvspan(strains.min(), LVE_STRAIN_LIMIT,
                color=C_LVE, alpha=0.15, label="LVE region")

    ys = features.get("yield_strain_pct", np.nan)
    if not np.isnan(ys):
        ax1.axvline(ys, color=C_YIELD, lw=2.0, ls=":",
                    label="Yield onset")

    ps = features.get("G_prime_peak_strain_pct", np.nan)
    if not np.isnan(ps):
        ax1.axvline(ps, color=C_PEAK, lw=1.8, ls="-.",
                    label="G' peak")

    co_s = features.get("crossover_strain_pct", np.nan)
    co_m = features.get("crossover_modulus_Pa",  np.nan)
    if not np.isnan(co_s):
        ax1.axvline(co_s, color=C_CROSS, lw=2.0, ls="-.",
                    label="G'/G'' crossover")
        ax1.scatter([co_s], [co_m], color=C_CROSS, zorder=5, s=80)

    ax1.set_ylabel("Modulus (Pa)", fontsize=FONT_LABEL)
    ax1.tick_params(axis="both", labelsize=FONT_TICK)
    ax1.legend(fontsize=FONT_LEGEND, framealpha=0.75,
               loc="lower left", **LEGEND_KW)
    ax1.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
    ax1.set_title("Storage (G') and loss (G'') modulus",
                  fontsize=FONT_TITLE, pad=6)

    # ── panel 2: tan(delta) ───────────────────────────────────────────────────
    ax2.semilogx(strains, tan_d, color=C_TAN, lw=2.5,
                 label="tan(\u03b4) = G''/G'")
    ax2.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.6,
                label="tan(\u03b4) = 1  (fluid-like)")

    ax2.axvspan(strains.min(), LVE_STRAIN_LIMIT,
                color=C_LVE, alpha=0.15, label="LVE region")

    for ref, col in [(10.0, C_REF10), (100.0, C_REF100)]:
        td = features.get(f"tan_delta_at_{int(ref)}pct_strain", np.nan)
        if not np.isnan(td):
            ax2.axvline(ref, color=col, lw=1.5, ls=":",
                        label=f"Reference strain  {int(ref)}%")
            ax2.scatter([ref], [td], color=col, zorder=5, s=60)

    ax2.set_xlabel("Oscillation strain (%)", fontsize=FONT_LABEL)
    ax2.set_ylabel("tan(\u03b4)", fontsize=FONT_LABEL)
    ax2.set_ylim(bottom=0, top=max(2.0, float(np.nanmax(tan_d)) * 1.1))
    ax2.tick_params(axis="both", labelsize=FONT_TICK)
    ax2.legend(fontsize=FONT_LEGEND, framealpha=0.75, **LEGEND_KW)
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
    ax2.set_title("Loss tangent tan(\u03b4)", fontsize=FONT_TITLE, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved  → {out_path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files found in {input_dir}")
        return

    all_features = []

    for filepath in files:
        stem = filepath.stem
        print(f"\nProcessing: {stem}")
        file_out = output_dir / stem
        file_out.mkdir(parents=True, exist_ok=True)

        try:
            df = load_amplitude_sweep(filepath)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        try:
            features = extract_features(df, stem)
        except Exception as e:
            print(f"  ERROR extracting features: {e}")
            continue

        # save per-file feature table
        feat_path = file_out / f"{stem}_features.csv"
        # pd.DataFrame([features]).to_csv(feat_path, index=False, sep=";")
        # pd.DataFrame([features]).to_csv(feat_path, index=False)


        with open(feat_path, "w", encoding="utf-8") as f:
            f.write("sep=;\n")
        pd.DataFrame([features]).round(3).to_csv(feat_path, index=False, sep=";", mode="a", decimal=",")

        print(f"  Features saved → {feat_path}")

        # save plot
        try:
            plot_sweep(df, features, file_out / f"{stem}_plot.png")
        except Exception as e:
            print(f"  WARNING: plot failed: {e}")

        # console summary
        def fv(k):
            v = features.get(k, np.nan)
            return f"{v:.2f}" if not np.isnan(v) else "n/a"

        print(f"  G' LVE              : {fv('G_prime_LVE_Pa')} Pa")
        print(f"  G'' LVE             : {fv('G_dprime_LVE_Pa')} Pa")
        print(f"  tan(delta) LVE      : {fv('tan_delta_LVE')}")
        print(f"  G' peak             : {fv('G_prime_peak_Pa')} Pa @ {fv('G_prime_peak_strain_pct')}% strain")
        print(f"  Yield strain        : {fv('yield_strain_pct')} %")
        print(f"  Yield stress        : {fv('yield_stress_Pa')} Pa")
        print(f"  tan(delta) @ 10%    : {fv('tan_delta_at_10pct_strain')}")
        print(f"  tan(delta) @ 100%   : {fv('tan_delta_at_100pct_strain')}")
        co = features.get("crossover_strain_pct", np.nan)
        if not np.isnan(co):
            print(f"  G'/G'' crossover    : {fv('crossover_strain_pct')} % strain, {fv('crossover_modulus_Pa')} Pa")
        else:
            print(f"  G'/G'' crossover    : not observed in sweep range")

        all_features.append(features)

    # save combined summary table
    if all_features:
        summary_path = output_dir / "rheological_features_all.csv"
        # pd.DataFrame(all_features).to_csv(summary_path, index=False, sep=";")
        # pd.DataFrame(all_features).to_csv(summary_path, index=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("sep=;\n")
        pd.DataFrame(all_features).round(3).to_csv(summary_path, index=False, sep=";", mode="a", decimal=",")


        print(f"\n{'='*60}")
        print(f"Summary table saved → {summary_path}")
        print(pd.DataFrame(all_features).to_string(index=False))
    else:
        print("No features extracted — check input files.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract rheological features from logarithmic amplitude sweep files."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing amplitude sweep files (.xls / .xlsx / .csv)"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Root output directory (rheological transformation result folder)"
    )
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
