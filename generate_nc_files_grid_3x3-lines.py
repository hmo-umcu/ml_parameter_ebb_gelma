#!/usr/bin/env python3
"""
generate_nc_files.py
--------------------
Reads the LHS sample CSV and generates NC files for a 48-well plate.
Each NC file contains 8 columns (= 8 samples), each column has 6 constructs (rows A-F).

GEOMETRY: 2D cross-hatch grid (single layer, 0-90 degree pattern)
  Pass 1: 3 horizontal strands running in X-direction (at fixed Y positions)
  Pass 2: 3 vertical strands running in Y-direction (at fixed X positions)
  Both passes at the same Z height (z_offset).
  Each strand is printed independently: M160 (on) -> G01 -> M161 (off).
  Z lifts to Z_TRAVEL before every G00 repositioning move between strands,
  so the nozzle never contacts already-printed material.

  Strand positions (from well centre): -3.0, 0.0, +3.0 mm  (3.0 mm pitch)
  Strand half-length: 3.5 mm  (7.0 mm total span per strand)
  Nominal construct footprint: ~7.85 mm x 7.85 mm (incl. strand width)
  Nominal pore gap: ~2.15 mm  (pitch 3.0 mm - strand width ~0.85 mm)
  Grid: 2x2 = 4 open pores per construct

Key design decisions:
  1. M151 (engage tool) called ONCE per column, at the first well only.
     Wells 2-6 reposition with G00 only. This prevents the
     "tool cannot be engaged at this height" collision error.
  2. Each strand has its own M160/M161 bracket — dispensing is strictly ON
     only during the G01 print move, OFF during all G00 travel.
  3. Z lifts to Z_TRAVEL before every inter-strand G00, so the nozzle tip
     cannot contact previously printed strands.
"""

import re
import math
import pandas as pd
from pathlib import Path

# == Paths ====================================================================
BASE_DIR      = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / "ai_poc_48_template.nc"
CSV_FILE      = BASE_DIR / "data" / "lhs" / "lhs_bioprint_samples_semicolon.csv"
OUTPUT_DIR    = BASE_DIR / "data" / "nc_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# == 48-well plate geometry ===================================================
ROWS_ALPHA = ['A', 'B', 'C', 'D', 'E', 'F']
COLS       = [1, 2, 3, 4, 5, 6, 7, 8]
N_COLS     = 8

Z_IMAGING = 20.00
Z_SAFE    = 40.00

X_A1, Y_A1 = -250.160,  53.250
X_A8        = -158.200
Y_F1        =  -11.790
X_STEP = (X_A8 - X_A1) / 7
Y_STEP = (Y_F1 - Y_A1) / 5

def build_well_map():
    wells = {}
    for r_idx, row in enumerate(ROWS_ALPHA):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells

WELL_MAP = build_well_map()

# == G55 origin positions (from original template) ============================
COL_X = {
    1: -45.785, 2: -32.705, 3: -19.625, 4: -6.545,
    5:   6.535, 6:  19.615, 7:  32.695, 8: 45.775,
}

ROW_Y = {
    'A':  32.720, 'B':  19.640, 'C':   6.560,
    'D':  -6.520, 'E': -19.600, 'F': -32.680,
}

# == Grid geometry (mm, relative to G55 well centre = 0,0) ====================
# 3 strands per pass, 2.5 mm pitch: -2.5, 0.0, +2.5 mm
# STRAND_EXT < max(STRAND_POS) → open corners (no overlap at crossings).
# Pore gap ≈ 2.5 - 0.85 = 1.65 mm
# Reach = 2.5 + 2.0 = 4.5 mm  (well radius ~5.0 mm → 0.5 mm clearance)

STRAND_POS        = [-2.5, 0.0, 2.5]   # 2.5 mm pitch, 3 strands
STRAND_EXT        =  2.0               # strand half-length → 4.0 mm strand, 4.5 mm reach
Z_TRAVEL          = 18.400             # safe Z for between-well moves
Z_LIFT_HV         =  1.0               # mm lift for H→V transition only

# G807 start/stop delays and far-end compensation.
# START_COMP extends the G01 far-end target to compensate for delayed valve opening.
# STOP_DELAY matches START_DELAY so all strand ends taper identically.
# No START_EXT needed — open corners do not require near-end compensation.
G807_START_DELAY  =  0.3               # mm
G807_STOP_DELAY   =  0.3               # mm
START_COMP        =  0.3               # mm — added to G01 far-end target
TAIL_EQ           =  0.3               # mm — G00 tail added after H3 and V3 M161
                                        # to match the residual deposition that H1,
                                        # H2, V1, V2 get from their following G00 moves


# =============================================================================
def grid_toolpath(z_mm):
    """
    G-code lines for the 3+3 strand cross-hatch grid in one well.
    Tool is already engaged and Z is at z_mm when this runs.

    Each strand: G807 → M160 → G01 → M161 → G00.
    No Z-lift between strands. Z-lift only for H→V transition.

    Strand-length equalisation:
      H1, H2, V1, V2 each have a short (~0.3mm) G00 following M161 that
      deposits a small residual tail, making them appear slightly longer
      than the nominal G01 endpoint.
      H3 and V3 had no such following move — H3 goes straight to Z-lift,
      V3 ends the well entirely — so they looked shorter than the others.
      Fix: after H3's M161, add a TAIL_EQ mm G00 in the strand direction
      before the Z-lift. After V3's M161, add a TAIL_EQ mm G00 as well.
      TAIL_EQ should equal the inter-strand G00 distance (~START_COMP).

    Snake pattern:
      H1: Y=sp[0] left→right   H2: Y=sp[1] right→left   H3: Y=sp[2] left→right
      V1: X=sp[0] top→bottom   V2: X=sp[1] bottom→top   V3: X=sp[2] top→bottom
    """
    e    = STRAND_EXT
    sp   = STRAND_POS
    z    = f"{z_mm:.3f}"
    zl   = f"{z_mm + Z_LIFT_HV:.3f}"
    g807 = f"G807[2, {G807_START_DELAY:.3f}, {G807_STOP_DELAY:.3f}]"
    sc   = START_COMP
    tq   = TAIL_EQ
    lines = []

    # ── Pass 1: horizontal strands ───────────────────────────────────────────
    for i, y in enumerate(sp):
        x_end_nom = e if i % 2 == 0 else -e
        x_end_ext = x_end_nom + sc if x_end_nom > 0 else x_end_nom - sc
        lines.append(f"{g807}")
        lines.append(f"M160                      ; H{i+1} ON")
        lines.append(f"G01 X{x_end_ext:.3f}      ; H-strand {i+1}")
        lines.append(f"M161                      ; H{i+1} OFF")
        if i < len(sp) - 1:
            x_next = -e if (i + 1) % 2 == 0 else e
            y_next = sp[i + 1]
            lines.append(f"G00 X{x_next:.3f} Y{y_next:.3f}  ; → H-strand {i+2} start")
        else:
            # H3: add tail-equalising move before Z-lift
            x_tail = x_end_ext + tq if x_end_ext > 0 else x_end_ext - tq
            lines.append(f"G00 X{x_tail:.3f}          ; tail-eq {tq}mm — match H1/H2 residual")

    # ── H→V transition ───────────────────────────────────────────────────────
    lines.append(f"G00 Z{zl}                   ; lift {Z_LIFT_HV}mm — clear H-strands")
    lines.append(f"G00 X{sp[0]:.3f} Y{-e:.3f}  ; → V-strand 1 start")
    lines.append(f"G00 Z{z}                    ; lower to print height")

    # ── Pass 2: vertical strands ─────────────────────────────────────────────
    for i, x in enumerate(sp):
        y_end_nom = e if i % 2 == 0 else -e
        y_end_ext = y_end_nom + sc if y_end_nom > 0 else y_end_nom - sc
        lines.append(f"{g807}")
        lines.append(f"M160                      ; V{i+1} ON")
        lines.append(f"G01 Y{y_end_ext:.3f}      ; V-strand {i+1}")
        lines.append(f"M161                      ; V{i+1} OFF")
        if i < len(sp) - 1:
            x_next       = sp[i + 1]
            y_next_start = -e if (i + 1) % 2 == 0 else e
            lines.append(f"G00 X{x_next:.3f} Y{y_next_start:.3f}  ; → V-strand {i+2} start")
        else:
            # V3: add tail-equalising move before well ends
            y_tail = y_end_ext + tq if y_end_ext > 0 else y_end_ext - tq
            lines.append(f"G00 Y{y_tail:.3f}          ; tail-eq {tq}mm — match V1/V2 residual")

    return lines


# =============================================================================
def build_column_block(col_num, sample_id, pressure_kpa, speed_mms, z_mm,
                       prog_start):
    """
    G-code for one full column (6 wells, rows A-F).

    First well (A):
        G805[x,y,z] / G55
        G00 X{xs} Y{ys}      <- move to first H-strand start (X=-e, Y=sp[0])
        M151                  <- engage tool
        Z{z_mm}              <- lower to print height
        M110 / grid_toolpath (includes all M160/M161/Z-lifts internally)

    Wells B-F:
        G805[x,y,z] / G55
        G00 Z{Z_TRAVEL}      <- lift to travel height first
        X{xs} Y{ys}          <- reposition XY (implicit G00)
        Z{z_mm}              <- lower to print height
        M110 / grid_toolpath
    """
    m200 = int(round(pressure_kpa * 10))
    fval = f"{speed_mms:.3f}"
    z    = f"{z_mm:.3f}"
    cx   = COL_X[col_num]
    # First strand start: X = -STRAND_EXT, Y = STRAND_POS[0]
    xs   = f"{-STRAND_EXT:.3f}"
    ys   = f"{STRAND_POS[0]:.3f}"

    lines = []
    lines.append("")
    lines.append(
        f"; -- Col {col_num} | Sample {sample_id} | "
        f"P={int(pressure_kpa)}kPa  F={speed_mms:.1f}mm/s  Z={z_mm:.3f}mm --"
    )

    prog = prog_start

    for w_idx, row in enumerate(ROWS_ALPHA):
        ry = ROW_Y[row]

        lines.append(f"G805[{cx:.3f}, {ry:.3f}, 2.620] ; G55 origin: {row}{col_num}")
        lines.append("G55")

        if w_idx == 0:
            lines.append("")
            lines.append("; Changing tool to 'PSD 1'")
            lines.append("#FLUSH WAIT")
            lines.append("T1")
            lines.append("G807[1, 0.002, 0.002] ; time-based start/stop delays [s]")
            lines.append(f"M200={m200} ; pressure {pressure_kpa:.0f}kPa")
            lines.append(f"F{fval}")
            lines.append(f"G00 X{xs} Y{ys}")    # XY to first H-strand start
            lines.append("M151 ; Engage tool for printing")
            lines.append(f"Z{z}")               # lower to print height
        else:
            lines.append(f"G00 Z{Z_TRAVEL:.3f}")  # lift to travel height
            lines.append(f"M200={m200}")
            lines.append(f"F{fval}")
            lines.append(f"X{xs} Y{ys}")          # XY reposition (implicit G00)
            lines.append(f"Z{z}")                  # lower to print height

        lines.append(f"M110={prog}")
        lines.extend(grid_toolpath(z_mm))
        # After grid_toolpath, dispensing is OFF (M161 was last command).
        # Lift to travel height before moving to next well.
        lines.append(f"G00 Z{Z_TRAVEL:.3f} ; lift after well {row}{col_num}")
        prog += 10

    return lines


# =============================================================================
def parse_template_header(template_path):
    raw    = template_path.read_text().splitlines()
    header = []
    for line in raw:
        if re.match(r'G805\[', line.strip()):
            break
        header.append(line.rstrip())
    return header


def update_header_pressure(header_lines, pressure_kpa):
    m200 = int(round(pressure_kpa * 10))
    out  = []
    for line in header_lines:
        s = line.strip()
        if re.match(r'M200=\d+', s):
            out.append(f"M200={m200} ; Set pressure to {pressure_kpa:.0f}kPa")
        elif re.match(r'M300=\d+', s) or re.match(r'M302', s):
            pass  # temperature set manually in Architect UI
        else:
            out.append(line.rstrip())
    return out


# =============================================================================
def generate_imaging_block(n_active_cols=8):
    well_names = [f"{r}{c}"
                  for c in range(1, n_active_cols + 1)
                  for r in ROWS_ALPHA]
    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Column-major order: col1 (A1-F1) -> col2 (A2-F2) -> ...")
    lines.append("; Slot 1 selected (T1) -- camera physically at slot 5.")
    lines.append(f"; Z safe = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1")
    lines.append("G803                       ; Move to system safe height")
    lines.append("")

    for well in well_names:
        x, y, z = WELL_MAP[well]
        lines.append(f"; --- Well {well} ---")
        lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Raise to safe Z first")
        lines.append(f"G00 X{x:.3f} Y{y:.3f}              ; Move XY to {well}")
        lines.append(f"G00 Z{z:.3f}                ; Lower to imaging height")
        lines.append(
            f'V.E.UserInteraction.Message = "Camera at {well} - trigger imaging, then click OK"'
        )
        lines.append("M121                       ; Pause for manual camera trigger")
        lines.append("")

    lines.append("; --- Return home ---")
    lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Safe Z before going home")
    lines.append("G800                       ; Go home")
    lines.append("M110=1000")
    lines.append("M30")
    return "\n".join(lines)


# =============================================================================
def batch_samples(df):
    n       = len(df)
    n_files = math.ceil(n / N_COLS)
    all_s   = [df.iloc[i] for i in range(n)]
    batches = []
    for k in range(n_files):
        batch = [all_s[k + j * n_files]
                 for j in range(N_COLS)
                 if k + j * n_files < n]
        batches.append(batch)
    return batches


# =============================================================================
def main():
    if not CSV_FILE.is_file():
        print(f"[ERROR] CSV not found: {CSV_FILE}")
        print("  Run lhs_sampling.py first.")
        return

    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from:\n  {CSV_FILE}")

    if not TEMPLATE_FILE.is_file():
        print(f"\n[ERROR] Template not found: {TEMPLATE_FILE}")
        return

    header_lines = parse_template_header(TEMPLATE_FILE)
    print(f"\nTemplate header parsed: {len(header_lines)} lines")

    batches = batch_samples(df)
    print(f"\nBatching into {len(batches)} NC files:")
    for idx, batch in enumerate(batches):
        sids = [int(s.name) for s in batch]
        print(f"  File {idx+1:>2}: {len(batch):>2} cols | sample IDs={sids}")

    print(f"\nGeometry: 2D cross-hatch grid (single layer)")
    print(f"  Strand positions: {STRAND_POS} mm  (pitch {STRAND_POS[1]-STRAND_POS[0]:.1f} mm)")
    print(f"  Strand half-ext:  +/-{STRAND_EXT} mm ({2*STRAND_EXT:.1f} mm span)")
    print(f"  Pores:            {len(STRAND_POS)-1}x{len(STRAND_POS)-1} = {(len(STRAND_POS)-1)**2} open pores")
    print(f"  Z between H-strands:  NO lift (travel at print height)")
    print(f"  Z between V-strands:  NO lift (travel at print height)")
    print(f"  Z for H→V transition: +{Z_LIFT_HV} mm lift (crosses printed H-strands)")
    print(f"  G807 start delay: {G807_START_DELAY} mm distance-based")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    for file_idx, batch in enumerate(batches):
        n_samples = len(batch)
        sids      = [int(s.name) for s in batch]
        fname     = f"data_collection_48well_8cols_s{sids[0]}-s{sids[-1]}.nc"
        fpath     = OUTPUT_DIR / fname

        first_pressure = float(batch[0]['Pressure_kPa'])

        file_comments = [f"% {fname}"]
        file_comments.append("; Generated by generate_nc_files.py")
        file_comments.append(
            "; Geometry: 2D cross-hatch grid | 3+3 strands | 2x2 open pores | single layer"
        )
        file_comments.append(f"; CSV: {CSV_FILE.name}")
        file_comments.append("; Temperature: set manually in Architect UI (not in G-code)")
        file_comments.append(";")
        file_comments.append(
            f"; {'Col':<5} {'SampleID':<10} {'P(kPa)':<10} {'F(mm/s)':<10} {'Z(mm)':<8}"
        )
        for col_num, s in enumerate(batch, start=1):
            file_comments.append(
                f"; {col_num:<5} {int(s.name):<10} {int(s['Pressure_kPa']):<10} "
                f"{float(s['NozzleSpeed_mms']):<10.1f} {float(s['Zoffset_mm']):<8}"
            )
        file_comments.append(";")

        updated_header = update_header_pressure(header_lines, first_pressure)
        all_lines      = file_comments + updated_header[1:]

        for col_num in COLS:
            if col_num > n_samples:
                break
            sample = batch[col_num - 1]
            col_block = build_column_block(
                col_num=col_num,
                sample_id=int(sample.name),
                pressure_kpa=float(sample['Pressure_kPa']),
                speed_mms=float(sample['NozzleSpeed_mms']),
                z_mm=float(sample['Zoffset_mm']),
                prog_start=40 + (col_num - 1) * 10,
            )
            all_lines.extend(col_block)

        all_lines.append("")
        all_lines.append("#FLUSH WAIT")
        all_lines.append(generate_imaging_block(n_active_cols=n_samples))

        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  [{file_idx+1:>2}] {fname}")

    print(f"\nDone. {len(batches)} NC files written to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
