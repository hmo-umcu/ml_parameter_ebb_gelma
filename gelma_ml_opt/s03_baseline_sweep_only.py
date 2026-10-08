
"""
s03_baseline_sweep_only.py
--------------------------
Baseline B2 from protocol 5.3: predict the held-out category's 32 LHS
conditions by interpolating that category's OWN pressure-sweep curve.
 
No training categories. No machine learning. Just the sweep, read at each
LHS pressure.
 
Why this baseline decides the Aim 1 claim
------------------------------------------
The fingerprint is measured on the held-out material, and SF is dominated by
pressure. The sweep is SF versus pressure. So the fingerprint is close to a
direct measurement of the thing being predicted, and "transfer" can collapse
into re-reading the target's own sweep.
 
  M beats B1 but not B2  ->  a cheap sweep predicts the surface. Useful, but
                             it is not cross-formulation transfer and should
                             not be written as such.
  M beats B2             ->  the model is genuinely adding structure learned
                             from the other formulations.
 
The sweep is at fixed speed and Z-offset (10 mm/s, 0.2 mm), so B2 is blind to
those two axes by construction. That is the point: it isolates how much of the
surface is pure pressure response.
 
Usage
-----
    python s03_baseline_sweep_only.py \
        --data_dir  <folder with *_sf_complete_48well.csv> \
        --sweep_dir <folder with *_sf_summary_sweep.csv> \
        --test_categories F
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import sf_data as S
 
SWEEP_SUFFIX = '_sf_summary_sweep.csv'
 
 
def load_sweep(sweep_dir, category):
    p = Path(sweep_dir) / f'{category}{SWEEP_SUFFIX}'
    if not p.exists():
        return None
    df = pd.read_csv(p, sep=S.sniff(p))
    for c in ('Pressure_kPa', S.TARGET):
        if c not in df.columns:
            raise SystemExit(f'{p.name} has no "{c}" column')
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Pressure_kPa', S.TARGET]).sort_values('Pressure_kPa')
    return df
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
    cats = sorted(df['category'].unique(), key=lambda c: S.NAME_TO_LETTER.get(c, 'Z'))
    targets = cats if a.loco else S.resolve_list(a.test_categories)
    if not targets:
        raise SystemExit('Give --test_categories or --loco.')
 
    print('=' * 74)
    print('  B2  SWEEP-ONLY BASELINE')
    print('=' * 74)
 
    rows, mets = [], []
    for cat in targets:
        sw = load_sweep(a.sweep_dir, cat)
        if sw is None:
            print(f'[SKIP] {S.label(cat)}: no {cat}{SWEEP_SUFFIX} in {a.sweep_dir}')
            continue
        te = df[df['category'] == cat].copy()
        P_lhs = te['Pressure_kPa'].values.astype(float)
        P_sw, S_sw = sw['Pressure_kPa'].values, sw[S.TARGET].values
 
        # np.interp clamps outside the sweep range rather than extrapolating.
        # Report how many LHS points that affects, since a clamped prediction
        # is a flat line and will look artificially good or bad.
        n_lo = int((P_lhs < P_sw.min()).sum())
        n_hi = int((P_lhs > P_sw.max()).sum())
        pred = np.interp(P_lhs, P_sw, S_sw)
 
        print(f'\n-- {S.label(cat)}')
        print(f'   sweep covers {P_sw.min():g}-{P_sw.max():g} kPa in {len(P_sw)} steps; '
              f'LHS covers {P_lhs.min():g}-{P_lhs.max():g} kPa')
        if n_lo or n_hi:
            print(f'   [NOTE] {n_lo} LHS point(s) below and {n_hi} above the sweep '
                  f'range were clamped to the\n          end value, not '
                  f'extrapolated. Those predictions are flat.')
 
        y = te[S.TARGET].values.astype(float)
        m = S.metrics(y, pred)
        m['model'] = 'B2'
        m['test_categories'] = S.NAME_TO_LETTER.get(cat, cat)
        m['n_clamped'] = n_lo + n_hi
        m['features'] = ''
        mets.append(m)
        S.print_metrics_table([m])
 
        for i in range(len(te)):
            rows.append({
                # B2 uses no fingerprint at all: it interpolates the target's
                # own sweep. The empty `features` column keeps the schema
                # identical to s02's predictions so s04 can concatenate them
                # without inventing a value.
                'model': 'B2', 'features': '', 'test_category': cat,
                'Sample': te['Sample'].iloc[i],
                **{c: te[c].iloc[i] for c in S.PRINT_FEATURES},
                'label_SF_mean': round(float(y[i]), 6),
                'pred_SF_mean': round(float(pred[i]), 6),
                'pred_std': '',
                'residual': round(float(pred[i] - y[i]), 6),
                'abs_error': round(float(abs(pred[i] - y[i])), 6),
            })
 
    if not rows:
        raise SystemExit('No sweep files matched. Nothing written.')
 
    tag = 'loco' if a.loco else 'test-' + '-'.join(
        S.NAME_TO_LETTER.get(c, c) for c in targets)
    pd.DataFrame(rows).to_csv(outdir / f'predictions_B2_{tag}.csv', sep=';', index=False)
    pd.DataFrame(mets).to_csv(outdir / f'metrics_B2_{tag}.csv', sep=';', index=False)
    print(f'\nWrote -> {outdir}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Baseline B2: target sweep only.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--sweep_dir', required=True,
                    help='Folder with the *_sf_summary_sweep.csv files')
    ap.add_argument('--outdir', default='results/03_b2')
    ap.add_argument('--test_categories', default=None)
    ap.add_argument('--loco', action='store_true')
    main(ap.parse_args())