"""
s06_aim2_paired_shift.py
------------------------
Part D, steps D1 to D3. Retrospective, no new printing.
 
For each matched pair (cell-free -> cell-laden, same concentration, DoF and
temperature), at every one of the 32 identical LHS settings:
 
  D1  DeltaSF_i = SF_laden,i - SF_free,i, with median, bootstrap CI and a
      Wilcoxon signed-rank test. Paired, because the two rows share the exact
      printer settings.
  D2  Fit SF_laden = b0 + b1 * SF_free to classify the shift:
        b1 ~ 1, b0 != 0   constant offset
        b1 != 1           scaling with baseline print quality
        poor fit          parameter-dependent, go to D3
  D3  Regress DeltaSF on P, Speed, Z to test whether the cell effect depends on
      the printing parameters.
 
Pairs are formed on the LHS coordinate, not on row order, so a category with a
missing or reordered condition cannot silently misalign.
 
Usage
-----
    python s06_aim2_paired_shift.py --data_dir <folder> [--pairs A/E,B/F]
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import sf_data as S
 
try:
    from scipy import stats
except ImportError:
    stats = None
 
 
def boot_ci(x, n=10000, seed=0, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))
 
 
def ols(X, y):
    A = np.column_stack([X, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float('nan')
    return b, r2, resid
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
    present = set(df['category'].unique())
 
    if a.pairs:
        want = []
        for tok in a.pairs.replace(';', ',').split(','):
            f, l = tok.split('/')
            want.append((f.strip().upper(), l.strip().upper()))
    else:
        want = S.PAIRS
 
    print('=' * 74)
    print('  PART D: MATCHED CELL-STATE SHIFT')
    print('=' * 74)
 
    all_rows, summary = [], []
    for fL, lL in want:
        fn, ln = S.LETTER_TO_NAME[fL], S.LETTER_TO_NAME[lL]
        if fn not in present or ln not in present:
            print(f'\n[SKIP] pair {fL}/{lL}: '
                  f'{"missing " + fn if fn not in present else ""}'
                  f'{" and " if fn not in present and ln not in present else ""}'
                  f'{"missing " + ln if ln not in present else ""}')
            continue
 
        free = df[df['category'] == fn].copy()
        lad = df[df['category'] == ln].copy()
        key = S.PRINT_FEATURES
        m = free.merge(lad, on=key, suffixes=('_free', '_laden'))
        if len(m) != len(free) or len(m) != len(lad):
            print(f'\n[WARN] pair {fL}/{lL}: matched {len(m)} of '
                  f'{len(free)}/{len(lad)} conditions. Only matched settings '
                  f'are compared.')
 
        d = (m[f'{S.TARGET}_laden'] - m[f'{S.TARGET}_free']).values
        lo, hi = boot_ci(d, seed=a.seed)
 
        print(f'\n{"=" * 74}\npair {fL}/{lL}   {fn}  ->  {ln}   '
              f'({len(m)} matched conditions)\n{"=" * 74}')
        print('-- D1 pointwise shift')
        print(f'   mean DeltaSF   {d.mean():+.4f}')
        print(f'   median DeltaSF {np.median(d):+.4f}')
        print(f'   95% bootstrap CI of the mean: [{lo:+.4f}, {hi:+.4f}]')
        print(f'   sign: {int((d > 0).sum())} up, {int((d < 0).sum())} down')
        crosses = lo <= 0 <= hi
        print(f'   CI {"includes" if crosses else "excludes"} zero -> '
              f'{"no detectable overall shift" if crosses else "a real shift"}')
        if stats is not None:
            w = stats.wilcoxon(d)
            print(f'   Wilcoxon signed-rank p = {w.pvalue:.4g} (secondary summary)')
        else:
            w = None
            print('   [skip] scipy not installed, no Wilcoxon test')
 
        print('-- D2 form of the shift')
        b, r2, _ = ols(m[f'{S.TARGET}_free'].values.reshape(-1, 1),
                       m[f'{S.TARGET}_laden'].values)
        b1, b0 = float(b[0]), float(b[1])
        print(f'   SF_laden = {b0:+.4f} + {b1:.4f} * SF_free   (R2 = {r2:.3f})')
        if r2 < 0.5:
            form = 'parameter-dependent (poor fit, go to D3)'
        elif abs(b1 - 1) < 0.15:
            form = ('constant offset' if abs(b0) > 0.02
                    else 'no meaningful change')
        else:
            form = f'scaling, slope {b1:.2f}'
        print(f'   interpretation: {form}')
 
        print('-- D3 does the cell effect depend on P, Speed, Z?')
        Xp = m[S.PRINT_FEATURES].values.astype(float)
        bb, r2d, _ = ols(Xp, d)
        for nme, coef in zip(S.PRINT_FEATURES, bb[:-1]):
            print(f'   d(DeltaSF)/d({nme}) = {coef:+.5f}')
        print(f'   R2 of DeltaSF on the printing parameters = {r2d:.3f}')
        if r2d > 0.3:
            print('   -> the cell effect is parameter-dependent, which is a '
                  'result in itself.')
        else:
            print('   -> little parameter dependence detected in this pair.')
 
        for i in range(len(m)):
            all_rows.append({
                'pair': f'{fL}/{lL}', 'cell_free': fn, 'cell_laden': ln,
                **{c: m[c].iloc[i] for c in S.PRINT_FEATURES},
                'SF_free': round(float(m[f'{S.TARGET}_free'].iloc[i]), 6),
                'SF_laden': round(float(m[f'{S.TARGET}_laden'].iloc[i]), 6),
                'DeltaSF': round(float(d[i]), 6),
            })
        summary.append({
            'pair': f'{fL}/{lL}', 'n_matched': len(m),
            'mean_DeltaSF': round(float(d.mean()), 4),
            'median_DeltaSF': round(float(np.median(d)), 4),
            'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
            'ci_excludes_zero': not crosses,
            'wilcoxon_p': (round(float(w.pvalue), 6) if w is not None else ''),
            'slope_b1': round(b1, 4), 'intercept_b0': round(b0, 4),
            'R2_laden_on_free': round(r2, 4), 'shift_form': form,
            'R2_delta_on_params': round(r2d, 4),
        })
 
    if not summary:
        raise SystemExit('\nNo complete pairs. With C and D missing, only A/E '
                         'and B/F can be analysed.')
 
    pd.DataFrame(all_rows).to_csv(outdir / 'paired_delta_sf.csv', sep=';', index=False)
    pd.DataFrame(summary).to_csv(outdir / 'paired_summary.csv', sep=';', index=False)
    print('\n' + '=' * 74)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'\nWrote -> {outdir}')
    print('\nBefore reading DeltaSF biologically, run the segmentation control '
          '(protocol 6.7):\n if cell-laden segmentation IoU is systematically '
          'lower, part of DeltaSF is a\n measurement artefact rather than a cell '
          'effect.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Part D steps D1-D3.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--outdir', default='results/06_aim2_paired')
    ap.add_argument('--pairs', default=None,
                    help='e.g. "A/E,B/F". Default: all four, skipping incomplete')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())