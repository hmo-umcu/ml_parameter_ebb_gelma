"""
s07_aim2_transfer_and_optimum.py
--------------------------------
Part D, steps D4 and D5. Retrospective, no new printing.
 
  D4  Can cell-free calibration predict cell-laden printing?
      Train a P/Speed/Z surrogate on the cell-free half of a pair ONLY, predict
      the cell-laden half at the same 32 settings. Compare against the identity
      baseline, which is simply using SF_free as the prediction of SF_laden.
      The identity baseline is the one that matters: a model that cannot beat
      "assume cells change nothing" has added nothing.
 
  D5  If you calibrate WITHOUT cells, how much do you lose?
      Take the optimum found from the cell-free data, evaluate it ON THE
      CELL-LADEN surface, and compare with what the cell-laden ink can achieve
      at its own optimum. The difference is the TRANSFER REGRET.
 
      This is deliberately not a comparison of two independently-fitted peaks.
      Two surfaces peaking in different places tells you nothing about whether
      one calibration works for the other, and if both peaks sit on the domain
      edge they agree for reasons that have nothing to do with the materials.
      The regret is a single number in SF units, comparable against replicate
      noise, and it names the two conditions to print.
 
RMSE is reported against the within-category replicate noise floor, because a
model cannot beat the measurement.
 
Usage
-----
    python s07_aim2_transfer_and_optimum.py --data_dir <folder> [--pairs A/E,B/F]
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
 
import sf_data as S
 
GRID = {'Pressure_kPa': (50.0, 130.0, 2.0),
        'NozzleSpeed_mms': (5.0, 15.0, 1.0),
        'Zoffset_mm': (0.1, 0.5, 0.05)}
 
 
def build_grid():
    axes = [np.round(np.arange(lo, hi + 1e-9, st), 6) for lo, hi, st in GRID.values()]
    mesh = np.meshgrid(*axes, indexing='ij')
    return pd.DataFrame({k: m.ravel() for k, m in zip(GRID, mesh)})
 
 
def fit(kind, sub, noise_mode, seed, n_restarts):
    X = sub[S.PRINT_FEATURES].values.astype(float)
    y = sub[S.TARGET].values.astype(float)
    sx = StandardScaler().fit(X)
    if kind == 'Ridge':
        m = Ridge(alpha=1.0).fit(sx.transform(X), y)
    else:
        alpha = S.alpha_from_std(sub, noise_mode)
        k = (ConstantKernel(1.0, (1e-3, 1e3))
             * Matern(length_scale=np.ones(X.shape[1]),
                      length_scale_bounds=(1e-2, 1e3), nu=2.5)
             + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))
        m = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                     n_restarts_optimizer=n_restarts,
                                     alpha=alpha if alpha is not None else 1e-10,
                                     random_state=seed).fit(sx.transform(X), y)
    return m, sx
 
 
def noise_floor(sub):
    if S.TARGET_STD not in sub.columns:
        return float('nan')
    return float(np.sqrt((sub[S.TARGET_STD] ** 2 / S.N_REPLICATES).mean()))
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
    present = set(df['category'].unique())
    pairs = ([tuple(t.strip().upper() for t in tok.split('/'))
              for tok in a.pairs.replace(';', ',').split(',')] if a.pairs
             else S.PAIRS)
    grid = build_grid()
 
    d4_rows, d5_rows = [], []
    print('=' * 74)
    print('  PART D: D4 cell-free -> cell-laden, D5 optimum movement')
    print('=' * 74)
 
    for fL, lL in pairs:
        fn, ln = S.LETTER_TO_NAME[fL], S.LETTER_TO_NAME[lL]
        if fn not in present or ln not in present:
            print(f'\n[SKIP] pair {fL}/{lL}: incomplete')
            continue
        free = df[df['category'] == fn].copy()
        lad = df[df['category'] == ln].copy()
        m = free.merge(lad, on=S.PRINT_FEATURES, suffixes=('_free', '_laden'))
 
        print(f'\n{"=" * 74}\npair {fL}/{lL}   {fn} -> {ln}   '
              f'({len(m)} matched)\n{"=" * 74}')
 
        # ---------------------------------------------------------- D4
        y_true = m[f'{S.TARGET}_laden'].values.astype(float)
        nf = noise_floor(lad)
        print(f'-- D4 predicting cell-laden   (replicate noise floor RMSE {nf:.4f})')
 
        ident = S.metrics(y_true, m[f'{S.TARGET}_free'].values.astype(float))
        ident['model'] = 'identity (SF_free as-is)'
        rows = [ident]
        for kind in ('Ridge', 'GPR'):
            mdl, sx = fit(kind, free, a.noise_mode, a.seed, a.n_restarts)
            p = mdl.predict(sx.transform(m[S.PRINT_FEATURES].values.astype(float)))
            r = S.metrics(y_true, p)
            r['model'] = f'trained on {fL} only ({kind})'
            rows.append(r)
        S.print_metrics_table(rows)
 
        best = min(rows, key=lambda r: r['RMSE'])
        if best['model'].startswith('identity'):
            print('   -> No model beats the identity baseline. On this pair, '
                  'assuming cells change\n      nothing is as good as any '
                  'surrogate trained cell-free.')
        else:
            print(f'   -> {best["model"]} beats identity '
                  f'({best["RMSE"]:.4f} vs {ident["RMSE"]:.4f}).')
        if np.isfinite(nf) and best['RMSE'] <= nf:
            print('   -> Best RMSE is at or below the replicate noise floor: '
                  'as good as the data allows.')
        for r in rows:
            d4_rows.append({'pair': f'{fL}/{lL}', **r,
                            'noise_floor_RMSE': round(nf, 4)})
 
        # ---------------------------------------------------------- D5
        # The question is NOT "where is each surface's peak", which invites the
        # useless comparison of two independently-fitted argmaxes. It is:
        #
        #   if you calibrate WITHOUT cells and use that optimum for the
        #   cell-laden ink, how much shape fidelity do you give up?
        #
        # So the cell-free optimum is evaluated ON THE CELL-LADEN SURFACE and
        # compared with the cell-laden ink's own optimum. That difference is the
        # transfer regret, and it maps directly onto two conditions to print.
        print('-- D5 does the cell-free optimum still work once cells are added?')
        m_free, sx_free = fit('GPR', free, a.noise_mode, a.seed, a.n_restarts)
        m_lad,  sx_lad  = fit('GPR', lad,  a.noise_mode, a.seed, a.n_restarts)
        Xg = grid[S.PRINT_FEATURES].values.astype(float)
 
        mu_free = m_free.predict(sx_free.transform(Xg))
        mu_lad, sd_lad = m_lad.predict(sx_lad.transform(Xg), return_std=True)
 
        k_free = int(np.argmax(mu_free))          # optimum from cell-free data
        k_lad = int(np.argmax(mu_lad))            # optimum from cell-laden data
        x_free = grid.iloc[k_free]
        x_lad = grid.iloc[k_lad]
 
        # the key number: laden performance AT the cell-free optimum
        sf_lad_at_free = float(mu_lad[k_free])
        sd_lad_at_free = float(sd_lad[k_free])
        sf_lad_at_lad = float(mu_lad[k_lad])
        regret = sf_lad_at_lad - sf_lad_at_free
        nf_lad = noise_floor(lad)
 
        def fmtx(r):
            return (f'P={r["Pressure_kPa"]:g} F={r["NozzleSpeed_mms"]:g} '
                    f'Z={r["Zoffset_mm"]:g}')
 
        print(f'   cell-free optimum   x*_free  = {fmtx(x_free)}')
        print(f'   cell-laden optimum  x*_laden = {fmtx(x_lad)}')
        print(f'   parameter distance: dP={x_lad["Pressure_kPa"] - x_free["Pressure_kPa"]:+g} kPa, '
              f'dSpeed={x_lad["NozzleSpeed_mms"] - x_free["NozzleSpeed_mms"]:+g} mm/s, '
              f'dZ={x_lad["Zoffset_mm"] - x_free["Zoffset_mm"]:+g} mm')
        print(f'   predicted SF on the CELL-LADEN surface:')
        print(f'     at x*_free  {sf_lad_at_free:.4f} +/- {sd_lad_at_free:.4f}'
              f'   <- what you get by calibrating without cells')
        print(f'     at x*_laden {sf_lad_at_lad:.4f}'
              f'   <- the best the cell-laden ink can do')
        print(f'   TRANSFER REGRET = {regret:.4f} SF'
              f'   ({regret / nf_lad:.1f}x the replicate noise floor {nf_lad:.4f})')
        if regret <= nf_lad:
            verdict = ('cell-free calibration transfers: the loss is within '
                       'measurement noise')
        elif regret <= 2 * nf_lad:
            verdict = 'small loss, 1-2x noise. Borderline; the print will decide'
        else:
            verdict = ('cell-free calibration does NOT transfer: you lose more '
                       'than 2x noise by skipping cells')
        print(f'   -> {verdict}')
 
        edge = []
        for c, (lo, hi, _) in GRID.items():
            if np.isclose(x_free[c], lo) or np.isclose(x_free[c], hi) \
               or np.isclose(x_lad[c], lo) or np.isclose(x_lad[c], hi):
                edge.append(c)
        if edge:
            print(f'   [NOTE] an optimum sits on the domain edge in {edge}. '
                  f'Both argmaxes are then\n          pinned by the box, so '
                  f'their agreement is partly forced by the design, not\n'
                  f'          by the materials behaving alike.')
 
        print(f'\n   TO TEST THIS PHYSICALLY, print in {ln} ink (cell-laden), '
              f'6 replicates each:')
        print(f'     cond 1  {fmtx(x_free)}   <- the cell-free recommendation')
        print(f'     cond 2  {fmtx(x_lad)}   <- the cell-laden recommendation')
        print(f'     If measured SF is the same within replicate error, cell-free '
              f'calibration is\n     sufficient for this formulation. If cond 2 '
              f'wins by more than noise, it is not.')
 
        d5_rows.append({
            'pair': f'{fL}/{lL}', 'cell_free': fn, 'cell_laden': ln,
            'x_free_P': x_free['Pressure_kPa'], 'x_free_Speed': x_free['NozzleSpeed_mms'],
            'x_free_Z': x_free['Zoffset_mm'],
            'x_laden_P': x_lad['Pressure_kPa'], 'x_laden_Speed': x_lad['NozzleSpeed_mms'],
            'x_laden_Z': x_lad['Zoffset_mm'],
            'dP': x_lad['Pressure_kPa'] - x_free['Pressure_kPa'],
            'dSpeed': x_lad['NozzleSpeed_mms'] - x_free['NozzleSpeed_mms'],
            'dZ': x_lad['Zoffset_mm'] - x_free['Zoffset_mm'],
            'SF_laden_at_x_free': round(sf_lad_at_free, 4),
            'SF_laden_at_x_free_sd': round(sd_lad_at_free, 4),
            'SF_laden_at_x_laden': round(sf_lad_at_lad, 4),
            'transfer_regret': round(regret, 4),
            'noise_floor': round(nf_lad, 4),
            'regret_over_noise': round(regret / nf_lad, 2) if nf_lad > 0 else '',
            'verdict': verdict,
            'optimum_on_domain_edge': ','.join(edge),
        })
 
    if not d4_rows:
        raise SystemExit('\nNo complete pairs to analyse.')
    pd.DataFrame(d4_rows).to_csv(outdir / 'd4_cellfree_to_cellladen.csv',
                                 sep=';', index=False)
    pd.DataFrame(d5_rows).to_csv(outdir / 'd5_optimum_movement.csv',
                                 sep=';', index=False)
    print(f'\nWrote -> {outdir}')
    print('\nD5 reports TRANSFER REGRET: how much cell-laden shape fidelity you '
          'lose by using\nthe cell-free optimum. It is not a comparison of two '
          'independent peaks, because\ntwo surfaces peaking in different places '
          'tells you nothing about whether one\ncalibration works for the other.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Part D steps D4 and D5.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--outdir', default='results/07_aim2_transfer')
    ap.add_argument('--pairs', default=None)
    ap.add_argument('--noise_mode', default='mean_of_6',
                    choices=['mean_of_6', 'raw', 'none'])
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())