
"""
s05_recommend_condition.py
--------------------------
Protocol stages 4 and 5.7: freeze the prospective recommendation for the target
category.
 
Fits the transfer model on the training categories only, attaches the TARGET's
fingerprint to a dense candidate grid inside the tested domain, predicts, and
writes the frozen recommendation.
 
Hard rules enforced here, from protocol 5.7:
  * The target's SF_mean values are never read. Only its fingerprint is used.
  * The grid stays inside the tested box (50-130 kPa, 5-15 mm/s, 0.1-0.5 mm).
    No extrapolation. The output is the best condition INSIDE the investigated
    domain, not a physical optimum.
  * The 32 historical LHS coordinates are excluded by default, so the
    recommendation is a genuinely untested point.
  * Selection is by highest predicted mean (5.7 item 12). Expected Improvement
    is for a sequential loop, not a one-shot recommendation.
 
The best observed LHS condition is scored under the same model and reported
alongside, so you can see whether the model would have found it.
 
Usage
-----
    python s05_recommend_condition.py --data_dir <folder> --target F \
        [--exclude_from_train B]          # C-F1 strict ablation
        [--features full|onset]
        [--include_historical]            # allow the 32 known coordinates
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
import fp_features as F
 
# Feature-name resolution lives in fp_features.py so that s02, s04 and s05 all
# make the same decision. sf_at_p_max_three_mean and tail_slope are the robust
# replacements for sf_at_p_max and collapse_slope; a table written with the new
# names against an sf_data.py listing the old ones used to fail here with a bare
# "Missing feature column(s)". See fp_features.py for the alias table.
 
 
# tested domain, protocol 5.7 item 9
GRID = {'Pressure_kPa': (50.0, 130.0, 1.0),
        'NozzleSpeed_mms': (5.0, 15.0, 0.5),
        'Zoffset_mm': (0.1, 0.5, 0.05)}
 
 
def build_grid():
    axes = [np.round(np.arange(lo, hi + 1e-9, st), 6) for lo, hi, st in GRID.values()]
    mesh = np.meshgrid(*axes, indexing='ij')
    return pd.DataFrame({k: m.ravel() for k, m in zip(GRID, mesh)})
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
    tgt = S.resolve(a.target)
    if tgt not in set(df['category']):
        raise SystemExit(f'{S.label(tgt)} not in the data.')
 
    train, test, excl = S.split_train_test(df, [tgt], a.exclude_from_train)
    tr = df[df['category'].isin(train)].copy()
    tg = df[df['category'] == tgt].copy()
 
    fp_feats, mapped, missing, available = F.resolve_spec(
        a.features, df, getattr(S, 'FINGERPRINT_ALL', []),
        getattr(S, 'FINGERPRINT_ONSET', ['onset_kPa']), S.PRINT_FEATURES,
        S.TARGET, S.TARGET_STD)
    if not F.report(mapped, missing, available, 's05'):
        if not a.allow_missing_features:
            print('\n  This script freezes a condition that gets printed and '
                  'consumes cells, so it\n  will not choose a different model '
                  'for you. Pass --allow_missing_features to\n  drop the '
                  'unresolved names and continue.')
            raise SystemExit(1)
        print(f'\n  [WARN] --allow_missing_features: continuing without '
              f'{missing}.')
    if not fp_feats:
        raise SystemExit('No fingerprint feature survived; nothing to fit on.')
 
    feats = list(S.PRINT_FEATURES) + fp_feats
    unused = [c for c in available if c not in fp_feats]
    if unused:
        print(f'[NOTE] fingerprint column(s) present but NOT used: {unused}')
 
    print('=' * 74)
    print(f'  FROZEN RECOMMENDATION for {S.label(tgt)}')
    print('=' * 74)
    print(f'  training categories : '
          f'{", ".join(S.NAME_TO_LETTER.get(c, c) for c in train)}')
    if excl:
        print(f'  withheld (ablation) : '
              f'{", ".join(S.NAME_TO_LETTER.get(c, c) for c in excl)}')
    print(f'  features            : {feats}')
    print(f'  fingerprint part    : {fp_feats}  ({len(fp_feats)} of '
          f'{len(available)} available)')
    w = F.budget_warning(len(fp_feats), len(train), 'this recommendation')
    if w:
        print('  ' + w.replace(chr(10), chr(10) + '  '))
 
    fp = tg[[f for f in feats if f not in S.PRINT_FEATURES]].drop_duplicates()
    if len(fp) != 1:
        raise SystemExit(f'Target fingerprint is not unique ({len(fp)} rows).')
    print(f'  target fingerprint  : {fp.iloc[0].to_dict()}')
 
    # Hard assertion, not a comment: raises if the target or an excluded
    # category reached the frame that gets fitted.
    S.audit_split(df, train, test, excl, fit_frame=tr)
    print(f'\n  Columns read FROM THE TARGET: {list(fp.columns)} (fingerprint)')
    print(f'                                {S.PRINT_FEATURES} (LHS coordinates, '
          f'to exclude them as candidates)')
    print(f'  NOT read from the target    : {S.TARGET}, {S.TARGET_STD}'
          + ('' if a.benchmark_csv else
             '  <- except the historical-best lookup below'))
 
    # -------------------------------------------- which model earned the job?
    # Reported, never enforced. Choosing a model that lost the comparison is a
    # legitimate thing to do on purpose.
    # Label the variant exactly as s02 does: s02 calls the full-fingerprint
    # variant M_full whatever list it was given, so calling it something else
    # here would look up a model name that does not exist in the comparison and
    # silently skip the consistency check. The `features` string is what
    # actually pins down which model this is.
    variant = 'M_onset' if a.features == 'onset' else 'M_full'
    want = f'{variant}:{"Ridge" if a.model == "ridge" else "GPR"}'
    heldout_rmse = None
    if a.comparison_csv:
        cmp_df = pd.read_csv(a.comparison_csv, sep=S.sniff(a.comparison_csv))
        sub = cmp_df[cmp_df['test_category'] == tgt]
        if sub.empty:
            raise SystemExit(f'{a.comparison_csv} has no rows for {tgt}.')
        # s02 records the features it fitted and s04 carries them here. A
        # mismatch means the model that won the comparison is not this model.
        # Compare against the row for the model actually SELECTED: B0 and B2 use
        # no fingerprint, so their features cell is empty and reading iloc[0]
        # would raise a mismatch against every run.
        if 'features' in cmp_df.columns:
            _rw = sub[sub['model'] == want]
            rec = ''
            if len(_rw):
                _v = _rw.iloc[0].get('features', '')
                rec = '' if pd.isna(_v) else str(_v).strip()
            if not rec:
                print(f'\n  [NOTE] {a.comparison_csv} records no feature list '
                      f'for {want}, so consistency\n  with s02 cannot be '
                      f'checked for this model.')
            elif rec != ','.join(feats):
                print(f'\n  [WARN] feature mismatch with {a.comparison_csv}:')
                print(f'         s04 recorded : {rec}')
                print(f'         s05 will use : {",".join(feats)}')
                print(f'         The winner reported below was selected on a '
                      f'different feature set, so\n         that selection does '
                      f'not transfer. Re-run s02 and s04 with these features.')
            else:
                print(f'\n  feature list matches {Path(a.comparison_csv).name}.')
        else:
            print(f'\n  [NOTE] {a.comparison_csv} has no `features` column, so '
                  f'consistency with s02\n  cannot be checked. Re-run s02 and '
                  f's04 with the updated scripts.')
        best_row = sub.loc[sub['RMSE'].idxmin()]
        row_want = sub[sub['model'] == want]
        heldout_rmse = float(row_want['RMSE'].iloc[0]) if len(row_want) else None
        print(f'\n  -- model selection ' + '-' * 52)
        print(f'     best in {Path(a.comparison_csv).name}: {best_row["model"]} '
              f'(RMSE {best_row["RMSE"]:.4f}, R2 {best_row["R2"]:+.3f})')
        print(f'     you selected           : {want}'
              + (f' (RMSE {heldout_rmse:.4f})' if heldout_rmse is not None else ''))
        if best_row['model'] != want:
            print(f'     [NOTE] {want} is not the best model for {S.label(tgt)}; '
                  f'{best_row["model"]} is. Continuing.')
        print('  ' + '-' * 70)
    else:
        print(f'\n  [NOTE] No --comparison_csv given, so {want} was not compared '
              f'against the\n  alternatives, and Ridge has no held-out RMSE to '
              f'report as uncertainty.')
 
    # ------------------------------------------------------------------ fit
    Xtr = tr[feats].values.astype(float)
    ytr = tr[S.TARGET].values.astype(float)
    sx = StandardScaler().fit(Xtr)
    if a.model == 'gpr':
        alpha = S.alpha_from_std(tr, a.noise_mode)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(Xtr.shape[1]),
                           length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         n_restarts_optimizer=a.n_restarts,
                                         alpha=alpha if alpha is not None else 1e-10,
                                         random_state=a.seed)
    else:
        model = Ridge(alpha=a.ridge_alpha)
    model.fit(sx.transform(Xtr), ytr)
    print(f'\n  model  : {want}')
    print(f'  fitted on {len(tr)} conditions from {len(train)} categories')
 
    # -------------------------------------------------------------- predict
    cand = build_grid()
    for f in feats:
        if f not in S.PRINT_FEATURES:
            cand[f] = fp.iloc[0][f]
    Xc = sx.transform(cand[feats].values.astype(float))
    if a.model == 'gpr':
        mu, sd = model.predict(Xc, return_std=True)
        unc_src = 'GPR posterior sd'
    else:
        mu = model.predict(Xc)
        # Ridge has no posterior. Using the held-out RMSE from the retrospective
        # comparison is the honest uncertainty here: it is the error the model
        # actually made on an unseen category, which is what this prediction is.
        sd = np.full(len(mu), heldout_rmse if heldout_rmse is not None else np.nan)
        unc_src = ('held-out RMSE from the comparison CSV'
                   if heldout_rmse is not None else 'unavailable (no --comparison_csv)')
    cand['pred_SF_mean'] = mu
    cand['pred_std'] = sd
 
    hist = set(map(tuple, tg[S.PRINT_FEATURES].round(6).values))
    cand['is_historical_LHS'] = [tuple(r) in hist for r in
                                 cand[S.PRINT_FEATURES].round(6).values]
    n_hist = int(cand['is_historical_LHS'].sum())
 
    pool = cand if a.include_historical else cand[~cand['is_historical_LHS']]
    if pool.empty:
        raise SystemExit('No candidates left after excluding historical points.')
    pool = pool.sort_values('pred_SF_mean', ascending=False).reset_index(drop=True)
    best = pool.iloc[0]
 
    print(f'\n  candidate grid      : {len(cand)} points, {n_hist} coincide with '
          f'historical LHS')
    print(f'  excluded historical : {"no" if a.include_historical else "yes"}')
 
    # ------------------------------- where does the known best rank under this model
    # Prefer the benchmark frozen by s01, so this script never opens the
    # target's SF_mean at all. Falling back to recomputing it is safe (the
    # recommendation above is already chosen and cannot be influenced by what
    # follows) but it is weaker provenance, so it is announced.
    if a.benchmark_csv:
        bm = pd.read_csv(a.benchmark_csv, sep=S.sniff(a.benchmark_csv))
        row = bm[bm['reference'].str.contains('best observed', case=False, na=False)]
        if row.empty:
            raise SystemExit(f'{a.benchmark_csv} has no "best observed" row. '
                             f'Re-run s01_freeze_dataset.py.')
        parts = [p.strip() for p in str(row.iloc[0]['params']).split(';')]
        hb = {'Pressure_kPa': float(parts[0].split()[0]),
              'NozzleSpeed_mms': float(parts[1].split()[0]),
              'Zoffset_mm': float(parts[2].split('=')[1].split()[0]),
              S.TARGET: float(row.iloc[0]['value'])}
        print(f'\n  historical best read from the frozen benchmark '
              f'({a.benchmark_csv}).')
    else:
        k = tg[S.TARGET].idxmax()
        hb = tg.loc[k]
        print('\n  [NOTE] No --benchmark_csv given, so the historical best was '
              'recomputed from the\n  target table. The recommendation above was '
              'already fixed and is unaffected, but\n  pass s01\'s '
              'target_benchmark.csv for cleaner provenance.')
    hb_row = pd.DataFrame([{**{c: hb[c] for c in S.PRINT_FEATURES},
                            **{f: fp.iloc[0][f] for f in feats
                               if f not in S.PRINT_FEATURES}}])
    Xh = sx.transform(hb_row[feats].values.astype(float))
    if a.model == 'gpr':
        hb_mu, hb_sd = model.predict(Xh, return_std=True)
    else:
        hb_mu = model.predict(Xh)
        hb_sd = np.array([heldout_rmse if heldout_rmse is not None else np.nan])
    rank = int((cand['pred_SF_mean'] > hb_mu[0]).sum()) + 1
 
    print('\n' + '-' * 74)
    print('  RECOMMENDATION (freeze this before printing)')
    print('-' * 74)
    print(f'  P*     = {best["Pressure_kPa"]:g} kPa')
    print(f'  Speed* = {best["NozzleSpeed_mms"]:g} mm/s')
    print(f'  Z*     = {best["Zoffset_mm"]:g} mm')
    print(f'  predicted SF_mean = {best["pred_SF_mean"]:.4f} '
          f'+/- {best["pred_std"]:.4f}   ({unc_src})')
    print(f'\n  historical best LHS: {hb["Pressure_kPa"]:g} kPa, '
          f'{hb["NozzleSpeed_mms"]:g} mm/s, Z={hb["Zoffset_mm"]:g} mm, '
          f'measured SF {hb[S.TARGET]:.4f}')
    print(f'  the same model scores it {hb_mu[0]:.4f} +/- {hb_sd[0]:.4f}, '
          f'rank {rank} of {len(cand)}')
 
    if best['Pressure_kPa'] >= GRID['Pressure_kPa'][1] - 1e-9:
        print('\n  [NOTE] The recommendation sits at the maximum grid pressure. '
              'The optimum is at or\n  beyond the domain edge, so report this as '
              'the best condition inside the tested\n  box, not a physical '
              'optimum.')
 
    out = {'category': tgt, 'model': want, 'uncertainty_source': unc_src,
           'heldout_RMSE': ('' if heldout_rmse is None else round(heldout_rmse, 6)),
           'features': ','.join(feats),
           'train_categories': ','.join(train),
           'excluded_from_train': ','.join(excl),
           'P_star': best['Pressure_kPa'], 'Speed_star': best['NozzleSpeed_mms'],
           'Z_star': best['Zoffset_mm'],
           'pred_SF_mean': round(float(best['pred_SF_mean']), 6),
           'pred_std': round(float(best['pred_std']), 6),
           'historical_best_SF': round(float(hb[S.TARGET]), 6),
           'historical_best_params': f'{hb["Pressure_kPa"]:g}/'
                                     f'{hb["NozzleSpeed_mms"]:g}/{hb["Zoffset_mm"]:g}',
           'historical_best_pred': round(float(hb_mu[0]), 6),
           'historical_best_rank_in_grid': rank,
           'grid_points': len(cand), 'excluded_historical': not a.include_historical,
           'noise_mode': a.noise_mode, 'seed': a.seed}
    pd.DataFrame([out]).to_csv(outdir / f'recommendation_{S.NAME_TO_LETTER.get(tgt, tgt)}.csv',
                               sep=';', index=False)
    cand.sort_values('pred_SF_mean', ascending=False).to_csv(
        outdir / f'candidate_ranking_{S.NAME_TO_LETTER.get(tgt, tgt)}.csv',
        sep=';', index=False)
 
    print(f'\n  Saved the full candidate ranking BEFORE printing, as required.')
    print(f'  -> {outdir}')
    print('\n  Print V1 (this recommendation) and V2 (the historical best) in the '
          'same fresh\n  batch, six replicates each, and do not retune V1 after '
          'seeing V2.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 5.7: freeze the recommendation.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--outdir', default='results/05_recommendation')
    ap.add_argument('--target', default='F')
    ap.add_argument('--exclude_from_train', default=None)
    ap.add_argument('--features', default='full',
                    help="'full' (sf_data.FINGERPRINT_ALL), 'onset', 'auto' "
                         "(every fingerprint column the tables carry), or an "
                         "explicit comma-separated list. Run s02 and s04 the "
                         "same way.")
    ap.add_argument('--allow_missing_features', action='store_true',
                    help='Drop unresolvable features and continue instead of '
                         'stopping. Changes the model relative to s02/s04.')
    ap.add_argument('--model', default='ridge', choices=['ridge', 'gpr'],
                    help='Model class for the recommendation. Default ridge.')
    ap.add_argument('--comparison_csv', default=None,
                    help="s04's model_comparison.csv. Used to report which "
                         "model won this fold and to supply Ridge's held-out "
                         "RMSE as the uncertainty.")
    ap.add_argument('--force', action='store_true',
                    help='Accepted for backward compatibility; has no effect')
    ap.add_argument('--ridge_alpha', type=float, default=1.0)
    ap.add_argument('--benchmark_csv', default=None,
                    help="s01's target_benchmark.csv. Supply it and this script "
                         "never reads the target's SF_mean at all.")
    ap.add_argument('--include_historical', action='store_true',
                    help='Allow the 32 known LHS coordinates as candidates')
    ap.add_argument('--noise_mode', default='mean_of_6',
                    choices=['mean_of_6', 'raw', 'none'])
    ap.add_argument('--n_restarts', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    main(ap.parse_args())