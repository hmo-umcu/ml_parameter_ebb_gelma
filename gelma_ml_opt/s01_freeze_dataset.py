"""
s01_freeze_dataset.py
---------------------
Protocol stage 1: data freeze.
 
Reads every <category>_sf_complete_48well.csv, checks the design, writes
combined_6category_table.csv, and freezes the benchmark numbers for the target
category BEFORE any model touches them.
 
Run this first and read the output. If it reports a problem, later steps will
produce numbers that look fine and are not.
 
Usage
-----
    python s01_freeze_dataset.py --data_dir <folder> --outdir results/01_freeze \
        [--target F]
 
Outputs
-------
    combined_6category_table.csv   one row per category x LHS condition
    category_manifest.csv          per-category provenance and fingerprint
    fingerprint_audit.csv          per-feature completeness, spread and rank
    target_benchmark.csv           frozen mean/median/best for the target
 
The fingerprint audit, and why it belongs here
----------------------------------------------
The fingerprint columns are CONSTANT within a category, so however many rows the
combined table has, in fingerprint space there are only as many distinct points
as there are categories. Leave-one-category-out then trains on N-1 of them.
 
A linear model in k features plus an intercept has k+1 parameters, so k = N-2 is
already exactly determined and anything beyond that is underdetermined: it can
reproduce every training category perfectly and still say nothing about a
held-out one. With 6 categories that budget is 4; with 8 it is 6.
 
Measured on pressure_sweep_c6_fingerprints.csv, five complete features already
reached rank 5 across six categories, the maximum those six points allow, with
91% of the variance in two principal components. Adding more descriptors to a
fingerprint does not add more information about a new material; the sweep only
contains so much, and the categories only support so many dimensions. What adds
information is more CATEGORIES and a WIDER sweep range.
 
This audit runs at freeze time precisely because a fit on too many features
looks healthy right up until it is asked about a category it has not seen.
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import sf_data as S
 
 
def fingerprint_columns(df):
    """
    Fingerprint columns: declared in sf_data if available, otherwise detected
    as the numeric columns that are CONSTANT within every category, which is
    what a material descriptor is by definition.
    """
    declared = [c for c in getattr(S, 'FINGERPRINT_ALL', []) if c in df.columns]
    if declared:
        return declared
    skip = {'category', S.TARGET, getattr(S, 'TARGET_STD', 'SF_std'),
            'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm', 'Sample',
            'Sample_ID', 'fold', 'n_images'}
    out = []
    for c in df.columns:
        if c in skip:
            continue
        v = pd.to_numeric(df[c], errors='coerce')
        if v.isna().all():
            continue
        if df.assign(_v=v).groupby('category')['_v'].nunique(dropna=False).max() == 1:
            out.append(c)
    return out
 
 
def fingerprint_audit(df, outdir):
    """Completeness, spread, collinearity and rank of the fingerprint block."""
    cols = fingerprint_columns(df)
    cats = sorted(df['category'].unique())
    n = len(cats)
    print('\n' + '=' * 74)
    print('  FINGERPRINT AUDIT')
    print('=' * 74)
    if not cols:
        print('  No fingerprint columns found in the combined table.')
        return True
    M = df.groupby('category')[cols].first().apply(pd.to_numeric, errors='coerce')
    M = M.loc[cats]
 
    budget = max(1, n - 2)
    print(f'  {n} categories -> leave-one-category-out trains on {n - 1} distinct '
          f'point(s)\n  in fingerprint space -> a linear model supports at most '
          f'{budget} feature(s).')
 
    rows, complete = [], []
    print(f'\n  {"feature":<26}{"missing":>9}{"SD":>14}{"usable":>9}')
    for c in cols:
        v = M[c]
        miss = int(v.isna().sum())
        sd = float(v.std(ddof=1)) if v.notna().sum() > 1 else float('nan')
        ok = (miss == 0) and np.isfinite(sd) and sd > 0
        if ok:
            complete.append(c)
        rows.append({'feature': c, 'n_missing': miss, 'sd': round(sd, 6)
                     if np.isfinite(sd) else '', 'usable': 'yes' if ok else 'no'})
        print(f'  {c:<26}{miss:>9}{sd:>14.6g}{("yes" if ok else "NO"):>9}')
    dropped = [c for c in cols if c not in complete]
    if dropped:
        print(f'\n  {len(dropped)} feature(s) cannot enter a design matrix '
              f'(blank for at least one\n  category, or constant across all of '
              f'them): {dropped}')
 
    ok_flag = True
    if len(complete) >= 2:
        Z = ((M[complete] - M[complete].mean()) / M[complete].std(ddof=1)).values
        Z = Z - Z.mean(0)
        sv = np.linalg.svd(Z, compute_uv=False)
        var = sv ** 2 / np.sum(sv ** 2)
        cum = np.cumsum(var)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        rank = int(np.linalg.matrix_rank(Z))
        print(f'\n  Effective dimensionality of the {len(complete)} usable '
              f'feature(s):')
        print('    ' + '  '.join(f'PC{i+1} {v:.0%}' for i, v in enumerate(var)
                                 if v > 0.005))
        print(f'    rank {rank} (cannot exceed {n - 1}); {k90} component(s) carry '
              f'90% of the variance')
        C = np.corrcoef(Z, rowvar=False)
        pairs = [(abs(C[i, j]), C[i, j], complete[i], complete[j])
                 for i in range(len(complete)) for j in range(i + 1, len(complete))
                 if abs(C[i, j]) >= 0.90]
        if pairs:
            print(f'\n  Near-duplicate pairs (|r| >= 0.90), one of each is spare:')
            for _a, r, x, y in sorted(pairs, reverse=True):
                print(f'    {x:<26} vs {y:<26} r = {r:+.3f}')
        if k90 > budget:
            print(f'\n  [WARN] the usable features span {k90} real dimension(s) '
                  f'but only {budget} can be\n         identified from '
                  f'{n - 1} training categories. Do not fit on all of them.')
            ok_flag = False
        else:
            print(f'\n  {k90} real dimension(s) against a budget of {budget}: '
                  f'within what the design supports.')
        print('\n  Protocol C3: run the transfer step at three settings and '
              'report all three,\n  one feature, PC1, and the full set. If one '
              'matches the full set, report one.')
    pd.DataFrame(rows).to_csv(outdir / 'fingerprint_audit.csv', sep=';',
                              index=False)
    return ok_flag
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = S.load_categories(a.data_dir)
 
    print('=' * 74)
    print('  DATA FREEZE')
    print('=' * 74)
    ok, common = S.check_design(df)
 
    # ---------------------------------------------------------------- manifest
    rows = []
    for c in sorted(df['category'].unique(), key=lambda x: S.NAME_TO_LETTER.get(x, 'Z')):
        sub = df[df['category'] == c]
        L = S.NAME_TO_LETTER.get(c, '?')
        conc, dof, cell, temp = S.META.get(L, (None,) * 4)
        r = {'letter': L, 'category': c, 'conc_pct': conc, 'DoF': dof,
             'cell_state': cell, 'print_temp_C': temp, 'n_conditions': len(sub),
             'SF_mean_mean': round(sub[S.TARGET].mean(), 4),
             'SF_mean_median': round(sub[S.TARGET].median(), 4),
             'SF_mean_best': round(sub[S.TARGET].max(), 4)}
        for f in S.FINGERPRINT_ALL:
            if f in sub.columns:
                r[f] = sub[f].iloc[0]
        if S.TARGET_STD in sub.columns:
            r['SF_std_median'] = round(sub[S.TARGET_STD].median(), 4)
        rows.append(r)
    man = pd.DataFrame(rows)
    man.to_csv(outdir / 'category_manifest.csv', sep=';', index=False)
 
    print('\n-- category manifest --')
    print(man[['letter', 'category', 'conc_pct', 'DoF', 'cell_state',
               'n_conditions', 'SF_mean_mean', 'SF_mean_best']].to_string(index=False))
 
    # ------------------------------------------------ pairs available for Aim 2
    present = set(df['category'].unique())
    print('\n-- Aim 2 matched pairs --')
    for f, l in S.PAIRS:
        fn, ln = S.LETTER_TO_NAME[f], S.LETTER_TO_NAME[l]
        have = fn in present and ln in present
        print(f'  {f}/{l}  {fn:<18} vs {ln:<18} '
              f'{"available" if have else "INCOMPLETE"}')
 
    # ------------------------------------------------------- target benchmark
    tgt = S.resolve(a.target)
    if tgt not in present:
        print(f'\n[WARN] target {S.label(tgt)} is not in the data; '
              f'benchmark skipped.')
    else:
        sub = df[df['category'] == tgt].copy()
        k = sub[S.TARGET].idxmax()
        best = sub.loc[k]
        ref = sub[(sub['Pressure_kPa'] == 90) & (sub['NozzleSpeed_mms'] == 10)
                  & (np.isclose(sub['Zoffset_mm'], 0.2))]
        bench = [
            {'reference': 'mean across LHS conditions',
             'value': round(sub[S.TARGET].mean(), 4), 'params': '',
             'use': 'secondary descriptive'},
            {'reference': 'median across LHS conditions',
             'value': round(sub[S.TARGET].median(), 4), 'params': '',
             'use': 'secondary robust'},
            {'reference': 'best observed LHS SF',
             'value': round(float(best[S.TARGET]), 4),
             'params': f'{best["Pressure_kPa"]:g} kPa; '
                       f'{best["NozzleSpeed_mms"]:g} mm/s; Z={best["Zoffset_mm"]:g} mm',
             'use': 'PRIMARY optimization baseline, reprint fresh'},
        ]
        if len(ref):
            bench.append({'reference': 'standard matched reference 90/10/0.2',
                          'value': round(float(ref[S.TARGET].iloc[0]), 4),
                          'params': '90 kPa; 10 mm/s; Z=0.2 mm',
                          'use': 'batch consistency check'})
        bdf = pd.DataFrame(bench)
        bdf.insert(0, 'category', tgt)
        bdf.to_csv(outdir / 'target_benchmark.csv', sep=';', index=False)
        print(f'\n-- frozen benchmark for target {S.label(tgt)} --')
        print(bdf.to_string(index=False))
        print('\n  The primary baseline is the BEST observed LHS condition, '
              'reprinted fresh in the\n  validation batch. The 32-point mean is '
              'context only, not the bar to beat.')
 
        # the protocol's warning: is the historical best at the domain edge?
        if float(best['Pressure_kPa']) >= sub['Pressure_kPa'].max():
            print(f'\n  [NOTE] The best observed condition sits at the maximum '
                  f'sampled pressure\n  ({best["Pressure_kPa"]:g} kPa). Any '
                  f'recommendation is the best inside the tested box,\n  not a '
                  f'physical optimum. Say so when reporting.')
 
    fp_ok = fingerprint_audit(df, outdir)
 
    df.to_csv(outdir / 'combined_6category_table.csv', sep=';', index=False)
    print(f'\nWrote {len(df)} rows -> {outdir / "combined_6category_table.csv"}')
    if not ok:
        print('\n[ACTION] The warnings above affect cross-category validity. '
              'Resolve them before running s02.')
    if not fp_ok:
        print('\n[ACTION] The fingerprint carries more dimensions than these '
              'categories can\n         identify. Reduce the feature set, or '
              'run s02 at the three C3 settings\n         and report them '
              'together, before quoting any transfer result.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 1: freeze the dataset.')
    ap.add_argument('--data_dir', required=True,
                    help='Folder with the *_sf_complete_48well.csv files')
    ap.add_argument('--outdir', default='results/01_freeze')
    ap.add_argument('--target', default='F',
                    help='Headline target category, letter or name (default: F)')
    main(ap.parse_args())