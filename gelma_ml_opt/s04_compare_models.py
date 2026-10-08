
"""
s04_compare_models.py
---------------------
Collects the prediction logs from s02 and s03, puts every model on one table,
and applies the protocol's decision rules (section 7).
 
Also compares each model's RMSE against the within-category replicate noise
floor. A model cannot meaningfully beat the measurement, so an RMSE at or below
the noise floor means "as good as the data allows", not "better".
 
The `features` column
---------------------
s02 records the fingerprint columns it actually fitted on, and this script
carries that string into model_comparison.csv. s05 reads it back and refuses to
quietly print a recommendation from a different feature set than the one that
won here. Without it, renaming a fingerprint column silently changes what
M_full means between stages, and the comparison stops meaning anything. Older
prediction files with no `features` column still work; the field is left empty
and s05 skips the check.
 
Usage
-----
    python s04_compare_models.py --data_dir <folder> \
        --pred_dirs results/02_transfer results/03_b2 \
        [--outdir results/04_comparison]
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import sf_data as S
 
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
 
    frames = []
    for d in a.pred_dirs:
        for p in sorted(Path(d).glob('predictions*.csv')):
            frames.append(pd.read_csv(p, sep=S.sniff(p)))
            print(f'read {p}')
    if not frames:
        raise SystemExit('No predictions*.csv found in the given --pred_dirs. '
                         'Run s02 and s03 first.')
    pred = pd.concat(frames, ignore_index=True)
 
    df = S.load_categories(a.data_dir)
    noise = {}
    if S.TARGET_STD in df.columns:
        for c, sub in df.groupby('category'):
            # RMSE floor for predicting the MEAN of six images
            noise[c] = float(np.sqrt((sub[S.TARGET_STD] ** 2 / S.N_REPLICATES).mean()))
 
    print('\n' + '=' * 74)
    print('  MODEL COMPARISON, per held-out category')
    print('=' * 74)
    rows = []
    if 'features' not in pred.columns:
        pred['features'] = ''
        print('[NOTE] the prediction files carry no `features` column, so '
              'model_comparison.csv\n       will not record which fingerprint '
              'columns were fitted and s05 cannot\n       check consistency. '
              'Re-run s02 with the updated script to get it.')
    for (cat, model), g in pred.groupby(['test_category', 'model']):
        m = S.metrics(g['label_SF_mean'], g['pred_SF_mean'])
        feats = sorted({str(x) for x in g['features'].fillna('') if str(x)})
        if len(feats) > 1:
            print(f'[WARN] {cat} / {model}: prediction rows disagree about the '
                  f'feature list.\n       Found {feats}. Two different runs were '
                  f'mixed in --pred_dirs.')
        m.update(test_category=cat, model=model,
                 features=(feats[0] if feats else ''),
                 noise_floor_RMSE=round(noise.get(cat, float('nan')), 4))
        m['RMSE_over_noise'] = (round(m['RMSE'] / noise[cat], 2)
                                if cat in noise and noise[cat] > 0 else float('nan'))
        rows.append(m)
    res = pd.DataFrame(rows)
 
    for cat, g in res.groupby('test_category'):
        print(f'\n-- held out: {S.label(cat)}   '
              f'(replicate noise floor RMSE {noise.get(cat, float("nan")):.4f})')
        S.print_metrics_table(g.to_dict('records'))
 
    res.to_csv(outdir / 'model_comparison.csv', sep=';', index=False)
 
    # ------------------------------------------------------- decision rules
    print('\n' + '=' * 74)
    print('  DECISION (protocol section 7)')
    print('=' * 74)
    for cat, g in res.groupby('test_category'):
        by = {r['model']: r for r in g.to_dict('records')}
        best_m = min((k for k in by if k.startswith('M_')),
                     key=lambda k: by[k]['RMSE'], default=None)
        best_b1 = min((k for k in by if k.startswith('B1')),
                      key=lambda k: by[k]['RMSE'], default=None)
        print(f'\n{S.label(cat)}')
        if best_m is None:
            print('  no fingerprint model in the logs')
            continue
        rm = by[best_m]['RMSE']
        print(f'  best fingerprint model : {best_m} RMSE {rm:.4f}')
        if best_b1:
            r1 = by[best_b1]['RMSE']
            print(f'  vs B1 ({best_b1}) RMSE {r1:.4f}   -> fingerprint '
                  f'{"helps" if rm < r1 else "does NOT help"} '
                  f'({(r1 - rm) / r1 * 100:+.1f}%)')
        if 'B0' in by:
            print(f'  vs B0 RMSE {by["B0"]["RMSE"]:.4f}   -> '
                  f'{"beats" if rm < by["B0"]["RMSE"] else "does NOT beat"} '
                  f'the formulation-blind average')
        if 'B2' in by:
            r2 = by['B2']['RMSE']
            print(f'  vs B2 RMSE {r2:.4f} (target sweep only)')
            if rm < r2:
                print('    -> Model beats the sweep-only baseline. This supports '
                      'the transfer claim:\n       the model adds structure '
                      'beyond re-reading the target measurement.')
            else:
                print('    -> Model does NOT beat the sweep-only baseline. The '
                      'honest claim is that a\n       cheap pressure sweep '
                      'predicts the surface. Do not describe this as\n'
                      '       cross-formulation transfer.')
        else:
            print('  [WARN] B2 is missing. Run s03. Without it you cannot '
                  'distinguish transfer\n         from re-reading the target '
                  'sweep, which is the whole question.')
 
        nf = by[best_m].get('RMSE_over_noise', float('nan'))
        if np.isfinite(nf):
            if nf <= 1.0:
                print(f'  RMSE is {nf:.2f}x the replicate noise floor: at the '
                      f'measurement limit.')
            elif nf <= 2.0:
                print(f'  RMSE is {nf:.2f}x the replicate noise floor.')
            else:
                print(f'  RMSE is {nf:.2f}x the replicate noise floor: well above '
                      f'measurement error,\n  so there is real unexplained '
                      f'structure left.')
 
    fl = sorted({f for f in res['features'] if f})
    if len(fl) > 1:
        print('\n  [WARN] more than one fingerprint feature list appears in '
              'this comparison:')
        for f in fl:
            print(f'         {f}')
        print('         Models fitted on different features are not directly '
              'comparable on RMSE.')
 
    print('\n  Ridge vs GPR: if they are within noise of each other, report the '
          'simpler one.\n  A flexible GPR is not automatically the better '
          'result (protocol 5.3).')
    print(f'\nWrote -> {outdir / "model_comparison.csv"}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 4: compare all models.')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--pred_dirs', nargs='+',
                    default=['results/02_transfer', 'results/03_b2'])
    ap.add_argument('--outdir', default='results/04_comparison')
    main(ap.parse_args())