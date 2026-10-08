"""
sf_data.py
----------
Shared loading, category handling and metrics for the Part C / Part D pipeline.
Not a pipeline step. Every s0*.py script imports from here so the data
contract is defined in exactly one place.
 
Category letters follow the protocol's convention:
 
    A gelma_10_60        10%   DoF60  cell-free
    B gelma_10_80        10%   DoF80  cell-free
    C gelma_7_60         7.5%  DoF60  cell-free    (deferred)
    D gelma_7_80         7.5%  DoF80  cell-free    (deferred)
    E cell_gelma_10_60   10%   DoF60  cell-laden
    F cell_gelma_10_80   10%   DoF80  cell-laden   (headline target)
    G cell_gelma_7_60    7.5%  DoF60  cell-laden
    H cell_gelma_7_80    7.5%  DoF80  cell-laden
 
Anywhere a category is named you may use either the letter or the folder name.
 
One condition is one row
------------------------
The 32 LHS settings are the design points. SF_mean is the target and SF_std is
the replicate dispersion of the six images behind it. The six images are NOT
six design points; treating them as such would overstate n by 6x.
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
 
LETTER_TO_NAME = {
    'A': 'gelma_10_60',      'B': 'gelma_10_80',
    'C': 'gelma_7_60',       'D': 'gelma_7_80',
    'E': 'cell_gelma_10_60', 'F': 'cell_gelma_10_80',
    'G': 'cell_gelma_7_60',  'H': 'cell_gelma_7_80',
}
NAME_TO_LETTER = {v: k for k, v in LETTER_TO_NAME.items()}
 
META = {   # concentration %, DoF, cell state, print temperature C
    'A': (10.0, 60, 'cell-free',  16), 'B': (10.0, 80, 'cell-free',  21),
    'C': (7.5,  60, 'cell-free',  16), 'D': (7.5,  80, 'cell-free',  21),
    'E': (10.0, 60, 'cell-laden', 16), 'F': (10.0, 80, 'cell-laden', 21),
    'G': (7.5,  60, 'cell-laden', 16), 'H': (7.5,  80, 'cell-laden', 21),
}
# Aim 2 matched pairs: cell-free -> cell-laden, same concentration and DoF
PAIRS = [('A', 'E'), ('B', 'F'), ('C', 'G'), ('D', 'H')]
 
PRINT_FEATURES = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
FINGERPRINT_ALL = ['onset_kPa', 'rise_slope', 'peak_sf', 'sf_at_p_max']
FINGERPRINT_ONSET = ['onset_kPa']
TARGET, TARGET_STD = 'SF_mean', 'SF_std'
SUFFIX = '_sf_complete_48well.csv'
 
N_REPLICATES = 6      # images behind each SF_mean, used for the SF_std**2/n rule
 
 
def resolve(cat):
    """'F', 'f', or 'cell_gelma_10_80' -> 'cell_gelma_10_80'."""
    c = str(cat).strip()
    if c.upper() in LETTER_TO_NAME:
        return LETTER_TO_NAME[c.upper()]
    if c in NAME_TO_LETTER:
        return c
    raise SystemExit(f'Unknown category {cat!r}. Use a letter A-H or one of '
                     f'{sorted(NAME_TO_LETTER)}')
 
 
def resolve_list(s):
    """Accepts None, a comma/semicolon string, or an iterable of names/letters."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return []
    if isinstance(s, (list, tuple, set)):
        items = list(s)
    else:
        items = str(s).replace(';', ',').split(',')
    return [resolve(x) for x in items if str(x).strip()]
 
 
def label(name):
    """'cell_gelma_10_80' -> 'F (cell_gelma_10_80)'."""
    return f'{NAME_TO_LETTER.get(name, "?")} ({name})'
 
 
def sniff(path):
    with open(path, newline='') as f:
        head = f.readline()
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def load_categories(data_dir, suffix=SUFFIX):
    """Read every <category>{suffix} in data_dir into one long DataFrame."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f'*{suffix}'))
    if not files:
        raise SystemExit(f'No *{suffix} files in {data_dir}')
    frames = []
    for p in files:
        df = pd.read_csv(p, sep=sniff(p))
        cat = p.name[:-len(suffix)]
        if 'category' not in df.columns:
            df['category'] = cat
        elif set(df['category'].unique()) != {cat}:
            print(f'[WARN] {p.name}: category column says '
                  f'{sorted(df["category"].unique())}, filename says {cat}. '
                  f'Using the column.')
        # the complete tables use "Sample"; older tables used "Sample_ID"
        if 'Sample' not in df.columns and 'Sample_ID' in df.columns:
            df = df.rename(columns={'Sample_ID': 'Sample'})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
 
    need = PRINT_FEATURES + [TARGET, 'category', 'Sample']
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise SystemExit(f'Missing required column(s): {missing}\n'
                         f'Found: {list(out.columns)}')
    numeric = PRINT_FEATURES + FINGERPRINT_ALL + [TARGET, TARGET_STD, 'Sample']
    for c in numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out
 
 
def check_design(df, verbose=True):
    """Report anything that would make cross-category comparison invalid."""
    ok = True
    cats = sorted(df['category'].unique(), key=lambda c: NAME_TO_LETTER.get(c, 'Z'))
    if verbose:
        print(f'Categories: {len(cats)}')
        for c in cats:
            sub = df[df['category'] == c]
            fp = sub[[f for f in FINGERPRINT_ALL if f in sub.columns]].drop_duplicates()
            print(f'  {label(c):<28} {len(sub):>3} rows  '
                  f'SF {sub[TARGET].min():.4f}-{sub[TARGET].max():.4f}  '
                  f'fingerprint rows: {len(fp)}')
            if len(fp) != 1:
                print(f'    [WARN] fingerprint is not constant within this '
                      f'category ({len(fp)} distinct rows)')
                ok = False
 
    # every category must share the same 32 LHS coordinates, or the paired and
    # matched-condition analyses silently compare different settings
    keys = {c: set(map(tuple, df.loc[df['category'] == c, PRINT_FEATURES].values))
            for c in cats}
    common = set.intersection(*keys.values()) if keys else set()
    for c in cats:
        extra = keys[c] - common
        if extra:
            print(f'  [WARN] {label(c)} has {len(extra)} LHS coordinate(s) not '
                  f'shared by all categories')
            ok = False
    if verbose:
        print(f'  shared LHS coordinates across all categories: {len(common)}')
    return ok, sorted(common)
 
 
def split_train_test(df, test_categories, exclude_from_train=()):
    """
    Category-level split. test_categories are evaluated on; exclude_from_train
    are additionally withheld from training but not evaluated (the protocol's
    C-F1 ablation withholds B while testing on F).
    """
    test = list(dict.fromkeys(resolve_list(test_categories)))
    excl = list(dict.fromkeys(resolve_list(exclude_from_train)))
    present = set(df['category'].unique())
    for c in test + excl:
        if c not in present:
            raise SystemExit(f'{label(c)} is not in the data. Present: '
                             f'{sorted(present)}')
    train = [c for c in present if c not in set(test) | set(excl)]
    if not train:
        raise SystemExit('No training categories left after the split.')
    return sorted(train), test, excl
 
 
def audit_split(df, train, test, excl=(), fit_frame=None, verbose=True):
    """
    Hard guarantee that no held-out category reaches the training frame.
 
    Raises rather than warns. The recommendation from s05 goes to a printer and
    consumes cells, so a silent leak is expensive in a way a warning is not.
 
    `fit_frame` is the DataFrame actually handed to .fit(). If given, its
    categories are checked directly, which catches a leak introduced by an
    edit further down the script rather than only one in the split itself.
    """
    train, test, excl = list(train), list(test), list(excl)
    bad = sorted(set(train) & (set(test) | set(excl)))
    if bad:
        raise SystemExit(f'LEAK: {bad} are both trained on and held out.')
 
    if fit_frame is not None:
        got = set(fit_frame['category'].unique())
        leaked = sorted(got & (set(test) | set(excl)))
        if leaked:
            raise SystemExit(
                f'LEAK: the frame passed to .fit() contains held-out '
                f'categor(ies) {leaked}. Expected only {sorted(train)}.')
        unexpected = sorted(got - set(train))
        if unexpected:
            raise SystemExit(f'LEAK: unexpected categor(ies) in the fit frame: '
                             f'{unexpected}')
 
    if verbose:
        print('\n  -- training-set audit ' + '-' * 50)
        roles = ([(c, 'TRAIN') for c in sorted(train)]
                 + [(c, 'TEST (scored, never fitted)') for c in test]
                 + [(c, 'EXCLUDED (not fitted, not scored)') for c in excl])
        for c, role in roles:
            n = int((df['category'] == c).sum())
            print(f'     {label(c):<28} {n:>3} rows   {role}')
        if fit_frame is not None:
            print(f'     fit frame: {len(fit_frame)} rows from '
                  f'{fit_frame["category"].nunique()} categor(ies)')
        print('  ' + '-' * 70)
    return True
 
 
def alpha_from_std(sub, mode='mean_of_6'):
    """
    Per-point GPR noise variance from the replicate spread (protocol 5.6).
 
      'mean_of_6' : SF_std**2 / 6, the variance of the MEAN of six images.
                    This is the right scale when the target is SF_mean.
      'raw'       : SF_std**2, the spread of a single image.
      'none'      : None, so a single fitted WhiteKernel is used instead.
 
    This is a documented sensitivity choice, not a silent redefinition of
    SF_std. Report which one was used.
    """
    if mode == 'none' or TARGET_STD not in sub.columns:
        return None
    s = pd.to_numeric(sub[TARGET_STD], errors='coerce').fillna(0.0).values
    v = s ** 2 / (N_REPLICATES if mode == 'mean_of_6' else 1.0)
    return np.maximum(v, 1e-10)
 
 
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        'n':    int(len(y_true)),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'MAE':  float(np.abs(err).mean()),
        # MAPE is deliberately absent: SF values near zero make it unstable,
        # and the protocol says it must not be a headline metric.
        'R2':   float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        'bias': float(err.mean()),
    }
 
 
def print_metrics_table(rows, key='model'):
    if not rows:
        print('  (nothing to report)')
        return
    w = max(len(str(r[key])) for r in rows) + 2
    print(f'{key:<{w}}{"n":>5}{"RMSE":>10}{"MAE":>10}{"R2":>9}{"bias":>10}')
    for r in sorted(rows, key=lambda r: r['RMSE']):
        print(f'{str(r[key]):<{w}}{r["n"]:>5}{r["RMSE"]:>10.4f}'
              f'{r["MAE"]:>10.4f}{r["R2"]:>9.3f}{r["bias"]:>+10.4f}')