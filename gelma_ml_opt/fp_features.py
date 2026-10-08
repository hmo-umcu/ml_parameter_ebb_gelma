"""
fp_features.py
--------------
One place where every stage decides WHICH fingerprint columns it fits on.
 
Why this module exists
----------------------
The fingerprint columns gained robust variants: sf_at_p_max_three_mean is the
mean of the last three sweep wells rather than the single last well, and
tail_slope is the always-defined replacement for collapse_slope, which is blank
whenever a sweep never peaked. Tables built with the new names against an
sf_data.py that still lists the old ones failed in two different and equally
unhelpful ways:
 
  s05 stopped with a bare `Missing feature column(s): ['sf_at_p_max']`.
  s02 SILENTLY DROPPED the M_full variant, because it filters variants with
      `set(f) <= have`. The run then completed, the comparison table simply had
      no M_full row, and nothing said why. That is the worse of the two.
 
Both stages now resolve their feature list through here, report every
substitution, and record the names actually used in their output CSVs. If s02
and s05 disagree about what M_full means, the model that "won" the comparison
is not the model that gets printed, and only a recorded list can catch that.
 
This is a compatibility layer, not the fix. The fix is to make
sf_data.FINGERPRINT_ALL name the columns build_complete_table.py actually
writes:
 
    FINGERPRINT_ALL = ['onset_kPa', 'rise_slope', 'peak_sf', 'auc_norm',
                       'tail_slope', 'sf_at_p_max_three_mean']
 
Each alias pair below measures the SAME quantity, so a substitution changes
precision, not meaning.
"""
 
import pandas as pd
 
FEATURE_ALIASES = {
    'sf_at_p_max':            ['sf_at_p_max_three_mean'],
    'sf_at_p_max_three_mean': ['sf_at_p_max'],
    'peak_sf':                ['peak_three_sf_mean'],
    'peak_three_sf_mean':     ['peak_sf'],
    # tail_slope is the always-defined stand-in; collapse_slope is blank
    # whenever the sweep never peaked, so this substitution only ever runs in
    # the direction that gains data.
    'collapse_slope':         ['tail_slope'],
}
 
NON_FEATURE_COLS = {'category', 'Sample', 'Sample_ID', 'fold', 'n_images',
                    'onset_censored', 'truncated'}
 
 
def detect_fingerprint_columns(df, print_features=(), target='SF_mean',
                               target_std='SF_std'):
    """
    Numeric columns that are CONSTANT within every category.
 
    That is what a material descriptor is by definition, so this finds the
    fingerprint block without needing a declared list, and keeps working when
    sweep_fingerprint.py adds a column.
    """
    skip = set(print_features) | NON_FEATURE_COLS | {target, target_std}
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
 
 
def resolve_features(wanted, columns):
    """(used, substitutions, still_missing) for a requested feature list."""
    used, mapped, missing = [], [], []
    for f in wanted:
        if f in columns:
            used.append(f)
            continue
        alt = next((a for a in FEATURE_ALIASES.get(f, []) if a in columns), None)
        if alt is not None:
            used.append(alt)
            mapped.append((f, alt))
        else:
            missing.append(f)
    return used, mapped, missing
 
 
def report(mapped, missing, available, where, fatal_hint=True):
    """Print substitutions and unresolved names. Returns True if all resolved."""
    for old, new in mapped:
        print(f'[MAP] {where}: "{old}" is not in these tables; using "{new}". '
              f'Same quantity,\n      different precision '
              f'(fp_features.FEATURE_ALIASES).')
    if not missing:
        return True
    print(f'\n[ERROR] {where}: {len(missing)} requested feature(s) are not in '
          f'the tables and have\n        no known equivalent: {missing}')
    print('\n  Fingerprint columns these tables DO carry:')
    for c in available:
        print(f'    {c}')
    if fatal_hint:
        print('\n  This normally means sf_data.FINGERPRINT_ALL is out of step '
              'with the columns\n  build_complete_table.py wrote. Either update '
              'that list and re-run every stage,\n  or pass the columns '
              'explicitly to every stage:')
        print(f'    --features {",".join(available)}')
        print('\n  Whatever you choose, s02, s04 and s05 must be run the SAME '
              'way. If they are not,\n  the model that wins the comparison is '
              'not the model that gets printed.')
    return False
 
 
def budget_warning(n_features, n_train_categories, what='this fit'):
    """
    Warn when the fingerprint has more dimensions than the categories identify.
 
    A fingerprint value is one point per category, so however many rows the
    table has, leave-one-category-out fits from n_train distinct points in that
    space. A linear model in k features plus an intercept has k+1 parameters,
    so k = n_train-1 is already exactly determined and anything beyond that can
    reproduce every training category while saying nothing about a held-out one.
    """
    budget = max(1, n_train_categories - 1)
    if n_features <= budget:
        return None
    return (f'[WARN] {what}: {n_features} fingerprint features from '
            f'{n_train_categories} training categories.\n'
            f'       Fingerprints are constant within a category, so there are '
            f'only {n_train_categories}\n'
            f'       distinct points in that space and k = {budget} is already '
            f'exactly determined.\n'
            f'       Report the one-feature and PC1 variants alongside this one '
            f'(plan item C3).')
 
 
def resolve_spec(spec, df, declared_all, declared_onset, print_features,
                 target='SF_mean', target_std='SF_std'):
    """
    Turn a --features value into a concrete list.
 
    spec may be 'full', 'onset', 'auto', or a comma-separated list of columns.
    Returns (used, mapped, missing, available).
    """
    available = detect_fingerprint_columns(df, print_features, target, target_std)
    if spec == 'onset':
        wanted = list(declared_onset)
    elif spec == 'full':
        wanted = list(declared_all)
    elif spec == 'auto':
        wanted = list(available)
    else:
        wanted = [f.strip() for f in str(spec).split(',') if f.strip()]
    used, mapped, missing = resolve_features(wanted, set(df.columns))
    return used, mapped, missing, available
 