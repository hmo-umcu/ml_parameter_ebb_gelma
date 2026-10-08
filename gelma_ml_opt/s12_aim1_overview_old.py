"""
s12_aim1_overview.py
--------------------
AIM 1, all test categories in one figure.
 
For every test category three kinds of shape fidelity are put on the same axis:
 
    best SF from the PRESSURE SWEEP    19 wells, pressure only, 1 well each,
                                       fixed 10 mm/s and Z = 0.2 mm, no replicates
    best SF from the LHS SCREEN        32 conditions x 6 replicates = 192 wells
    SF from the VALIDATION PRINT       6 replicates at the model's recommendation
 
The two benchmarks are drawn as horizontal lines spanning each category slot, so
a validation marker sitting above both lines is immediately readable as "the
model found something better than either screen did".
 
Validation variants
-------------------
A category can have more than one validation print, differing in what the model
was allowed to train on. The variant label is read from the recommendation CSV's
own `excluded_from_train` field, not guessed:
 
    (empty)                      trained on all other
    the cell-free partner        cell-free partner held out
    anything else                that category held out
 
with one correction applied from the data rather than from the filename: if the
target's own cell-free partner was never collected, an empty field cannot mean
"trained on the partner too", so the run is labelled "cell-free partner held
out". GelMA 7.5% DoF 60 cell-laden is that case, because gelma_7_60 does not
exist in the dataset yet.
 
By default only the cell-free-partner-held-out prints are drawn, so each test
category shows three markers. --show_full_training adds the rest.
 
Everything for one category is drawn at the SAME x position as a marker, with no
error bars, distinguished by marker shape and colour rather than by horizontal
offset. When two markers coincide, either lower --marker_alpha to see through
them or set --jitter 0.06 to separate them slightly.
 
Figure appearance is fully parameterised: --marker_size, --marker_alpha,
--tick_size, --label_size, --legend_size, --legend_loc, --y_pad, --jitter,
--fig_width, --fig_height. The replicate standard deviations are NOT drawn; they
are in aim1_overview.csv along with every Welch test.
 
By default only the Gaussian process results are shown (--models). The Ridge
recommendation is the design-box corner and is the same point for every
category, so it says more about the model class than about any formulation.
 
Dropping runs
-------------
    --list_runs            print every run key and stop
    --exclude excl_H       drop runs whose folder name, category or variant
                           matches any of these substrings
Unlike the other scripts, --exclude here affects BOTH the table and the figure,
because this script exists to produce one curated overview. Everything dropped
is printed so nothing disappears silently.
 
For example, the 7.5% DoF 60 category has a run that holds out the 7.5% DoF 80
category. That is a different degree of functionalisation rather than a
cell-free / cell-laden contrast, so it is not comparable to the E and F variants
and is normally dropped with --exclude excl_H.
 
Usage
-----
    python s12_aim1_overview.py \
        --validation_csv validation_sf_summary.csv \
        --rec_dir results/05_recommendation \
        --combined_csv combined_6category_table.csv \
        --sweep_dir sweep \
        --exclude excl_H \
        --outdir results/12_aim1_overview
"""
 
import argparse
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
 
try:
    from scipy import stats as _st
except ImportError:
    _st = None
 
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
INK, INK_SOFT, GRIDC, SURFACE = '#1a1a19', '#6b6a63', '#e5e4df', '#fcfcfb'
LHS_C, SWEEP_C = '#4a3aa7', '#6b6a63'
 
LETTER_TO_NAME = {'A': 'gelma_10_60', 'B': 'gelma_10_80',
                  'C': 'gelma_7_60', 'D': 'gelma_7_80',
                  'E': 'cell_gelma_10_60', 'F': 'cell_gelma_10_80',
                  'G': 'cell_gelma_7_60', 'H': 'cell_gelma_7_80'}
 
PF = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
TARGET, TARGET_STD = 'SF_mean', 'SF_std'
 
FOLDER_RE = re.compile(r'^validation(?:_excl_(?P<excl>[A-H]))?_t_(?P<tgt>[A-H])'
                       r'(?:_(?P<rest>.*?))?_col(?P<col>\d+)$', re.IGNORECASE)
 
 
# --------------------------------------------------------------------------
 
def legend_label(cat):
    m = re.fullmatch(r'(cell_)?gelma_(\d+(?:\.\d+)?)_(\d+)', str(cat))
    if not m:
        return str(cat)
    cell, conc, dof = m.groups()
    if conc == '7':
        conc = '7.5'
    return f'GelMA {conc}% DoF {dof}  {"cell-laden" if cell else "cell-free"}'
 
 
def two_line_label(cat):
    lab = legend_label(cat)
    head, _, tail = lab.rpartition('  ')
    return f'{head}\n{tail}' if head else lab
 
 
def sniff(path):
    head = Path(path).read_text(errors='replace').split('\n', 1)[0]
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def blank_list(v):
    return [x.strip() for x in str(v).split(',')
            if x.strip() and x.strip().lower() not in ('nan', 'none')]
 
 
HELD_OUT = 'cell-free partner held out'
FULL_TRAIN = 'trained on all other'
 
 
def variant_label(target, excluded, available=()):
    """What the model was allowed to train on.
 
    A run with an empty `excluded_from_train` normally means the full training
    set. But if the target's own CELL-FREE PARTNER was never collected, it
    cannot have been in the training set either, so that run is in the same
    condition as an explicit hold-out and is labelled as such. GelMA 7.5% DoF 60
    cell-laden is this case: its cell-free partner gelma_7_60 does not exist in
    the dataset yet.
    """
    excl = blank_list(excluded)
    partner = target[len('cell_'):] if str(target).startswith('cell_') else None
    partner_missing = partner is not None and partner not in set(available)
    if not excl:
        return HELD_OUT if partner_missing else FULL_TRAIN
    if partner and set(excl) == {partner}:
        return HELD_OUT
    return f'{", ".join(legend_label(e) for e in excl)} held out'
 
 
def model_class(code):
    c = str(code)
    return ('Gaussian process' if 'GPR' in c
            else 'Ridge regression' if 'Ridge' in c else c)
 
 
def welch(m1, s1, n1, m2, s2, n2):
    diff = m1 - m2
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = np.sqrt(v1 + v2)
    if not np.isfinite(se) or se <= 0:
        return diff, np.nan
    t = diff / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    if _st is not None:
        return diff, float(2 * _st.t.sf(abs(t), df))
    import math
    return diff, float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
 
 
def pfmt(p):
    if not np.isfinite(p):
        return ''
    return '<0.0001' if p < 1e-4 else round(float(p), 4)
 
 
def pval(v):
    s = str(v).strip()
    if not s:
        return np.nan
    return 1e-5 if s.startswith('<') else float(s)
 
 
# --------------------------------------------------------------------------
 
def load_recs(rec_dir):
    rows = []
    for p in sorted(Path(rec_dir).glob('recommendation_*.csv')):
        d = pd.read_csv(p, sep=sniff(p))
        if d.empty:
            continue
        r = d.iloc[0].to_dict()
        r['rec_file'] = p.name
        rows.append(r)
    if not rows:
        raise SystemExit(f'No recommendation_*.csv in {rec_dir}')
    return pd.DataFrame(rows)
 
 
def benchmarks(combined, sweep_dir):
    lhs, swp = {}, {}
    for cat, sub in combined.groupby('category'):
        b = sub.loc[sub[TARGET].idxmax()]
        lhs[cat] = {'SF': float(b[TARGET]),
                    'SD': float(b[TARGET_STD]) if TARGET_STD in sub else np.nan,
                    'P': float(b['Pressure_kPa']), 'F': float(b['NozzleSpeed_mms']),
                    'Z': float(b['Zoffset_mm'])}
    if sweep_dir:
        for p in sorted(Path(sweep_dir).glob('*_sf_summary_sweep.csv')):
            cat = p.name[:-len('_sf_summary_sweep.csv')]
            d = pd.read_csv(p, sep=sniff(p)).dropna(subset=['Pressure_kPa', TARGET])
            if d.empty:
                continue
            b = d.loc[d[TARGET].idxmax()]
            swp[cat] = {'SF': float(b[TARGET]), 'P': float(b['Pressure_kPa'])}
    return lhs, swp
 
 
def build(val, recs, lhs, swp, targets, n, alpha):
    available = set(lhs)
    rows, skipped = [], []
    for _, v in val.iterrows():
        name = str(v['Folder_Name'])
        if name.lower().startswith('d5_'):
            continue
        m = FOLDER_RE.match(name)
        if not m:
            skipped.append((name, 'folder name does not match the validation pattern'))
            continue
        letter = m.group('tgt').upper()
        tgt = LETTER_TO_NAME.get(letter)
        if targets and letter not in targets:
            skipped.append((name, f'target {letter} not in --targets'))
            continue
        k = (round(float(v['Pressure_kPa']), 3),
             round(float(v['NozzleSpeed_mms']), 3), round(float(v['Zoffset_mm']), 3))
        cand = recs[recs['category'] == tgt]
        hit = cand[[(round(float(r['P_star']), 3), round(float(r['Speed_star']), 3),
                     round(float(r['Z_star']), 3)) == k for _, r in cand.iterrows()]] \
            if len(cand) else cand
        if len(hit) == 0:
            skipped.append((name, 'no frozen recommendation at this condition'))
            continue
        rest = m.group('rest') or ''
        if len(hit) > 1 and re.search(r'(^|_)(gpr|ridge)($|_)', rest, re.I):
            want = 'GPR' if re.search(r'(^|_)gpr($|_)', rest, re.I) else 'Ridge'
            nar = hit[hit['model'].astype(str).str.contains(want)]
            if len(nar):
                hit = nar
        r = hit.iloc[0]
 
        L, S = lhs.get(tgt, {}), swp.get(tgt)
        meas, sd = float(v[TARGET]), float(v[TARGET_STD])
        dL, pL = welch(meas, sd, n, L.get('SF', np.nan), L.get('SD', np.nan), n)
 
        rows.append({
            'target_letter': letter, 'category': tgt,
            'category_label': legend_label(tgt),
            'run': name,
            'variant': variant_label(tgt, r.get('excluded_from_train', ''),
                                     available),
            'partner_in_dataset': (
                (tgt[len('cell_'):] in available)
                if str(tgt).startswith('cell_') else ''),
            'model': model_class(r.get('model', '')),
            'P_kPa': k[0], 'Speed_mms': k[1], 'Z_mm': k[2],
            'validation_SF': round(meas, 4), 'validation_sd': round(sd, 4),
            'LHS_best_SF': round(L.get('SF', np.nan), 4),
            'LHS_best_sd': round(L.get('SD', np.nan), 4),
            'LHS_best_condition': (f'{L["P"]:g}/{L["F"]:g}/{L["Z"]:g}' if L else ''),
            'sweep_best_SF': round(S['SF'], 4) if S else '',
            'sweep_best_pressure_kPa': S['P'] if S else '',
            'delta_vs_LHS_best': round(dL, 4),
            'p_vs_LHS_best': pfmt(pL),
            'beats_LHS_best': ('yes, significantly' if np.isfinite(pL) and pL < alpha
                               and dL > 0 else
                               'no, significantly worse' if np.isfinite(pL)
                               and pL < alpha and dL < 0 else
                               'no significant difference'),
            'delta_vs_sweep_best': (round(meas - S['SF'], 4) if S else ''),
            'rec_file': r['rec_file'],
        })
    return pd.DataFrame(rows), skipped
 
 
def drop(df, exclude):
    if df.empty or not exclude:
        return df, []
    pats = [s.strip().lower() for s in exclude.split(',') if s.strip()]
    hay = (df['run'] + ' | ' + df['category_label'] + ' | ' + df['variant']
           + ' | ' + df['model']).str.lower()
    mask = hay.apply(lambda h: any(p in h for p in pats))
    return df[~mask], list(df[mask]['run'])
 
 
# --------------------------------------------------------------------------
 
def make_figure(d, outdir, alpha, order, a):
    """Markers only, no error bars. Everything for a category sits on the same
    x tick, distinguished by marker shape and colour."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
 
    cats = [c for c in order if c in set(d['category'])]
    cats += [c for c in d['category'].unique() if c not in cats]
 
    variants = sorted(d['variant'].unique())
    VSHAPE = ['s', 'D', 'P', 'X', 'h']
    vstyle = {v: (VSHAPE[i % len(VSHAPE)], SERIES[i % len(SERIES)])
              for i, v in enumerate(variants)}
 
    ms = a.marker_size
    fw = a.fig_width if a.fig_width > 0 else 1.85 * len(cats) + 2.4
    fh = a.fig_height if a.fig_height > 0 else 5.6
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
 
    for i, cat in enumerate(cats):
        sub = d[d['category'] == cat].sort_values(['variant', 'model'])
        r0 = sub.iloc[0]
 
        if str(r0['sweep_best_SF']).strip():
            ax.plot(i, float(r0['sweep_best_SF']), marker='v', ms=ms, ls='none',
                    color=SWEEP_C, markerfacecolor=SWEEP_C, markeredgecolor=INK,
                    markeredgewidth=.8, alpha=a.marker_alpha, zorder=4)
 
        ax.plot(i, r0['LHS_best_SF'], marker='o', ms=ms, ls='none',
                color=LHS_C, markerfacecolor=LHS_C, markeredgecolor=INK,
                markeredgewidth=.8, alpha=a.marker_alpha, zorder=5)
 
        k = len(sub)
        xs = (np.full(k, float(i)) if a.jitter <= 0 or k == 1
              else np.linspace(i - a.jitter, i + a.jitter, k))
        for j, (x, (_, r)) in enumerate(zip(xs, sub.iterrows())):
            shape, colour = vstyle[r['variant']]
            ax.plot(x, r['validation_SF'], marker=shape, ms=ms - 2 * j,
                    ls='none', color=colour, markerfacecolor=colour,
                    markeredgecolor=INK, markeredgewidth=.8,
                    alpha=a.marker_alpha, zorder=6 + j)
            if a.show_significance and pval(r['p_vs_LHS_best']) < alpha:
                ax.annotate('*', (x, r['validation_SF']),
                            textcoords='offset points',
                            xytext=(ms * 0.75, -1), fontsize=a.label_size,
                            color=colour, ha='left', va='center')
 
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([two_line_label(c) for c in cats], fontsize=a.tick_size,
                       color=INK)
    ax.set_xlim(-0.45, len(cats) - 0.55)
 
    sb = pd.to_numeric(d['sweep_best_SF'], errors='coerce')
    vals = list(d['validation_SF']) + list(d['LHS_best_SF']) + \
        list(sb.dropna())
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = max(0.015, a.y_pad * (hi - lo))
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax.set_ylabel('Shape fidelity', fontsize=a.label_size, color=INK)
    ax.grid(True, axis='y', color=GRIDC, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRIDC)
    ax.tick_params(colors=INK_SOFT, labelsize=a.tick_size)
 
    handles = [
        Line2D([], [], marker='v', ls='none', ms=ms * .8, color=SWEEP_C,
               markerfacecolor=SWEEP_C, markeredgecolor=INK,
               label='Pressure sweep best SF'),
        Line2D([], [], marker='o', ls='none', ms=ms * .8, color=LHS_C,
               markerfacecolor=LHS_C, markeredgecolor=INK,
               label='LHS sampling best SF'),
    ]
    for v in variants:
        shape, colour = vstyle[v]
        handles.append(Line2D([], [], marker=shape, ls='none', ms=ms * .8,
                              color=colour, markerfacecolor=colour,
                              markeredgecolor=INK,
                              label=f'ML validation SF, {v}'))
    leg = ax.legend(handles=handles, frameon=True, framealpha=.92,
                    edgecolor=GRIDC, facecolor=SURFACE,
                    fontsize=a.legend_size, loc=a.legend_loc)
    for t in leg.get_texts():
        t.set_color(INK)
 
    fig.tight_layout()
    p = outdir / 'fig_aim1_overview.png'
    fig.savefig(p, facecolor=SURFACE)
    fig.savefig(p.with_suffix('.eps'), format='eps', facecolor=SURFACE,
            dpi=900)
    plt.close(fig)
    return p
 
 
# --------------------------------------------------------------------------
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if _st is None:
        print('[WARN] scipy missing: p-values use a normal approximation.')
 
    val = pd.read_csv(a.validation_csv, sep=sniff(a.validation_csv))
    need = ['Folder_Name'] + PF + [TARGET, TARGET_STD]
    missing = [c for c in need if c not in val.columns]
    if missing:
        raise SystemExit(f'{a.validation_csv} is missing {missing}')
    combined = pd.read_csv(a.combined_csv, sep=sniff(a.combined_csv))
    recs = load_recs(a.rec_dir)
    lhs, swp = benchmarks(combined, a.sweep_dir)
 
    targets = ([t.strip().upper() for t in a.targets.split(',') if t.strip()]
               if a.targets else None)
    d, skipped = build(val, recs, lhs, swp, targets, a.n_replicates, a.alpha)
    if d.empty:
        raise SystemExit('No validation prints matched a frozen recommendation.')
 
    if a.list_runs:
        print('Run keys for --exclude:\n')
        for _, r in d.iterrows():
            print(f'  {r["run"]}')
            print(f'      {r["category_label"]}  |  {r["variant"]}  |  {r["model"]}')
        return
 
    d, dropped = drop(d, a.exclude)
    dropped_full, dropped_model = [], []
    if not a.show_full_training:
        dropped_full = list(d[d['variant'] == FULL_TRAIN]['run'])
        d = d[d['variant'] != FULL_TRAIN]
    if a.models != 'all':
        want = 'Gaussian process' if a.models == 'gpr' else 'Ridge regression'
        dropped_model = list(d[d['model'] != want]['run'])
        d = d[d['model'] == want]
    if d.empty:
        raise SystemExit('The filters removed every run.')
 
    order = [LETTER_TO_NAME[t] for t in (targets or [])
             if LETTER_TO_NAME.get(t) in set(d['category'])]
 
    print('=' * 78)
    print('  AIM 1 OVERVIEW: three SF values per test category')
    print('=' * 78)
    for cat in (order or list(d['category'].unique())):
        sub = d[d['category'] == cat]
        r0 = sub.iloc[0]
        print(f'\n  {r0["category_label"]}')
        sb = (f'{float(r0["sweep_best_SF"]):.4f} at '
              f'{r0["sweep_best_pressure_kPa"]:g} kPa'
              if str(r0['sweep_best_SF']).strip() else 'no sweep file')
        print(f'    best SF, pressure sweep : {sb}')
        print(f'    best SF, LHS screen     : {r0["LHS_best_SF"]:.4f} +/- '
              f'{r0["LHS_best_sd"]:.4f} at {r0["LHS_best_condition"]}')
        for _, r in sub.sort_values(['variant', 'model']).iterrows():
            print(f'    validation print        : {r["validation_SF"]:.4f} +/- '
                  f'{r["validation_sd"]:.4f} at '
                  f'{r["P_kPa"]:g}/{r["Speed_mms"]:g}/{r["Z_mm"]:g}')
            print(f'        {r["variant"]}, {r["model"]}')
            print(f'        vs LHS best  {r["delta_vs_LHS_best"]:+.4f}  '
                  f'p={r["p_vs_LHS_best"]}  ({r["beats_LHS_best"]})')
            if str(r['delta_vs_sweep_best']).strip():
                print(f'        vs sweep best {float(r["delta_vs_sweep_best"]):+.4f}'
                      f'  (no test, the sweep has 1 well per pressure)')
 
    for runs, why in ((dropped, f'--exclude "{a.exclude}"'),
                      (dropped_full, 'trained on the full set '
                                     '(--show_full_training to include)'),
                      (dropped_model, f'--models {a.models}')):
        if runs:
            print(f'\n  Dropped, {why}:')
            for r in runs:
                print(f'    {r}')
    if skipped:
        print('\n  Not evaluated:')
        for name, why in skipped:
            print(f'    {name}: {why}')
 
    d.to_csv(outdir / 'aim1_overview.csv', sep=';', index=False)
    fig = None
    try:
        fig = make_figure(d, outdir, a.alpha, order, a)
    except ImportError:
        print('  [WARN] matplotlib missing, no figure written')
    print(f'\n  Wrote aim1_overview.csv'
          + (f' and {fig.name}' if fig else '') + f'\n  -> {outdir}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='One figure covering every Aim 1 test category.')
    ap.add_argument('--validation_csv', required=True)
    ap.add_argument('--rec_dir', required=True)
    ap.add_argument('--combined_csv', required=True)
    ap.add_argument('--sweep_dir', default=None)
    ap.add_argument('--outdir', default='results/12_aim1_overview')
    ap.add_argument('--targets', default='E,F,G',
                    help='Test categories to include, in plot order '
                         '(default E,F,G)')
    ap.add_argument('--exclude', default=None,
                    help='Comma-separated substrings; drop matching runs from '
                         'BOTH the table and the figure. e.g. "excl_H"')
    ap.add_argument('--models', default='gpr', choices=['gpr', 'ridge', 'all'],
                    help='Which model class to show. Default gpr, because the '
                         'Ridge recommendation is the design-box corner and is '
                         'the same point for every category.')
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='Horizontal spread within a category, in x units. '
                         'Default 0, i.e. everything on the same tick. Raise it '
                         'to about 0.06 if two markers coincide.')
    ap.add_argument('--tick_size', type=float, default=15,
                    help='Tick label font size (default 15)')
    ap.add_argument('--label_size', type=float, default=17,
                    help='Axis label font size (default 17)')
    ap.add_argument('--legend_size', type=float, default=12,
                    help='Legend font size (default 12)')
    ap.add_argument('--marker_size', type=float, default=17,
                    help='Marker size in points (default 17)')
    ap.add_argument('--marker_alpha', type=float, default=0.85,
                    help='Marker opacity, 0 to 1. Lower it to see markers that '
                         'sit on top of each other (default 0.85)')
    ap.add_argument('--y_pad', type=float, default=0.20,
                    help='Vertical margin as a fraction of the data range. '
                         'Lower zooms in further (default 0.20)')
    ap.add_argument('--fig_width', type=float, default=0,
                    help='Figure width in inches. 0 = auto, tight around the '
                         'ticks')
    ap.add_argument('--fig_height', type=float, default=0,
                    help='Figure height in inches. 0 = auto')
    ap.add_argument('--legend_loc', default='best',
                    help="Legend position, e.g. 'best', 'lower right', "
                         "'upper left'")
    ap.add_argument('--show_full_training', action='store_true',
                    help='Also show validation prints whose model was trained on '
                         'every other category. Off by default, so each test '
                         'category shows three markers: sweep best, LHS best, '
                         'and the cell-free-partner-held-out validation print.')
    ap.add_argument('--show_significance', action='store_true',
                    help='Draw * beside a validation marker whose Welch test '
                         'against the LHS best is significant. Off by default; '
                         'the p-values are always in aim1_overview.csv.')
    ap.add_argument('--list_runs', action='store_true')
    ap.add_argument('--n_replicates', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.05)
    main(ap.parse_args())