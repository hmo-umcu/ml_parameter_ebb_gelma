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
 
Figure labels follow the manuscript (figure 9): "Ramp best" (highest
single-well SF of the 23-well pressure ramp), "LHS best" (highest six-well mean
of the 32 LHS conditions) and "Recommended condition" (six-well mean of the
prospective print). The x axis shows the formulation and the bioink letter, as
in table 4, for example "10% DoF 60 (E)". --xtick_style legacy restores the
old "GelMA 10% DoF 60 cell-laden" labels.

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
 
NAME_TO_LETTER = {v: k for k, v in LETTER_TO_NAME.items()}

# Legend labels, worded as in the manuscript text and the caption of figure 9
LAB_RAMP_BEST = 'Ramp best'
LAB_LHS_BEST = 'LHS best'
LAB_RECOMMENDED = 'Recommended condition'

PF = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
TARGET, TARGET_STD = 'SF_mean', 'SF_std'
 
# Current naming:  validation_<T>[_excl-<X>]_col<N>   e.g. validation_E_excl-A_col1
# Older naming:    validation[_excl_<X>]_t_<T>[_rest]_col<N>, still accepted.
FOLDER_RE_NEW = re.compile(r'^validation_(?P<tgt>[A-H])(?:_excl-(?P<excl>[A-H]))?'
                           r'(?:_(?P<rest>.*?))?_col(?P<col>\d+)$', re.IGNORECASE)
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


def wrapped_label(cat, width=0):
    """Category name over several short lines.

    An x tick label as wide as 'GelMA 10% DoF 60' next to two others needs a very
    wide figure, and a wide figure means small text once it is placed in the
    manuscript. Wrapping the name instead keeps the figure narrow and the font
    large. width 0 leaves the name on one line.
    """
    import textwrap
    lab = legend_label(cat)
    head, _, tail = lab.rpartition('  ')
    if not head:
        head, tail = lab, ''
    lines = textwrap.wrap(head, width) if width and width > 0 else [head]
    return '\n'.join(lines + ([tail] if tail else []))
 
 
def manuscript_label(cat):
    """Formulation and bioink letter, as in table 4: '10% DoF 60' over '(E)'."""
    m = re.fullmatch(r'(cell_)?gelma_(\d+(?:\.\d+)?)_(\d+)', str(cat))
    if not m:
        return str(cat)
    _, conc, dof = m.groups()
    if conc == '7':
        conc = '7.5'
    letter = NAME_TO_LETTER.get(str(cat), '')
    return f'{conc}% DoF {dof}' + (f'\n({letter})' if letter else '')


def sniff(path):
    head = Path(path).read_text(errors='replace').split('\n', 1)[0]
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def blank_list(v):
    return [x.strip() for x in str(v).split(',')
            if x.strip() and x.strip().lower() not in ('nan', 'none')]
 
 
HELD_OUT = ''
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
    excl = [LETTER_TO_NAME.get(e.upper(), e) for e in blank_list(excluded)]
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
        m = FOLDER_RE_NEW.match(name) or FOLDER_RE.match(name)
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
        # 1st choice: the recommendation file named after this validation run,
        #   recommendation_<Folder_Name>.csv
        by_name = recs[recs['rec_file'].str.lower()
                       == f'recommendation_{name}.csv'.lower()]
        if len(by_name):
            hit = by_name
            r0 = hit.iloc[0]
            kr = (round(float(r0['P_star']), 3), round(float(r0['Speed_star']), 3),
                  round(float(r0['Z_star']), 3))
            if kr != k:
                print(f'  [WARN] {name}: printed {k} but {r0["rec_file"]} '
                      f'recommends {kr}. Using the file matched by name.')
        else:
            # fallback: match by condition within the target category
            cand = recs[recs['category'] == tgt]
            hit = cand[[(round(float(r['P_star']), 3), round(float(r['Speed_star']), 3),
                         round(float(r['Z_star']), 3)) == k for _, r in cand.iterrows()]] \
                if len(cand) else cand
        if len(hit) == 0:
            skipped.append((name, f'no recommendation_{name}.csv in --rec_dir and '
                                  f'no frozen recommendation at this condition'))
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
            'variant': variant_label(
                tgt,
                (r.get('excluded_from_train', '')
                 if blank_list(r.get('excluded_from_train', ''))
                 else (m.group('excl') or '')),
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
 
# --------------------------------------------------------------------------
# figure sizing and legend placement
#
# These figures are printed small in a manuscript, so the fonts are set for the
# printed size rather than for the screen. What makes text look small on the page
# is not the font size but the ratio between the font and the figure WIDTH: a
# wide figure scaled down to one column shrinks every letter with it. So the
# figure is kept narrow, it is never widened past --max_fig_width, and a legend
# that would not fit is wrapped and re-columned instead of stretching the figure.
#
#   _needed_size   measures every label, tick, title and legend box and returns
#                  the size that holds them. Matplotlib does not grow the canvas
#                  for a label longer than the figure, it just cuts the text, and
#                  Figure.get_tightbbox clips to the canvas so it does not report
#                  the overflow either. The same function widens the figure when
#                  neighbouring x tick labels would touch.
#   _render        builds, measures and rebuilds until nothing overflows, within
#                  the maximum size. Text is never shrunk.
#   _legend        wraps long labels at --legend_wrap characters, picks the most
#                  columns that still fit the figure width, and places the legend
#                  where it hides the least: 'inside' at --legend_loc, 'above' or
#                  'right' outside the data area, 'auto' inside unless it covers
#                  the data.
#   _save          writes .png, .pdf and .eps side by side. The pdf and eps are
#                  vector, so LaTeX can scale them without softening the text.
# --------------------------------------------------------------------------

def _new_fig(a, size):
    """One axis, constrained layout where available."""
    import matplotlib.pyplot as plt
    try:
        fig, ax = plt.subplots(figsize=size, dpi=a.dpi, layout='constrained')
        fig.get_layout_engine().set(w_pad=a.pad_pt / 72, h_pad=a.pad_pt / 72)
    except (TypeError, AttributeError):
        fig, ax = plt.subplots(figsize=size, dpi=a.dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _needed_size(fig, a):
    from matplotlib.transforms import Bbox
    try:
        if fig.get_layout_engine() is None:
            fig.tight_layout()
    except AttributeError:
        fig.tight_layout()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes, xticks = [], []
    for ax in fig.axes:
        boxes.append(ax.get_window_extent(r))
        arts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        arts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for art in arts:
            if art.get_text():
                boxes.append(art.get_window_extent(r))
        xticks.append(sorted([t.get_window_extent(r) for t in ax.get_xticklabels()
                              if t.get_text()], key=lambda b: b.x0))
        for art in ax.texts:
            boxes.append(art.get_window_extent(r))
        if ax.get_legend() is not None:
            boxes.append(ax.get_legend().get_window_extent(r))
    for leg in fig.legends:
        boxes.append(leg.get_window_extent(r))
    b = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    w, h = fig.get_size_inches()
    pad = 2 * a.pad_pt / 72
    need_w = w + max(0.0, -b.x0) + max(0.0, b.x1 - w) + pad
    need_h = h + max(0.0, -b.y0) + max(0.0, b.y1 - h) + pad
    gap = 0.07 * fig.dpi        # clear space wanted between x tick labels
    for row in xticks:
        over = sum(max(0.0, row[i - 1].x1 + gap - row[i].x0)
                   for i in range(1, len(row)))
        need_w = max(need_w, w + over / fig.dpi)
    return need_w, need_h


def _render(build, a, size):
    """Draw, measure, and grow the figure until no text is clipped."""
    import matplotlib.pyplot as plt
    size = (min(size[0], a.max_fig_width), min(size[1], a.max_fig_height))
    start = size
    fig = build(size)
    if getattr(a, 'autofit', True):
        for _ in range(4):
            w, h = _needed_size(fig, a)
            w = min(max(size[0], w), a.max_fig_width)
            h = min(max(size[1], h), a.max_fig_height)
            if w <= size[0] + 0.01 and h <= size[1] + 0.01:
                break
            plt.close(fig)
            size = (round(w, 2), round(h, 2))
            fig = build(size)
    w, h = fig.get_size_inches()
    if (round(w, 2), round(h, 2)) != (round(start[0], 2), round(start[1], 2)):
        print(f'  [NOTE] figure resized to {w:.2f} x {h:.2f} in so the labels '
              f'and the legend fit at this font size '
              f'(limit {a.max_fig_width:g} x {a.max_fig_height:g}; '
              f'raise --max_fig_width if a label is still tight)')
    return fig


def _legend_covers_data(ax, leg):
    """Fraction of the legend box drawn on top of data."""
    from matplotlib.transforms import Bbox
    r = ax.figure.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    area = max(lb.width * lb.height, 1e-9)
    cov = 0.0
    for art in list(ax.lines) + list(ax.patches) + list(ax.collections):
        try:
            bb = art.get_window_extent(r)
        except Exception:
            continue
        ov = Bbox.intersection(lb, bb)
        if ov is not None:
            cov += max(0.0, ov.width) * max(0.0, ov.height)
    return min(1.0, cov / area)


def _legend(fig, ax, handles, labels, a, style):
    """Wrap, column and place the legend so it neither covers the data nor
    widens the figure.

    Outside the axes the legend is attached to the FIGURE, not to the axes.
    An axes legend anchored above is laid out inside the axes' own space, so a
    wide legend squeezes the plot into a narrow column; a figure legend at
    'outside upper center' reserves a band of its own and leaves the axes full
    width. The number of columns is then chosen as the largest that still fits
    the figure width, so the legend never forces the figure wider.
    """
    import textwrap
    if a.legend_wrap > 0:
        labels = [textwrap.fill(str(l), a.legend_wrap) for l in labels]

    def put(where, ncol):
        if where == 'above':
            try:
                return fig.legend(handles, labels, loc='outside upper center',
                                  ncol=ncol, **style)
            except (ValueError, TypeError):
                return ax.legend(handles, labels, loc='lower center',
                                 bbox_to_anchor=(0.5, 1.02), ncol=ncol, **style)
        if where == 'right':
            try:
                return fig.legend(handles, labels, loc='outside right upper',
                                  ncol=1, **style)
            except (ValueError, TypeError):
                return ax.legend(handles, labels, loc='upper left',
                                 bbox_to_anchor=(1.02, 1.0), ncol=1, **style)
        return ax.legend(handles, labels, loc=a.legend_loc, ncol=ncol, **style)

    def fitted(where):
        """Most columns whose legend still fits across the figure."""
        if a.legend_ncol > 0:
            return put(where, a.legend_ncol)
        room = (fig.get_size_inches()[0] - 2 * a.pad_pt / 72) * fig.dpi
        leg = None
        for ncol in range(min(len(labels), 3), 0, -1):
            if leg is not None:
                leg.remove()
            leg = put(where, ncol)
            fig.canvas.draw()
            if leg.get_window_extent(fig.canvas.get_renderer()).width <= room:
                break
        return leg

    where = a.legend_placement
    leg = fitted('inside' if where == 'auto' else where)
    if where == 'auto':
        fig.canvas.draw()
        if _legend_covers_data(ax, leg) > a.legend_overlap_tol:
            leg.remove()
            leg = fitted('above')
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _save(fig, path, a):
    """png for viewing, pdf and eps for the manuscript."""
    from pathlib import Path as _P
    path = _P(path)
    fig.savefig(path, facecolor=SURFACE)
    fig.savefig(path.with_suffix('.pdf'), facecolor=SURFACE)
    fig.savefig(path.with_suffix('.eps'), format='eps', facecolor=SURFACE,
                dpi=getattr(a, 'eps_dpi', 900))


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

    def build(size):
        fig, ax = _new_fig(a, size)
        for i, cat in enumerate(cats):
            sub = d[d['category'] == cat].sort_values(['variant', 'model'])
            r0 = sub.iloc[0]

            if str(r0['sweep_best_SF']).strip():
                ax.plot(i, float(r0['sweep_best_SF']), marker='v', ms=ms,
                        ls='none', color=SWEEP_C, markerfacecolor=SWEEP_C,
                        markeredgecolor=INK, markeredgewidth=.8,
                        alpha=a.marker_alpha, zorder=4)

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
        ticklab = (manuscript_label if a.xtick_style == 'manuscript'
                   else (lambda c: wrapped_label(c, a.xtick_wrap)))
        ax.set_xticklabels([ticklab(c) for c in cats],
                           fontsize=a.tick_size, color=INK)
        ax.set_xlim(-0.45, len(cats) - 0.55)

        sb = pd.to_numeric(d['sweep_best_SF'], errors='coerce')
        vals = list(d['validation_SF']) + list(d['LHS_best_SF']) + list(sb.dropna())
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
                   label=LAB_RAMP_BEST),
            Line2D([], [], marker='o', ls='none', ms=ms * .8, color=LHS_C,
                   markerfacecolor=LHS_C, markeredgecolor=INK,
                   label=LAB_LHS_BEST),
        ]
        for v in variants:
            shape, colour = vstyle[v]
            handles.append(Line2D([], [], marker=shape, ls='none', ms=ms * .8,
                                  color=colour, markerfacecolor=colour,
                                  markeredgecolor=INK,
                                  label=(LAB_RECOMMENDED if v == HELD_OUT
                                         else f'{LAB_RECOMMENDED}, {v}')))
        _legend(fig, ax, handles, [h.get_label() for h in handles], a,
                dict(frameon=True, framealpha=.92, edgecolor=GRIDC,
                     facecolor=SURFACE, fontsize=a.legend_size))
        return fig

    # narrow on purpose: the printed text size is set by the ratio of the
    # font to the figure width, not by the font size alone
    fw = a.fig_width if a.fig_width > 0 else 1.45 * len(cats) + 1.3
    fh = a.fig_height if a.fig_height > 0 else 4.2
    fig = _render(build, a, (fw, fh))
    p = outdir / 'fig_aim1_overview.png'
    _save(fig, p, a)
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
    ap.add_argument('--targets', default='E,F,G,H',
                    help='Test categories to include, in plot order '
                         '(default E,F,G,H)')
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
    ap.add_argument('--tick_size', type=float, default=16,
                    help='Tick label font size (default 16)')
    ap.add_argument('--label_size', type=float, default=18,
                    help='Axis label font size (default 18)')
    ap.add_argument('--legend_size', type=float, default=14,
                    help='Legend font size (default 14)')
    ap.add_argument('--xtick_style', default='manuscript',
                    choices=['manuscript', 'legacy'],
                    help="x tick labels. 'manuscript' (default): formulation and "
                         "bioink letter as in table 4, e.g. '10%% DoF 60 (E)'. "
                         "'legacy': the old 'GelMA 10%% DoF 60 cell-laden'.")
    ap.add_argument('--xtick_wrap', type=int, default=11,
                    help='Wrap each category name on the x axis at this many '
                         'characters, so the names stay narrow and the figure '
                         'can stay small (0 keeps one line, default 11)')
    ap.add_argument('--marker_size', type=float, default=22,
                    help='Marker size in points (default 22)')
    ap.add_argument('--marker_alpha', type=float, default=0.80,
                    help='Marker opacity, 0 to 1. Lower it to see markers that '
                         'sit on top of each other (default 0.80)')
    ap.add_argument('--y_pad', type=float, default=0.20,
                    help='Vertical margin as a fraction of the data range. '
                         'Lower zooms in further (default 0.20)')
    ap.add_argument('--fig_width', type=float, default=0,
                    help='Starting figure width in inches. 0 = auto. It is '
                         'grown only as far as --max_fig_width if something '
                         'does not fit')
    ap.add_argument('--fig_height', type=float, default=0,
                    help='Figure height in inches. 0 = auto')
    ap.add_argument('--legend_loc', default='best',
                    help="Legend position when it is placed INSIDE the axes, "
                         "e.g. 'best', 'lower right', 'upper left'")
    ap.add_argument('--legend_placement', default='above',
                    choices=['above', 'inside', 'right', 'auto'],
                    help="Where the legend goes. Default 'above': these labels "
                         "are long, and a legend inside either covers markers or "
                         "eats the plot area. 'auto' tries inside first and "
                         "moves it above only if it covers data.")
    ap.add_argument('--legend_ncol', type=int, default=0,
                    help='Legend columns. 0 picks the most columns that still '
                         'fit across the figure')
    ap.add_argument('--legend_wrap', type=int, default=30,
                    help='Wrap legend labels at this many characters, so a long '
                         'label makes the legend taller rather than the figure '
                         'wider (0 disables)')
    ap.add_argument('--legend_overlap_tol', type=float, default=0.02,
                    help='With --legend_placement auto, how much of the legend '
                         'box may sit on the data before it moves out')
    ap.add_argument('--dpi', type=int, default=300,
                    help='Saved PNG resolution (default 300). The pdf and eps '
                         'written beside it are vector.')
    ap.add_argument('--eps_dpi', type=int, default=900)
    ap.add_argument('--pad_pt', type=float, default=2.0,
                    help='Padding in points between the content and the figure '
                         'edge (default 2)')
    ap.add_argument('--max_fig_width', type=float, default=7.2,
                    help='The figure is never grown wider than this, in inches. '
                         'A wide figure scaled down to one manuscript column '
                         'makes all of its text small (default 7.2)')
    ap.add_argument('--max_fig_height', type=float, default=6.0,
                    help='The figure is never grown taller than this (default 6)')
    ap.add_argument('--no_autofit', dest='autofit', action='store_false',
                    help='Keep the figure size exactly as given, even if that '
                         'clips a label')
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