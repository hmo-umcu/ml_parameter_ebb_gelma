"""
s11_aim2_evaluate.py
--------------------
AIM 2. Can a formulation be calibrated WITHOUT cells and the result reused once
cells are in?
 
What was printed
----------------
For each formulation, two conditions were printed IN THE CELL-LADEN INK. They
differ only in which dataset the setting was derived from:
 
    condition 1   the optimum of a model fitted to the CELL-FREE ink
    condition 2   the optimum of a model fitted to the CELL-LADEN ink itself
 
Nothing was printed cell-free. "Cell-free" and "cell-laden" here name the SOURCE
of the number, not what went in the cartridge.
 
Why condition 2 is not an upper bound
-------------------------------------
It is tempting to read condition 2 as "the best the cell-laden ink can do", in
which case condition 1 beating it would be impossible. It is not that. It is a
model's argmax over a surface fitted to the cell-laden 32-point screen, and that
model can be wrong, especially when its argmax lands on the edge of the design
box. So condition 1 CAN beat condition 2, and when it does the honest reading is
that the cell-laden model was poor, not that cells help.
 
That is exactly why this script adds a third, purely measured reference:
 
    reference     the best condition actually MEASURED in the cell-laden
                  32-point LHS screen, and where available the best measured
                  point of the cell-laden pressure sweep
 
The reference is not printed here; it is read from the existing screen data.
With it, two different questions separate cleanly:
 
    Q1  does the calibration source matter?     condition 1 vs condition 2
    Q2  is the cell-free-derived condition any
        good in absolute terms?                 condition 1 vs the reference
 
Aim 2 is supported only when Q1 says "no difference" AND Q2 says condition 1 is
at least as good as the measured reference. If both printed conditions fall
below the reference, the pair says nothing about transfer, because both models
failed and the comparison is between two bad answers. The script says so
explicitly rather than reporting a tidy verdict.
 
Figure (figure 10 of the manuscript)
------------------------------------
By default the figure is drawn as in the manuscript: per formulation, one bar
for the cell-free-derived condition and one for the cell-laden-derived
condition, both printed with the bioink (six-well mean, error bar SD). No LHS
best or ramp best is drawn. Labels use the manuscript terms:

    legend   "Cell-free-derived condition", "Cell-laden-derived condition"
    x axis   the formulation, e.g. "10% DoF 60"

--show_screen_best adds the LHS best ("LHS best"), --show_sweep_best the ramp
best ("Ramp best"), --hide_cell_laden_derived drops the second bar, and
--xtick_style legacy restores "GelMA 10% DoF 60 / printed cell-laden".

Choosing what appears in the figure
-----------------------------------
    --list_pairs          print the pair keys and stop
    --include 10_80       keep only pairs matching any of these substrings
    --exclude 7.5         drop pairs matching any of these
Matching is case-insensitive against the formulation label. Filtering affects
the FIGURE only; the CSV always contains every pair.
 
Usage
-----
    python s11_aim2_evaluate.py \
        --validation_csv validation_sf_summary.csv \
        --manifest_csv d5_print_manifest.csv \
        --combined_csv combined_6category_table.csv \
        --sweep_dir sweep \
        --outdir results/11_aim2
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
 
TARGET, TARGET_STD = 'SF_mean', 'SF_std'
 
D5_RE = re.compile(r'^d5_(?P<pair>[A-H]-[A-H])_(?P<ink>.+?)_c(?P<cond>[12])_'
                   r'(?P<tag>freeopt|ladenopt)_', re.IGNORECASE)
 
 
def legend_label(cat):
    m = re.fullmatch(r'(cell_)?gelma_(\d+(?:\.\d+)?)_(\d+)', str(cat))
    if not m:
        return str(cat)
    cell, conc, dof = m.groups()
    if conc == '7':
        conc = '7.5'
    return f'GelMA {conc}% DoF {dof}  {"cell-laden" if cell else "cell-free"}'
 
 
def formulation(cat):
    """Formulation without the cell state, e.g. 'GelMA 10% DoF 60'."""
    return legend_label(cat).split('  ')[0]
 
 
def short_formulation(cat):
    """Formulation as written in the manuscript, e.g. '10% DoF 60'."""
    return formulation(cat).replace('GelMA ', '')


def slug(cat):
    return formulation(cat).replace('%', 'pct').replace('.', '_').replace(' ', '_')
 
 
def fmt_cond(p, f, z):
    return f'{p:g} kPa / {f:g} mm/s / {z:g} mm'
 
 
def sniff(path):
    head = Path(path).read_text(errors='replace').split('\n', 1)[0]
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def welch(m1, s1, n1, m2, s2, n2):
    diff = m1 - m2
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = np.sqrt(v1 + v2)
    if not np.isfinite(se) or se <= 0:
        return diff, np.nan, np.nan, np.nan
    t = diff / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    if _st is not None:
        p = 2 * _st.t.sf(abs(t), df)
    else:
        import math
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return diff, t, df, p
 
 
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
 
def load_printed(val):
    rows = []
    for _, r in val.iterrows():
        m = D5_RE.match(str(r['Folder_Name']))
        if not m:
            continue
        rows.append({'pair_code': m.group('pair').replace('-', '/'),
                     'ink': m.group('ink'),
                     'source': ('cell-free' if m.group('tag').lower() == 'freeopt'
                                else 'cell-laden'),
                     'P': float(r['Pressure_kPa']), 'F': float(r['NozzleSpeed_mms']),
                     'Z': float(r['Zoffset_mm']),
                     'SF': float(r[TARGET]), 'SD': float(r[TARGET_STD]),
                     'folder': str(r['Folder_Name'])})
    return pd.DataFrame(rows)
 
 
def lhs_best(combined, cat):
    sub = combined[combined['category'] == cat]
    if sub.empty:
        return None
    b = sub.loc[sub[TARGET].idxmax()]
    return {'SF': float(b[TARGET]),
            'SD': float(b[TARGET_STD]) if TARGET_STD in sub.columns else np.nan,
            'cond': fmt_cond(float(b['Pressure_kPa']), float(b['NozzleSpeed_mms']),
                             float(b['Zoffset_mm']))}
 
 
def sweep_best(sweep_dir, cat):
    if not sweep_dir:
        return None
    p = Path(sweep_dir) / f'{cat}_sf_summary_sweep.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p, sep=sniff(p)).dropna(subset=['Pressure_kPa', TARGET])
    if d.empty:
        return None
    b = d.loc[d[TARGET].idxmax()]
    return {'SF': float(b[TARGET]), 'P': float(b['Pressure_kPa'])}
 
 
def evaluate(printed, manifest, combined, sweep_dir, n, alpha):
    pred = {}
    if manifest is not None:
        for _, r in manifest.iterrows():
            pred[str(r['pair'])] = {
                'predicted_gain': float(r.get('transfer_regret', np.nan)),
                'noise_floor': float(r.get('noise_floor', np.nan)),
                'predicted_verdict': str(r.get('verdict', '')),
                'cell_free_source': str(r.get('derived_from', ''))
                if str(r.get('derivation', '')).lower() == 'freeopt' else '',
            }
 
    out = []
    for code, sub in printed.groupby('pair_code'):
        fr = sub[sub['source'] == 'cell-free']
        la = sub[sub['source'] == 'cell-laden']
        if len(fr) != 1 or len(la) != 1:
            print(f'  [WARN] {code}: expected one cell-free-derived and one '
                  f'cell-laden-derived print, found {len(fr)} and {len(la)}')
            continue
        f, l = fr.iloc[0], la.iloc[0]
        ink = l['ink']
 
        # source formulation: the cell-free partner of this ink
        src = ink[len('cell_'):] if ink.startswith('cell_') else ink
        mf = pred.get(code, {})
        if mf.get('cell_free_source'):
            src = mf['cell_free_source']
 
        L = lhs_best(combined, ink)
        SW = sweep_best(sweep_dir, ink)
 
        # Q1: does the calibration source matter?
        d1, t1, df1, p1 = welch(l['SF'], l['SD'], n, f['SF'], f['SD'], n)
        # Q2: is the cell-free-derived condition good in absolute terms?
        d2 = p2 = np.nan
        if L:
            d2, _, _, p2 = welch(f['SF'], f['SD'], n, L['SF'], L['SD'], n)
        # sanity on the cell-laden model itself
        d3 = p3 = np.nan
        if L:
            d3, _, _, p3 = welch(l['SF'], l['SD'], n, L['SF'], L['SD'], n)
 
        if not np.isfinite(p1):
            q1 = 'indeterminate'
        elif p1 >= alpha:
            q1 = 'the calibration source did not matter'
        elif d1 > 0:
            q1 = 'the cell-laden-derived condition was better'
        else:
            q1 = 'the cell-free-derived condition was better'
 
        if not L or not np.isfinite(p2):
            q2 = 'no cell-laden screen data to compare against'
        elif p2 >= alpha:
            q2 = 'matched the best condition the 32-point screen found'
        elif d2 > 0:
            q2 = 'beat the best condition the 32-point screen found'
        else:
            q2 = 'fell short of the best condition the 32-point screen found'
 
        both_short = (L is not None and np.isfinite(p2) and np.isfinite(p3)
                      and p2 < alpha and d2 < 0 and p3 < alpha and d3 < 0)
        if both_short:
            aim2 = ('NOT INFORMATIVE: both printed conditions fell significantly '
                    'below the best condition already measured, so this pair '
                    'compares two poor answers and says nothing about transfer')
        elif q1.startswith('the calibration source did not matter') and \
                (not L or d2 >= 0 or p2 >= alpha):
            aim2 = ('SUPPORTED: calibrating without cells cost nothing, and the '
                    'resulting condition was at least as good as the best the '
                    'cell-laden screen found')
        elif q1 == 'the cell-laden-derived condition was better':
            aim2 = ('NOT SUPPORTED: skipping cells cost significant shape '
                    'fidelity for this formulation')
        elif q1 == 'the cell-free-derived condition was better':
            aim2 = ('SUPPORTED, with a caveat: the cell-free-derived condition '
                    'won outright, which means the cell-laden model was the '
                    'weaker of the two rather than that cells help')
        else:
            aim2 = 'indeterminate'
 
        pg = mf.get('predicted_gain', np.nan)
        nf = mf.get('noise_floor', np.nan)
        agree = ''
        if np.isfinite(pg) and np.isfinite(nf) and nf > 0:
            meas_transfers = np.isfinite(p1) and (p1 >= alpha or d1 < 0)
            if pg <= nf:
                agree = 'yes' if meas_transfers else 'no'
            elif pg <= 2 * nf:
                agree = 'borderline (the prediction declined to call it)'
            else:
                agree = 'no' if meas_transfers else 'yes'
 
        out.append({
            'formulation': formulation(ink),
            'ink_printed': legend_label(ink),
            'calibration_source_ink': legend_label(src),
            'pair_code': code,
 
            'cond_from_cell_free': fmt_cond(f['P'], f['F'], f['Z']),
            'SF_from_cell_free': round(f['SF'], 4),
            'sd_from_cell_free': round(f['SD'], 4),
            'cond_from_cell_laden': fmt_cond(l['P'], l['F'], l['Z']),
            'SF_from_cell_laden': round(l['SF'], 4),
            'sd_from_cell_laden': round(l['SD'], 4),
 
            'Q1_difference_laden_minus_free': round(d1, 4),
            'Q1_p': pfmt(p1), 'Q1_result': q1,
 
            'screen_best_SF': round(L['SF'], 4) if L else '',
            'screen_best_sd': round(L['SD'], 4) if L else '',
            'screen_best_condition': L['cond'] if L else '',
            'Q2_difference_free_minus_screen': round(d2, 4) if np.isfinite(d2) else '',
            'Q2_p': pfmt(p2), 'Q2_result': q2,
            'laden_minus_screen': round(d3, 4) if np.isfinite(d3) else '',
            'laden_vs_screen_p': pfmt(p3),
 
            'sweep_best_SF': round(SW['SF'], 4) if SW else '',
            'sweep_best_pressure_kPa': SW['P'] if SW else '',
 
            'aim2_conclusion': aim2,
            'predicted_gain_from_using_cells': (round(pg, 4)
                                                if np.isfinite(pg) else ''),
            'noise_floor': nf if np.isfinite(nf) else '',
            'prediction_matched_the_print': agree,
            'predicted_verdict_before_printing': mf.get('predicted_verdict', ''),
            'folder_from_cell_free': f['folder'],
            'folder_from_cell_laden': l['folder'],
        })
    return pd.DataFrame(out)
 
 
def apply_filter(df, include, exclude):
    if df.empty:
        return df
    hay = (df['formulation'] + ' | ' + df['ink_printed']).str.lower()
    keep = pd.Series(True, index=df.index)
    if include:
        pats = [s.strip().lower() for s in include.split(',') if s.strip()]
        keep &= hay.apply(lambda h: any(p in h for p in pats))
    if exclude:
        pats = [s.strip().lower() for s in exclude.split(',') if s.strip()]
        keep &= hay.apply(lambda h: not any(p in h for p in pats))
    return df[keep]
 
 
# --------------------------------------------------------------------------
 
LHS_C, SWEEP_C = '#4a3aa7', '#6b6a63'

# Labels, worded as in the manuscript text and the caption of figure 10
LAB_FREE_DERIVED = 'Cell-free-derived condition'
LAB_LADEN_DERIVED = 'Cell-laden-derived condition'
LAB_LHS_BEST = 'LHS best'
LAB_RAMP_BEST = 'Ramp best'
 
 
# --------------------------------------------------------------------------
# figure sizing and legend placement
#
# These figures are printed small in a manuscript, so the fonts are set for the
# printed size rather than for the screen. Two things then need care: the text
# must not be cut off, and it must not sit on top of the data.
#
#   _needed_size   measures every label, tick, title and legend box and returns
#                  the figure size that holds them all. Matplotlib does not grow
#                  the canvas for a label longer than the figure, it just cuts
#                  the text, and Figure.get_tightbbox clips to the canvas so it
#                  does not report the overflow either. The same function widens
#                  the figure when neighbouring x tick labels would touch.
#   _render        builds the figure, measures it, and rebuilds it larger until
#                  nothing overflows. Text is never shrunk.
#   _legend        'inside' uses --legend_loc, 'above' and 'right' move the
#                  legend out of the data area completely, and 'auto' (default)
#                  starts inside and moves it above only if it covers the data.
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
    b = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    w, h = fig.get_size_inches()
    pad = 2 * a.pad_pt / 72
    need_w = w + max(0.0, -b.x0) + max(0.0, b.x1 - w) + pad
    need_h = h + max(0.0, -b.y0) + max(0.0, b.y1 - h) + pad
    gap = 0.07 * fig.dpi        # points of clear space between tick labels
    for row in xticks:
        over = sum(max(0.0, row[i - 1].x1 + gap - row[i].x0)
                   for i in range(1, len(row)))
        need_w = max(need_w, w + over / fig.dpi)
    return need_w, need_h


def _render(build, a, size):
    """Draw, measure, and grow the figure until no text is clipped."""
    import matplotlib.pyplot as plt
    fig = build(size)
    if not getattr(a, 'autofit', True):
        return fig
    for _ in range(4):
        w, h = _needed_size(fig, a)
        if w <= size[0] + 0.01 and h <= size[1] + 0.01:
            break
        plt.close(fig)
        size = (round(max(size[0], w), 2), round(max(size[1], h), 2))
        fig = build(size)
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


def _legend(fig, ax, handles, labels, a, style, ncol=None):
    if ncol is None:
        ncol = a.legend_ncol if a.legend_ncol > 0 else (2 if len(labels) > 3 else 1)

    def put(where):
        if where == 'above':
            return ax.legend(handles, labels, loc='lower center',
                             bbox_to_anchor=(0.5, 1.02), ncol=ncol, **style)
        if where == 'right':
            return ax.legend(handles, labels, loc='upper left',
                             bbox_to_anchor=(1.02, 1.0), ncol=1, **style)
        return ax.legend(handles, labels, loc=a.legend_loc, **style)

    where = a.legend_placement
    leg = put('inside' if where == 'auto' else where)
    if where == 'auto':
        fig.canvas.draw()
        if _legend_covers_data(ax, leg) > a.legend_overlap_tol:
            leg.remove()
            leg = put('above')
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _save(fig, path, a):
    from pathlib import Path as _P
    fig.savefig(path, facecolor=SURFACE)
    fig.savefig(_P(path).with_suffix('.eps'), format='eps', facecolor=SURFACE,
                dpi=getattr(a, 'eps_dpi', 900))


def make_figure(d, outdir, a):
    """Grouped bars with replicate error bars, one group per formulation.

    Every bar was printed in, or measured in, the CELL-LADEN ink. The default
    view is figure 10 of the manuscript: the cell-free-derived and the
    cell-laden-derived condition side by side, with no LHS best or ramp best.
    --show_screen_best and --show_sweep_best add those references, and
    --hide_cell_laden_derived drops the second bar.

    The pressure-sweep bar has no error bar: that screen prints one well per
    pressure, so it carries no replicate spread.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = d.reset_index(drop=True)
    sb = pd.to_numeric(d['screen_best_SF'], errors='coerce')
    sbs = pd.to_numeric(d['screen_best_sd'], errors='coerce')
    wb = pd.to_numeric(d['sweep_best_SF'], errors='coerce')

    series = []
    if a.show_sweep_best:
        series.append(('sweep', wb, None, SWEEP_C, LAB_RAMP_BEST))
    if a.show_screen_best and not a.hide_screen_best:
        series.append(('screen', sb, sbs, LHS_C, LAB_LHS_BEST))
    series.append(('free', d['SF_from_cell_free'], d['sd_from_cell_free'],
                   SERIES[0], a.label_cellfree))
    if not a.hide_cell_laden_derived:
        series.append(('laden', d['SF_from_cell_laden'], d['sd_from_cell_laden'],
                       SERIES[1], a.label_cellladen))

    k = len(series)
    w = a.bar_width / k
    x = np.arange(len(d))

    def build(size):
        fig, ax = _new_fig(a, size)
        for j, (_, vals, errs, colour, label) in enumerate(series):
            off = (j - (k - 1) / 2) * w
            v = pd.to_numeric(vals, errors='coerce')
            e = pd.to_numeric(errs, errors='coerce') if errs is not None else None
            ax.bar(x + off, v.fillna(0), width=w * .92,
                   yerr=(e.fillna(0) if e is not None else None),
                   capsize=a.capsize, color=colour, edgecolor=INK, linewidth=.7,
                   alpha=a.bar_alpha, label=label, zorder=3,
                   error_kw=dict(ecolor=INK, elinewidth=1.2, capthick=1.2))

        ax.set_xticks(x)
        if a.xtick_style == 'legacy':
            ticks = [f'{r["formulation"]}\nprinted cell-laden'
                     for _, r in d.iterrows()]
        else:
            ticks = [str(r['formulation']).replace('GelMA ', '')
                     for _, r in d.iterrows()]
        ax.set_xticklabels(ticks, fontsize=a.tick_size, color=INK)
        ax.set_xlim(-0.55, len(d) - 0.45)
        ax.set_ylim(a.ymin, a.ymax)
        ax.set_ylabel('Shape fidelity', fontsize=a.label_size, color=INK)
        if a.title:
            ax.set_title(a.title, fontsize=a.label_size, color=INK, loc='left',
                         pad=8)
        ax.grid(True, axis='y', color=GRIDC, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRIDC)
        ax.tick_params(colors=INK_SOFT, labelsize=a.tick_size)

        handles, labels = ax.get_legend_handles_labels()
        _legend(fig, ax, handles, labels, a,
                dict(frameon=True, framealpha=.92, edgecolor=GRIDC,
                     facecolor=SURFACE, fontsize=a.legend_size))
        return fig

    fw = a.fig_width if a.fig_width > 0 else 1.9 * len(d) + 1.6
    fh = a.fig_height if a.fig_height > 0 else 4.4
    fig = _render(build, a, (fw, fh))
    fw2, fh2 = fig.get_size_inches()
    if (round(fw2, 2), round(fh2, 2)) != (round(fw, 2), round(fh, 2)):
        print(f'  [NOTE] figure enlarged to {fw2:.2f} x {fh2:.2f} in so the '
              f'labels and the legend fit at this font size')
    p = outdir / 'fig_aim2_cellfree_calibration.png'
    # same filename, .eps. PostScript has no transparency, so bar_alpha is
    # rendered opaque in the EPS; everything else is vector.
    _save(fig, p, a)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if _st is None:
        print('[WARN] scipy missing: p-values use a normal approximation.')
 
    val = pd.read_csv(a.validation_csv, sep=sniff(a.validation_csv))
    printed = load_printed(val)
    if printed.empty:
        raise SystemExit(f'No d5_* rows in {a.validation_csv}. This script only '
                         f'reads the paired transfer prints; recommendations go '
                         f'through s10_aim1_evaluate.py.')
    combined = pd.read_csv(a.combined_csv, sep=sniff(a.combined_csv))
    manifest = (pd.read_csv(a.manifest_csv, sep=sniff(a.manifest_csv))
                if a.manifest_csv else None)
 
    d = evaluate(printed, manifest, combined, a.sweep_dir, a.n_replicates, a.alpha)
    if d.empty:
        raise SystemExit('No complete pairs found.')
 
    if a.list_pairs:
        print('Pair keys for --include / --exclude:\n')
        for _, r in d.iterrows():
            print(f'  {r["formulation"]}   (printed in {r["ink_printed"]})')
        return
 
    print('=' * 78)
    print('  AIM 2: can a formulation be calibrated without cells?')
    print('=' * 78)
    print('  Both conditions in each pair were printed in the CELL-LADEN ink.')
    print('  They differ only in which dataset the setting was derived from.')
 
    for _, r in d.iterrows():
        print(f'\n{"-" * 78}\n  {r["formulation"]}   printed in '
              f'{r["ink_printed"]}\n{"-" * 78}')
        print(f'    from the CELL-FREE  ink   {r["cond_from_cell_free"]:>28s}   '
              f'SF {r["SF_from_cell_free"]:.4f} +/- {r["sd_from_cell_free"]:.4f}')
        print(f'      (source: {r["calibration_source_ink"]})')
        print(f'    from the CELL-LADEN ink   {r["cond_from_cell_laden"]:>28s}   '
              f'SF {r["SF_from_cell_laden"]:.4f} +/- {r["sd_from_cell_laden"]:.4f}')
        if str(r['screen_best_SF']).strip():
            print(f'    reference, best MEASURED  {r["screen_best_condition"]:>28s}   '
                  f'SF {float(r["screen_best_SF"]):.4f} +/- '
                  f'{float(r["screen_best_sd"]):.4f}')
        if str(r['sweep_best_SF']).strip():
            sc = f'{r["sweep_best_pressure_kPa"]:g} kPa, 10 mm/s, Z=0.2 mm'
            print(f'    reference, sweep best     {sc:>28s}   '
                  f'SF {float(r["sweep_best_SF"]):.4f}   (1 well, no test)')
 
        print(f'\n    Q1 does the calibration source matter?')
        print(f'       cell-laden minus cell-free = '
              f'{r["Q1_difference_laden_minus_free"]:+.4f}  p={r["Q1_p"]}')
        print(f'       -> {r["Q1_result"]}')
        print(f'    Q2 is the cell-free-derived condition good in absolute terms?')
        if str(r['Q2_difference_free_minus_screen']).strip():
            print(f'       cell-free minus screen best = '
                  f'{float(r["Q2_difference_free_minus_screen"]):+.4f}  '
                  f'p={r["Q2_p"]}')
        print(f'       -> {r["Q2_result"]}')
        print(f'\n    AIM 2: {r["aim2_conclusion"]}')
        if str(r['prediction_matched_the_print']).strip():
            print(f'    prediction before printing: '
                  f'{r["predicted_verdict_before_printing"]}')
            print(f'    prediction matched the print: '
                  f'{r["prediction_matched_the_print"]}')
 
    sup = int(d['aim2_conclusion'].str.startswith('SUPPORTED').sum())
    print(f'\n{"=" * 78}')
    print(f'  Aim 2 supported in {sup} of {len(d)} formulation(s).')
    print('  Read the caveat lines: "supported" because the two conditions tie is')
    print('  a different claim from "supported" because the cell-free one won.')
 
    d.to_csv(outdir / 'aim2_summary.csv', sep=';', index=False)
    for _, r in d.iterrows():
        pd.DataFrame([r]).to_csv(
            outdir / f'aim2_{slug(r["ink_printed"])}.csv', sep=';', index=False)
 
    plotd = apply_filter(d, a.include, a.exclude)
    figs = []
    if a.include or a.exclude:
        print(f'\n  Figure filter: {len(plotd)} of {len(d)} pair(s) plotted')
    if plotd.empty:
        print('  [WARN] the filter removed every pair, so no figure was drawn.')
    else:
        try:
            figs.append(make_figure(plotd, outdir, a))
        except ImportError:
            print('  [WARN] matplotlib missing, no figure written')
    print(f'\n  Wrote {len(list(outdir.glob("*.csv")))} table(s) and '
          f'{len(figs)} figure(s)\n  -> {outdir}')
 
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Aim 2 evaluation of the paired '
                                             'transfer prints.')
    ap.add_argument('--validation_csv', required=True)
    ap.add_argument('--combined_csv', required=True,
                    help='combined_6category_table.csv, for the measured reference')
    ap.add_argument('--manifest_csv', default=None,
                    help='d5_print_manifest.csv, for what was predicted beforehand')
    ap.add_argument('--sweep_dir', default=None,
                    help='Folder of <category>_sf_summary_sweep.csv')
    ap.add_argument('--outdir', default='results/11_aim2')
    ap.add_argument('--n_replicates', type=int, default=6)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--include', default=None,
                    help='Comma-separated substrings; keep only matching '
                         'formulations in the FIGURE')
    ap.add_argument('--exclude', default=None,
                    help='Comma-separated substrings; drop matching '
                         'formulations from the FIGURE')
    ap.add_argument('--show_sweep_best', action='store_true',
                    help='Also draw the ramp best ("Ramp best"). Off by default: '
                         'the ramp has one well per pressure, so it carries '
                         'no error bar and is not comparable to the others.')
    ap.add_argument('--show_screen_best', action='store_true',
                    help='Also draw the LHS best ("LHS best"). Off by default, '
                         'as in figure 10 of the manuscript. The values stay in '
                         'the CSV.')
    ap.add_argument('--hide_screen_best', action='store_true',
                    help='Kept for older command lines; the LHS best is now '
                         'hidden unless --show_screen_best is given.')
    ap.add_argument('--dpi', type=int, default=300,
                    help='Saved PNG resolution (default 300). The EPS is vector '
                         'and is written at --eps_dpi for its rasterised parts.')
    ap.add_argument('--eps_dpi', type=int, default=900,
                    help='Resolution used inside the EPS (default 900)')
    ap.add_argument('--show_cell_laden_derived', action='store_true',
                    help='Kept for older command lines; the cell-laden-derived '
                         'condition is now drawn by default, as in figure 10.')
    ap.add_argument('--hide_cell_laden_derived', action='store_true',
                    help='Do not draw the cell-laden-derived condition.')
    ap.add_argument('--label_cellfree', default=LAB_FREE_DERIVED,
                    help='Legend text for the cell-free-derived condition '
                         '(default "Cell-free-derived condition")')
    ap.add_argument('--label_cellladen', default=LAB_LADEN_DERIVED,
                    help='Legend text for the cell-laden-derived condition '
                         '(default "Cell-laden-derived condition")')
    ap.add_argument('--xtick_style', default='manuscript',
                    choices=['manuscript', 'legacy'],
                    help="x tick labels. 'manuscript' (default): the formulation "
                         "as in the text, e.g. '10%% DoF 60'. 'legacy': "
                         "'GelMA 10%% DoF 60' over 'printed cell-laden'.")
    ap.add_argument('--title', default='',
                    help='Figure title. Empty by default.')
    ap.add_argument('--ymin', type=float, default=0.4)
    ap.add_argument('--ymax', type=float, default=0.8)
    ap.add_argument('--tick_size', type=float, default=15)
    ap.add_argument('--label_size', type=float, default=17)
    ap.add_argument('--legend_size', type=float, default=15)
    ap.add_argument('--bar_width', type=float, default=0.72,
                    help='Total width of one formulation group (default 0.72)')
    ap.add_argument('--bar_alpha', type=float, default=0.92)
    ap.add_argument('--capsize', type=float, default=5,
                    help='Error-bar cap size (default 5)')
    ap.add_argument('--legend_loc', default='best',
                    help='Legend position when it is inside the axes')
    ap.add_argument('--legend_placement', default='auto',
                    choices=['auto', 'inside', 'above', 'right'],
                    help="Where the legend goes. 'auto' keeps it inside unless "
                         "it covers the bars, in which case it moves above the "
                         "axes (default: auto)")
    ap.add_argument('--legend_ncol', type=int, default=0,
                    help='Legend columns, 0 chooses 1 or 2 automatically')
    ap.add_argument('--legend_overlap_tol', type=float, default=0.02,
                    help='How much of the legend box may sit on the bars before '
                         'it is moved out, as a fraction (default 0.02)')
    ap.add_argument('--pad_pt', type=float, default=2.0,
                    help='Padding in points between the content and the figure '
                         'edge (default 2)')
    ap.add_argument('--no_autofit', dest='autofit', action='store_false',
                    help='Keep the figure size exactly as given, even if that '
                         'clips a label')
    ap.add_argument('--fig_width', type=float, default=0,
                    help='Starting figure width in inches. 0 = auto. The figure '
                         'is grown automatically if a label or the legend does '
                         'not fit')
    ap.add_argument('--fig_height', type=float, default=0,
                    help='Starting figure height in inches. 0 = auto')
    ap.add_argument('--list_pairs', action='store_true')
    main(ap.parse_args())