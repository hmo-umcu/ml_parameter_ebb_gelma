"""
s13_cellfree_vs_cellladen.py
----------------------------
How different are the cell-free and cell-laden inks, and what does calibrating
on the cell-free one actually buy you?
 
Two questions, kept separate because they have different answers.
 
PART 1  HOW DIFFERENT ARE THE INKS?
    Uses only the 32-point LHS screens, paired condition by condition, so every
    comparison is like for like.
 
      offset                mean(SF_laden - SF_free) with a paired t-test.
                            A signed number: cells can help or hurt.
      Pearson r             agreement in level.
      Spearman rho          agreement in RANK. This is the one that governs
                            transferability: if the cell-free screen ranks
                            conditions the same way the cell-laden screen does,
                            a condition chosen cell-free will be a good
                            cell-laden condition even when the levels differ.
      identity RMSE         error from using SF_free directly as a prediction
                            of SF_laden.
      top-k overlap         how many of the best k cell-free conditions are
                            also among the best k cell-laden ones.
      sweep RMS             distance between the two pressure-sweep curves,
                            where both sweeps exist.
 
PART 2  WHAT DOES THE CELL-FREE ROUTE BUY?
    Three shape fidelities on the same cell-laden ink:
 
      naive transfer        take the BEST MEASURED cell-free condition and use
                            it unchanged. Measured directly in the cell-laden
                            screen, no model involved. This is what a lab would
                            do with no ML at all.
      model transfer        the condition a model trained on cell-free data
                            recommended, as actually printed in the cell-laden
                            ink. Read from the validation summary.
      ceiling               the best condition measured anywhere in the
                            cell-laden 32-point screen.
 
    From those:
      transfer regret       ceiling - transfer, in SF units
      fraction of ceiling   transfer / ceiling
      percentile            where the transferred condition sits in the
                            cell-laden screen's own ranking
      value added by the model
                            model transfer - naive transfer. This isolates what
                            the ML step contributed on top of simply reusing the
                            cell-free winner.
      cell-laden wells      192 for the full screen against 6 for a validated
                            recommendation.
 
Pairs are discovered from the data: for each cell_<X> category present, if <X>
is also present the two are paired. Nothing is hardcoded.
 
Usage
-----
    python s13_cellfree_vs_cellladen.py \
        --combined_csv combined_6category_table.csv \
        --sweep_dir sweep \
        --validation_csv validation_sf_summary.csv \
        --outdir results/13_ink_difference
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
CEIL_C, NAIVE_C, MODEL_C = '#4a3aa7', '#6b6a63', '#2a78d6'
 
PF = ['Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
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
    return legend_label(cat).split('  ')[0]
 
 
def slug(cat):
    return formulation(cat).replace('%', 'pct').replace('.', '_').replace(' ', '_')
 
 
def sniff(path):
    head = Path(path).read_text(errors='replace').split('\n', 1)[0]
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def fmt_cond(p, f, z):
    return f'{p:g} kPa / {f:g} mm/s / {z:g} mm'
 
 
def pfmt(p):
    if not np.isfinite(p):
        return ''
    return '<0.0001' if p < 1e-4 else round(float(p), 4)
 
 
# --------------------------------------------------------------------------
 
def find_pairs(combined):
    cats = set(combined['category'].unique())
    pairs = []
    for c in sorted(cats):
        if c.startswith('cell_') and c[len('cell_'):] in cats:
            pairs.append((c[len('cell_'):], c))
    return pairs
 
 
def load_sweeps(sweep_dir):
    out = {}
    if not sweep_dir:
        return out
    for p in sorted(Path(sweep_dir).glob('*_sf_summary_sweep.csv')):
        cat = p.name[:-len('_sf_summary_sweep.csv')]
        d = pd.read_csv(p, sep=sniff(p)).dropna(subset=['Pressure_kPa', TARGET])
        out[cat] = d.sort_values('Pressure_kPa').reset_index(drop=True)
    return out
 
 
def load_model_transfers(validation_csv):
    """SF actually printed at each cell-free-derived recommendation."""
    if not validation_csv:
        return {}
    val = pd.read_csv(validation_csv, sep=sniff(validation_csv))
    out = {}
    for _, r in val.iterrows():
        m = D5_RE.match(str(r['Folder_Name']))
        if not m or m.group('tag').lower() != 'freeopt':
            continue
        out[m.group('ink')] = {
            'SF': float(r[TARGET]), 'SD': float(r[TARGET_STD]),
            'cond': fmt_cond(float(r['Pressure_kPa']),
                             float(r['NozzleSpeed_mms']), float(r['Zoffset_mm'])),
            'folder': str(r['Folder_Name'])}
    return out
 
 
# --------------------------------------------------------------------------
 
def onset_of(sw, thr):
    """Pressure at which SF first crosses thr, linearly interpolated.
 
    One well per pressure, so this is a weak estimate. It is reported for
    direction, not magnitude.
    """
    P = sw['Pressure_kPa'].values.astype(float)
    S = sw[TARGET].values.astype(float)
    above = np.flatnonzero(S > thr)
    if above.size == 0:
        return np.nan
    i = int(above[0])
    if i == 0:
        return float(P[0])
    if S[i] == S[i - 1]:
        return float(P[i])
    return float(P[i - 1] + (thr - S[i - 1]) * (P[i] - P[i - 1])
                 / (S[i] - S[i - 1]))
 
 
def part1(combined, sweeps, pairs, top_k, onset_sf):
    rows, paired = [], {}
    for free, lad in pairs:
        f = combined[combined['category'] == free]
        l = combined[combined['category'] == lad]
        m = f.merge(l, on=PF, suffixes=('_free', '_laden'))
        if m.empty:
            print(f'  [WARN] {free} and {lad} share no LHS conditions, skipped')
            continue
        if len(m) != len(f) or len(m) != len(l):
            print(f'  [NOTE] {formulation(lad)}: {len(m)} of {len(f)}/{len(l)} '
                  f'conditions matched between the two screens')
        x = m[f'{TARGET}_free'].values.astype(float)
        y = m[f'{TARGET}_laden'].values.astype(float)
        d = y - x
        n = len(d)
        sd = float(np.std(d, ddof=1))
        se = sd / np.sqrt(n)
        if _st is not None:
            tcrit = float(_st.t.ppf(0.975, n - 1))
            _, p_paired = _st.ttest_rel(y, x)
            r, _ = _st.pearsonr(x, y)
            rho, p_rho = _st.spearmanr(x, y)
        else:
            tcrit, p_paired = 1.96, np.nan
            r = float(np.corrcoef(x, y)[0, 1])
            rho = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
            p_rho = np.nan
 
        kf = set(pd.Series(x).nlargest(top_k).index)
        kl = set(pd.Series(y).nlargest(top_k).index)
 
        # Is the effect of cells a constant shift or proportional to SF?
        if _st is not None:
            sl, _, _, p_sl, _ = _st.linregress((x + y) / 2, d)
        else:
            sl, p_sl = np.nan, np.nan
 
        # Sweep-derived diagnostics. The sweep is a single well per pressure at
        # a fixed 10 mm/s and Z = 0.2 mm, so it is both noisier than the screen
        # and a different slice of the design box. It is reported as an
        # INDEPENDENT check, and a sign disagreement with the screen is a
        # finding rather than a rounding error.
        sw = sw_off = on_shift = pk_shift = ''
        agree = ''
        if free in sweeps and lad in sweeps:
            grid = np.union1d(sweeps[free]['Pressure_kPa'],
                              sweeps[lad]['Pressure_kPa'])
            av = np.interp(grid, sweeps[free]['Pressure_kPa'], sweeps[free][TARGET])
            bv = np.interp(grid, sweeps[lad]['Pressure_kPa'], sweeps[lad][TARGET])
            sw = round(float(np.sqrt(np.mean((bv - av) ** 2))), 4)
            sw_off = round(float(np.mean(bv - av)), 4)
            o_f, o_l = onset_of(sweeps[free], onset_sf), onset_of(sweeps[lad], onset_sf)
            if np.isfinite(o_f) and np.isfinite(o_l):
                on_shift = round(o_l - o_f, 2)
            pk_shift = round(float(sweeps[lad][TARGET].max()
                                   - sweeps[free][TARGET].max()), 4)
            agree = ('yes' if np.sign(sw_off) == np.sign(np.mean(d))
                     else 'NO - the screen and the sweep disagree on the SIGN '
                          'of the cell effect')
 
        rows.append({
            'formulation': formulation(lad),
            'cell_free_category': free, 'cell_laden_category': lad,
            'n_paired_conditions': n,
            'offset_laden_minus_free': round(float(np.mean(d)), 4),
            'offset_sd': round(sd, 4),
            'offset_CI95_low': round(float(np.mean(d) - tcrit * se), 4),
            'offset_CI95_high': round(float(np.mean(d) + tcrit * se), 4),
            'offset_p_paired_t': pfmt(p_paired),
            'cells_effect': ('cells improve shape fidelity' if np.mean(d) > 0
                             else 'cells reduce shape fidelity'),
            'pearson_r': round(float(r), 3),
            'spearman_rho': round(float(rho), 3),
            'spearman_p': pfmt(p_rho),
            'identity_RMSE': round(float(np.sqrt(np.mean(d ** 2))), 4),
            f'top{top_k}_overlap': f'{len(kf & kl)}/{top_k}',
            'offset_vs_meanSF_slope': round(float(sl), 3) if np.isfinite(sl) else '',
            'offset_vs_meanSF_p': pfmt(p_sl),
            'offset_shape': ('proportional to SF' if np.isfinite(p_sl)
                             and p_sl < 0.05 else 'roughly a constant shift'),
            'mean_replicate_sd_cell_free': round(float(
                combined[combined['category'] == free][TARGET_STD].mean()), 4),
            'mean_replicate_sd_cell_laden': round(float(
                combined[combined['category'] == lad][TARGET_STD].mean()), 4),
            'sweep_curve_RMS_difference': sw,
            'sweep_offset_laden_minus_free': sw_off,
            'sweep_onset_shift_kPa': on_shift,
            'sweep_peak_SF_shift': pk_shift,
            'screen_and_sweep_agree_on_sign': agree,
        })
        paired[lad] = m
    return pd.DataFrame(rows), paired
 
 
def part2(combined, paired, model_tr, pairs):
    rows = []
    for free, lad in pairs:
        m = paired.get(lad)
        if m is None:
            continue
        x = m[f'{TARGET}_free'].values.astype(float)
        y = m[f'{TARGET}_laden'].values.astype(float)
        ysd = (m[f'{TARGET_STD}_laden'].values.astype(float)
               if f'{TARGET_STD}_laden' in m else np.full(len(y), np.nan))
 
        kf, kl = int(np.argmax(x)), int(np.argmax(y))
        naive, ceiling = float(y[kf]), float(y[kl])
        pct = float((y < naive).mean() * 100)
 
        mt = model_tr.get(lad)
        mv = float(mt['SF']) if mt else np.nan
 
        rows.append({
            'formulation': formulation(lad),
            'cell_laden_category': lad,
 
            'naive_transfer_condition': fmt_cond(*m.iloc[kf][PF]),
            'naive_transfer_SF': round(naive, 4),
            'naive_transfer_sd': round(float(ysd[kf]), 4) if np.isfinite(ysd[kf]) else '',
            'naive_transfer_percentile_in_cell_laden_screen': round(pct, 1),
 
            'model_transfer_condition': mt['cond'] if mt else '',
            'model_transfer_SF': round(mv, 4) if np.isfinite(mv) else '',
            'model_transfer_sd': round(float(mt['SD']), 4) if mt else '',
 
            'ceiling_condition': fmt_cond(*m.iloc[kl][PF]),
            'ceiling_SF': round(ceiling, 4),
            'ceiling_sd': round(float(ysd[kl]), 4) if np.isfinite(ysd[kl]) else '',
 
            'naive_regret': round(ceiling - naive, 4),
            'naive_fraction_of_ceiling': round(naive / ceiling, 3) if ceiling else '',
            'model_regret': (round(ceiling - mv, 4) if np.isfinite(mv) else ''),
            'model_fraction_of_ceiling': (round(mv / ceiling, 3)
                                          if np.isfinite(mv) and ceiling else ''),
            'value_added_by_the_model': (round(mv - naive, 4)
                                         if np.isfinite(mv) else ''),
 
            'cell_laden_wells_for_the_ceiling': len(y) * 6,
            'cell_laden_wells_for_the_transfer': 6 if mt else '',
        })
    return pd.DataFrame(rows)
 
 
# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
 
def _style(ax, a):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDC, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_SOFT, labelsize=a.tick_size)
 
 
def _new_fig(nc, a, size):
    """Figure with a constrained layout, falling back on older matplotlib."""
    import matplotlib.pyplot as plt
    try:
        fig, axes = plt.subplots(1, nc, figsize=size, dpi=a.dpi,
                                 squeeze=False, layout='constrained')
        fig.get_layout_engine().set(w_pad=a.pad_pt / 72, h_pad=a.pad_pt / 72)
        constrained = True
    except (TypeError, AttributeError):
        fig, axes = plt.subplots(1, nc, figsize=size, dpi=a.dpi, squeeze=False)
        constrained = False
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0], constrained


def _needed_size(fig, a):
    """Figure size in inches that holds every label without clipping.

    A small panel with large fonts can need an axis label longer than the panel
    itself. Matplotlib does not grow the canvas for that, it cuts the text off,
    and Figure.get_tightbbox clips to the canvas so it does not see the overflow
    either. The label boxes are therefore measured directly and the figure is
    rebuilt larger. Text is never shrunk.
    """
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        boxes.append(ax.get_window_extent(r))
        arts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        arts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for art in arts:
            if art.get_text():
                boxes.append(art.get_window_extent(r))
        if ax.get_legend() is not None:
            boxes.append(ax.get_legend().get_window_extent(r))
    b = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    w, h = fig.get_size_inches()
    pad = 2 * a.pad_pt / 72
    return (w + max(0.0, -b.x0) + max(0.0, b.x1 - w) + pad,
            h + max(0.0, -b.y0) + max(0.0, b.y1 - h) + pad)


def scatter_limits(paired, a):
    """One set of axis limits and tick positions for every panel.

    Shared so that the grid lines of the separate panel figures land in exactly
    the same places and the subfigures line up in the manuscript.
    """
    hi = a.axis_max
    if hi <= 0:
        hi = max(max(float(m[f'{TARGET}_free'].max()),
                     float(m[f'{TARGET}_laden'].max())) for m in paired.values())
        hi = float(np.ceil(hi * 1.06 / a.tick_step) * a.tick_step)
    lo = a.axis_min
    ticks = np.arange(lo, hi + 1e-9, a.tick_step)
    return (lo, hi), ticks


def _scatter_panel(ax, i, cat, m, p1, a, lim, ticks, show_ylabel=True,
                   show_legend=True):
    """One cell-free vs cell-laden panel, drawn on a given axis."""
    row = p1[p1['cell_laden_category'] == cat].iloc[0]
    x = m[f'{TARGET}_free']
    y = m[f'{TARGET}_laden']
    ls_ = a.label_size * a.panel_font_scale
    ts_ = a.tick_size * a.panel_font_scale
    gs_ = a.legend_size * a.panel_font_scale

    ax.plot(lim, lim, color=INK_SOFT, lw=1.2, ls=(0, (4, 4)), zorder=2,
            label='no effect of cells')
    ax.plot(x, y, 'o', ms=a.scatter_size, color=SERIES[i % len(SERIES)],
            markeredgecolor=INK, markeredgewidth=.5,
            alpha=a.marker_alpha, ls='none', zorder=3)
    if a.show_fit_lines:
        # ordinary least squares, drawn only over the measured x range
        # (no extrapolation). The coefficients are in the CSV, not the legend.
        slope, icpt = np.polyfit(x.astype(float), y.astype(float), 1)
        xf = np.array([float(x.min()), float(x.max())])
        ax.plot(xf, slope * xf + icpt, color=SERIES[i % len(SERIES)],
                lw=2.0, zorder=4, label='best fit')

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_box_aspect(1)
    ax.set_xlabel('Shape fidelity, cell-free', fontsize=ls_, color=INK)
    if show_ylabel:
        ax.set_ylabel('Shape fidelity, cell-laden', fontsize=ls_, color=INK)

    if a.panel_title == 'full':
        ax.set_title(f'{row["formulation"]}\n'
                     f'n = {row["n_paired_conditions"]} paired LHS conditions\n'
                     f'Spearman rho = {row["spearman_rho"]:+.2f},  '
                     f'offset = {row["offset_laden_minus_free"]:+.3f}',
                     fontsize=ls_ - 3, color=INK, loc='left', pad=6)
    elif a.panel_title == 'short':
        ax.set_title(f'{row["formulation"]}\n'
                     f'Spearman rho = {row["spearman_rho"]:+.2f}',
                     fontsize=ls_ - 3, color=INK, loc='left', pad=6)

    _style(ax, a)
    ax.tick_params(labelsize=ts_)
    if show_legend:
        leg = ax.legend(frameon=False, fontsize=gs_, loc='upper left')
        for t in leg.get_texts():
            t.set_color(INK)


def _save(fig, path, a, constrained):
    """Write png and eps.

    With the constrained layout every label is already inside the figure, so
    nothing is cropped afterwards: bbox_inches tight is skipped there because
    it can cut a long axis label on a square panel. Without it (old matplotlib)
    tight_layout runs first and the crop is applied.
    """
    kw = {}
    if not constrained:
        fig.tight_layout(pad=a.pad_pt / 6)
        kw = dict(bbox_inches='tight', pad_inches=0.05)
    if a.bbox_tight:
        kw = dict(bbox_inches='tight', pad_inches=a.pad_inches)
    fig.savefig(path, facecolor=SURFACE, **kw)
    fig.savefig(path.with_suffix('.eps'), format='eps', facecolor=SURFACE,
                dpi=900, **kw)


def write_tex_snippet(paths, p1, outdir, a):
    """A ready-to-paste LaTeX figure with one subfigure per panel."""
    n = max(1, len(paths))
    w = a.subfig_width if a.subfig_width > 0 else round(0.98 / n - 0.015, 3)
    lines = ['% requires \\usepackage{graphicx} and \\usepackage{subcaption}',
             '\\begin{figure}[t]', '  \\centering']
    for k, (cat, path) in enumerate(paths):
        row = p1[p1['cell_laden_category'] == cat].iloc[0]
        cap = str(row['formulation']).replace('%', '\\%')   # LaTeX escape
        lines += [f'  \\begin{{subfigure}}[t]{{{w}\\linewidth}}',
                  '    \\centering',
                  f'    \\includegraphics[width=\\linewidth]{{{path.name}}}',
                  f'    \\caption{{{cap}}}',
                  f'    \\label{{fig:paired_scatter_{slug(cat)}}}',
                  '  \\end{subfigure}' + ('\\hfill' if k < len(paths) - 1 else '')]
    lines += ['  \\caption{Shape fidelity of the cell-laden ink against the '
              'cell-free ink of the same formulation, one point per LHS '
              'condition. The dashed line is equality.}',
              '  \\label{fig:paired_scatter}', '\\end{figure}', '']
    t = outdir / 'fig_paired_scatter.tex'
    t.write_text('\n'.join(lines))
    return t


def fig_scatter(paired, p1, outdir, a):
    """One standalone figure per formulation, sized for use as a TeX subfigure.

    Every panel uses the same axis limits, the same tick positions and the same
    figure size, so the grid lines coincide and the subfigures line up when they
    are placed side by side in the manuscript. The old single multi-panel figure
    is still available with --combined_scatter.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cats = list(paired)
    lim, ticks = scatter_limits(paired, a)

    def build(nc, idx, size):
        fig, axes, constrained = _new_fig(nc, a, size)
        for k, i in enumerate(idx):
            _scatter_panel(axes[k], i, cats[i], paired[cats[i]], p1, a, lim,
                           ticks, show_ylabel=(nc == 1 or k == 0),
                           show_legend=(nc == 1 and
                                        not a.legend_first_panel_only) or k == 0)
        return fig, constrained

    # First pass: find the size that fits the longest label and title, so that
    # every panel ends up identical in size.
    size = (a.panel_width, a.panel_height)
    if a.autofit:
        # repeated, because growing the figure moves the labels and can reveal
        # a little more overflow on the other axis
        for _ in range(4):
            need = list(size)
            for i in range(len(cats)):
                fig, _ = build(1, [i], size)
                w, h = _needed_size(fig, a)
                plt.close(fig)
                need = [max(need[0], w), max(need[1], h)]
            grown = (round(need[0], 3), round(need[1], 3))
            if grown[0] <= size[0] + 0.01 and grown[1] <= size[1] + 0.01:
                break
            size = grown
        if size != (a.panel_width, a.panel_height):
            print(f'  [NOTE] paired-scatter panels enlarged to '
                  f'{size[0]:.2f} x {size[1]:.2f} in so the axis labels fit at '
                  f'this font size')

    out, named = [], []
    for i, cat in enumerate(cats):
        fig, constrained = build(1, [i], size)
        path = outdir / f'fig_paired_scatter_{slug(cat)}.png'
        _save(fig, path, a, constrained)
        plt.close(fig)
        out.append(path)
        named.append((cat, path))

    if a.combined_scatter:
        fig, constrained = build(len(cats), list(range(len(cats))),
                                 (size[0] * len(cats), size[1]))
        path = outdir / 'fig_paired_scatter.png'
        _save(fig, path, a, constrained)
        plt.close(fig)
        out.append(path)

    if named and a.write_tex:
        out.append(write_tex_snippet(named, p1, outdir, a))
    return out


def fig_offset(paired, p1, outdir, a):
    """Bland-Altman: is the effect of cells a constant shift or SF-dependent?
 
    One point per LHS condition, each the mean of its replicate wells. No fitted
    line is drawn. The mean offset and its 95% limits of agreement are optional
    (--show_offset_lines) and appear in the legend when drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for i, (cat, m) in enumerate(paired.items()):
        row = p1[p1['cell_laden_category'] == cat].iloc[0]
        x = m[f'{TARGET}_free'].values
        y = m[f'{TARGET}_laden'].values
        c = SERIES[i % len(SERIES)]
        ax.plot((x + y) / 2, y - x, 'o', ms=a.scatter_size, color=c,
                markeredgecolor=INK, markeredgewidth=.5, ls='none',
                alpha=a.marker_alpha, zorder=3,
                label=f'{row["formulation"]}  (n = {len(x)})')
        if a.show_offset_lines:
            mu, sd = float(np.mean(y - x)), float(np.std(y - x, ddof=1))
            ax.axhline(mu, color=c, lw=1.8, zorder=2,
                       label=f'{row["formulation"]}: mean offset {mu:+.3f}')
            ax.axhline(mu + 1.96 * sd, color=c, lw=1.1, ls=(0, (4, 4)), zorder=2,
                       label=f'{row["formulation"]}: 95% limits of agreement')
            ax.axhline(mu - 1.96 * sd, color=c, lw=1.1, ls=(0, (4, 4)), zorder=2)
    ax.axhline(0, color=INK, lw=1.2, zorder=1, label='no effect of cells')
    ax.set_xlabel('Mean shape fidelity of the pair', fontsize=a.label_size,
                  color=INK)
    ax.set_ylabel('Cell-laden minus cell-free', fontsize=a.label_size,
                  color=INK)
    _style(ax, a)
    handles, labels = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    leg = ax.legend(hh, ll, frameon=True, framealpha=.92, edgecolor=GRIDC,
                    facecolor=SURFACE, fontsize=a.legend_size, loc=a.legend_loc)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    p = outdir / 'fig_offset_distribution.png'
    fig.savefig(p, facecolor=SURFACE); plt.close(fig)
    fig.savefig(p.with_suffix('.eps'), format='eps', facecolor=SURFACE,
    dpi=900)
    return p
 
 
def fig_benefit(p2, outdir, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
 
    d = p2.reset_index(drop=True)
    ms = a.marker_size
    fw = a.fig_width if a.fig_width > 0 else 1.95 * len(d) + 2.6
    fh = a.fig_height if a.fig_height > 0 else 5.6
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
 
    for i, r in d.iterrows():
        ax.plot(i, r['ceiling_SF'], marker='o', ms=ms, ls='none', color=CEIL_C,
                markerfacecolor=CEIL_C, markeredgecolor=INK, markeredgewidth=.8,
                alpha=a.marker_alpha, zorder=5)
        ax.plot(i, r['naive_transfer_SF'], marker='v', ms=ms, ls='none',
                color=NAIVE_C, markerfacecolor=NAIVE_C, markeredgecolor=INK,
                markeredgewidth=.8, alpha=a.marker_alpha, zorder=4)
        if str(r['model_transfer_SF']).strip():
            mv = float(r['model_transfer_SF'])
            ax.plot(i, mv, marker='s', ms=ms, ls='none', color=MODEL_C,
                    markerfacecolor=MODEL_C, markeredgecolor=INK,
                    markeredgewidth=.8, alpha=a.marker_alpha, zorder=6)
            ax.annotate('', xy=(i, mv), xytext=(i, r['naive_transfer_SF']),
                        arrowprops=dict(arrowstyle='-|>', color=MODEL_C, lw=1.6,
                                        shrinkA=ms * .55, shrinkB=ms * .55),
                        zorder=3)
            ax.annotate(f'{mv - r["naive_transfer_SF"]:+.3f}',
                        ((i), (mv + r['naive_transfer_SF']) / 2),
                        textcoords='offset points', xytext=(ms * .8, 0),
                        fontsize=a.legend_size, color=MODEL_C, va='center')
 
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f'{r["formulation"]}\nprinted cell-laden'
                        for _, r in d.iterrows()],
                       fontsize=a.tick_size, color=INK)
    ax.set_xlim(-0.45, len(d) - 0.55)
    ax.set_ylim(a.ymin, a.ymax)
    ax.set_ylabel('Shape fidelity', fontsize=a.label_size, color=INK)
    if a.title:
        ax.set_title(a.title, fontsize=a.label_size, color=INK, loc='left', pad=10)
    ax.grid(True, axis='y', color=GRIDC, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_SOFT, labelsize=a.tick_size)
 
    handles = [
        Line2D([], [], marker='o', ls='none', ms=ms * .8, color=CEIL_C,
               markerfacecolor=CEIL_C, markeredgecolor=INK,
               label='ceiling: best measured in the cell-laden screen'),
        Line2D([], [], marker='v', ls='none', ms=ms * .8, color=NAIVE_C,
               markerfacecolor=NAIVE_C, markeredgecolor=INK,
               label='naive transfer: best cell-free condition, reused'),
        Line2D([], [], marker='s', ls='none', ms=ms * .8, color=MODEL_C,
               markerfacecolor=MODEL_C, markeredgecolor=INK,
               label='model transfer: recommended from cell-free, printed'),
    ]
    leg = ax.legend(handles=handles, frameon=True, framealpha=.92,
                    edgecolor=GRIDC, facecolor=SURFACE,
                    fontsize=a.legend_size, loc=a.legend_loc)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    p = outdir / 'fig_transfer_benefit.png'
    fig.savefig(p, facecolor=SURFACE); plt.close(fig)
    fig.savefig(p.with_suffix('.eps'), format='eps', facecolor=SURFACE,
    dpi=900)
    return p
 
 
# --------------------------------------------------------------------------
 
def main(a):
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if _st is None:
        print('[WARN] scipy missing: p-values and rank tests are approximate.')
 
    combined = pd.read_csv(a.combined_csv, sep=sniff(a.combined_csv))
    sweeps = load_sweeps(a.sweep_dir)
    model_tr = load_model_transfers(a.validation_csv)
    pairs = find_pairs(combined)
    if not pairs:
        raise SystemExit('No cell-free / cell-laden pair is present in '
                         f'{a.combined_csv}.')
 
    print('=' * 78)
    print('  CELL-FREE vs CELL-LADEN')
    print('=' * 78)
    print(f'  pairs found: '
          + ', '.join(formulation(l) for _, l in pairs))
    unpaired = sorted(c for c in combined['category'].unique()
                      if c not in [x for pr in pairs for x in pr])
    if unpaired:
        print(f'  no partner, excluded: '
              + ', '.join(legend_label(c) for c in unpaired))
 
    p1, paired = part1(combined, sweeps, pairs, a.top_k, a.onset_sf)
    p2 = part2(combined, paired, model_tr, pairs)
 
    print('\n' + '-' * 78)
    print('  PART 1: how different are the inks? (paired over the LHS screen)')
    print('-' * 78)
    for _, r in p1.iterrows():
        print(f'\n  {r["formulation"]}   n = {r["n_paired_conditions"]} paired '
              f'conditions')
        print(f'    offset (cell-laden minus cell-free) '
              f'{r["offset_laden_minus_free"]:+.4f}  '
              f'95% CI [{r["offset_CI95_low"]:+.4f}, {r["offset_CI95_high"]:+.4f}]  '
              f'p={r["offset_p_paired_t"]}')
        print(f'    -> {r["cells_effect"]}')
        print(f'    rank agreement  Spearman rho {r["spearman_rho"]:+.3f} '
              f'(p={r["spearman_p"]})   Pearson r {r["pearson_r"]:+.3f}')
        print(f'    using cell-free SF as-is to predict cell-laden: '
              f'RMSE {r["identity_RMSE"]:.4f}')
        print(f'    top-{a.top_k} overlap {r[f"top{a.top_k}_overlap"]}')
        print(f'    offset shape: {r["offset_shape"]} '
              f'(slope vs mean SF {r["offset_vs_meanSF_slope"]}, '
              f'p={r["offset_vs_meanSF_p"]})')
        print(f'    replicate sd  cell-free {r["mean_replicate_sd_cell_free"]:.4f}'
              f'   cell-laden {r["mean_replicate_sd_cell_laden"]:.4f}')
        if str(r['sweep_curve_RMS_difference']).strip():
            print(f'    independent check from the pressure sweeps '
                  f'(10 mm/s, Z=0.2 mm, 1 well per pressure):')
            print(f'      sweep offset {float(r["sweep_offset_laden_minus_free"]):+.4f}'
                  f'   curve RMS {float(r["sweep_curve_RMS_difference"]):.4f}')
            if str(r['sweep_onset_shift_kPa']).strip():
                print(f'      extrusion onset shift '
                      f'{float(r["sweep_onset_shift_kPa"]):+.2f} kPa   '
                      f'peak SF shift {float(r["sweep_peak_SF_shift"]):+.4f}')
            print(f'      sign agreement with the screen: '
                  f'{r["screen_and_sweep_agree_on_sign"]}')
 
    print('\n' + '-' * 78)
    print('  PART 2: what does the cell-free route buy?')
    print('-' * 78)
    for _, r in p2.iterrows():
        print(f'\n  {r["formulation"]}')
        print(f'    ceiling         SF {r["ceiling_SF"]:.4f}  at '
              f'{r["ceiling_condition"]}')
        print(f'    naive transfer  SF {r["naive_transfer_SF"]:.4f}  at '
              f'{r["naive_transfer_condition"]}')
        print(f'                    regret {r["naive_regret"]:+.4f}, '
              f'{100 * r["naive_fraction_of_ceiling"]:.1f}% of the ceiling, '
              f'{r["naive_transfer_percentile_in_cell_laden_screen"]:.0f}th '
              f'percentile of the cell-laden screen')
        if str(r['model_transfer_SF']).strip():
            print(f'    model transfer  SF {float(r["model_transfer_SF"]):.4f}  at '
                  f'{r["model_transfer_condition"]}')
            print(f'                    regret {float(r["model_regret"]):+.4f}, '
                  f'{100 * float(r["model_fraction_of_ceiling"]):.1f}% of the '
                  f'ceiling')
            print(f'    VALUE ADDED BY THE MODEL '
                  f'{float(r["value_added_by_the_model"]):+.4f} SF over simply '
                  f'reusing the cell-free winner')
            print(f'    cell-laden wells: {r["cell_laden_wells_for_the_transfer"]} '
                  f'for the validated recommendation against '
                  f'{r["cell_laden_wells_for_the_ceiling"]} for the full screen')
        else:
            print(f'    model transfer  not available '
                  f'(no cell-free-derived print found for this formulation)')
 
    p1.to_csv(outdir / 'ink_difference_metrics.csv', sep=';', index=False)
    p2.to_csv(outdir / 'transfer_benefit_metrics.csv', sep=';', index=False)
    for _, r in p1.iterrows():
        pd.DataFrame([r]).to_csv(
            outdir / f'ink_difference_{slug(r["cell_laden_category"])}.csv',
            sep=';', index=False)
 
    figs = []
    try:
        figs.extend(fig_scatter(paired, p1, outdir, a))
        figs.append(fig_offset(paired, p1, outdir, a))
        if not p2.empty:
            figs.append(fig_benefit(p2, outdir, a))
    except ImportError:
        print('\n  [WARN] matplotlib missing, no figures written')
 
    print(f'\n  Wrote {len(list(outdir.glob("*.csv")))} table(s) and '
          f'{len(figs)} figure(s)\n  -> {outdir}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Quantify the cell-free / cell-laden difference and what '
                    'transferring a cell-free calibration is worth.')
    ap.add_argument('--combined_csv', required=True)
    ap.add_argument('--sweep_dir', default=None)
    ap.add_argument('--validation_csv', default=None,
                    help='validation_sf_summary.csv, for the printed '
                         'cell-free-derived recommendation')
    ap.add_argument('--outdir', default='results/13_ink_difference')
    ap.add_argument('--top_k', type=int, default=5)
    ap.add_argument('--title', default='')
    ap.add_argument('--ymin', type=float, default=0.3)
    ap.add_argument('--ymax', type=float, default=0.8)
    ap.add_argument('--tick_size', type=float, default=16)
    ap.add_argument('--label_size', type=float, default=18)
    ap.add_argument('--legend_size', type=float, default=13)
    ap.add_argument('--marker_size', type=float, default=18,
                    help='Marker size for the benefit figure')
    ap.add_argument('--show_offset_lines', action='store_true',
                    help='Draw the mean offset and its 95 percent limits of agreement '
                         'on the Bland-Altman figure. Off by default; they are '
                         'labelled in the legend when on.')
    ap.add_argument('--show_fit_lines', action='store_true',
                    help='Draw a least-squares best-fit line in each panel of '
                         'fig_paired_scatter, labelled simply "best fit". '
                         'Off by default.')
    ap.add_argument('--n_replicates', type=int, default=6,
                    help='Wells per LHS condition, stated in the figure titles '
                         '(default 6)')
    ap.add_argument('--scatter_size', type=float, default=12,
                    help='Point size in the paired scatter and Bland-Altman '
                         'figures (default 12)')
    ap.add_argument('--onset_sf', type=float, default=0.1,
                    help='SF threshold used for the sweep onset shift '
                         '(default 0.1, matching sweep_fingerprint.py)')
    ap.add_argument('--panel_width', type=float, default=3.8,
                    help='Width in inches of each standalone paired-scatter '
                         'panel. Smaller means larger text once LaTeX scales '
                         'the image (default 3.8)')
    ap.add_argument('--panel_height', type=float, default=3.8,
                    help='Height in inches of each standalone panel '
                         '(default 3.8)')
    ap.add_argument('--panel_font_scale', type=float, default=1.0,
                    help='Multiplies tick, label and legend sizes in the '
                         'paired-scatter panels only (default 1.0)')
    ap.add_argument('--panel_title', choices=['full', 'short', 'none'],
                    default='none',
                    help="Title on each panel. Default 'none': the formulation "
                         "belongs in the subfigure caption, and repeating it "
                         "above the panel is redundant. 'short' adds the "
                         "formulation and Spearman rho, 'full' adds n as well.")
    ap.add_argument('--axis_min', type=float, default=0.0,
                    help='Lower limit shared by both axes of every panel '
                         '(default 0)')
    ap.add_argument('--axis_max', type=float, default=0.0,
                    help='Upper limit shared by both axes of every panel. '
                         '0 means take it from the data, rounded up to a whole '
                         'tick, so all panels still match (default 0)')
    ap.add_argument('--tick_step', type=float, default=0.2,
                    help='Spacing of the ticks and grid lines in the paired '
                         'scatter, identical in every panel (default 0.2)')
    ap.add_argument('--combined_scatter', action='store_true',
                    help='Also write the old single figure with all panels '
                         'side by side')
    ap.add_argument('--legend_first_panel_only', action='store_true',
                    help='Draw the legend only on the first panel. Off by '
                         'default, since each panel is now a standalone '
                         'subfigure')
    ap.add_argument('--no_tex', dest='write_tex', action='store_false',
                    help='Do not write the LaTeX subfigure snippet')
    ap.add_argument('--subfig_width', type=float, default=0,
                    help='Subfigure width as a fraction of \\linewidth in the '
                         'TeX snippet (default: fit all panels on one row)')
    ap.add_argument('--no_autofit', dest='autofit', action='store_false',
                    help='Keep --panel_width and --panel_height exactly as '
                         'given, even if that cuts off an axis label')
    ap.add_argument('--pad_pt', type=float, default=2.0,
                    help='Padding in points between the panel content and the '
                         'figure edge (default 2)')
    ap.add_argument('--bbox_tight', action='store_true',
                    help='Crop the saved panels to their content. Off by '
                         'default: the layout already leaves no excess margin, '
                         'and cropping can clip an axis label')
    ap.add_argument('--pad_inches', type=float, default=0.02,
                    help='Margin kept when --bbox_tight is used (default 0.02)')
    ap.add_argument('--dpi', type=int, default=300,
                    help='Raster resolution of the png files (default 300)')
    ap.add_argument('--marker_alpha', type=float, default=0.85)
    ap.add_argument('--legend_loc', default='best')
    ap.add_argument('--fig_width', type=float, default=0)
    ap.add_argument('--fig_height', type=float, default=0)
    main(ap.parse_args())