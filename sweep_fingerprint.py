"""
sweep_fingerprint.py
--------------------
Step 1 of the intermediate analysis: turn each category's pressure-sweep curve
into the per-category fingerprint the plan uses as the model input, plot all
curves together, and test whether they actually separate the formulations.
 
Why this runs first
-------------------
The plan makes Aim 1 conditional on one check: "the pressure-sweep curves must
actually separate the formulations in a replicate-consistent way ... If they do
not, the fingerprint carries little information and Aim 1 should fall back to
the optimum-comparison form." Everything downstream depends on the answer, and
this is the cheapest thing to compute, so it goes first.
 
Descriptors, and which are actually measurable
-----------------------------------------------
The plan names six: extrusion onset, printable-window width, peak-SF pressure,
rise slope, collapse slope, area under the curve.
 
A sweep that is still rising at its highest pressure has no peak inside the
sampled range, so peak pressure, window width and collapse slope do not exist
for it. This script reports those as empty rather than inventing a value at the
range edge, and sets `truncated=yes`. On cell_gelma_10_60 that is the case: SF
climbs monotonically to the last well at 120 kPa.
 
Noise for the separation check
-------------------------------
Each sweep pressure is printed once, so the sweep carries NO replicate-noise
estimate of its own. The check the plan specifies ("beyond replicate noise")
therefore cannot be computed from sweep data alone. As a stand-in this script
uses the median SF_std from the SAME category's 48-well plate, where each
condition has 6 replicates. That is a proxy from a different plate, not the
sweep's own repeatability, and it is labelled as such everywhere it is used.
Treat a "separated" verdict from it as suggestive, not settled.
 
Usage
-----
    python sweep_fingerprint.py --data_dir <folder with the summary CSVs> \
        [--output_dir <folder>] [--onset_sf 0.02]
 
    The folder should hold the per-category files named
        <category>_sf_summary_sweep.csv     (required)
        <category>_sf_summary_48well.csv    (optional, used for the noise proxy)
 
Outputs
-------
    sweep_curves.png     all curves on one axis
    fingerprints.csv     one row per category
    console              the separation check
"""
 
import argparse
import csv
import re
from pathlib import Path
 
import numpy as np
 
# Validated categorical palette (light mode). Checked with the dataviz
# validator: worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6.
# The contrast WARN on slots 3/4/5 is discharged by the legend plus the
# fingerprints.csv table view.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
INK      = '#1a1a19'
INK_SOFT = '#6b6a63'
GRID     = '#e5e4df'
SURFACE  = '#fcfcfb'
 
 
def load_summary(path):
    with open(path, newline='') as f:
        head = f.readline()
    delim = ';' if head.count(';') >= head.count(',') else ','
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=delim))
 
 
def category_of(path, suffix):
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem
 
 
def legend_label(cat):
    m = re.fullmatch(r'(cell_)?gelma_(\d+(?:\.\d+)?)_(\d+)', cat)
    if not m:
        return cat
    cell_prefix, concentration, dof = m.groups()
    if concentration == '7':
        concentration = '7.5'
    state = 'cell-laden' if cell_prefix else 'cell-free'
    # return f'GelMA {concentration}% DoF {dof}  {state}'
    return f'GelMA {concentration}% DoF {dof}  {state}'
 
 
def curve_from(rows):
    """(pressures, SF) sorted by pressure."""
    pts = []
    for r in rows:
        try:
            pts.append((float(r['Pressure_kPa']), float(r['SF_mean'])))
        except (KeyError, TypeError, ValueError):
            continue
    pts.sort()
    return np.array([p for p, _ in pts]), np.array([s for _, s in pts])
 
 
def _tail_only(P, S, tail_kPa, tail_n):
    """tail_slope alone, for the branch where nothing cleared the onset."""
    if tail_n is not None:
        n = int(min(max(2, tail_n), len(S)))
    else:
        n = int(min(max(2, np.sum(P >= P[-1] - float(tail_kPa))), len(S)))
    return {'tail_slope': round(float(np.polyfit(P[-n:], S[-n:], 1)[0]), 6),
            'tail_slope_n': n, 'tail_p_lo': round(float(P[-n]), 2)}
 
 
def fingerprint(P, S, onset_sf, smooth_n=3, tail_kPa=40.0, tail_n=None):
    """Descriptors from one SF-vs-pressure curve. Unmeasurable ones are None."""
    fp = {
        'n_points':      len(P),
        'n_zero':        int(np.sum(S <= 0)),
        'p_min':         float(P.min()),
        'p_max':         float(P.max()),
        'sf_at_p_max':   float(S[-1]),
    }
 
    above = np.flatnonzero(S > onset_sf)
    if above.size == 0:
        fp.update(onset_kPa=None, rise_slope=None, peak_sf=None, peak_kPa=None,
                  truncated='n/a', window_kPa=None, collapse_slope=None,
                  auc_norm=0.0, sf_median=round(float(np.median(S)), 4),
                  sf_iqr=0.0, peak_three_sf_mean=None, peak_three_sf_std=None,
                  peak_three_n=None, peak_three_p_lo=None, peak_three_p_hi=None,
                  sf_at_p_max_three_mean=None, last_three_sf_std=None,
                  last_three_n=None, last_three_p_lo=None,
                  **_tail_only(P, S, tail_kPa, tail_n))
        return fp
 
    i = int(above[0])
    if i == 0:
        onset = float(P[0])          # already extruding at the lowest pressure
        fp['onset_censored'] = 'left'
    else:
        # linear interpolation between the last sub-threshold point and this one
        p0, p1, s0, s1 = P[i - 1], P[i], S[i - 1], S[i]
        onset = float(p0 + (onset_sf - s0) * (p1 - p0) / (s1 - s0)) \
            if s1 != s0 else float(p1)
        fp['onset_censored'] = ''
    fp['onset_kPa'] = round(onset, 2)
 
    k = int(np.argmax(S))
    fp['peak_sf'] = round(float(S[k]), 4)
    truncated = (k == len(S) - 1)
    fp['truncated'] = 'yes' if truncated else 'no'
 
    # rise slope over onset -> peak
    if k > i:
        sl = np.polyfit(P[i:k + 1], S[i:k + 1], 1)[0]
        fp['rise_slope'] = round(float(sl), 6)
    else:
        fp['rise_slope'] = None
 
    if truncated:
        # No maximum inside the sampled range: these three are not measurable.
        # Reporting P[-1] as "the peak" would be an artefact of where the sweep
        # stopped, and window width and collapse slope would be pure fiction.
        fp['peak_kPa'] = None
        fp['window_kPa'] = None
        fp['collapse_slope'] = None
    else:
        fp['peak_kPa'] = round(float(P[k]), 2)
        half = 0.5 * S[k]
        inw = np.flatnonzero(S >= half)
        fp['window_kPa'] = round(float(P[inw[-1]] - P[inw[0]]), 2)
        sl = np.polyfit(P[k:], S[k:], 1)[0] if len(P) - k >= 3 else None
        fp['collapse_slope'] = round(float(sl), 6) if sl is not None else None
 
    # AUC normalised by pressure span, so it is a mean SF over the swept range
    fp['auc_norm'] = round(float(np.trapezoid(S, P) / (P[-1] - P[0])), 4)
 
    # ---- level descriptors over the swept range -----------------------------
    # There is deliberately NO sf_mean column. auc_norm above already IS the
    # mean: on a uniform pressure grid the trapezoid integral over the span is
    # the arithmetic mean with the endpoints half-weighted. Measured over 2000
    # simulated sweeps the two correlated at 0.9998 and differed by at most
    # 0.013, and on the four real 7.5% curves they ordered the categories
    # identically (Spearman +1.000). A second column for the same number would
    # only add a dimension to a fingerprint that finding A5 already says has
    # too many for the number of training categories.
    #
    # sf_median survives because each sweep pressure is printed ONCE, so a
    # single bad well moves the mean while the median ignores it. That matters
    # given finding A4, where one sweep point sat 4.3 LHS standard deviations
    # below the plate value at the same parameters.
    #
    # sf_iqr is the spread of SF across the swept range: steep curves have a
    # wide IQR, flat or dead ones a narrow one.
    fp['sf_median'] = round(float(np.median(S)), 4)
    q1, q3 = np.percentile(S, [25, 75])
    fp['sf_iqr']    = round(float(q3 - q1), 4)
 
    # ---- neighbourhood-averaged descriptors ---------------------------------
    # Both of the single-well descriptors above are read off ONE printed well,
    # and a sweep has no replicates, so a single bad well sets the value with
    # nothing to contradict it. These two average a short run of adjacent
    # pressures instead.
    #
    #   peak_three_sf_mean   the peak well and its two neighbours
    #   sf_at_p_max_three_mean   the last three wells of the sweep
    #
    # The window is centred on the RAW argmax, not on a smoothed curve, so the
    # location still comes from the data as measured. n and the pressure span
    # actually averaged are recorded, because at an edge only two wells exist
    # and a 3-well window at a 5 kPa step spans 10 kPa of real pressure, which
    # is a different quantity from a point reading.
    half = max(0, (smooth_n - 1) // 2)
 
    lo = max(0, k - half)
    hi = min(len(S), k + half + 1)
    seg = S[lo:hi]
    fp['peak_three_sf_mean'] = round(float(np.mean(seg)), 4)
    fp['peak_three_sf_std']  = round(float(np.std(seg)), 4)
    fp['peak_three_n']       = int(seg.size)
    fp['peak_three_p_lo']    = round(float(P[lo]), 2)
    fp['peak_three_p_hi']    = round(float(P[hi - 1]), 2)
 
    # ---- tail_slope: the always-defined replacement for collapse_slope ------
    # collapse_slope is blank whenever the sweep never peaked, and blank again
    # when it peaked too near the end to fit a line through the descent. On the
    # six-category file that is 4 blanks out of 6, and a single blank disqualifies
    # the whole column from a design matrix.
    #
    # Filling those blanks with 0 would be WRONG. A sweep still climbing at its
    # last well has not been shown not to collapse; it was not swept far enough
    # to find out. Zero would assert "no collapse" from no evidence, and put an
    # untested category at the same value as a genuinely flat one.
    #
    # tail_slope instead fits a line through the LAST tail_n wells regardless of
    # where the peak is. It is defined for every curve, and its sign carries the
    # meaning collapse_slope was reaching for:
    #     positive  still rising at the top of the swept range
    #     ~zero     plateaued
    #     negative  collapsing
    # collapse_slope is kept unchanged alongside it, for the protocol's six
    # named descriptors and for reporting.
    # The window is specified in kPa, not in wells, because a well count means
    # different things at different sweep steps. Measured on the four 30-140
    # curves, a 40 kPa tail reproduces collapse_slope almost exactly wherever
    # collapse_slope exists (-0.00155 vs -0.00155, -0.00411 vs -0.00407) while
    # a 20 kPa tail reported the one genuinely collapsing category as flat.
    if tail_n is not None:
        n_tail = int(min(max(2, tail_n), len(S)))
    else:
        n_tail = int(max(2, np.sum(P >= P[-1] - float(tail_kPa))))
        n_tail = min(n_tail, len(S))
    tail_fit = S[-n_tail:]
    P_tail = P[-n_tail:]
    fp['tail_slope']   = round(float(np.polyfit(P_tail, tail_fit, 1)[0]), 6)
    fp['tail_slope_n'] = n_tail
    fp['tail_p_lo']    = round(float(P_tail[0]), 2)
 
    n_last = min(smooth_n, len(S))
    tail = S[-n_last:]
    fp['sf_at_p_max_three_mean'] = round(float(np.mean(tail)), 4)
    # Spread of the last few wells. On a plateau these are near-replicates of
    # the same SF at slightly different pressures, so this is the closest thing
    # the sweep has to a repeatability estimate of its OWN. It is not a true
    # replicate std, because the pressures genuinely differ and any residual
    # slope inflates it, but it beats borrowing SF_std from a different plate,
    # which is what plan item B2 objects to.
    fp['last_three_sf_std'] = round(float(np.std(tail)), 4)
    fp['last_three_n']      = int(n_last)
    fp['last_three_p_lo']   = round(float(P[-n_last]), 2)
    return fp
 
 
def common_window_stats(cats, curves, step=None):
    """
    Level descriptors recomputed on the pressure window COMMON to every
    category, so they can be compared across categories at all.
 
    Why this is not optional. sf_mean, sf_median and auc_norm are averages over
    whatever range that category happened to be swept. Some plates here were
    printed 30-120 kPa and later ones 30-140 kPa. The extra four wells sit at
    the top of the rising flank where SF is highest, so a category swept to 140
    gets a higher mean than an identical material swept to 120, purely from
    where the sweep stopped. Feeding that to a cross-category model teaches it
    the sweep protocol, not the material.
 
    The experiment plan says the same thing in B1: "if some categories are
    characterised over a wider range than others, the fingerprints are not
    comparable and the transfer input is inconsistent by construction."
 
    Returns (lo, hi, {cat: {...}}) or (None, None, {}) if there is no overlap.
    """
    los = [P.min() for P, _ in curves.values()]
    his = [P.max() for P, _ in curves.values()]
    lo, hi = max(los), min(his)
    if not (hi > lo):
        return None, None, {}
    if step is None:
        gaps = []
        for P, _ in curves.values():
            gaps += list(np.diff(np.unique(P)))
        step = float(np.median(gaps)) if gaps else 5.0
    grid = np.arange(lo, hi + 1e-9, step)
    out = {}
    for c in cats:
        P, S = curves[c]
        Sc = np.interp(grid, P, S)
        # cw_auc_norm is the common-window mean; no separate cw_sf_mean for the
        # same reason there is no sf_mean.
        n_last = min(3, len(Sc))
        out[c] = {
            'cw_sf_median':      round(float(np.median(Sc)), 4),
            'cw_auc_norm':       round(float(np.trapezoid(Sc, grid) / (hi - lo)), 4),
            'cw_sf_at_hi':       round(float(Sc[-1]), 4),
            'cw_sf_at_hi_three': round(float(np.mean(Sc[-n_last:])), 4),
        }
    return float(lo), float(hi), out
 
 
def _rank(a):
    """Average ranks, ties shared. Avoids a scipy dependency."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    # average ranks within tie groups
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r
 
 
def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])
 
 
def collinearity_report(cats, fps, extra, thresh=0.90):
    """
    Rank-correlate every pair of descriptors ACROSS categories and name the
    redundant ones.
 
    The fingerprint is the model input, and finding A5 already notes that only
    four descriptors are usable while leave-one-category-out fits them from 5
    to 7 training points. Adding columns that carry the same ordering makes
    that worse, not better. This is the check that says whether a new
    descriptor earns its place, on the real data rather than by argument.
    """
    names = ['onset_kPa', 'rise_slope', 'tail_slope', 'peak_sf',
             'peak_three_sf_mean',
             'sf_at_p_max', 'sf_at_p_max_three_mean', 'auc_norm',
             'sf_median', 'sf_iqr',
             'cw_sf_median', 'cw_auc_norm', 'cw_sf_at_hi', 'cw_sf_at_hi_three']
    cols = {}
    for n in names:
        vals = []
        for c in cats:
            v = extra.get(c, {}).get(n, fps[c].get(n))
            vals.append(v)
        if all(v is not None for v in vals) and len(set(vals)) > 1:
            cols[n] = np.array(vals, float)
    if len(cols) < 2:
        print('  Not enough complete descriptors to correlate.')
        return
    if len(cats) < 4:
        print(f'  [NOTE] only {len(cats)} categories, so these correlations '
              f'are very noisy.')
        print( '         Treat them as a smell test, not evidence.')
    ks = list(cols)
    pairs = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r = spearman(cols[ks[i]], cols[ks[j]])
            if np.isfinite(r):
                pairs.append((abs(r), r, ks[i], ks[j]))
    pairs.sort(reverse=True)
    hi = [p for p in pairs if p[0] >= thresh]
    if hi:
        print(f'  Descriptor pairs with |Spearman| >= {thresh:g} '
              f'(one of each pair is redundant):')
        for a, r, x, y in hi:
            print(f'    {x:<14} vs {y:<14}  rho = {r:+.3f}')
    else:
        print(f'  No descriptor pair reaches |Spearman| {thresh:g}.')
    print(f'\n  Least redundant descriptors (lowest max |rho| against any other):')
    worst = {}
    for a, r, x, y in pairs:
        worst[x] = max(worst.get(x, 0), a)
        worst[y] = max(worst.get(y, 0), a)
    for k, v in sorted(worst.items(), key=lambda t: t[1])[:5]:
        print(f'    {k:<14} max |rho| {v:.3f}')
 
 
def noise_proxy(rows48):
    """Median SF_std across the 48-well plate: a stand-in for sweep repeatability."""
    stds = []
    for r in rows48 or []:
        try:
            if int(r.get('n_images', 0)) > 1:
                stds.append(float(r['SF_std']))
        except (TypeError, ValueError):
            continue
    return float(np.median(stds)) if stds else None
 
 
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


def strip_common_prefix(labels):
    """Pull the word every legend entry starts with out into a legend title.

    All six entries begin with 'GelMA', which costs about a sixth of the legend
    width on every row and pushes the entries into fewer, taller columns. Moving
    it to the legend title says it once and lets three columns fit, which keeps
    the figure short. Nothing is lost: the title is part of the legend.
    """
    parts = [str(l).split(' ') for l in labels]
    n = 0
    while all(len(p) > n + 1 for p in parts) and \
            len({p[n] for p in parts}) == 1:
        n += 1
    if n == 0:
        return list(labels), None
    prefix = ' '.join(parts[0][:n])
    return [' '.join(p[n:]) for p in parts], prefix


def plot_curves(cats, curves, out_path, onset_sf, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def build(size):
        fig, ax = _new_fig(args, size)
        for n, cat in enumerate(cats):
            P, S = curves[cat]
            ax.plot(P, S, color=SERIES[n % len(SERIES)],
                    linewidth=args.line_width, marker='o',
                    markersize=args.marker_size, markeredgecolor=SURFACE,
                    markeredgewidth=args.marker_edge_width,
                    label=legend_label(cat), zorder=3)

        ax.axhline(onset_sf, color=INK_SOFT,
                   linewidth=args.threshold_line_width,
                   linestyle=(0, (4, 4)), zorder=1)
        # The label sits on the dashed line itself, with a patch of background
        # behind it, so it stays readable at this size without covering a curve.
        left = args.annotation_loc == 'left'
        ax.annotate(f'onset SF = {onset_sf:g}',
                    xy=(ax.get_xlim()[0 if left else 1], onset_sf),
                    xytext=(6 if left else -6, 5), textcoords='offset points',
                    ha='left' if left else 'right', va='bottom',
                    fontsize=args.annotation_font_size, color=INK, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor=SURFACE,
                              edgecolor='none', alpha=0.85))

        ax.set_xlabel('Pressure (kPa)', fontsize=args.axis_title_size, color=INK)
        ax.set_ylabel('Shape fidelity SF', fontsize=args.axis_title_size,
                      color=INK)
        ax.grid(True, color=GRID, linewidth=args.grid_line_width, zorder=0)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=args.label_tick_size)

        handles, labels = ax.get_legend_handles_labels()
        title = None
        if args.legend_strip_common:
            # The shared formulation ('GelMA 10% DoF 60') is removed from the
            # entries. By default it is NOT shown as a legend title either: in a
            # manuscript each panel carries it in the subfigure caption, so a
            # title here is redundant and costs vertical space.
            labels, prefix = strip_common_prefix(labels)
            if args.legend_title_common:
                title = prefix
        if args.legend_two_line:
            # break at the gap between the formulation and the cell state, so
            # 'cell-laden' is never split across two lines by a character wrap
            labels = [re.sub(r'\s{2,}', '\n', l) for l in labels]
        _legend(fig, ax, handles, labels, args,
                dict(frameon=False, fontsize=args.legend_font_size,
                     title=title, title_fontsize=args.legend_font_size,
                     handlelength=1.5, handletextpad=0.5, columnspacing=1.2,
                     labelspacing=0.35, borderaxespad=0.1, borderpad=0.2))
        return fig

    fig = _render(build, args, (args.figure_width, args.figure_height))
    _save(fig, out_path, args)
    plt.close(fig)


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    SW, LH = '_sf_summary_sweep.csv', '_sf_summary_48well.csv'
    sweep_files = sorted(data_dir.glob(f'*{SW}'))
    if not sweep_files:
        raise SystemExit(f'No *{SW} files in {data_dir}')
 
    cats, curves, fps = [], {}, {}
    for p in sweep_files:
        cat = category_of(p, SW)
        P, S = curve_from(load_summary(p))
        if len(P) < 3:
            print(f'[SKIP] {cat}: only {len(P)} usable point(s)')
            continue
        cats.append(cat)
        curves[cat] = (P, S)
        fp = fingerprint(P, S, args.onset_sf, args.smooth_n,
                         args.tail_kPa, args.tail_n)
        fp['onset_sf_threshold'] = args.onset_sf
        # B4: onset is one of only a few surviving descriptors and it moves a
        # lot with the threshold, so the threshold is recorded in the output and
        # two alternatives are reported alongside it.
        for t in (0.04, 0.20):
            fp[f'onset_kPa_at_{t:g}'] = fingerprint(
                P, S, t, args.smooth_n, args.tail_kPa,
                args.tail_n).get('onset_kPa')
        lh = data_dir / f'{cat}{LH}'
        fp['noise_proxy_sf_std'] = (round(noise_proxy(load_summary(lh)), 4)
                                    if lh.exists() else None)
        fps[cat] = fp
 
    print(f'Categories: {len(cats)}')
    print(f'{"category":<24}{"onset":>8}{"peak SF":>9}{"peak P":>8}'
          f'{"trunc":>7}{"rise":>10}{"AUC":>8}{"zeros":>7}')
    for cat in cats:
        f = fps[cat]
        def g(k, fmtstr='{:.4g}'):
            v = f.get(k)
            return '-' if v is None else fmtstr.format(v)
        print(f'{cat:<24}{g("onset_kPa"):>8}{g("peak_sf"):>9}'
              f'{g("peak_kPa"):>8}{f["truncated"]:>7}{g("rise_slope"):>10}'
              f'{g("auc_norm"):>8}{f["n_zero"]:>7}')
 
    print(f'\n{"category":<24}{"collapse_slope":>16}{"tail_slope":>13}'
          f'{"over kPa":>12}  meaning')
    for cat in cats:
        f = fps[cat]
        cs = f.get('collapse_slope')
        ts = f.get('tail_slope')
        # label only. 1e-3 SF/kPa over a 20 kPa tail is 0.02 SF, about one
        # single-well noise, so anything flatter is called a plateau.
        mean_txt = ('still rising' if ts is not None and ts > 1e-3 else
                    'collapsing' if ts is not None and ts < -1e-3 else
                    'plateaued')
        print(f'{cat:<24}{("-" if cs is None else f"{cs:+.6f}"):>16}'
              f'{("-" if ts is None else f"{ts:+.6f}"):>13}'
              f'{f.get("tail_p_lo", "-"):>8}-{f["p_max"]:g}  {mean_txt}')
    n_blank = sum(1 for c in cats if fps[c].get('collapse_slope') is None)
    if n_blank:
        print(f'\ncollapse_slope is blank for {n_blank}/{len(cats)} categor(ies): '
              f'either the sweep never\npeaked, or it peaked too close to the '
              f'end to fit a line through the descent.\nA blank disqualifies the '
              f'whole column from a design matrix, and filling it with 0\nwould '
              f'assert "does not collapse" from no evidence. Use tail_slope as '
              f'the feature\nand keep collapse_slope for reporting.')
 
    trunc = [c for c in cats if fps[c]['truncated'] == 'yes']
    if trunc:
        print(f'\n[NOTE] {len(trunc)}/{len(cats)} categor(ies) are still rising at '
              f'the highest swept pressure:')
        print(f'       {trunc}')
        print('       For these, peak pressure / window width / collapse slope do '
              'not exist in the\n       sampled range and are left empty. The '
              'fingerprint reduces to onset, rise\n       slope and AUC.')
 
    # ------------------------------------------- common-window level metrics
    print('\n' + '=' * 72)
    print('  LEVEL DESCRIPTORS  (mean / median / AUC)')
    print('=' * 72)
    lo, hi, extra = common_window_stats(cats, curves)
    if lo is None:
        print('  Categories share no overlapping pressure window, so mean, '
              'median and AUC\n  cannot be compared across them at all.')
        extra = {}
    else:
        spans = {c: (curves[c][0].min(), curves[c][0].max()) for c in cats}
        widest = max(b - a for a, b in spans.values())
        print(f'  Common window: {lo:g} to {hi:g} kPa '
              f'({hi - lo:g} kPa wide; widest single sweep is {widest:g} kPa)')
        odd = [c for c in cats if spans[c] != (lo, hi)]
        if odd:
            print(f'  {len(odd)} categor(ies) are swept OUTSIDE this window and '
                  f'are truncated to it:')
            for c in odd:
                a, b = spans[c]
                print(f'    {c:<24} own range {a:g}-{b:g} kPa')
            print('  Their raw sf_mean / sf_median / auc_norm are NOT comparable '
                  'with the others.\n  Use the cw_ columns for anything '
                  'cross-category.')
        print(f'\n  {"category":<24}{"sf_median":>11}{"sf_iqr":>8}'
              f'{"cw_median":>11}{"cw_auc":>9}{"cw_hi":>8}{"cw_hi_3":>9}')
        for c in cats:
            f, e = fps[c], extra[c]
            print(f'  {c:<24}{f["sf_median"]:>11.4f}{f["sf_iqr"]:>8.4f}'
                  f'{e["cw_sf_median"]:>11.4f}{e["cw_auc_norm"]:>9.4f}'
                  f'{e["cw_sf_at_hi"]:>8.4f}{e["cw_sf_at_hi_three"]:>9.4f}')
        for c in cats:
            fps[c].update(extra[c])
 
    print('\n' + '=' * 72)
    print(f'  SINGLE-WELL vs {args.smooth_n}-WELL AVERAGED DESCRIPTORS')
    print('=' * 72)
    print('  A sweep prints each pressure ONCE, so peak_sf and sf_at_p_max are '
          'each read off a\n  single well with no replicate to contradict it. '
          'The averaged columns take a run\n  of adjacent pressures instead.')
    print(f'\n  {"category":<24}{"peak_sf":>9}{"peak_3":>9}{"delta":>8}'
          f'{"span kPa":>11}{"sf@pmax":>9}{"last_3":>8}{"delta":>8}{"sd_3":>7}')
    for c in cats:
        f = fps[c]
        if f.get('peak_three_sf_mean') is None:
            print(f'  {c:<24}  (no extrusion above the onset threshold)')
            continue
        dp = f['peak_three_sf_mean'] - f['peak_sf']
        dl = f['sf_at_p_max_three_mean'] - f['sf_at_p_max']
        span = f'{f["peak_three_p_lo"]:g}-{f["peak_three_p_hi"]:g}'
        flag = '' if f['peak_three_n'] == args.smooth_n else f' n={f["peak_three_n"]}'
        print(f'  {c:<24}{f["peak_sf"]:>9.4f}{f["peak_three_sf_mean"]:>9.4f}'
              f'{dp:>+8.4f}{span:>11}{f["sf_at_p_max"]:>9.4f}'
              f'{f["sf_at_p_max_three_mean"]:>8.4f}{dl:>+8.4f}'
              f'{f["last_three_sf_std"]:>7.4f}{flag}')
    edge = [c for c in cats
            if fps[c].get('peak_three_n') not in (None, args.smooth_n)]
    if edge:
        print(f'\n  [NOTE] {len(edge)} categor(ies) peak at an END of the swept '
              f'range, so only {args.smooth_n - 1} wells\n         were '
              f'available to average: {edge}. Their peak_three_sf_mean is not on '
              f'the\n         same footing as the others.')
    sds = [fps[c]['last_three_sf_std'] for c in cats
           if fps[c].get('last_three_sf_std') is not None]
    if sds:
        print(f'\n  Median spread of the last {args.smooth_n} wells across '
              f'categories: {np.median(sds):.4f}')
        print('  On a plateau those wells are near-replicates, so this is the '
              'closest thing the\n  sweep has to a repeatability estimate of '
              'its OWN. It is inflated by any residual\n  slope, so it is an '
              'UPPER bound on the noise, but unlike the 48-well proxy below it\n'
              '  comes from the same plate.')
 
    print('\n' + '=' * 72)
    print('  DESCRIPTOR REDUNDANCY')
    print('=' * 72)
    collinearity_report(cats, fps, extra)
 
    # ------------------------------------------------------- separation check
    print('\n' + '=' * 72)
    print('  SEPARATION CHECK')
    print('=' * 72)
    if len(cats) < 2:
        print('  Need at least 2 categories.')
    else:
        grid = curves[cats[0]][0]
        for c in cats[1:]:
            grid = np.union1d(grid, curves[c][0])
        interp = {c: np.interp(grid, *curves[c]) for c in cats}
        proxies = [fps[c]['noise_proxy_sf_std'] for c in cats
                   if fps[c]['noise_proxy_sf_std'] is not None]
        noise = float(np.median(proxies)) if proxies else None
 
        print(f'  Common pressure grid: {len(grid)} point(s), '
              f'{grid.min():g} to {grid.max():g} kPa')
        if noise is None:
            print('  No 48-well files found, so there is no noise proxy at all. '
                  'Pairwise\n  distances are reported without a reference scale.')
        else:
            print(f'  Noise proxy (median 48-well SF_std, NOT sweep '
                  f'repeatability): {noise:.4f}')
        print(f'\n  Pairwise RMS difference between curves'
              f'{" (x noise proxy)" if noise else ""}:')
        pairs = []
        for a in range(len(cats)):
            for b in range(a + 1, len(cats)):
                d = float(np.sqrt(np.mean((interp[cats[a]] - interp[cats[b]]) ** 2)))
                pairs.append((d, cats[a], cats[b]))
        pairs.sort()
        for d, ca, cb in pairs:
            ratio = f'{d / noise:>6.1f}x' if noise else ''
            print(f'    {ca:<24} vs {cb:<24} RMS {d:.4f}  {ratio}')
 
        if noise:
            worst, ca, cb = pairs[0]
            print(f'\n  Closest pair: {ca} vs {cb}, RMS {worst:.4f} '
                  f'= {worst / noise:.1f}x the noise proxy.')
            if worst < 2 * noise:
                print('  VERDICT: the closest pair is within ~2x the noise proxy. '
                      'The curves may not\n  separate these formulations. This is '
                      'the case the plan says should trigger the\n  fallback '
                      'framing for Aim 1.')
            else:
                print('  VERDICT: every pair is separated by more than 2x the '
                      'noise proxy. Suggestive\n  that the fingerprint carries '
                      'category information, but remember the proxy comes\n  '
                      'from a different plate, so this is not the replicate-'
                      'consistency test itself.')
 
    # ------------------------------------------------------------- write out
    cols = ['category', 'onset_kPa', 'onset_censored', 'onset_sf_threshold',
            'onset_kPa_at_0.02', 'onset_kPa_at_0.1',
            'rise_slope', 'peak_sf', 'peak_kPa', 'window_kPa', 'collapse_slope',
            'tail_slope', 'tail_slope_n', 'tail_p_lo',
            'peak_three_sf_mean', 'peak_three_sf_std', 'peak_three_n',
            'peak_three_p_lo', 'peak_three_p_hi',
            'auc_norm', 'sf_median', 'sf_iqr',
            'sf_at_p_max_three_mean', 'last_three_sf_std', 'last_three_n',
            'last_three_p_lo',
            'cw_sf_median', 'cw_auc_norm', 'cw_sf_at_hi', 'cw_sf_at_hi_three',
            'truncated', 'n_points', 'n_zero', 'p_min', 'p_max', 'sf_at_p_max',
            'noise_proxy_sf_std']
    fp_path = out_dir / 'fingerprints.csv'
    with open(fp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=';',
                           extrasaction='ignore')
        w.writeheader()
        for cat in cats:
            row = {'category': cat}
            row.update({k: ('' if fps[cat].get(k) is None else fps[cat].get(k))
                        for k in cols[1:]})
            w.writerow(row)
 
    png = out_dir / 'sweep_curves.png'
    try:
        plot_curves(cats, curves, png, args.onset_sf, args)
        print(f'\nCurves       -> {png}')
    except ImportError:
        print('\n[WARN] matplotlib not installed, no plot written. '
              'pip install matplotlib')
    print(f'Fingerprints -> {fp_path}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract pressure-sweep fingerprints and test separation.')
    ap.add_argument('--data_dir', required=True,
                    help='Folder holding *_sf_summary_sweep.csv (and optionally '
                         '*_sf_summary_48well.csv for the noise proxy)')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--tail_kPa', type=float, default=40.0,
                    help='Width in kPa of the window at the top of the sweep '
                         'that tail_slope is fitted through (default: 40). '
                         'tail_slope is the always-defined stand-in for '
                         'collapse_slope, which is blank whenever the sweep '
                         'never peaked. Specified in kPa rather than in wells '
                         'so it means the same thing at any sweep step. At 40 '
                         'kPa it reproduced collapse_slope on the 30-140 '
                         'curves; at 20 kPa it read a genuinely collapsing '
                         'category as flat.')
    ap.add_argument('--tail_n', type=int, default=None,
                    help='Override --tail_kPa with an exact well count')
    ap.add_argument('--smooth_n', type=int, default=3,
                    help='How many adjacent wells to average for '
                         'peak_three_sf_mean and sf_at_p_max_three_mean '
                         '(default: 3). Odd values keep the peak window '
                         'centred on the peak well.')
    ap.add_argument('--onset_sf', type=float, default=0.2,
                    help='SF above which extrusion counts as started '
                         '(default: 0.2). NOTE: the help here used to say 0.02 '
                         'while the code used 0.2, so any onset value produced '
                         'before this was fixed was computed at 0.2, not 0.02. '
                         'The threshold is now written into fingerprints.csv '
                         'and onset is also reported at 0.02 and 0.1.')
    plot = ap.add_argument_group('plot appearance')
    plot.add_argument('--label_tick_size', type=float, default=15,
                      help='Axis tick-label font size (default: 15)')
    plot.add_argument('--legend_font_size', type=float, default=15,
                      help='Legend font size (default: 15)')
    plot.add_argument('--axis_title_size', type=float, default=17,
                      help='X- and y-axis title font size (default: 17)')
    plot.add_argument('--plot_title_size', type=float, default=16,
                      help='Main plot title font size (default: 12)')
    plot.add_argument('--plot_title_pad', type=float, default=10,
                      help='Padding above the main plot title (default: 10)')
    plot.add_argument('--annotation_font_size', type=float, default=15,
                      help='Onset-threshold annotation font size (default: 15)')
    plot.add_argument('--annotation_loc', default='left',
                      choices=['left', 'right'],
                      help='Which end of the onset line carries its label '
                           '(default: left)')
    plot.add_argument('--marker_size', type=float, default=6,
                      help='Curve marker size (default: 6)')
    plot.add_argument('--line_width', type=float, default=2,
                      help='Curve line width (default: 2)')
    plot.add_argument('--marker_edge_width', type=float, default=1.2,
                      help='Curve marker-edge width (default: 1.2)')
    plot.add_argument('--grid_line_width', type=float, default=0.8,
                      help='Grid line width (default: 0.8)')
    plot.add_argument('--threshold_line_width', type=float, default=1,
                      help='Onset-threshold line width (default: 1)')
    plot.add_argument('--figure_width', type=float, default=4.2,
                      help='Starting figure width in inches. Narrow on purpose: '
                           'the printed text size is set by the ratio of the '
                           'font to the figure width (default: 4.2)')
    plot.add_argument('--figure_height', type=float, default=3.2,
                      help='Starting figure height in inches (default: 3.2)')
    plot.add_argument('--dpi', type=int, default=300,
                      help='Saved PNG resolution (default: 300). The pdf and '
                           'eps written beside it are vector.')
    plot.add_argument('--eps_dpi', type=int, default=900)
    plot.add_argument('--legend_placement', default='auto',
                      choices=['auto', 'inside', 'above', 'right'],
                      help="Where the legend goes. 'auto' keeps it inside "
                           "unless it covers the curves, in which case it moves "
                           "above the axes (default: auto)")
    plot.add_argument('--legend_loc', default='lower right',
                      help='Legend position when it is placed INSIDE the axes')
    plot.add_argument('--legend_ncol', type=int, default=0,
                      help='Legend columns. 0 picks the most that fit the width')
    plot.add_argument('--no_legend_two_line', dest='legend_two_line',
                      action='store_false',
                      help='Keep each legend entry on one line instead of '
                           'breaking it between the formulation and the cell '
                           'state. One line is wider, so fewer columns fit.')
    plot.add_argument('--no_legend_strip', dest='legend_strip_common',
                      action='store_false',
                      help='Keep the shared word (GelMA) on every legend entry '
                           'instead of moving it to the legend title. Keeping '
                           'it makes the legend wider and therefore taller.')
    plot.add_argument('--legend_title_common', action='store_true',
                      help='Show the shared formulation (e.g. "GelMA 10%% DoF '
                           '60") as a legend title above the entries. Off by '
                           'default: as a subfigure it belongs in the caption.')
    plot.add_argument('--legend_wrap', type=int, default=0,
                      help='Wrap legend labels at this many characters, so the '
                           'ink names fit in narrow columns instead of '
                           'stretching the figure (0 off)')
    plot.add_argument('--legend_overlap_tol', type=float, default=0.02,
                      help='With auto placement, how much of the legend box may '
                           'sit on the curves before it moves out')
    plot.add_argument('--pad_pt', type=float, default=1.5,
                      help='Padding in points between content and figure edge '
                           '(default: 1.5)')
    plot.add_argument('--max_fig_width', type=float, default=5.2,
                      help='The figure is never grown wider than this, in '
                           'inches (default: 5.2). Keeping it small is what '
                           'makes the text large relative to the panel once the '
                           'panel is scaled into a subfigure.')
    plot.add_argument('--max_fig_height', type=float, default=4.2,
                      help='The figure is never grown taller than this '
                           '(default: 4.2)')
    plot.add_argument('--no_autofit', dest='autofit', action='store_false',
                      help='Keep the figure size exactly as given')
    main(ap.parse_args())