"""
draw_target_geometry_pressure-sweep.py
--------------------------------------
Same as draw_target_geometry.py, but it also understands the pressure-sweep
naming `sweep_N.tif` and converts N into the (col, row) the well actually
occupied, so ROW_STEP / COL_STEP apply to sweeps exactly as they do to the
48-well plates.
 
    sweep_0  .. sweep_5   -> column 0, rows 0..5
    sweep_6  .. sweep_11  -> column 1, rows 0..5
    sweep_12 .. sweep_17  -> column 2, rows 0..5
    sweep_18              -> column 3, row 0
    (19 wells: three full columns plus one)
 
`col_row.tif` stems still work, so this script handles both folder types.
 
Overlay the target G-code geometry on printed scaffold images and compute IoU.
 
Why the target used to land in a weird place
--------------------------------------------
The printer deposits the scaffold at the TRUE centre of each well (it follows
the printer protocol / plate definition). Imaging is a separate, retrospective
step: the print head moves to a *roughly* estimated position just to frame the
whole well, so the well appears at a slightly different pixel position in every
image. The well centre in the image is therefore the correct anchor for the
target geometry, but it must be found reliably.
 
The previous detector failed because:
  1. cv2.HoughCircles was run once with fixed parameters and the candidate with
     the LARGEST radius was kept. On a shiny well bottom, specular highlights
     and the plate shadow can produce a large spurious circle.
  2. The sanity gate was `|cx - w/2| < 0.35*w`, i.e. +/-448 px on a 1280 px
     image. A centre ~445 px away from the image centre still passed, which is
     exactly the failure seen in the bad overlays.
  3. On failure it silently returned the image centre, and `well_src` was
     printed as 'JSON' even when the JSON contained no well shape.
 
What this version does instead
-------------------------------
  * Canny edges + a gradient-direction voting accumulator (own implementation,
    so the behaviour is inspectable) generates centre candidates.
  * Every candidate is scored by ANGULAR COVERAGE: the fraction of 360 angular
    bins that contain an edge pixel at radius ~r whose gradient points radially.
    A real well rim covers many bins; a specular blob covers a few. This works
    on partially visible rims (the arc only needs to cover part of the circle).
  * The best candidate is refined by an algebraic least-squares circle fit on
    its inlier edge points (sub-pixel centre).
  * TWO-PASS, plate-wide consensus:
        Pass A - detect on every image with a loose radius range, take the
                 median radius of the confident detections. The well rim shows
                 several concentric rings; locking the radius makes every image
                 latch onto the SAME ring. (Concentric rings share a centre, so
                 this does not bias the centre, it only stabilises the score.)
        Pass B - re-detect with the radius locked, then reject any detection
                 that is low-confidence or too far from the plate-wide median
                 centre, and substitute the median centre for those.
  * Every centre is written to `well_centres.csv`. Re-running with
    `--centres_csv <file>` uses those values verbatim, so a well that still
    looks wrong can be corrected by hand once and stays corrected.
  * `--debug_dir` writes a picture of the detected circle + chosen centre per
    image so failures are visible instead of silent.
 
Anchor correction
-----------------
GLOBAL_OFFSET_MM + ROW_STEP/COL_STEP are added to the detected well centre.
All three are ZERO. The old ROW_STEP (-0.15, 0.0) / COL_STEP (0.0, 0.10) were
hand-tuned against the broken detector; with the detector fixed they inject
error. Measured on well 1_3 (col=1, row=3), ROW_STEP contributed exactly
-0.45 mm in x, which was the entire x-misalignment there. Zeroing it raised
that well's IoU from 0.454 to 0.540. Use --no_drift to ignore them entirely.
 
No annotation is required
-------------------------
Nothing here needs a labelme JSON. Well detection is fully automatic from the
image. If a `{stem}.json` sits next to the image AND contains a shape labelled
'well' it is used as ground truth; otherwise it is ignored. JSONs holding only
'strands' and 'pores' never trigger that path.
 
The script DOES need the printed-strand mask. IoU is the overlap of TWO
shapes: this script builds the target from the G-code, and something must
measure what was actually printed. That half of the metric cannot be omitted.
 
During annotation it came from labelme. In deployment it comes from the
trained WP1 segmentation model. Two ways to supply it:
 
  1. Your inference script already writes `{stem}-mask.png` somewhere
     (0 = background, non-zero = strand). Point --mask_dir at that folder.
     No change to this script is needed.
 
  2. One command instead of two: pass --segmenter py:<file.py>:<func> and the
     script calls your model itself. The function takes (img_rgb, stem) and
     returns an HxW array, non-zero on strand, at the ORIGINAL image size.
     See segmenter_template.py. Generated masks are written to --save_masks
     (default: output_dir), so they are inspectable and reusable via
     --mask_dir without re-running the model.
 
With --segmenter the script enumerates IMAGES rather than masks, so a folder
of unannotated deployment images is a valid input.
 
With a single image there is no plate to build consensus from, so the rim
radius is not locked and a failed detection falls back to the image centre
rather than the plate centre. Batch a whole plate when you can.
 
Usage
-----
    python draw_target_geometry.py <img_dir>
        [--mask_dir <dir>]         folder with *-mask.png (default: img_dir)
        [--output_dir <dir>]       where to save overlays  (default: img_dir)
        [--strand_width_mm <f>]    strand width in mm       (default: 0.41)
        [--strand_gap_mm <f>]      centre-to-centre spacing (default: 2.5)
        [--alpha <f>]              overlay opacity 0-1      (default: 0.5)
        [--iou_threshold <f>]      flag results above this IoU with a star
        [--no_drift]               ignore the anchor corrections entirely
        [--centres_csv <file>]     use these well centres verbatim (overrides
                                   detection); written automatically otherwise
        [--debug_dir <dir>]        save detection debug images
        [--min_score <f>]          min angular coverage to trust a detection
        [--radius_tol <f>]         allowed rim-radius deviation from plate median
        [--centre_tol_mm <f>]      floor of the adaptive outlier gate
        [--search_frac <f>]        centre search half-window as frac of image
        [--no_consensus]           per-image detection only, no plate median
        [--segmenter py:f.py:fn]   run your model to produce the printed mask
                                   when no *-mask.png exists (deployment)
        [--save_masks <dir>]       where to write segmenter masks
        [--pred_masks]             read *-pred-mask.png (U-Net++ output) instead
                                   of the annotation *-mask.png
        [--mask_suffix=<s>]        any other suffix (needs '=', value starts '-')
 
Output
------
    <output_dir>/
        {stem}-target-overlay.png   coloured red/green/yellow overlay
        {stem}-target-mask.png      exact binary target geometry mask (0/255)
        well_centres.csv            stem,cx,cy,r,score,source
    <debug_dir>/
        {stem}-welldet.png          detected circle + centre drawn on image
"""
 
import argparse
import csv
import re
import sys
from pathlib import Path
 
import numpy as np
from PIL import Image, ImageDraw
import json
 
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
 
 
# -- geometry constants -------------------------------------------------------
PX_PER_MM               = 67.0
DEFAULT_STRAND_WIDTH_MM = 0.41    # 22G nozzle inner diameter
DEFAULT_STRAND_GAP_MM   = 2.5     # centre-to-centre between adjacent strands
N_STRANDS               = 3       # 3 H-strands + 3 V-strands
 
 
# -- pressure-sweep well indexing ---------------------------------------------
# The 48-well folders name files `col_row.tif` (0_0 ... 7_5).
# The pressure_sweep folders name them `sweep_N.tif`, N = 0 ... 18, printed
# COLUMN BY COLUMN: sweep_0..5 -> column 0 rows 0..5, sweep_6..11 -> column 1,
# sweep_12..17 -> column 2, sweep_18 -> column 3 row 0.
# So three full columns plus one well in the fourth, 19 wells total.
#
#     col = N // WELLS_PER_COL      row = N % WELLS_PER_COL
#
# This yields exactly the same (col, row) a 48-well file would have, so
# ROW_STEP / COL_STEP mean the same physical thing in both scripts and the
# offsets you tuned on the 48-well plates carry over unchanged.
WELLS_PER_COL = 6
 
 
# -- anchor corrections (mm) --------------------------------------------------
# One preset is active at a time. In the 48-well version these were written as
# a stack of bare assignments, where the LAST uncommented block silently won:
# the cell_gelma_7_60 values (0.30, -0.30) were overridden by cell_gelma_10_80
# (0.30, -0.20) with no warning. Selecting by name makes that impossible.
#
# These are hand-tuned by eye. That is fine as a working practice, but it means
# they are only valid for the material/pressure combination they were tuned on.
PRESETS = {
    #  name                 GLOBAL_OFFSET_MM   ROW_STEP        COL_STEP
    'cell_gelma_7_60':   ((0.30, -0.30),   (-0.15, 0.0),   (0.0, 0.10)),
    'cell_gelma_10_80':  ((0.30, -0.20),   (-0.15, 0.0),   (0.0, 0.10)),
    'gelma_10_80':  ((0.40, -0.30),   (-0.15, 0.0),   (0.0, 0.10)),
    'gelma_10_80_sweep':  ((-0.25,  0.00),   (0.00, 0.0),   (0.0, 0.10)),
    'gelma_10_60_sweep':  ((-0.10, -0.20),   (-0.05, 0.0),   (0.0, 0.00)),
    'cell_gelma_10_60_sweep':  ((-0.20, -0.00),   (-0.00, 0.0),   (0.0, 0.00)),
    'gelma_rgen200':     ((0.40, -0.40),   (-0.15, 0.0),   (0.0, 0.10)),
    'gelma_alt':         ((0.60,  0.10),   (-0.20, 0.0),   (0.0, 0.10)),
    'pluronic_rgen100':  ((0.00,  0.00),   (-0.15, 0.0),   (0.0, 0.10)),
    'none':              ((0.00,  0.00),   ( 0.00, 0.0),   (0.0, 0.00)),
}

 
# <<< SET THIS, or override on the command line with --preset >>>
ACTIVE_PRESET = 'cell_gelma_10_80'
 
GLOBAL_OFFSET_MM, ROW_STEP, COL_STEP = PRESETS[ACTIVE_PRESET]
 
 
# -- detection defaults -------------------------------------------------------
DEFAULT_MIN_SCORE      = 0.30   # fraction of the 360 angular bins supported
DEFAULT_RADIUS_TOL     = 0.10   # allowed |r - r_median| / r_median
DEFAULT_CENTRE_TOL_MM  = 2.00   # FLOOR of the adaptive distance gate (see below)
DEFAULT_CENTRE_TOL_K   = 3.0    # gate = max(floor, K * median distance to median)
DEFAULT_SEARCH_FRAC    = 0.22   # centre must lie within +/-frac * (w, h)
RADIUS_FRAC_MIN        = 0.28   # of min(h, w)
RADIUS_FRAC_MAX        = 0.85   # of min(h, w)
WORK_MAX_DIM           = 640    # detection is done on a downscaled copy
ANGULAR_BINS           = 360
RADIAL_BAND_PX         = 3.0    # inlier band around the candidate radius (work scale)
GRAD_RADIAL_MIN        = 0.80   # |cos(angle between gradient and radius)|
 
 
# =============================================================================
# filename parsing / drift
# =============================================================================
def parse_col_row(stem):
    """
    Map a filename stem to (col, row).
 
      '3_2'      -> (3, 2)                      48-well grid, col_row
      'sweep_14' -> (14 // 6, 14 % 6) = (2, 2)  pressure sweep, printed
                                                column by column
 
    Returns (None, None) if the stem matches neither form.
    """
    m = re.match(r'^(\d+)_(\d+)$', stem)
    if m:
        return int(m.group(1)), int(m.group(2))
 
    m = re.match(r'^sweep[_-]?(\d+)$', stem, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        return idx // WELLS_PER_COL, idx % WELLS_PER_COL
 
    return None, None
 
 
def get_drift_offset(stem, apply_drift=True):
    """
    Total anchor correction (dx_mm, dy_mm, description) for a well:
    GLOBAL_OFFSET_MM (all wells) + ROW_STEP/COL_STEP (print-order drift).
 
    NOTE on the (col - 1): this is carried over verbatim from the 48-well
    script. Your files start at column 0, so column 0 receives -1 x COL_STEP
    rather than zero. That is very likely an off-by-one, BUT you tuned
    GLOBAL_OFFSET_MM by eye against exactly this behaviour, so the constant
    has already absorbed it. Changing it here would shift every well by
    +COL_STEP (0.10 mm = 6.7 px) and invalidate your tuning, so it is left
    alone deliberately. Keeping it identical is also what makes a sweep well
    and a 48-well well at the same (col, row) get the same correction.
    """
    if not apply_drift:
        return 0.0, 0.0, 'disabled'
    dx, dy = GLOBAL_OFFSET_MM
    col, row = parse_col_row(stem)
    if col is None:
        return dx, dy, 'global only (stem is neither col_row nor sweep_N)'
    dx += row * ROW_STEP[0] + (col - 1) * COL_STEP[0]
    dy += row * ROW_STEP[1] + (col - 1) * COL_STEP[1]
    return dx, dy, f'col={col} row={row}'
 
 
# =============================================================================
# well centre from labelme JSON (optional ground truth)
# =============================================================================
def read_well_centre_from_json(json_path):
    """
    Read a well annotation from a labelme JSON.
    Accepts a shape labelled 'well' (case-insensitive) that is either a
    'circle' (2 points: centre, rim) or a polygon/points shape (>=5 points,
    least-squares circle fit). Returns (cx, cy, r) or None.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
 
    for shape in data.get('shapes', []):
        if str(shape.get('label', '')).strip().lower() != 'well':
            continue
        pts = np.asarray(shape.get('points', []), dtype=float)
        if shape.get('shape_type') == 'circle' and len(pts) >= 2:
            cx, cy = pts[0]
            r = float(np.hypot(pts[1][0] - cx, pts[1][1] - cy))
            return float(cx), float(cy), r
        if len(pts) >= 5:
            fit = fit_circle_ls(pts[:, 0], pts[:, 1])
            if fit is not None:
                return fit
    return None
 
 
# =============================================================================
# circle fitting / scoring primitives
# =============================================================================
def fit_circle_ls(xs, ys):
    """
    Algebraic (Kasa) least-squares circle fit.
    Returns (cx, cy, r) or None if degenerate.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return None
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    val = sol[2] + cx ** 2 + cy ** 2
    if not np.isfinite(val) or val <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(val))
 
 
def angular_coverage(ex, ey, gux, guy, cx, cy, r,
                     band=RADIAL_BAND_PX,
                     n_bins=ANGULAR_BINS,
                     grad_min=GRAD_RADIAL_MIN):
    """
    Score a candidate circle by the fraction of angular bins that contain an
    edge pixel lying within `band` of radius r AND whose gradient is radial.
 
    ex, ey     : edge pixel coordinates (arrays)
    gux, guy   : unit gradient vector at each edge pixel
    Returns (score, inlier_boolean_mask).
 
    Rationale: a real rim is an arc, so it lights up a contiguous run of bins
    even when partly out of frame. A specular highlight or a shadow blob lights
    up only a few bins, so it scores low no matter how large its radius is.
    """
    dx = ex - cx
    dy = ey - cy
    rho = np.hypot(dx, dy)
    near = np.abs(rho - r) <= band
    if not np.any(near):
        return 0.0, near
 
    # gradient must point along the radius (either inward or outward)
    inv = np.zeros_like(rho)
    np.divide(1.0, rho, out=inv, where=rho > 1e-6)
    radial = np.abs(dx * inv * gux + dy * inv * guy)
    inl = near & (radial >= grad_min)
    if not np.any(inl):
        return 0.0, inl
 
    theta = np.arctan2(dy[inl], dx[inl])
    bins = ((theta + np.pi) / (2 * np.pi) * n_bins).astype(np.int32) % n_bins
    return float(np.unique(bins).size) / n_bins, inl
 
 
def _vote_centres(ex, ey, gux, guy, radii, shape, max_pts=15000, rng=None):
    """
    Gradient-direction Hough accumulator for circle centres.
    Each edge pixel votes for `p - r*n` and `p + r*n` for every candidate r.
    Returns a float32 accumulator of the given shape.
    """
    h, w = shape
    n = ex.size
    if n > max_pts:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, size=max_pts, replace=False)
        ex, ey, gux, guy = ex[idx], ey[idx], gux[idx], guy[idx]
 
    acc = np.zeros((h, w), dtype=np.float32)
    for r in radii:
        for sgn in (-1.0, 1.0):
            cxs = np.rint(ex + sgn * r * gux).astype(np.int32)
            cys = np.rint(ey + sgn * r * guy).astype(np.int32)
            ok = (cxs >= 0) & (cxs < w) & (cys >= 0) & (cys < h)
            if np.any(ok):
                np.add.at(acc, (cys[ok], cxs[ok]), 1.0)
    return acc
 
 
def _peak_candidates(acc, k=12, min_sep=8):
    """Top-k local maxima of the accumulator, separated by `min_sep` px."""
    sm = cv2.GaussianBlur(acc, (0, 0), 3.0)
    h, w = sm.shape
    flat = np.argsort(sm.ravel())[::-1]
    picks = []
    for f in flat:
        if len(picks) >= k:
            break
        y, x = divmod(int(f), w)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2 for px, py in picks):
            picks.append((x, y))
    return picks
 
 
# =============================================================================
# well detection
# =============================================================================
def detect_well_circle(img_rgb,
                       r_lock=None,
                       search_frac=DEFAULT_SEARCH_FRAC,
                       r_frac_min=RADIUS_FRAC_MIN,
                       r_frac_max=RADIUS_FRAC_MAX):
    """
    Detect the well rim.
 
    r_lock : if given (full-resolution px), restrict the search to
             [0.90*r_lock, 1.10*r_lock]. Used in pass B so that every image
             latches onto the same concentric ring.
 
    Returns dict(cx, cy, r, score, method) in FULL-RESOLUTION pixels.
    score is the angular coverage in [0, 1]; 0.0 means detection failed.
    """
    h, w = img_rgb.shape[:2]
    fail = dict(cx=w / 2.0, cy=h / 2.0, r=float('nan'), score=0.0, method='failed')
    if cv2 is None:
        return fail
 
    scale = min(1.0, WORK_MAX_DIM / float(max(h, w)))
    sw, sh = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
 
    # Canny thresholds from the image statistics so exposure changes are handled
    med = float(np.median(gray))
    lo = int(max(10, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
 
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
 
    ey, ex = np.nonzero(edges)
    if ex.size < 200:
        return fail
    m = mag[ey, ex]
    keep = m > 1e-6
    ex, ey, m = ex[keep].astype(np.float32), ey[keep].astype(np.float32), m[keep]
    gux = gx[ey.astype(int), ex.astype(int)] / m
    guy = gy[ey.astype(int), ex.astype(int)] / m
 
    min_dim_s = min(sh, sw)
    if r_lock is not None and np.isfinite(r_lock):
        rl = r_lock * scale
        r_min_s, r_max_s = 0.90 * rl, 1.10 * rl
    else:
        r_min_s = r_frac_min * min_dim_s
        r_max_s = r_frac_max * min_dim_s
    r_min_s = max(8.0, r_min_s)
    r_max_s = max(r_min_s + 4.0, r_max_s)
 
    step = max(2.0, (r_max_s - r_min_s) / 24.0)
    radii = np.arange(r_min_s, r_max_s + 1e-6, step)
 
    acc = _vote_centres(ex, ey, gux, guy, radii, (sh, sw))
 
    # hard gate: the well must be roughly framed, so its centre cannot be far
    # from the image centre. This is what the old 0.35*w gate was too loose for.
    cx0, cy0 = sw / 2.0, sh / 2.0
    win_x, win_y = search_frac * sw, search_frac * sh
    yy, xx = np.mgrid[0:sh, 0:sw]
    acc[(np.abs(xx - cx0) > win_x) | (np.abs(yy - cy0) > win_y)] = 0.0
 
    best = None
    for (px, py) in _peak_candidates(acc, k=12, min_sep=max(6, int(0.02 * min_dim_s))):
        # refine radius for this candidate by scanning the allowed range
        rho = np.hypot(ex - px, ey - py)
        sel = (rho >= r_min_s - 4) & (rho <= r_max_s + 4)
        if sel.sum() < 60:
            continue
        hist, edges_r = np.histogram(rho[sel],
                                     bins=max(8, int((r_max_s - r_min_s) / 2)),
                                     range=(r_min_s, r_max_s))
        for bi in np.argsort(hist)[::-1][:4]:
            r_try = 0.5 * (edges_r[bi] + edges_r[bi + 1])
            score, inl = angular_coverage(ex, ey, gux, guy, px, py, r_try)
            if score <= 0:
                continue
            # sub-pixel refinement on the inliers, then re-score
            fit = fit_circle_ls(ex[inl], ey[inl])
            if fit is not None:
                fcx, fcy, fr = fit
                if (abs(fcx - cx0) <= win_x and abs(fcy - cy0) <= win_y
                        and r_min_s * 0.85 <= fr <= r_max_s * 1.15):
                    s2, _ = angular_coverage(ex, ey, gux, guy, fcx, fcy, fr)
                    if s2 >= score:
                        px_f, py_f, r_f, score = fcx, fcy, fr, s2
                    else:
                        px_f, py_f, r_f = float(px), float(py), r_try
                else:
                    px_f, py_f, r_f = float(px), float(py), r_try
            else:
                px_f, py_f, r_f = float(px), float(py), r_try
 
            if best is None or score > best['score']:
                best = dict(cx=px_f, cy=py_f, r=r_f, score=score)
 
    if best is None:
        return fail
 
    inv = 1.0 / scale
    return dict(cx=best['cx'] * inv, cy=best['cy'] * inv, r=best['r'] * inv,
                score=best['score'],
                method='rim-lock' if r_lock is not None else 'rim')
 
 
def detect_well_centre_auto(img_rgb):
    """Backwards-compatible wrapper returning (cx, cy)."""
    d = detect_well_circle(img_rgb)
    return d['cx'], d['cy']
 
 
# =============================================================================
# consensus
# =============================================================================
def robust_median(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else float('nan')
 
 
 
# =============================================================================
# target geometry mask
# =============================================================================
def make_target_mask(img_h, img_w, cx, cy,
                     strand_width_mm=DEFAULT_STRAND_WIDTH_MM,
                     strand_gap_mm=DEFAULT_STRAND_GAP_MM,
                     n_strands=N_STRANDS):
    """Render the ideal G-code crosshatch as a binary mask (0/1 uint8)."""
    half_w_px      = (strand_width_mm / 2) * PX_PER_MM
    half_span      = (n_strands - 1) / 2.0 * strand_gap_mm
    offsets_mm     = [-half_span + i * strand_gap_mm for i in range(n_strands)]
    half_extent_px = (half_span + strand_width_mm / 2) * PX_PER_MM
 
    canvas = Image.new('L', (img_w, img_h), 0)
    draw   = ImageDraw.Draw(canvas)
 
    for y_off_mm in offsets_mm:          # horizontal strands
        y_px = cy + y_off_mm * PX_PER_MM
        draw.rectangle([cx - half_extent_px, y_px - half_w_px,
                        cx + half_extent_px, y_px + half_w_px], fill=1)
 
    for x_off_mm in offsets_mm:          # vertical strands
        x_px = cx + x_off_mm * PX_PER_MM
        draw.rectangle([x_px - half_w_px, cy - half_extent_px,
                        x_px + half_w_px, cy + half_extent_px], fill=1)
 
    return np.array(canvas, dtype=np.uint8)
 
 
# =============================================================================
# IoU / overlay
# =============================================================================
def load_rgb(path):
    """
    Load an image as uint8 RGB. Handles 8-bit and 16-bit / float TIFs
    (microscope cameras often write 12- or 16-bit), which PIL's plain
    convert('RGB') would clip or refuse.
    """
    im = Image.open(path)
    if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
        a = np.asarray(im).astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
        a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
        return np.repeat(a.astype(np.uint8)[:, :, None], 3, axis=2)
    return np.array(im.convert('RGB'))
 
 
def load_binary_mask(path):
    """Load a mask PNG robustly: handles L / RGB / RGBA and 0-1 or 0-255."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            arr = arr[..., :3].max(axis=2) if alpha.min() == alpha.max() else alpha
        else:
            arr = arr.max(axis=2)
    return (arr > 0).astype(np.uint8)
 
 
def compute_iou(pred_mask, target_mask):
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0
 
 
def render_overlay(img_rgb, pred_mask, target_mask, alpha=0.5):
    """Red=printed only, Green=target only, Yellow=overlap."""
    overlay = img_rgb.astype(np.float32).copy()
    pred    = pred_mask   > 0
    target  = target_mask > 0
    RED    = np.array([255,  60,  60], dtype=np.float32)
    GREEN  = np.array([ 60, 220,  60], dtype=np.float32)
    YELLOW = np.array([255, 220,   0], dtype=np.float32)
    for mask, colour in [(pred & ~target, RED),
                         (target & ~pred, GREEN),
                         (pred & target,  YELLOW)]:
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colour
    return np.clip(overlay, 0, 255).astype(np.uint8)
 
 
def save_debug(img_rgb, det, cx, cy, path):
    """Draw the detected rim (cyan) and the final anchor (magenta cross)."""
    if cv2 is None:
        return
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    if np.isfinite(det.get('r', float('nan'))):
        cv2.circle(vis, (int(round(det['cx'])), int(round(det['cy']))),
                   int(round(det['r'])), (255, 255, 0), 2)
        cv2.drawMarker(vis, (int(round(det['cx'])), int(round(det['cy']))),
                       (255, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.drawMarker(vis, (int(round(cx)), int(round(cy))),
                   (255, 0, 255), cv2.MARKER_CROSS, 40, 2)
    cv2.putText(vis, f"score={det.get('score', 0):.2f} src={det.get('source', '?')}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imwrite(str(path), vis)
 
 
# =============================================================================
# pluggable segmenter (deployment: no annotation available)
# =============================================================================
# IoU needs BOTH shapes. This script builds the target from the G-code; the
# printed shape has to be measured from the image by a segmentation model.
# During annotation that came from labelme. In deployment it comes from the
# trained WP1 model.
#
# You do NOT need this hook if your inference code already writes
# `{stem}-mask.png` somewhere: just point --mask_dir at that folder and the
# script works unchanged. The hook exists only so one command does both.
#
# It is deliberately framework-agnostic. Rather than guess at PyTorch vs ONNX
# vs Ultralytics pre/post-processing, you supply a Python function:
#
#     --segmenter py:/path/to/my_segmenter.py:predict
#
# where predict(img_rgb, stem) -> HxW array, non-zero on strand. See
# segmenter_template.py.
def load_segmenter(spec):
    """Load a segmenter from 'py:<file>[:<func>]'. Returns callable(img_rgb, stem)."""
    if not spec.startswith('py:'):
        raise SystemExit(f"Unsupported --segmenter spec {spec!r}. "
                         f"Expected 'py:<file.py>[:<func>]'.")
    parts = spec[3:].rsplit(':', 1)
    if len(parts) == 2 and not parts[1].endswith('.py'):
        path, func = parts
    else:
        path, func = spec[3:], 'predict'
    path = Path(path)
    if not path.exists():
        raise SystemExit(f'Segmenter file not found: {path}')
 
    import importlib.util
    mod_spec = importlib.util.spec_from_file_location('user_segmenter', path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    if not hasattr(mod, func):
        raise SystemExit(f'{path} has no function {func!r}')
    fn = getattr(mod, func)
    print(f'Segmenter loaded: {path}:{func}')
 
    def run(img_rgb, stem):
        out = fn(img_rgb, stem)
        if out is None:
            raise RuntimeError(f'segmenter returned None for {stem}')
        out = np.asarray(out)
        if out.ndim == 3:            # (1,H,W) or (H,W,1) or logits (C,H,W)
            out = out.squeeze()
        if out.ndim != 2:
            raise RuntimeError(f'segmenter returned shape {out.shape} for {stem}; '
                               f'expected a 2-D HxW mask')
        if out.shape != img_rgb.shape[:2]:
            raise RuntimeError(f'segmenter mask {out.shape} != image '
                               f'{img_rgb.shape[:2]} for {stem}; resize it to the '
                               f'original image size before returning')
        return (out > 0).astype(np.uint8)
 
    return run
 
 
# =============================================================================
# batch processing
# =============================================================================
def find_tif(img_dir, stem):
    for ext in ('.tif', '.tiff', '.TIF', '.TIFF', '.png', '.jpg', '.jpeg'):
        p = img_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None
 
 
def read_centres_csv(path):
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[row['stem']] = (float(row['cx']), float(row['cy']),
                                    float(row.get('r') or 'nan'))
            except (KeyError, TypeError, ValueError):
                continue
    return out
 
 
def process_folder(img_dir, mask_dir, output_dir,
                   strand_width_mm, strand_gap_mm, alpha,
                   iou_threshold, apply_drift,
                   centres_csv=None, debug_dir=None,
                   min_score=DEFAULT_MIN_SCORE,
                   centre_tol_mm=DEFAULT_CENTRE_TOL_MM,
                   radius_tol=DEFAULT_RADIUS_TOL,
                   search_frac=DEFAULT_SEARCH_FRAC,
                   use_consensus=True, segmenter=None, save_masks=None,
                   mask_suffix='-mask.png'):
 
    img_dir    = Path(img_dir)
    mask_dir   = Path(mask_dir)   if mask_dir   else img_dir
    output_dir = Path(output_dir) if output_dir else img_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
 
    segment = load_segmenter(segmenter) if segmenter else None
 
    if segment is None:
        mask_files = sorted(
            p for p in mask_dir.glob(f'*{mask_suffix}')
            if 'visible' not in p.name and 'target' not in p.name
        )
        if not mask_files:
            print(f'No *{mask_suffix} files found in {mask_dir}.\n'
                  f'IoU compares the TARGET geometry against the PRINTED shape, so a '
                  f'mask of\nwhat was actually printed is required - it is half the '
                  f'metric, not an optional\nextra. Either point --mask_dir at where '
                  f'your segmentation model writes\n{{stem}}-mask.png, or pass '
                  f'--segmenter py:<file.py>:<func> to run the model here.')
            return
        cut = mask_suffix.rsplit('.', 1)[0]      # '-pred-mask.png' -> '-pred-mask'
        stems = [(p.stem[:-len(cut)] if p.stem.endswith(cut) else p.stem, p)
                 for p in mask_files]
    else:
        # deployment: enumerate IMAGES, masks are produced on the fly
        seen, stems = set(), []
        for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.jpg', '*.jpeg'):
            for p in sorted(img_dir.glob(ext)):
                st = p.stem
                if st.endswith('-mask') or 'target' in st or 'welldet' in st:
                    continue
                if st not in seen:
                    seen.add(st)
                    stems.append((st, None))
        if not stems:
            print(f'No images found in {img_dir}.')
            return
 
    manual = read_centres_csv(centres_csv) if centres_csv and Path(centres_csv).exists() else {}
    if manual:
        print(f'Loaded {len(manual)} manual centre(s) from {centres_csv}')
 
    centre_tol_px = centre_tol_mm * PX_PER_MM
 
    print(f'Found {len(stems)} '
          f'{"image(s), masks from segmenter" if segment else "mask(s)"}'
          f'  strand_width={strand_width_mm}mm'
          f'  strand_gap={strand_gap_mm}mm'
          f'  drift={"on" if apply_drift else "off"}\n')
 
    # ---------------------------------------------------------------- pass A
    # collect a loose detection per image, purely to learn the plate-wide
    # rim radius (the well rim has several concentric rings; without this,
    # different images latch onto different rings).
    items = []
    for stem, mask_path in stems:
        tif_path = find_tif(img_dir, stem)
        if tif_path is None:
            print(f'[SKIP] no image file for {stem}')
            continue
        items.append((stem, mask_path, tif_path))
 
    if not items:
        print('Nothing to process.')
        return
 
    r_lock = None
    if use_consensus and cv2 is not None:
        print('Pass A: learning plate-wide rim radius ...')
        radii = []
        for stem, _, tif_path in items:
            if stem in manual:
                continue
            if (img_dir / f'{stem}.json').exists():
                j = read_well_centre_from_json(img_dir / f'{stem}.json')
                if j:
                    radii.append(j[2])
                    continue
            img_rgb = load_rgb(tif_path)
            d = detect_well_circle(img_rgb, search_frac=search_frac)
            if d['score'] >= min_score:
                radii.append(d['r'])
        r_lock = robust_median(radii) if len(radii) >= 3 else None
        if r_lock is not None and np.isfinite(r_lock):
            print(f'  consensus rim radius = {r_lock:.1f} px '
                  f'({r_lock / PX_PER_MM:.2f} mm) from {len(radii)} detection(s)\n')
        else:
            print('  not enough confident detections; radius lock disabled\n')
 
    # ---------------------------------------------------------------- pass B
    print('Pass B: detecting well centres ...')
    dets = {}
    for stem, mask_path, tif_path in items:
        if stem in manual:
            mcx, mcy, mr = manual[stem]
            dets[stem] = dict(cx=mcx, cy=mcy, r=mr, score=1.0, source='csv')
            continue
        wj = img_dir / f'{stem}.json'
        if wj.exists():
            j = read_well_centre_from_json(wj)
            if j:
                dets[stem] = dict(cx=j[0], cy=j[1], r=j[2], score=1.0, source='json')
                continue
        img_rgb = load_rgb(tif_path)
        d = detect_well_circle(img_rgb, r_lock=r_lock, search_frac=search_frac)
        d['source'] = 'detect'
        dets[stem] = d
 
    # --------------------------------------------------- consensus + rejection
    #
    # IMPORTANT: the imaging position is only roughly estimated, so the well
    # genuinely sits at a different pixel in every image. A fixed
    # "distance from the plate median" threshold would therefore throw away
    # perfectly good detections. Two checks are used instead:
    #
    #   (1) RADIUS CONSISTENCY - position independent and physically hard:
    #       the well is the same size in every image at the same
    #       magnification, so r must match the plate median radius.
    #   (2) ADAPTIVE DISTANCE - the gate scales with how much the imaging
    #       position actually wanders on this plate, with an absolute floor.
    #       It only catches gross outliers, not normal locating scatter.
    #
    trusted = [d for d in dets.values() if d['score'] >= min_score]
    if use_consensus and len(trusted) >= 3:
        med_cx = robust_median([d['cx'] for d in trusted])
        med_cy = robust_median([d['cy'] for d in trusted])
        med_r  = robust_median([d['r'] for d in trusted])
        dists  = [float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy)) for d in trusted]
        med_d  = robust_median(dists)
        gate   = max(centre_tol_px, DEFAULT_CENTRE_TOL_K * med_d)
        print(f'  plate median centre = ({med_cx:.0f}, {med_cy:.0f})  '
              f'median radius = {med_r:.0f} px  '
              f'from {len(trusted)}/{len(dets)} confident detection(s)')
        print(f'  imaging scatter: median {med_d:.0f} px ({med_d / PX_PER_MM:.2f} mm) '
              f'-> outlier gate {gate:.0f} px ({gate / PX_PER_MM:.2f} mm)')
    else:
        med_cx = med_cy = med_r = med_d = float('nan')
        gate = centre_tol_px
        if use_consensus:
            print('  fewer than 3 confident detections; consensus disabled. '
                  'A failed\n  detection will fall back to the image centre, '
                  'so check the overlays.')
 
    for stem, d in dets.items():
        if d['source'] in ('csv', 'json'):
            continue
        reason = None
        if d['score'] < min_score:
            reason = f"low score {d['score']:.2f} < {min_score}"
        elif np.isfinite(med_r) and np.isfinite(d['r']) and \
                abs(d['r'] - med_r) > radius_tol * med_r:
            reason = (f"radius {d['r']:.0f}px off plate median {med_r:.0f}px "
                      f"by >{radius_tol:.0%}")
        elif np.isfinite(med_cx):
            dist = float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy))
            if dist > gate:
                reason = (f'{dist:.0f}px ({dist / PX_PER_MM:.2f}mm) from plate '
                          f'median, gate {gate:.0f}px')
        if reason:
            if np.isfinite(med_cx):
                print(f'  [FIX] {stem}: rejected ({reason}) -> plate median centre')
                d.update(cx=med_cx, cy=med_cy, r=med_r, source='consensus')
            else:
                print(f'  [WARN] {stem}: rejected ({reason}) and no consensus '
                      f'available -> image centre; check this one manually')
                d['source'] = 'image-centre'
    print()
 
    # ---------------------------------------------------------------- render
    results = []
    for stem, mask_path, tif_path in items:
        d = dets[stem]
        img_rgb   = load_rgb(tif_path)
        h, w      = img_rgb.shape[:2]
        if mask_path is not None:
            pred_mask = load_binary_mask(mask_path)
        else:
            try:
                pred_mask = segment(img_rgb, stem)
            except Exception as exc:
                print(f'[SKIP] {stem}: segmenter failed: {exc}')
                continue
            out_masks = Path(save_masks) if save_masks else output_dir
            out_masks.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                out_masks / f'{stem}-mask.png')
        if pred_mask.shape != (h, w):
            print(f'[SKIP] {stem}: mask shape {pred_mask.shape} != image {(h, w)}')
            continue
 
        wcx, wcy = d['cx'], d['cy']
        dx_mm, dy_mm, _ = get_drift_offset(stem, apply_drift)
        cx = wcx + dx_mm * PX_PER_MM
        cy = wcy + dy_mm * PX_PER_MM
 
        target_mask = make_target_mask(h, w, cx, cy, strand_width_mm, strand_gap_mm)
        iou = compute_iou(pred_mask, target_mask)
 
        Image.fromarray((target_mask * 255).astype(np.uint8)).save(
            output_dir / f'{stem}-target-mask.png')
        Image.fromarray(render_overlay(img_rgb, pred_mask, target_mask, alpha)).save(
            output_dir / f'{stem}-target-overlay.png')
        if debug_dir:
            save_debug(img_rgb, d, cx, cy, debug_dir / f'{stem}-welldet.png')
 
        flag = '  *' if iou_threshold > 0 and iou >= iou_threshold else ''
        print(f'[OK] {stem}  IoU={iou:.3f}{flag}'
              f'  well=({wcx:.0f},{wcy:.0f})[{d["source"]},score={d["score"]:.2f}]'
              f'  corr=({dx_mm:+.2f},{dy_mm:+.2f})mm'
              f'  target=({cx:.0f},{cy:.0f})')
        results.append((stem, iou, d))
 
    # ---------------------------------------------------------------- summary
    csv_out = output_dir / 'well_centres.csv'
    with open(csv_out, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['stem', 'cx', 'cy', 'r', 'score', 'source', 'iou'])
        for stem, iou, d in results:
            wtr.writerow([stem, f'{d["cx"]:.2f}', f'{d["cy"]:.2f}',
                          f'{d["r"]:.2f}' if np.isfinite(d.get('r', np.nan)) else '',
                          f'{d["score"]:.3f}', d['source'], f'{iou:.4f}'])
 
    if results:
        print('\n-- IoU summary (sorted) --')
        for stem, iou, d in sorted(results, key=lambda x: -x[1]):
            flag = '  *' if iou_threshold > 0 and iou >= iou_threshold else ''
            print(f'  {stem}: {iou:.3f}{flag}   [{d["source"]}]')
        n_fixed = sum(1 for _, _, d in results if d['source'] in ('consensus', 'image-centre'))
        if n_fixed:
            print(f'\n{n_fixed}/{len(results)} well(s) used a fallback centre. '
                  f'Inspect them in --debug_dir, correct their rows in '
                  f'{csv_out.name}, then re-run with --centres_csv {csv_out.name}')
            if n_fixed > 0.3 * len(results):
                print('  Many fallbacks: if the wells are framed loosely, the centre '
                      f'search window may be too tight - try --search_frac 0.30. '
                      f'If the rims are low contrast, try --min_score 0.20.')
 
    print(f'\nDone. Overlays -> {output_dir}\nCentres -> {csv_out}')
 
 
# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Overlay G-code target geometry on printed scaffold images.')
    parser.add_argument('img_dir', help='Folder containing .tif images')
    parser.add_argument('--mask_dir', '-m', default=None,
                        help='Folder with *-mask.png files (default: img_dir)')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='Where to save overlay PNGs (default: img_dir)')
    parser.add_argument('--strand_width_mm', type=float, default=DEFAULT_STRAND_WIDTH_MM,
                        help=f'Strand width mm (default: {DEFAULT_STRAND_WIDTH_MM})')
    parser.add_argument('--strand_gap_mm', type=float, default=DEFAULT_STRAND_GAP_MM,
                        help=f'Centre-to-centre strand spacing mm (default: {DEFAULT_STRAND_GAP_MM})')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Overlay opacity 0-1 (default: 0.5)')
    parser.add_argument('--iou_threshold', type=float, default=0.0,
                        help='Flag results above this IoU (default: 0.0 = off)')
    parser.add_argument('--no_drift', action='store_true',
                        help='Disable drift correction - use raw well centre only')
    parser.add_argument('--centres_csv', default=None,
                        help='CSV of stem,cx,cy[,r] used verbatim, overriding detection')
    parser.add_argument('--debug_dir', default=None,
                        help='Save per-image well-detection debug pictures here')
    parser.add_argument('--min_score', type=float, default=DEFAULT_MIN_SCORE,
                        help=f'Min angular coverage to trust a detection (default: {DEFAULT_MIN_SCORE})')
    parser.add_argument('--centre_tol_mm', type=float, default=DEFAULT_CENTRE_TOL_MM,
                        help='Floor of the adaptive outlier gate, in mm '
                             f'(default: {DEFAULT_CENTRE_TOL_MM}). The actual gate is '
                             f'max(this, {DEFAULT_CENTRE_TOL_K} x median imaging scatter)')
    parser.add_argument('--radius_tol', type=float, default=DEFAULT_RADIUS_TOL,
                        help='Allowed relative deviation of the detected rim radius '
                             f'from the plate median (default: {DEFAULT_RADIUS_TOL})')
    parser.add_argument('--search_frac', type=float, default=DEFAULT_SEARCH_FRAC,
                        help=f'Centre search half-window as fraction of image size (default: {DEFAULT_SEARCH_FRAC})')
    parser.add_argument('--mask_suffix', default='-mask.png',
                        help="Suffix of the printed-strand mask files "
                             "(default: '-mask.png'). NOTE: the value starts with a "
                             "dash, so it must be written with '=', e.g. "
                             "--mask_suffix=-pred-mask.png. Use --pred_masks instead.")
    parser.add_argument('--pred_masks', action='store_true',
                        help="Shorthand for --mask_suffix=-pred-mask.png, i.e. read the "
                             "U-Net++ predictions written by unetplusplus_test.py.")
    parser.add_argument('--segmenter', default=None,
                        help="Produce the printed-strand mask by running your own model: "
                             "'py:<file.py>[:<func>]'. The function takes (img_rgb, stem) "
                             "and returns an HxW array, non-zero on strand. Use in "
                             "deployment where no annotation exists.")
    parser.add_argument('--save_masks', default=None,
                        help='Where to write segmenter-produced masks '
                             '(default: output_dir). They are reusable via --mask_dir.')
    parser.add_argument('--preset', default=None, choices=sorted(PRESETS),
                        help=f'Anchor-offset preset (default: {ACTIVE_PRESET}, set at '
                             f'the top of this file). Use "none" for zero offsets.')
    parser.add_argument('--no_consensus', action='store_true',
                        help='Per-image detection only, no plate-wide median fallback')
    args = parser.parse_args()
 
    if args.preset:
        GLOBAL_OFFSET_MM, ROW_STEP, COL_STEP = PRESETS[args.preset]
        ACTIVE_PRESET = args.preset
    print(f'Anchor preset: {ACTIVE_PRESET}  global={GLOBAL_OFFSET_MM}  '
          f'row_step={ROW_STEP}  col_step={COL_STEP}')
 
    if cv2 is None:
        sys.exit('OpenCV (cv2) is required for well detection. pip install opencv-python')
 
    process_folder(
        args.img_dir, args.mask_dir, args.output_dir,
        args.strand_width_mm, args.strand_gap_mm,
        args.alpha, args.iou_threshold,
        apply_drift=not args.no_drift,
        centres_csv=args.centres_csv,
        debug_dir=args.debug_dir,
        min_score=args.min_score,
        centre_tol_mm=args.centre_tol_mm,
        radius_tol=args.radius_tol,
        search_frac=args.search_frac,
        use_consensus=not args.no_consensus,
        segmenter=args.segmenter,
        save_masks=args.save_masks,
        mask_suffix='-pred-mask.png' if args.pred_masks else args.mask_suffix,
    )