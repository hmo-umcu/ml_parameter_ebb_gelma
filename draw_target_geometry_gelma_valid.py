"""
draw_target_geometry.py
-----------------------
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
                 centre.
  * Every centre is written to `well_centres.csv`. Re-running with
    `--centres_csv <file>` uses those values verbatim, so a well that still
    looks wrong can be corrected by hand once and stays corrected.
  * `--debug_dir` writes a picture of the detected circle + chosen centre per
    image so failures are visible instead of silent.
 
Automatic detection on a bright, specular well bottom
------------------------------------------------------
The original detector assumes the well rim is the strongest circular edge
structure in the frame. On a polished, brightly lit well bottom that is false,
and it fails in a specific, diagnosable way:
 
    med = np.median(gray); lo = 0.66 * med; hi = 1.33 * med
    edges = cv2.Canny(gray, lo, hi)
 
The Canny thresholds come from the INTENSITY median. A bright frame has a high
median, so lo lands near 120 and hi near 240, and the soft multi-ring rim is
thresholded away before it can be scored. What survives is the boundary of the
specular highlights, which is strong and arc-shaped, so the accumulator votes
for it. The symptom is angular-coverage scores of 0.16 to 0.24 against a 0.30
bar, with rim radii that disagree across one plate by 50 to 70 percent.
 
Rather than retune that detector and risk moving results that are already
correct, detection is now a LADDER. Each stage has its own acceptance bar and
the first stage to clear its bar wins:
 
    stage 1  the original, unchanged: legacy Canny, the given search window.
             bar = --min_score. An image this stage already handles never
             reaches stage 2, so its numbers are bit-identical to before.
    stage 2  robust edges: saturated pixels and a dilated collar excluded,
             CLAHE before the gradient, Canny thresholds from percentiles of
             the GRADIENT MAGNITUDE rather than the intensity median. Wider
             search window. bar = --min_score.
    stage 3  gradient-profile detector, no thresholding anywhere. For a
             candidate circle it samples the gradient on the circle itself and
             measures how much of the circumference carries an outward
             gradient of CONSISTENT SIGN. A real rim keeps one sign the whole
             way round; a specular boundary or a scaffold edge flips sign, so
             it cannot score highly however strong it is.
             bar = --soft_min_score.
 
Stages 2 and 3 both run when stage 1 fails, and the winner is whichever scores
higher as a fraction of its own bar, because their scores are not on the same
scale. Stage 3 runs with NO radius lock: unlike the Canny stages it does not
hop between concentric rings, and forcing it onto a radius median learned from
a weaker detector measurably moved its centres. Cross-image radius consistency
is still enforced afterwards, per stage.
 
Measured on a six-well synthetic plate built to reproduce this failure (bright
frame, 40 px soft rim ramp, three blown-out highlights): stage 1 alone rendered
nothing at all, and the ladder recovered all six with a mean centre error of
4.8 px, which is 0.07 mm. On a plate the original detector already handles, the
output is byte-identical with the ladder on or off. Escalation costs nothing
when stage 1 succeeds and roughly 2.5 s per image when it does not.
 
Use --no_escalate to get the original stage-1-only detector back.
 
Outliers
--------
A well can clear its confidence bar and still be in the wrong place, because
the detector locked confidently onto the wrong circle. The old response was to
substitute the plate median centre, i.e. discard the image's own evidence and
replace it with the average of the others.
 
Instead, a well that sits further from the plate median than the outlier gate
is now RECONCILED: the other detectors are re-run on that same image with the
plate radius locked, and among the hypotheses that clear their own bar the one
nearest the plate median is chosen. The plate is used as a prior to choose
between circles found in this image, never as a substitute for them. A well
that is not an outlier is not touched.
 
Why a target could still land far from the print, and what changed
-------------------------------------------------------------------
A rejected well used to be given the PLATE MEDIAN CENTRE. On a full 48-well
plate that is a reasonable stand-in. On a small plate (a validation column of
6 or 12 wells) it is not: the camera genuinely moves between wells, so the
median is off by the whole imaging scatter, and the substituted well ends up
visibly shifted while its neighbours look correct. That substitution was the
main source of the large target-geometry shifts.
 
It is no longer the default. `--on_reject` now controls it:
 
    skip    (default) the well is NOT rendered. No {stem}-target-mask.png and
            no overlay are written, the well is listed in
            well_centres_review.csv, and its row in well_centres.csv is marked
            status=REVIEW. pore_analysis.py will then refuse to score it rather
            than score it against a wrong anchor. Correct cx,cy by eye, set
            status to ok, and re-run with --centres_csv.
    median  the old behaviour, kept only to reproduce earlier runs. The
            substitution distance is now printed in mm so it is visible.
    keep    render the raw per-image detection unchanged, flagged
            'kept-unverified'.
 
Two further causes of a shifted anchor, both now diagnosable:
 
  * WRONG RING. Pass A takes the median rim radius and pass B forces every
    image onto it. The comment above claims concentric rings share a centre.
    That is true in orthographic projection and FALSE for an off-axis
    perspective view of a deep well: the top rim and the bottom rim do not
    project to the same image centre, and the scaffold sits on the bottom. If
    pass A locked the top rim, every centre carries a parallax offset that
    grows with distance from the optical axis. Pass A now prints the spread of
    its radii and warns above 20%. Use `--radius_mm` to lock the radius to the
    known well-bottom geometry, or `--no_radius_lock` to test the hypothesis.
  * SEARCH WINDOW. `--search_frac` zeroes the accumulator outside
    +/-frac*(w,h) of the image centre. A well framed more loosely than that has
    its true centre erased before scoring, and the detector is forced onto a
    wrong peak. Raise it to 0.30 if many wells are rejected.
 
The per-well distance from the plate median is now printed in mm (`d_med`) and
the three worst offenders are listed at the end. A well high on that list with
a poor IoU is an anchor problem, not a printing problem.
 
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
        [--on_reject skip|median|keep]
                                   what to do with an untrustworthy centre
                                   (default: skip, see above)
        [--no_escalate]            stage 1 only, i.e. the original detector
        [--sat_thresh <int>]       grey value counting as a specular blow-out
        [--soft_min_score <f>]     acceptance bar for the stage-3 detector
        [--px_per_mm <f>]          camera scale; THE knob for a calibration error
        [--radius_mm <f>]          lock the rim radius to a known physical value
        [--no_radius_lock]         skip pass A, per-image radius
        [--global_offset_mm DX DY] override GLOBAL_OFFSET_MM
        [--row_step_mm DX DY]      override ROW_STEP
        [--col_step_mm DX DY]      override COL_STEP
        [--no_consensus]           legacy: --no_radius_lock --on_reject keep
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
                                    NOT written for a well marked REVIEW
        well_centres.csv            stem,cx,cy,r,score,source,iou,status,reason
        well_centres_review.csv     only the wells that were not rendered
    <debug_dir>/
        {stem}-welldet.png          detected circle + centre drawn on image
                                    (written for rejected wells too)
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
 
 
# -- anchor corrections (mm) --------------------------------------------------
# All three are ZERO. GLOBAL_OFFSET_MM shifts every well equally; ROW_STEP and
# COL_STEP model drift along print order (stem `col_row`, col 1-8, row 0-5).
#
# The old ROW_STEP (-0.15, 0.0) / COL_STEP (0.0, 0.10) were hand-tuned against
# the broken detector. With the detector fixed they inject error: on well 1_3
# (col=1, row=3) ROW_STEP contributed exactly -0.45 mm in x, which was the
# entire x-misalignment there. Zeroing it raised that well's IoU 0.454 -> 0.540.
# If you ever set these again, verify against measured data, not by eye.
## R-GEN 200, GelMA
 
# GLOBAL_OFFSET_MM = (0.40, -0.40)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
# ## cell_gelma_7_60
# GLOBAL_OFFSET_MM = (0.30, -0.30)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
## cell_gelma_7_60
GLOBAL_OFFSET_MM = (0.30, -0.30)   # (dx_mm, dy_mm) applied to every well
ROW_STEP         = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
 
## validation
GLOBAL_OFFSET_MM = (0.00, -0.00)   # (dx_mm, dy_mm) applied to every well
ROW_STEP         = (-0.00, 0.0)   # (dx_mm, dy_mm) per row increment
COL_STEP         = (0.0, 0.00)   # (dx_mm, dy_mm) per column increment
 
 
# ## cell_gelma_10_80
# GLOBAL_OFFSET_MM = (0.30, -0.20)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
# ## cell_gelma_10_60
# GLOBAL_OFFSET_MM = (-0.20, -0.20)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.00, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.00)   # (dx_mm, dy_mm) per column increment
 
# ## gelma_10_80
# GLOBAL_OFFSET_MM = (0.40, -0.30)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
# ## gelma_10_60
# GLOBAL_OFFSET_MM = (-0.10, -0.20)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.05, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.00)   # (dx_mm, dy_mm) per column increment
 
# GLOBAL_OFFSET_MM = (0.60, 0.10)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP         = (-0.20, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP         = (0.0, 0.10)   # (dx_mm, dy_mm) per column increment
 
## R-GEN 100, Pluronic
# GLOBAL_OFFSET_MM = (0.0, 0.0)   # (dx_mm, dy_mm) applied to every well
# ROW_STEP = (-0.15, 0.0)   # (dx_mm, dy_mm) per row increment
# COL_STEP = (0.0, 0.10)    # (dx_mm, dy_mm) per column increment
 
 
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
 
# -- robust-mode / escalation defaults ----------------------------------------
DEFAULT_SAT_THRESH     = 248    # >= this grey value counts as a specular blow-out
SPEC_DILATE_PX         = 5      # collar grown around each saturated region
WIDE_SEARCH_FRAC       = 0.32   # search window used from stage 2 onwards
SOFT_MIN_SCORE         = 0.35   # acceptance for the gradient-profile detector
SOFT_N_ANGLES          = 180
SOFT_COARSE_STEP       = 5      # centre grid step, work-scale px
SOFT_N_RADII           = 28
 
 
# =============================================================================
# filename parsing / drift
# =============================================================================
def parse_col_row(stem):
    """Parse 'col_row' stem -> (col, row). Returns (None, None) if no match."""
    m = re.match(r'^(\d+)_(\d+)$', stem)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)
 
 
def get_drift_offset(stem, apply_drift=True):
    """
    Total anchor correction (dx_mm, dy_mm, description) for a well:
    GLOBAL_OFFSET_MM (all wells) + ROW_STEP/COL_STEP (print-order drift).
    """
    if not apply_drift:
        return 0.0, 0.0, 'disabled'
    dx, dy = GLOBAL_OFFSET_MM
    col, row = parse_col_row(stem)
    if col is None:
        return dx, dy, 'global only (stem not col_row)'
    dx += row * ROW_STEP[0] + (col - 1) * COL_STEP[0]
    dy += row * ROW_STEP[1] + (col - 1) * COL_STEP[1]
    return dx, dy, f'row={row} col={col}'
 
 
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
 
 
def specular_mask(gray, sat_thresh=DEFAULT_SAT_THRESH, dilate_px=SPEC_DILATE_PX):
    """
    Boolean mask of blown-out highlights plus a collar around them.
 
    On a polished well bottom the highlight boundary is the strongest edge in
    the frame and it is roughly arc-shaped, so it both dominates the Canny
    output and scores well on angular coverage. Excluding it is what lets the
    real rim compete.
    """
    m = (gray >= sat_thresh).astype(np.uint8)
    if m.any() and dilate_px > 1:
        k = np.ones((int(dilate_px), int(dilate_px)), np.uint8)
        m = cv2.dilate(m, k)
    return m.astype(bool)
 
 
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
                       r_frac_max=RADIUS_FRAC_MAX,
                       edge_mode='legacy',
                       sat_thresh=DEFAULT_SAT_THRESH):
    """
    Detect the well rim.
 
    r_lock : if given (full-resolution px), restrict the search to
             [0.90*r_lock, 1.10*r_lock]. Used in pass B so that every image
             latches onto the same concentric ring.
 
    edge_mode :
      'legacy'  the original edge extraction, unchanged. Canny thresholds are
                derived from the INTENSITY median. Kept as the default and as
                stage 1 of the ladder so that every image the detector already
                handles produces bit-identical numbers.
      'robust'  for bright, specular well bottoms, where 'legacy' fails. Three
                differences, all aimed at the same failure:
                  1. Saturated pixels (>= sat_thresh) and a dilated collar
                     around them are excluded. On a polished well bottom the
                     blown-out highlights carry the strongest edges in the
                     frame, and those edges are what the accumulator was
                     voting on.
                  2. CLAHE before the gradient, so a soft low-contrast rim in
                     a bright frame is lifted without amplifying the
                     highlights (CLAHE clips).
                  3. Canny thresholds from percentiles of the GRADIENT
                     MAGNITUDE rather than 0.66/1.33 x the intensity median.
                     A bright image has a high median, so the legacy rule sets
                     lo ~ 120 and hi ~ 240 and throws the rim away before it
                     can be scored. This is the direct cause of the 0.16-0.24
                     angular-coverage scores seen on the validation plate.
 
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
 
    spec = None
    if edge_mode == 'robust':
        spec = specular_mask(gray, sat_thresh)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
 
    gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
 
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
 
    if edge_mode == 'robust':
        pool = mag[~spec] if spec is not None and (~spec).any() else mag
        hi = float(np.percentile(pool, 92.0))
        lo = max(4.0, 0.40 * hi)
        edges = cv2.Canny(gray, int(lo), int(max(lo + 1, hi)), L2gradient=True)
        if spec is not None:
            edges[spec] = 0
    else:
        # Canny thresholds from the image statistics so exposure changes are handled
        med = float(np.median(gray))
        lo = int(max(10, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        edges = cv2.Canny(gray, lo, hi, L2gradient=True)
 
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
    tag = 'rim-lock' if r_lock is not None else 'rim'
    if edge_mode != 'legacy':
        tag += f'-{edge_mode}'
    return dict(cx=best['cx'] * inv, cy=best['cy'] * inv, r=best['r'] * inv,
                score=best['score'], method=tag)
 
 
# =============================================================================
# gradient-profile detector (stage 3)
# =============================================================================
def detect_well_circle_soft(img_rgb,
                            r_lock=None,
                            search_frac=WIDE_SEARCH_FRAC,
                            r_frac_min=RADIUS_FRAC_MIN,
                            r_frac_max=RADIUS_FRAC_MAX,
                            sat_thresh=DEFAULT_SAT_THRESH,
                            n_ang=SOFT_N_ANGLES):
    """
    Find the rim without binarising anything.
 
    The two Canny-based stages share one weakness: an edge either survives the
    threshold or it does not, and a soft rim spread over 30-60 px of gentle
    gradient can fall below any threshold that also suppresses the highlights.
    This detector never thresholds. For a candidate circle it samples the
    gradient field on the circle itself and asks how much of the circumference
    carries an outward-pointing gradient of CONSISTENT SIGN.
 
    Consistency of sign is the discriminating part. A real rim is a transition
    from the bright well bottom to the darker wall, so the radial derivative
    has the same sign the whole way round. A specular boundary or the scaffold
    edge flips sign as you go round, so it cannot score highly however strong
    it is.
 
    Returns the same dict shape as detect_well_circle, in FULL-RESOLUTION px.
    """
    h, w = img_rgb.shape[:2]
    fail = dict(cx=w / 2.0, cy=h / 2.0, r=float('nan'), score=0.0, method='failed')
    if cv2 is None:
        return fail
 
    scale = min(1.0, WORK_MAX_DIM / float(max(h, w)))
    sw, sh = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
 
    spec = specular_mask(gray, sat_thresh)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    g = cv2.GaussianBlur(g, (0, 0), 2.5).astype(np.float32)
 
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    # a saturated pixel carries no usable gradient, so zero it rather than
    # letting the highlight edge dominate the profile
    gx[spec] = 0.0
    gy[spec] = 0.0
 
    mag = np.hypot(gx, gy)
    valid = mag[~spec] if (~spec).any() else mag
    tau = float(np.percentile(valid, 70.0)) if valid.size else 0.0
    if tau <= 0:
        return fail
 
    min_dim_s = min(sh, sw)
    if r_lock is not None and np.isfinite(r_lock):
        rl = r_lock * scale
        r_lo, r_hi = 0.88 * rl, 1.12 * rl
    else:
        r_lo, r_hi = r_frac_min * min_dim_s, r_frac_max * min_dim_s
    r_lo = max(8.0, r_lo)
    r_hi = max(r_lo + 4.0, r_hi)
    radii = np.linspace(r_lo, r_hi, SOFT_N_RADII)
 
    th = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
 
    def score_grid(cxs, cys, radii):
        """
        (best_score, best_cx, best_cy, best_r) over the given candidates.
 
        Ranking is coverage plus a small bounded bonus for rim STRENGTH. On a
        well whose rim is fully visible the coverage saturates, so several
        radii tie at the same score and the choice between them becomes
        arbitrary, which moves the centre. The strength term breaks those ties
        towards the sharpest circle, i.e. the actual rim. The bonus is capped
        at RANK_BETA so it can never outweigh a real coverage difference.
        """
        RANK_BETA = 0.02
        best = (0.0, np.nan, np.nan, np.nan, -1.0)
        cxs = np.asarray(cxs, np.float32)
        cys = np.asarray(cys, np.float32)
        for r in radii:
            # sample points: (n_cand, n_ang)
            px = cxs[:, None] + r * ct[None, :]
            py = cys[:, None] + r * st[None, :]
            ix = np.rint(px).astype(np.int32)
            iy = np.rint(py).astype(np.int32)
            ok = (ix >= 0) & (ix < sw) & (iy >= 0) & (iy < sh)
            ixc = np.clip(ix, 0, sw - 1)
            iyc = np.clip(iy, 0, sh - 1)
            # outward radial component of the gradient
            v = gx[iyc, ixc] * ct[None, :] + gy[iyc, ixc] * st[None, :]
            v = np.where(ok, v, 0.0)
            nvis = ok.sum(axis=1).astype(np.float32)
            nvis[nvis < 1] = 1.0
            # sign that the majority of the circumference agrees on
            s = np.sign((v > 0).sum(axis=1) - (v < 0).sum(axis=1))
            s[s == 0] = 1.0
            sv  = s[:, None] * v
            cov = (sv > tau).sum(axis=1) / nvis
            # mean rim strength over the supporting arc, in units of tau
            strength = np.clip(sv, 0.0, None).sum(axis=1) / nvis / tau
            # a circle mostly out of frame cannot be trusted
            keep_fr = ok.sum(axis=1) >= 0.55 * n_ang
            cov = np.where(keep_fr, cov, 0.0)
            strength = np.where(keep_fr, strength, 0.0)
            rank = cov + RANK_BETA * np.tanh(strength)
            j = int(np.argmax(rank))
            if rank[j] > best[4]:
                best = (float(cov[j]), float(cxs[j]), float(cys[j]), float(r),
                        float(rank[j]))
        return best
 
    cx0, cy0 = sw / 2.0, sh / 2.0
    wx, wy = search_frac * sw, search_frac * sh
    gxs = np.arange(cx0 - wx, cx0 + wx + 1e-6, SOFT_COARSE_STEP)
    gys = np.arange(cy0 - wy, cy0 + wy + 1e-6, SOFT_COARSE_STEP)
    GX, GY = np.meshgrid(gxs, gys)
    coarse = score_grid(GX.ravel(), GY.ravel(), radii)
    if coarse[0] <= 0 or not np.isfinite(coarse[1]):
        return fail
 
    # local refinement, 1 px centre steps and a finer radius comb
    _, bcx, bcy, br, _rk = coarse
    fx = np.arange(bcx - SOFT_COARSE_STEP, bcx + SOFT_COARSE_STEP + 1e-6, 1.0)
    fy = np.arange(bcy - SOFT_COARSE_STEP, bcy + SOFT_COARSE_STEP + 1e-6, 1.0)
    FX, FY = np.meshgrid(fx, fy)
    fine_r = np.linspace(max(8.0, br - 6), br + 6, 13)
    fine = score_grid(FX.ravel(), FY.ravel(), fine_r)
    best = fine if fine[4] >= coarse[4] else coarse
 
    inv = 1.0 / scale
    return dict(cx=best[1] * inv, cy=best[2] * inv, r=best[3] * inv,
                score=best[0],
                method='soft-lock' if r_lock is not None else 'soft')
 
 
def detect_with_escalation(img_rgb, r_lock, search_frac, min_score,
                           escalate=True, sat_thresh=DEFAULT_SAT_THRESH,
                           soft_min_score=SOFT_MIN_SCORE,
                           wide_frac=WIDE_SEARCH_FRAC):
    """
    Try progressively more tolerant detectors and stop at the first that clears
    its own acceptance bar.
 
      stage 1  legacy edges, the given search window        bar: min_score
      stage 2  robust edges (specular mask + CLAHE +        bar: min_score
               gradient-percentile Canny), wider window
      stage 3  gradient-profile detector, no thresholding   bar: soft_min_score
 
    Stage 1 is the unmodified original. An image the current detector already
    handles never reaches stage 2, so its numbers cannot move. Only images that
    the current detector FAILS see any new code at all, which is the point:
    there is nothing to regress on a well that already works.
 
    The returned dict carries `stage` and `cleared`. If nothing cleared its
    bar, the best available evidence is returned with cleared=False, and the
    caller decides what to do with it.
    """
    d = detect_well_circle(img_rgb, r_lock=r_lock, search_frac=search_frac)
    d['stage'], d['cleared'] = 1, d['score'] >= min_score
    if d['cleared'] or not escalate:
        return d
 
    wide = max(search_frac, wide_frac)
 
    d2 = detect_well_circle(img_rgb, r_lock=r_lock, search_frac=wide,
                            edge_mode='robust', sat_thresh=sat_thresh)
    d2['stage'], d2['cleared'] = 2, d2['score'] >= min_score
 
    # Stage 3 runs UNLOCKED on purpose. The radius lock exists to stop the
    # Canny-based detector hopping between the well's concentric rings. The
    # gradient-profile detector does not hop: on a six-image test plate it
    # returned radii within 0.4% of each other with no lock at all. Forcing it
    # onto a radius median learned from the weaker detector measurably moved
    # its centres (mean error 4.8 px unlocked against 9.5 px locked), so the
    # lock is not applied here. Cross-image radius consistency is still
    # enforced afterwards by the --radius_tol check.
    d3 = detect_well_circle_soft(img_rgb, r_lock=None, search_frac=wide,
                                 sat_thresh=sat_thresh)
    d3['stage'], d3['cleared'] = 3, d3['score'] >= soft_min_score
 
    # Both stages ran, so choose between them rather than taking whichever
    # happens to be tried first. Scores from the two are not on the same scale,
    # so they are compared as a fraction of their own acceptance bar.
    tried = [(d2, min_score), (d3, soft_min_score)]
    cleared = [t for t in tried if t[0]['cleared']]
    pool = cleared if cleared else tried
    best, _bar = max(pool, key=lambda t: t[0]['score'] / max(t[1], 1e-6))
    best = dict(best)
    if not cleared:
        best['cleared'] = False
    return best
 
 
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
    """
    Read stem,cx,cy[,r] rows to be used verbatim as well centres.
 
    Rows whose `status` column still says REVIEW are SKIPPED, not used. That
    column is written for wells this script refused to anchor: their cx,cy are
    the failed detection, so handing them straight back would silently
    reintroduce the wrong anchor. Correct the row by eye, set status to ok,
    then it is honoured.
    """
    out, pending = {}, []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            status = str(row.get('status', '') or '').strip().upper()
            if status.startswith('REVIEW'):
                pending.append(str(row.get('stem', '')).strip())
                continue
            try:
                out[row['stem']] = (float(row['cx']), float(row['cy']),
                                    float(row.get('r') or 'nan'))
            except (KeyError, TypeError, ValueError):
                continue
    if pending:
        print(f'[NOTE] {len(pending)} row(s) in {path} are still marked REVIEW '
              f'and were ignored:\n       {pending}\n'
              f'       Fix their cx,cy by eye and set status to "ok" to use them.')
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
                   mask_suffix='-mask.png',
                   on_reject='skip', use_radius_lock=True, radius_mm=None,
                   escalate=True, sat_thresh=DEFAULT_SAT_THRESH,
                   soft_min_score=SOFT_MIN_SCORE):
 
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
 
    # --centres_csv given as a BARE FILENAME (no folder separator) is resolved
    # inside img_dir. That makes a per-folder loop trivial: the same literal
    # argument picks up each folder's own file, with no shell expansion.
    #     for /D %D in (...\*) do python ... "%D" --centres_csv well_centres_manual.csv
    manual = {}
    if centres_csv:
        cpath = Path(centres_csv)
        if not cpath.is_absolute() and cpath.parent in (Path('.'), Path('')):
            cpath = img_dir / cpath.name
        if cpath.exists():
            manual = read_centres_csv(cpath)
            print(f'Loaded {len(manual)} manual centre(s) from {cpath}')
        else:
            # Never silent. A missing centres file used to fall through to
            # automatic detection with no message, which on this imaging setup
            # means every well is rejected and the folder produces nothing.
            print(f'[WARN] --centres_csv "{centres_csv}" not found '
                  f'(looked for {cpath}).\n'
                  f'       Falling back to automatic detection for every well '
                  f'in this folder.')
 
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
    if radius_mm is not None:
        # Physical lock. The well bottom has a known diameter, so the correct
        # ring can be named instead of inferred. This is the strongest way to
        # stop different images latching onto different concentric rings.
        r_lock = float(radius_mm) * PX_PER_MM
        print(f'Rim radius locked from --radius_mm: {radius_mm:.2f} mm '
              f'= {r_lock:.1f} px (pass A skipped)\n')
    elif use_radius_lock and cv2 is not None:
        print('Pass A: learning plate-wide rim radius ...')
        radii = []
        stage_a = {}
        for stem, _, tif_path in items:
            if stem in manual:
                continue
            if (img_dir / f'{stem}.json').exists():
                j = read_well_centre_from_json(img_dir / f'{stem}.json')
                if j:
                    radii.append(j[2])
                    continue
            img_rgb = load_rgb(tif_path)
            d = detect_with_escalation(img_rgb, None, search_frac, min_score,
                                       escalate=escalate, sat_thresh=sat_thresh,
                                       soft_min_score=soft_min_score)
            if d['cleared'] and np.isfinite(d['r']):
                stage_a[d['stage']] = stage_a.get(d['stage'], 0) + 1
                # Only stages 1 and 2 consume r_lock, and the concentric ring a
                # detector settles on is detector-specific. Mixing a stage-3
                # radius into the lock would push stages 1 and 2 onto a ring
                # they never chose.
                if d['stage'] in (1, 2):
                    radii.append(d['r'])
        r_lock = robust_median(radii) if len(radii) >= 3 else None
        if r_lock is not None and np.isfinite(r_lock):
            ra = np.asarray(radii, dtype=float)
            spread = (ra.max() - ra.min()) / r_lock if r_lock > 0 else float('nan')
            print(f'  consensus rim radius = {r_lock:.1f} px '
                  f'({r_lock / PX_PER_MM:.2f} mm) from {len(radii)} detection(s)')
            print(f'  pass-A radii: min {ra.min():.0f}  median {r_lock:.0f}  '
                  f'max {ra.max():.0f} px  (spread {spread:.0%})')
            if stage_a:
                print('  detector stage used: ' + '  '.join(
                    f'{n} image(s) at stage {s}'
                    for s, n in sorted(stage_a.items())))
            if np.isfinite(spread) and spread > 0.20:
                # Wide spread means the detector is NOT seeing one ring. Locking
                # the median then forces every image onto a ring that may not be
                # the one the scaffold sits in, which shifts every centre.
                print('  [WARN] the pass-A radii disagree by more than 20%. The '
                      'detector is not\n         seeing a single ring. Locking '
                      'the median here can bias every centre.\n'
                      '         Measure the well bottom diameter and pass '
                      '--radius_mm instead.')
            print()
        else:
            print('  not enough confident detections; radius lock disabled\n')
    else:
        print('Radius lock disabled.\n')
 
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
        d = detect_with_escalation(img_rgb, r_lock, search_frac, min_score,
                                   escalate=escalate, sat_thresh=sat_thresh,
                                   soft_min_score=soft_min_score)
        d['source'] = 'detect'
        dets[stem] = d
        if d['stage'] > 1:
            print(f'  {stem}: stage 1 scored {"below bar" if not d["cleared"] else "low"}, '
                  f'used stage {d["stage"]} ({d["method"]}, score {d["score"]:.2f})'
                  f'{"" if d["cleared"] else "  [still below bar]"}')
 
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
    def _cleared(d):
        """Did this detection clear the bar appropriate to the stage that found it."""
        return d.get('cleared', d['score'] >= min_score)
 
    trusted = [d for d in dets.values() if _cleared(d)]
    if len(trusted) >= 3:
        med_cx = robust_median([d['cx'] for d in trusted])
        med_cy = robust_median([d['cy'] for d in trusted])
        med_r  = robust_median([d['r'] for d in trusted])
        # Different detector stages settle on different concentric rings, so a
        # single plate-wide median radius would flag every well found at the
        # minority stage as an outlier. Compare like with like.
        med_r_stage = {}
        for st in {d.get('stage', 1) for d in trusted}:
            rs = [d['r'] for d in trusted if d.get('stage', 1) == st]
            if len(rs) >= 3:
                med_r_stage[st] = robust_median(rs)
        dists  = [float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy)) for d in trusted]
        med_d  = robust_median(dists)
        gate   = max(centre_tol_px, DEFAULT_CENTRE_TOL_K * med_d)
        print(f'  plate median centre = ({med_cx:.0f}, {med_cy:.0f})  '
              f'median radius = {med_r:.0f} px  '
              f'from {len(trusted)}/{len(dets)} confident detection(s)')
        print(f'  imaging scatter: median {med_d:.0f} px ({med_d / PX_PER_MM:.2f} mm) '
              f'-> outlier gate {gate:.0f} px ({gate / PX_PER_MM:.2f} mm)')
        if len(trusted) < 6:
            print(f'  [NOTE] the plate median is built from only {len(trusted)} '
                  f'detection(s). On a small\n         plate it is not a reliable '
                  f'stand-in for a well whose own detection failed.')
    else:
        med_cx = med_cy = med_r = med_d = float('nan')
        med_r_stage = {}
        gate = centre_tol_px
        print('  fewer than 3 confident detections; no plate median available.')
 
    # ---- what happens to a well that fails its checks ------------------------
    #
    # The old behaviour was to substitute the plate median centre for any well
    # that failed. That is defensible on a full 48-well plate and WRONG on a
    # small one: the camera genuinely moves between wells, so the median is off
    # by the imaging scatter, and one well ends up visibly shifted while its
    # neighbours look fine. THAT substitution is the main source of the large
    # target-geometry shifts. It is no longer the default.
    #
    #   on_reject='skip'    (default) no target mask, no overlay, listed for review
    #   on_reject='median'            old behaviour, kept to reproduce old runs
    #   on_reject='keep'              use the raw per-image detection unchanged
    #
    # ---- outlier reconciliation ---------------------------------------------
    #
    # A well can clear its confidence bar and STILL be in the wrong place: the
    # detector locked confidently onto the wrong circle. Previously the only
    # response was to reject it and substitute the plate median, i.e. throw the
    # image's own evidence away.
    #
    # Instead, for a well that is a spatial outlier, re-run the other detectors
    # on THAT image with the plate radius locked, and choose among the
    # hypotheses that clear their own bar the one nearest the plate median. The
    # plate is used as a PRIOR to disambiguate between candidate circles found
    # in this image, never as a substitute for them. A well that is not an
    # outlier is not touched, so nothing that currently works can move.
    #
    if escalate and np.isfinite(med_cx):
        tif_by_stem = {s: t for s, _m, t in items}
        wide = max(search_frac, WIDE_SEARCH_FRAC)
        for stem, d in dets.items():
            if d['source'] in ('csv', 'json') or not _cleared(d):
                continue
            dist0 = float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy))
            if dist0 <= gate:
                continue
            img_rgb = load_rgb(tif_by_stem[stem])
            cands = [d]
            alt1 = detect_well_circle(img_rgb, r_lock=med_r, search_frac=wide,
                                      edge_mode='robust', sat_thresh=sat_thresh)
            if alt1['score'] >= min_score:
                alt1.update(stage=2, cleared=True, source='detect')
                cands.append(alt1)
            alt2 = detect_well_circle_soft(img_rgb, r_lock=med_r,
                                           search_frac=wide, sat_thresh=sat_thresh)
            if alt2['score'] >= soft_min_score:
                alt2.update(stage=3, cleared=True, source='detect')
                cands.append(alt2)
            best = min(cands, key=lambda x: np.hypot(x['cx'] - med_cx,
                                                     x['cy'] - med_cy))
            if best is not d:
                dist1 = float(np.hypot(best['cx'] - med_cx, best['cy'] - med_cy))
                print(f'  [ALT]  {stem}: {dist0 / PX_PER_MM:.2f}mm from the plate '
                      f'median exceeded the gate;')
                print(f'         a stage-{best["stage"]} circle in the SAME image '
                      f'sits {dist1 / PX_PER_MM:.2f}mm away '
                      f'(score {best["score"]:.2f}) -> using that')
                d.update(best)
                d['method'] = best['method'] + '-alt'
 
    rejected = {}
    for stem, d in dets.items():
        if d['source'] in ('csv', 'json'):
            continue
        reason = None
        if not _cleared(d):
            bar = soft_min_score if d.get('stage') == 3 else min_score
            reason = (f"score {d['score']:.2f} < {bar} after "
                      f"{'escalation to stage %d' % abs(d.get('stage', 1)) if escalate else 'stage 1'}")
        else:
            # Reference radius for THIS well's detector stage. If fewer than
            # three wells were found at that stage there is no reliable
            # reference, and falling back to the plate-wide median would be
            # actively wrong: the stages settle on different rings, so the
            # minority stage would be rejected for disagreeing with the
            # majority's ring rather than for being a bad detection.
            ref_r = med_r_stage.get(d.get('stage', 1))
            if ref_r is not None and np.isfinite(ref_r) and np.isfinite(d['r']) and \
                    abs(d['r'] - ref_r) > radius_tol * ref_r:
                reason = (f"radius {d['r']:.0f}px off the stage-"
                          f"{d.get('stage', 1)} plate median {ref_r:.0f}px "
                          f"by >{radius_tol:.0%}")
            elif np.isfinite(med_cx):
                dist = float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy))
                if dist > gate:
                    reason = (f'{dist:.0f}px ({dist / PX_PER_MM:.2f}mm) from '
                              f'plate median, gate {gate:.0f}px')
        if not reason:
            continue
 
        d['reject_reason'] = reason
        if on_reject == 'median' and np.isfinite(med_cx):
            sub = float(np.hypot(d['cx'] - med_cx, d['cy'] - med_cy))
            print(f'  [FIX]  {stem}: rejected ({reason})')
            print(f'         -> plate median centre, target moved '
                  f'{sub:.0f}px ({sub / PX_PER_MM:.2f}mm)')
            d.update(cx=med_cx, cy=med_cy, r=med_r, source='consensus')
        elif on_reject == 'keep':
            print(f'  [KEEP] {stem}: rejected ({reason}) but kept as detected; '
                  f'verify this overlay by eye')
            d['source'] = 'kept-unverified'
        else:
            print(f'  [SKIP] {stem}: rejected ({reason}) -> NOT rendered')
            d['source'] = 'rejected'
            rejected[stem] = reason
    print()
 
    # ---------------------------------------------------------------- render
    results = []
    for stem, mask_path, tif_path in items:
        d = dets[stem]
 
        if d['source'] == 'rejected':
            # No target mask and no overlay are written, so this well cannot
            # enter pore_analysis.py / the SF table with a wrong anchor. The
            # debug picture IS still written so you can see what went wrong.
            if debug_dir:
                img_rgb = load_rgb(tif_path)
                save_debug(img_rgb, d, d['cx'], d['cy'],
                           debug_dir / f'{stem}-welldet.png')
            continue
 
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
 
        # distance of this well's anchor from the plate median, in mm. This is
        # the number to watch: a well that is far from the rest is the one whose
        # target geometry will look shifted.
        if np.isfinite(med_cx):
            dist_mm = float(np.hypot(wcx - med_cx, wcy - med_cy)) / PX_PER_MM
            d['dist_mm'] = dist_mm
            dist_str = f'  d_med={dist_mm:.2f}mm'
        else:
            dist_str = ''
 
        flag = '  *' if iou_threshold > 0 and iou >= iou_threshold else ''
        print(f'[OK] {stem}  IoU={iou:.3f}{flag}'
              f'  well=({wcx:.0f},{wcy:.0f})[{d["source"]},score={d["score"]:.2f}]'
              f'{dist_str}'
              f'  corr=({dx_mm:+.2f},{dy_mm:+.2f})mm'
              f'  target=({cx:.0f},{cy:.0f})')
        results.append((stem, iou, d))
 
    # ---------------------------------------------------------------- summary
    csv_out = output_dir / 'well_centres.csv'
    iou_by_stem = {stem: iou for stem, iou, _ in results}
    with open(csv_out, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['stem', 'cx', 'cy', 'r', 'score', 'source', 'iou',
                      'status', 'reason', 'stage', 'method'])
        for stem, _mask_path, _tif_path in items:
            d = dets[stem]
            is_rej = d['source'] == 'rejected'
            wtr.writerow([
                stem, f'{d["cx"]:.2f}', f'{d["cy"]:.2f}',
                f'{d["r"]:.2f}' if np.isfinite(d.get('r', np.nan)) else '',
                f'{d["score"]:.3f}', d['source'],
                f'{iou_by_stem[stem]:.4f}' if stem in iou_by_stem else '',
                'REVIEW' if is_rej else 'ok',
                d.get('reject_reason', ''),
                d.get('stage', ''), d.get('method', ''),
            ])
 
    review_out = None
    if rejected:
        review_out = output_dir / 'well_centres_review.csv'
        with open(review_out, 'w', newline='') as f:
            wtr = csv.writer(f)
            wtr.writerow(['stem', 'cx', 'cy', 'r', 'score', 'status', 'reason'])
            for stem, reason in rejected.items():
                d = dets[stem]
                wtr.writerow([
                    stem, f'{d["cx"]:.2f}', f'{d["cy"]:.2f}',
                    f'{d["r"]:.2f}' if np.isfinite(d.get('r', np.nan)) else '',
                    f'{d["score"]:.3f}', 'REVIEW', reason])
 
    if results:
        print('\n-- IoU summary (sorted) --')
        for stem, iou, d in sorted(results, key=lambda x: -x[1]):
            flag = '  *' if iou_threshold > 0 and iou >= iou_threshold else ''
            dm = d.get('dist_mm')
            dstr = f'   d_med={dm:.2f}mm' if dm is not None else ''
            print(f'  {stem}: {iou:.3f}{flag}   [{d["source"]}]{dstr}')
 
        # The anchor, not the print, is what a large d_med points at.
        dms = [(s, d['dist_mm']) for s, _, d in results if d.get('dist_mm') is not None]
        if dms:
            worst = sorted(dms, key=lambda x: -x[1])[:3]
            print('\n-- anchors furthest from the plate median --')
            for s, dm in worst:
                print(f'  {s}: {dm:.2f} mm')
            print('  A well near the top of this list with a poor IoU is an '
                  'anchor problem,\n  not a printing problem. Check its '
                  '-welldet.png before believing its SF.')
 
        n_fixed = sum(1 for _, _, d in results
                      if d['source'] in ('consensus', 'image-centre',
                                         'kept-unverified'))
        if n_fixed:
            print(f'\n{n_fixed}/{len(results)} rendered well(s) did NOT use a '
                  f'verified own detection. Inspect them in --debug_dir.')
 
    if rejected:
        print(f'\n{len(rejected)} well(s) were NOT rendered because their centre '
              f'could not be trusted:')
        for stem, reason in rejected.items():
            print(f'  {stem}: {reason}')
        print(f'\nNo -target-mask.png was written for these, so pore_analysis.py '
              f'will refuse to\nscore them rather than score them wrongly. To '
              f'recover them:')
        print(f'  1. look at <debug_dir>/{{stem}}-welldet.png')
        print(f'  2. correct cx,cy in {csv_out.name} and set status to "ok"')
        print(f'  3. re-run with --centres_csv {csv_out.name}')
        print(f'Review list also written to {review_out.name}')
        if len(rejected) > 0.3 * len(items):
            print('\nMany rejections. Before hand-correcting, try in this order:')
            print('  --search_frac 0.30      wells framed more loosely than +/-22%')
            print('  --min_score 0.20        low-contrast rims')
            print('  --radius_mm <r>         lock the rim radius from the known '
                  'well geometry')
 
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
    parser.add_argument('--no_consensus', action='store_true',
                        help='Legacy shorthand for --no_radius_lock --on_reject keep')
    parser.add_argument('--on_reject', choices=['skip', 'median', 'keep'],
                        default='skip',
                        help="What to do with a well whose centre fails the "
                             "checks. 'skip' (default) writes no target mask and "
                             "lists it for review, so a wrong anchor can never "
                             "reach the SF table. 'median' is the OLD behaviour "
                             "(substitute the plate median centre), which is what "
                             "produced the large target shifts on small plates. "
                             "'keep' renders the raw detection unchanged.")
    parser.add_argument('--no_radius_lock', action='store_true',
                        help='Skip pass A. Each image picks its own rim radius. '
                             'Use this to test whether the plate-wide radius lock '
                             'is latching onto the wrong concentric ring.')
    parser.add_argument('--no_escalate', action='store_true',
                        help='Stage 1 only. Reproduces the detector exactly as '
                             'it was before the escalation ladder was added.')
    parser.add_argument('--sat_thresh', type=int, default=DEFAULT_SAT_THRESH,
                        help=f'Grey value at or above which a pixel counts as a '
                             f'specular blow-out and is excluded from stages 2 '
                             f'and 3 (default: {DEFAULT_SAT_THRESH})')
    parser.add_argument('--soft_min_score', type=float, default=SOFT_MIN_SCORE,
                        help=f'Acceptance bar for the stage-3 gradient-profile '
                             f'detector (default: {SOFT_MIN_SCORE})')
    parser.add_argument('--px_per_mm', type=float, default=None,
                        help=f'Camera scale in pixels per mm (default: '
                             f'{PX_PER_MM}). THIS is the knob for a camera '
                             f'calibration error. Scaling --strand_width_mm and '
                             f'--strand_gap_mm instead only works if you apply '
                             f'the SAME factor to both, and it silently breaks '
                             f'--centre_tol_mm and every distance this script '
                             f'reports in mm.')
    parser.add_argument('--radius_mm', type=float, default=None,
                        help='Lock the rim radius to this physical value in mm '
                             'instead of inferring it from the plate. Strongest '
                             'way to force every image onto the same ring.')
    parser.add_argument('--global_offset_mm', type=float, nargs=2,
                        metavar=('DX', 'DY'), default=None,
                        help='Override GLOBAL_OFFSET_MM from the command line')
    parser.add_argument('--row_step_mm', type=float, nargs=2,
                        metavar=('DX', 'DY'), default=None,
                        help='Override ROW_STEP from the command line')
    parser.add_argument('--col_step_mm', type=float, nargs=2,
                        metavar=('DX', 'DY'), default=None,
                        help='Override COL_STEP from the command line')
    args = parser.parse_args()
 
    if cv2 is None:
        sys.exit('OpenCV (cv2) is required for well detection. pip install opencv-python')
 
    if args.px_per_mm is not None:
        PX_PER_MM = float(args.px_per_mm)
    print(f'Camera scale: PX_PER_MM = {PX_PER_MM}')
 
    # Sanity check on the target geometry. A camera calibration error is ONE
    # scale factor and must be applied to both strand parameters equally, or
    # to PX_PER_MM. Different factors change the target's fill fraction, which
    # changes IoU and SF systematically and makes this plate incomparable to
    # any plate scored with different numbers.
    _wf = args.strand_width_mm / DEFAULT_STRAND_WIDTH_MM
    _gf = args.strand_gap_mm / DEFAULT_STRAND_GAP_MM
    if abs(_wf - _gf) > 0.05 * max(_wf, _gf):
        print(f'[WARN] --strand_width_mm is {_wf:.2f}x the 22G default and '
              f'--strand_gap_mm is {_gf:.2f}x\n'
              f'       the design default. These are different factors, so this '
              f'is not a pure\n'
              f'       scale correction: the target strands are '
              f'{_wf / _gf:.2f}x thicker relative to\n'
              f'       the pitch than the G-code design. The target fill '
              f'fraction, and therefore\n'
              f'       IoU and SF, are shifted. Use --px_per_mm for a '
              f'calibration error instead.')
 
    # The module-level anchor blocks are stacked presets and the LAST one
    # executed wins, which is easy to get wrong when a block is uncommented
    # further down the file. Allow a CLI override and always print what is
    # actually in force.
    if args.global_offset_mm is not None:
        GLOBAL_OFFSET_MM = tuple(args.global_offset_mm)
    if args.row_step_mm is not None:
        ROW_STEP = tuple(args.row_step_mm)
    if args.col_step_mm is not None:
        COL_STEP = tuple(args.col_step_mm)
    print(f'Anchor correction in force: GLOBAL_OFFSET_MM={GLOBAL_OFFSET_MM}  '
          f'ROW_STEP={ROW_STEP}  COL_STEP={COL_STEP}'
          f'{"  (disabled by --no_drift)" if args.no_drift else ""}')
 
    on_reject      = 'keep' if args.no_consensus else args.on_reject
    use_radius_lock = not (args.no_consensus or args.no_radius_lock)
    print(f'Detector: stage 1 only' if args.no_escalate else
          f'Detector: escalation ladder stage 1 -> 2 -> 3  '
          f'(sat_thresh={args.sat_thresh}, soft bar={args.soft_min_score})')
    print(f'Rejected wells: {on_reject}   radius lock: '
          f'{"--radius_mm %.2f" % args.radius_mm if args.radius_mm is not None else ("pass A" if use_radius_lock else "off")}\n')
 
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
        on_reject=on_reject,
        use_radius_lock=use_radius_lock,
        radius_mm=args.radius_mm,
        escalate=not args.no_escalate,
        sat_thresh=args.sat_thresh,
        soft_min_score=args.soft_min_score,
    )