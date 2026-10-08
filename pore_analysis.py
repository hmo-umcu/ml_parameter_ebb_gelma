"""
pore_analysis.py
-----------------
Per-image pore-position scoring for shape fidelity.

For each {stem}, requires:
    {stem}.tif                       raw image                    REQUIRED
    {stem}-target-mask.png           exact binary target geometry mask,
                                      written by draw_target_geometry.py    PREFERRED
    {stem}-target-overlay.png        target overlay (draw_target_geometry.py)
                                      used as FALLBACK only if -target-mask.png
                                      is missing — lossy colour-threshold
                                      extraction, kept for backward
                                      compatibility with older data
    {stem}-pred-mask.png             binary predicted mask (unetplusplus_test.py)
    {stem}-pred-visible.png          fallback ONLY if -pred-mask.png is missing
                                      (predicted mask is reconstructed from the
                                      green overlay colour — less exact, use
                                      -pred-mask.png whenever it exists)

Produces five PNGs per image, written to <output_dir>:
    {stem}-0-target-mask-bw.png   target footprint, pure black/white, extracted
                                   directly from -target-overlay.png BEFORE
                                   anything else — inspect this first if pore
                                   counts/positions look wrong, since every
                                   downstream step (target pore positions,
                                   IoU, matching) derives from this mask
    {stem}-1-target-on-raw.png    raw image + target geometry overlay only
    {stem}-2-pred-mask-bw.png     predicted mask, pure black/white, no raw image
    {stem}-3-pred-pores.png       enclosed pores extracted from predicted mask
                                   only (black/white; empty/black if none found)
    {stem}-4-scored-overlay.png   raw image + target geometry + matched,
                                   scored predicted pores (score text on each)

Also writes:
    pore_scores.csv     per-image SF, IoU(pred|target), per-pore Pr_i, etc.

Pore-position logic
--------------------
1. Target pore positions are derived from the target-overlay file itself:
   the target footprint (green+yellow pixels) has N enclosed background
   regions = the N expected pore positions for this geometry (typically 4
   for a 3x3 crosshatch).
2. Predicted pores are enclosed background regions found inside the
   predicted mask's own footprint (same detection logic, applied to the
   predicted mask instead of the target mask).
3. A predicted pore is matched to a target pore position i if its centroid
   falls inside that target region (primary rule — robust to predicted
   pores being smaller/larger/misshapen relative to target, which is what
   Pr_i is meant to capture), OR if it overlaps that target region by
   >= --match_overlap_frac of the smaller of the two areas (fallback for
   off-centre partial overlaps). Multiple predicted components matching
   the same target position are merged (union) before scoring.
4. Pr_i = IoU(matched predicted pixels, target pore_i)  if matched
         = 0                                            if target pore_i
                                                          was never matched
   Predicted pores that do not overlap ANY target pore position by the
   threshold are excluded entirely: not counted, not drawn in the final
   scored overlay.

SF formula
----------
    SF = (1 - w) * IoU(pred|target) + (w / N) * sum(Pr_i)

    where N = number of target pore positions found in this image,
    w = --w (default 0.25), weighting the pore bonus against the IoU term.
    Normalized so a perfect print (IoU=1, every Pr_i=1) gives SF=1.0 exactly.

Usage
-----
    python pore_analysis.py \
        --data_dir   /path/to/folder/with/tif_overlay_predmask \
        --output_dir /path/to/output \
        [--w 0.25] \
        [--min_pore_px 10000] \
        [--max_pore_px 150000] \
        [--max_aspect_ratio 3.0] \
        [--match_overlap_frac 0.3]
"""

import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


# ── colours ────────────────────────────────────────────────────────────────
TARGET_COLOUR   = np.array([60,  180, 255], dtype=np.float32)   # sky blue, target footprint
TARGET_PORE_OUT = (255, 255, 0)                                  # cyan-ish outline, BGR-safe via PIL RGB -> (0,255,255)? use RGB tuple
MATCHED_FILL    = np.array([255, 60,  60 ], dtype=np.float32)   # red, matched predicted pore
ALPHA_FOOTPRINT = 0.40
ALPHA_PORE      = 0.55

PRED_OVERLAY_COLOUR = np.array([60, 220, 60], dtype=np.float32)  # must match unetplusplus_test.py PRED_COLOUR
PRED_OVERLAY_ALPHA  = 0.45                                       # must match unetplusplus_test.py ALPHA


# ── target mask extraction (same logic as unetplusplus_evaluate.py) ─────────
def extract_target_mask(overlay_path):
    """GREEN + YELLOW pixels in *-target-overlay.png => target footprint.

    Fallback only — lossy/noisy due to anti-aliasing and specular
    highlights on the gel surface. Prefer load_target_mask() below, which
    uses the exact binary mask saved by draw_target_geometry.py.
    """
    img = np.array(Image.open(overlay_path).convert('RGB'))
    R = img[:, :, 0].astype(int)
    G = img[:, :, 1].astype(int)
    B = img[:, :, 2].astype(int)

    green_only = (G - R > 50) & (G - B > 50)
    yellow     = (R > 140) & (G > 130) & (B < 100) & (np.abs(R - G) < 60)

    return (green_only | yellow).astype(np.uint8)


def load_target_mask(data_dir, stem):
    """
    Load the target geometry mask for `stem`, preferring the exact binary
    mask written by draw_target_geometry.py ({stem}-target-mask.png —
    0/255, no colour thresholding needed) and falling back to lossy
    colour-threshold extraction from {stem}-target-overlay.png only if
    that file doesn't exist (e.g. data generated before that script was
    updated to save it).

    Returns (target_mask, source) where source is 'exact-mask' or
    'overlay-extraction' (logged so silently-degraded data is visible).
    """
    data_dir = Path(data_dir)
    exact_path = data_dir / f'{stem}-target-mask.png'

    if exact_path.exists():
        arr = np.array(Image.open(exact_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return (arr > 0).astype(np.uint8), 'exact-mask'

    overlay_path = data_dir / f'{stem}-target-overlay.png'
    if overlay_path.exists():
        return extract_target_mask(overlay_path), 'overlay-extraction'

    return None, 'missing'


def extract_pred_mask_from_overlay(overlay_path):
    """
    Fallback: reconstruct binary predicted mask from a *-pred-visible.png
    overlay (green-tinted prediction blended onto raw image at known alpha).
    Only used when *-pred-mask.png is not available.
    """
    img = np.array(Image.open(overlay_path).convert('RGB')).astype(np.float32)
    # pixels where green channel is boosted relative to red/blue, consistent
    # with PRED_OVERLAY_COLOUR blended at PRED_OVERLAY_ALPHA
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    boosted = (G - R > 25) & (G - B > 25)
    return boosted.astype(np.uint8)


# ── enclosed-hole detection (shared by target and predicted masks) ──────────
def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def find_enclosed_holes(mask, bbox, min_px, max_px, max_ar, close_kernel=21):
    """
    mask: binary uint8 (1=foreground/strand, 0=background)
    bbox: (x0,y0,x1,y1) restrict hole search to this region (the mask's own
          footprint bounding box) so stray background elsewhere in the image
          is never picked up.
    close_kernel: morphological closing kernel size (px) applied to `mask`
          before hole detection. Seals gaps in the colour-threshold mask —
          both thin anti-aliasing gaps along strand edges AND, importantly,
          speckled noise from specular highlights on the wet gel surface
          (bright reflections fail the green/yellow RGB thresholds, leaving
          scattered background-classified pixels inside the target
          footprint). If this speckle noise forms a connected chain from
          the image border into an enclosed pore, flood-fill will
          mis-classify that pore as "outside" and it silently disappears
          from detection. Default raised to 21px after diagnosing exactly
          this failure mode (kernel=9 lost one of four pores on a noisy
          sample; kernel=15+ reliably recovered all four). Pores here are
          ~hundreds of px across with ~100px+ gaps between them, so this
          kernel size closes leaks without merging genuinely separate pores
          — but re-check --close_kernel against your own pore spacing if
          you see it merging adjacent pores in -0-target-mask-bw.png.

    Returns:
        holes_labelled : uint16 array, 0 = not a hole, i = hole id (1..n)
        hole_list       : list of dicts {id, area, bbox(x,y,w,h), centroid, mask}
    """
    h, w = mask.shape
    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bg = (mask == 0).astype(np.uint8)

    # flood fill from a border corner to mark "outside" background as 2,
    # leaving truly enclosed background regions at value 1
    ff = bg.copy()
    flood_scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_scratch, (0, 0), 2)
    interior_bg = (ff == 1).astype(np.uint8)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        restrict = np.zeros_like(interior_bg)
        restrict[y0:y1, x0:x1] = 1
        interior_bg = interior_bg & restrict

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        interior_bg, connectivity=8)

    holes_labelled = np.zeros_like(labels, dtype=np.uint16)
    hole_list = []
    next_id = 0

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw_  = stats[i, cv2.CC_STAT_WIDTH]
        bh_  = stats[i, cv2.CC_STAT_HEIGHT]
        ar   = max(bw_, bh_) / max(1, min(bw_, bh_))

        if not (min_px <= area <= max_px):
            continue
        if ar > max_ar:
            continue

        next_id += 1
        comp_mask = (labels == i)
        holes_labelled[comp_mask] = next_id
        hole_list.append({
            'id':       next_id,
            'area':     int(area),
            'bbox':     (int(stats[i, cv2.CC_STAT_LEFT]),
                         int(stats[i, cv2.CC_STAT_TOP]), int(bw_), int(bh_)),
            'centroid': (float(centroids[i][0]), float(centroids[i][1])),
            'mask':     comp_mask,
        })

    return holes_labelled, hole_list


# ── metrics ───────────────────────────────────────────────────────────────────
def binary_iou(a, b):
    a, b = a > 0, b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ── matching predicted pores to target pore positions ────────────────────────
def match_pores(target_holes, pred_holes, match_overlap_frac):
    """
    For each target pore position, find predicted pore components that are
    spatially AT that position — primarily judged by the predicted pore's
    centroid falling inside the target pore region (robust to predicted
    pores being legitimately smaller/larger/odd-shaped than the target,
    which is exactly what Pr_i is meant to score). As a fallback for
    off-centre but still overlapping pores, also match if the intersection
    covers >= match_overlap_frac of whichever pore (target or predicted)
    is smaller.

    Multiple predicted components matching the same target position are
    merged (union) before scoring.

    Returns:
        per_target: list of dicts {target_id, matched (bool), pr, mask_pred (merged
                    predicted pixels for this target, or None), target_mask}
        unmatched_pred_ids: list of predicted hole ids that matched no target
                    position at all (excluded from scoring/drawing)
    """
    per_target = []
    claimed_pred_ids = set()

    for t in target_holes:
        t_mask = t['mask']
        t_area = t['area']
        matched_pred_masks = []

        for p in pred_holes:
            cx, cy = p['centroid']
            centroid_inside = t_mask[int(round(cy)), int(round(cx))] > 0 \
                if 0 <= int(round(cy)) < t_mask.shape[0] \
                and 0 <= int(round(cx)) < t_mask.shape[1] else False

            inter = np.logical_and(t_mask, p['mask']).sum()
            smaller_area = min(t_area, p['area'])
            overlap_ok = (smaller_area > 0) and \
                         (inter / smaller_area) >= match_overlap_frac

            if centroid_inside or overlap_ok:
                matched_pred_masks.append(p['mask'])
                claimed_pred_ids.add(p['id'])

        if matched_pred_masks:
            merged = np.zeros_like(t_mask)
            for m in matched_pred_masks:
                merged |= m
            pr = binary_iou(merged, t_mask)
            per_target.append({
                'target_id': t['id'], 'matched': True, 'pr': pr,
                'mask_pred': merged, 'target_mask': t_mask,
                'target_bbox': t['bbox'], 'target_centroid': t['centroid'],
            })
        else:
            per_target.append({
                'target_id': t['id'], 'matched': False, 'pr': 0.0,
                'mask_pred': None, 'target_mask': t_mask,
                'target_bbox': t['bbox'], 'target_centroid': t['centroid'],
            })

    unmatched_pred_ids = [p['id'] for p in pred_holes
                           if p['id'] not in claimed_pred_ids]
    return per_target, unmatched_pred_ids


# ── drawing helpers ───────────────────────────────────────────────────────────
def blend(img_f, mask, colour, alpha):
    img_f = img_f.copy()
    m = mask > 0
    img_f[m] = (1 - alpha) * img_f[m] + alpha * colour
    return img_f


def draw_dashed_rect_outline(canvas_u8, bbox, colour_bgr, thickness=2):
    x, y, w, h = bbox
    cv2.rectangle(canvas_u8, (x, y), (x + w, y + h), colour_bgr, thickness)


# ── per-image processing ──────────────────────────────────────────────────────
def process_image(stem, data_dir, output_dir, w, min_px, max_px, max_ar,
                   match_overlap_frac, close_kernel=21):

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    tif_path     = data_dir / f'{stem}.tif'
    predmask_path  = data_dir / f'{stem}-pred-mask.png'
    predvis_path   = data_dir / f'{stem}-pred-visible.png'

    if not tif_path.exists():
        print(f'  [SKIP] {stem}: missing .tif')
        return None

    raw = np.array(Image.open(tif_path).convert('RGB'))

    target_mask, target_src = load_target_mask(data_dir, stem)
    if target_mask is None:
        print(f'  [SKIP] {stem}: no -target-mask.png or -target-overlay.png found')
        return None
    if target_src == 'overlay-extraction':
        print(f'  [WARN] {stem}: -target-mask.png not found, falling back to '
              f'colour-threshold extraction from -target-overlay.png '
              f'(less exact — re-run draw_target_geometry.py to fix)')

    # ── predicted mask ─────────────────────────────────────────────────────
    if predmask_path.exists():
        arr = np.array(Image.open(predmask_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        pred_mask = (arr > 0).astype(np.uint8)
    elif predvis_path.exists():
        print(f'  [WARN] {stem}: -pred-mask.png not found, '
              f'reconstructing from -pred-visible.png (less exact)')
        pred_mask = extract_pred_mask_from_overlay(predvis_path)
    else:
        print(f'  [SKIP] {stem}: no -pred-mask.png or -pred-visible.png found')
        return None

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT 0 — target footprint, pure black/white, no raw image, no blend.
    # Saved BEFORE anything else derives from target_mask, so it reflects
    # exactly what the colour-threshold extraction produced. If a pore
    # position is missing downstream (in target pore count, IoU, or the
    # scored overlay), check this file first — every later step depends on
    # this mask, so a gap here propagates everywhere.
    # ════════════════════════════════════════════════════════════════════
    out0 = (target_mask * 255).astype(np.uint8)
    Image.fromarray(out0).save(output_dir / f'{stem}-0-target-mask-bw.png')

    # ── overall shape-fidelity IoU ────────────────────────────────────────
    iou_pred_target = binary_iou(pred_mask, target_mask)

    # ── target / predicted pore detection ───────────────────────────────────
    target_bbox = mask_bbox(target_mask)
    pred_bbox   = mask_bbox(pred_mask)

    _, target_holes = find_enclosed_holes(
        target_mask, target_bbox, min_px, max_px, max_ar, close_kernel)
    _, pred_holes = find_enclosed_holes(
        pred_mask, pred_bbox, min_px, max_px, max_ar, close_kernel)

    per_target, unmatched_pred_ids = match_pores(
        target_holes, pred_holes, match_overlap_frac)

    n_target = len(per_target)
    sum_pr   = sum(t['pr'] for t in per_target)
    sf       = (1 - w) * iou_pred_target + (w / n_target) * sum_pr if n_target > 0 \
               else iou_pred_target

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT 1 — raw + target geometry overlay only
    # ════════════════════════════════════════════════════════════════════
    out1 = blend(raw.astype(np.float32), target_mask, TARGET_COLOUR,
                 ALPHA_FOOTPRINT)
    out1 = np.clip(out1, 0, 255).astype(np.uint8)
    Image.fromarray(out1).save(output_dir / f'{stem}-1-target-on-raw.png')

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT 2 — predicted mask, pure black/white, no raw image
    # ════════════════════════════════════════════════════════════════════
    out2 = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(out2).save(output_dir / f'{stem}-2-pred-mask-bw.png')

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT 3 — pores extracted from predicted segments only (black/white)
    # ════════════════════════════════════════════════════════════════════
    out3 = np.zeros_like(pred_mask, dtype=np.uint8)
    for p in pred_holes:
        out3[p['mask']] = 255
    Image.fromarray(out3).save(output_dir / f'{stem}-3-pred-pores.png')

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT 4 — raw + target geometry + matched, scored predicted pores
    # ════════════════════════════════════════════════════════════════════
    out4_f = blend(raw.astype(np.float32), target_mask, TARGET_COLOUR,
                   ALPHA_FOOTPRINT)

    for t in per_target:
        if t['matched']:
            out4_f = blend(out4_f, t['mask_pred'], MATCHED_FILL, ALPHA_PORE)

    out4 = np.clip(out4_f, 0, 255).astype(np.uint8)
    out4_bgr = cv2.cvtColor(out4, cv2.COLOR_RGB2BGR)

    # outline every expected target pore position (cyan) so misses are visible
    for t in per_target:
        x, y, bw_, bh_ = t['target_bbox']
        cv2.rectangle(out4_bgr, (x, y), (x + bw_, y + bh_), (255, 255, 0), 2)

    # write score text on matched pores; mark unmatched target positions
    for t in per_target:
        cx, cy = t['target_centroid']
        if t['matched']:
            text = f'{t["pr"]:.2f}'
            colour = (255, 255, 255)
        else:
            text = 'x'
            colour = (0, 0, 255)
        (tw_, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        org = (int(cx - tw_ / 2), int(cy + th_ / 2))
        cv2.putText(out4_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)   # outline for legibility
        cv2.putText(out4_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    colour, 2, cv2.LINE_AA)

    # SF score banner
    sf_text = f'SF = {sf:.3f}   (IoU={iou_pred_target:.3f}, ' \
              f'pores matched={sum(1 for t in per_target if t["matched"])}/{n_target})'
    cv2.putText(out4_bgr, sf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out4_bgr, sf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)

    out4_final = cv2.cvtColor(out4_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out4_final).save(output_dir / f'{stem}-4-scored-overlay.png')

    # ── CSV row ────────────────────────────────────────────────────────────
    pr_values = [f'{t["pr"]:.4f}' for t in per_target]
    return {
        'stem':                 stem,
        'iou_pred_target':      f'{iou_pred_target:.4f}',
        'n_target_pores':       n_target,
        'n_matched_pores':      sum(1 for t in per_target if t['matched']),
        'n_unmatched_pred_pores': len(unmatched_pred_ids),
        'sum_pr':               f'{sum_pr:.4f}',
        'w':                    w,
        'SF':                   f'{sf:.4f}',
        'pore_scores':          ';'.join(pr_values),
    }


# ── batch driver ──────────────────────────────────────────────────────────────
def discover_stems(data_dir):
    data_dir = Path(data_dir)
    stems = sorted(
        p.stem for p in data_dir.glob('*.tif')
        if not any(x in p.name for x in ['visible', 'overlay', 'target', 'pred'])
    )
    return stems


def run(data_dir, output_dir, w, min_px, max_px, max_ar, match_overlap_frac,
        close_kernel=21):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = discover_stems(data_dir)
    print(f'Found {len(stems)} candidate images in {data_dir}\n')

    rows = []
    for stem in stems:
        print(f'Processing {stem} ...')
        row = process_image(stem, data_dir, output_dir, w, min_px, max_px,
                            max_ar, match_overlap_frac, close_kernel)
        if row is not None:
            rows.append(row)
            print(f'  SF={row["SF"]}  IoU={row["iou_pred_target"]}  '
                  f'matched={row["n_matched_pores"]}/{row["n_target_pores"]}  '
                  f'scores=[{row["pore_scores"]}]')

    if rows:
        csv_path = output_dir / 'pore_scores.csv'
        fieldnames = ['stem', 'iou_pred_target', 'n_target_pores',
                      'n_matched_pores', 'n_unmatched_pred_pores',
                      'sum_pr', 'w', 'SF', 'pore_scores']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n✓ Scores written → {csv_path}')

    print(f'✓ Outputs in       → {output_dir}')
    return rows


# ── multi-fold batch driver ───────────────────────────────────────────────────
def run_cv_folds(parent_dir, w, min_px, max_px, max_ar, match_overlap_frac,
                  close_kernel=21, fold_glob='fold_*', n_pore_cols=4):
    """
    For each fold directory matching `fold_glob` inside `parent_dir` (the
    layout written by unetplusplus_cross_validate.py — fold_0, fold_1, ...,
    each containing a predictions/ subfolder), run the full pore_analysis
    pipeline on <fold>/predictions and write outputs to <fold>/pore_analysis.

    Also collects every image's score across all folds into one master
    table at <parent_dir>/pore_scores_all_folds.csv, with the per-pore
    scores (already shown as text on each -4-scored-overlay.png) split out
    into individual pore_1 .. pore_N columns rather than one joined string,
    so they're directly usable for downstream stats/plots per pore index.

    Note: requires -target-mask.png to have been propagated into each
    fold's predictions/ dir (i.e. unetplusplus_cross_validate.py's
    copy_split() and unetplusplus_test.py's COPY_SUFFIXES must include
    '-target-mask.png' — both updated for this).
    """
    parent_dir = Path(parent_dir)
    fold_dirs = sorted(
        d for d in parent_dir.glob(fold_glob)
        if d.is_dir() and (d / 'predictions').exists()
    )
    if not fold_dirs:
        print(f'[ERROR] No fold directories matching "{fold_glob}" with a '
              f'predictions/ subfolder found in {parent_dir}')
        return

    print(f'Found {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}\n')

    all_rows = []
    for fold_dir in fold_dirs:
        data_dir   = fold_dir / 'predictions'
        output_dir = fold_dir / 'pore_analysis'
        print(f'{"="*60}')
        print(f'  {fold_dir.name}')
        print(f'{"="*60}')

        fold_rows = run(
            data_dir=data_dir, output_dir=output_dir,
            w=w, min_px=min_px, max_px=max_px, max_ar=max_ar,
            match_overlap_frac=match_overlap_frac, close_kernel=close_kernel,
        )
        for row in (fold_rows or []):
            row = dict(row)
            row['fold'] = fold_dir.name
            all_rows.append(row)
        print()

    if not all_rows:
        print('[WARNING] No rows collected across any fold — nothing to '
              'write for the master CSV.')
        return

    max_pores_seen = max(
        (len(r['pore_scores'].split(';')) if r['pore_scores'] else 0)
        for r in all_rows
    )
    if max_pores_seen > n_pore_cols:
        print(f'[WARNING] Found images with {max_pores_seen} target pores, '
              f'more than --n_pore_cols={n_pore_cols}. Extra pore scores '
              f'will be dropped from the master CSV — raise --n_pore_cols '
              f'to keep them (per-fold pore_scores.csv files still have '
              f'everything).')

    master_fieldnames = (
        ['fold', 'stem', 'iou_pred_target', 'n_target_pores',
         'n_matched_pores', 'n_unmatched_pred_pores', 'w', 'SF']
        + [f'pore_{i+1}' for i in range(n_pore_cols)]
    )

    master_path = parent_dir / 'pore_scores_all_folds.csv'
    with open(master_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=master_fieldnames,
                                delimiter=';', extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            scores = row['pore_scores'].split(';') if row['pore_scores'] else []
            out_row = dict(row)
            for i in range(n_pore_cols):
                out_row[f'pore_{i+1}'] = scores[i] if i < len(scores) else ''
            writer.writerow(out_row)

    print(f'✓ Master CSV across all {len(fold_dirs)} fold(s) '
          f'({len(all_rows)} images) → {master_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pore-position shape-fidelity scoring against target geometry.'
    )
    parser.add_argument('--data_dir',
        help='Folder with .tif/-target-mask.png/-pred-mask.png triplets '
             '(single-folder mode)')
    parser.add_argument('--output_dir',
        help='Where to save outputs (single-folder mode)')
    parser.add_argument('--cv_parent_dir',
        help='Parent folder containing fold_0, fold_1, ... subdirectories '
             '(multi-fold mode — the layout written by '
             'unetplusplus_cross_validate.py). Each fold_i/predictions is '
             'processed into fold_i/pore_analysis, plus a master '
             'pore_scores_all_folds.csv is written in this parent folder.')
    parser.add_argument('--fold_glob', default='fold_*',
        help='Glob pattern for fold directories under --cv_parent_dir '
             '(default: fold_*)')
    parser.add_argument('--n_pore_cols', type=int, default=4,
        help='Number of pore_N columns in the master CSV for multi-fold '
             'mode (default: 4, matching the 3x3 crosshatch geometry)')
    parser.add_argument('--w', type=float, default=0.25,
        help='Weight on the pore bonus term relative to IoU (default: 0.25)')
    parser.add_argument('--min_pore_px', type=int, default=10000)
    parser.add_argument('--max_pore_px', type=int, default=150000)
    parser.add_argument('--max_aspect_ratio', type=float, default=3.0)
    parser.add_argument('--match_overlap_frac', type=float, default=0.3,
        help='Min fraction of a target pore region a predicted pore must '
             'overlap to be considered a match for that position (default: 0.3)')
    parser.add_argument('--close_kernel', type=int, default=21,
        help='Morphological closing kernel (px) applied before hole '
             'detection, to seal gaps in the colour-threshold mask '
             '(anti-aliasing + specular-highlight speckle noise) that '
             'would otherwise let a pore leak out as "outside" background '
             'or merge two adjacent pores (default: 21, 0=disable)')
    args = parser.parse_args()

    if args.cv_parent_dir:
        if args.data_dir or args.output_dir:
            print('[ERROR] --cv_parent_dir is mutually exclusive with '
                  '--data_dir/--output_dir.')
            raise SystemExit(1)
        run_cv_folds(
            parent_dir=args.cv_parent_dir,
            w=args.w,
            min_px=args.min_pore_px,
            max_px=args.max_pore_px,
            max_ar=args.max_aspect_ratio,
            match_overlap_frac=args.match_overlap_frac,
            close_kernel=args.close_kernel,
            fold_glob=args.fold_glob,
            n_pore_cols=args.n_pore_cols,
        )
    else:
        if not args.data_dir or not args.output_dir:
            print('[ERROR] Single-folder mode requires both --data_dir '
                  'and --output_dir (or use --cv_parent_dir for multi-fold '
                  'mode).')
            raise SystemExit(1)
        run(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            w=args.w,
            min_px=args.min_pore_px,
            max_px=args.max_pore_px,
            max_ar=args.max_aspect_ratio,
            match_overlap_frac=args.match_overlap_frac,
            close_kernel=args.close_kernel,
        )