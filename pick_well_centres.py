"""
pick_well_centres.py
--------------------
Set the well centres by hand, once, and write a well_centres.csv that
draw_target_geometry*.py consumes verbatim via --centres_csv.
 
Why this exists
---------------
The automatic rim detector assumes the well rim is the strongest circular
edge structure in the image. On a bright, highly reflective well bottom with
large specular highlights that assumption fails: the Canny thresholds are
derived from the global median, the median is high, and the soft rim edges are
thresholded away before they can be scored. The detector then latches onto
whatever arc the highlights provide, which is why a failing folder shows
angular-coverage scores near 0.2 and rim radii that disagree by 50-70%.
 
No amount of fallback logic recovers a centre that was never found. For a
validation plate of 6 or 12 wells, clicking each centre once takes a couple of
minutes, is exactly correct, and stays correct forever because the CSV is
reused.
 
What you see
------------
Per image:
  * the photo
  * the printed-strand mask outline in RED (from -pred-mask.png, so you can
    see what was actually printed)
  * the target G-code crosshatch in GREEN, drawn at the current centre
  * the live IoU in the title
 
You move the green crosshatch until it sits where the printer was TOLD to
print, then accept. Note that this is a judgement about the well, not about
the print: if the print is genuinely offset from the well centre, the green
should NOT be moved onto it. That offset is real and the IoU should show it.
 
Controls
--------
  left click        place the centre here
  arrow keys        nudge 1 px          (shift + arrow = 10 px)
  [  ]              px_per_mm -1 / +1   ({ } = -5 / +5)
                    Use this to CALIBRATE the camera scale: the G-code strand
                    pitch is a known physical distance, so matching the green
                    pitch to the printed lattice measures px_per_mm. That is a
                    calibration, not a fit of the metric.
  enter / space     accept this well and go to the next
  s                 skip this well (no row written)
  r                 reset to the initial guess
  b                 go back one well
  q                 save what has been accepted so far and quit
 
Usage
-----
    python pick_well_centres.py <img_dir> [--pred_masks] \
        [--output_csv <path>]        default: <img_dir>/well_centres_manual.csv
                                     deliberately NOT well_centres.csv, which
                                     draw_target_geometry overwrites on every run
        [--strand_width_mm <f>]      must MATCH what you pass to
        [--strand_gap_mm <f>]        draw_target_geometry, or the preview lies
        [--px_per_mm <f>]            default 67.0, same caveat
        [--init csv|image|print]     where the first guess comes from
        [--centres_csv <path>]       existing centres to start from
        [--only 0_1,0_3]             work on these stems only
 
Then:
    python draw_target_geometry_gelma.py <img_dir> --pred_masks \
        --centres_csv well_centres_manual.csv
 
A bare filename there is resolved inside img_dir, so one literal argument works
for every folder in a loop.
 
Rows written carry status=ok, so draw_target_geometry uses them directly.
"""
 
import argparse
import csv
import sys
from pathlib import Path
 
import numpy as np
from PIL import Image, ImageDraw
 
import matplotlib
import matplotlib.pyplot as plt
 
 
DEFAULT_PX_PER_MM       = 67.0
DEFAULT_STRAND_WIDTH_MM = 0.41
DEFAULT_STRAND_GAP_MM   = 2.5
N_STRANDS               = 3
 
FIELDNAMES = ['stem', 'cx', 'cy', 'r', 'score', 'source', 'iou',
              'status', 'reason']
 
 
# =============================================================================
# io  (kept byte-identical in behaviour to draw_target_geometry.py)
# =============================================================================
def load_rgb(path):
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
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            arr = arr[..., :3].max(axis=2) if alpha.min() == alpha.max() else alpha
        else:
            arr = arr.max(axis=2)
    return (arr > 0).astype(np.uint8)
 
 
def make_target_mask(img_h, img_w, cx, cy, strand_width_mm, strand_gap_mm,
                     px_per_mm, n_strands=N_STRANDS):
    """Identical geometry to draw_target_geometry.make_target_mask."""
    half_w_px      = (strand_width_mm / 2) * px_per_mm
    half_span      = (n_strands - 1) / 2.0 * strand_gap_mm
    offsets_mm     = [-half_span + i * strand_gap_mm for i in range(n_strands)]
    half_extent_px = (half_span + strand_width_mm / 2) * px_per_mm
 
    canvas = Image.new('L', (img_w, img_h), 0)
    draw   = ImageDraw.Draw(canvas)
    for y_off_mm in offsets_mm:
        y_px = cy + y_off_mm * px_per_mm
        draw.rectangle([cx - half_extent_px, y_px - half_w_px,
                        cx + half_extent_px, y_px + half_w_px], fill=1)
    for x_off_mm in offsets_mm:
        x_px = cx + x_off_mm * px_per_mm
        draw.rectangle([x_px - half_w_px, cy - half_extent_px,
                        x_px + half_w_px, cy + half_extent_px], fill=1)
    return np.array(canvas, dtype=np.uint8)
 
 
def compute_iou(pred_mask, target_mask):
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0
 
 
def find_image(img_dir, stem):
    for ext in ('.tif', '.tiff', '.TIF', '.TIFF', '.png', '.jpg', '.jpeg'):
        p = img_dir / f'{stem}{ext}'
        if p.exists():
            return p
    return None
 
 
def enumerate_stems(img_dir, mask_suffix):
    """Every image stem in the folder, excluding anything this pipeline made."""
    skip = ('mask', 'overlay', 'visible', 'pred', 'target', 'welldet')
    seen, out = set(), []
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.jpg', '*.jpeg'):
        for p in sorted(img_dir.glob(ext)):
            st = p.stem
            if any(t in st.lower() for t in skip):
                continue
            if st not in seen:
                seen.add(st)
                out.append(st)
 
    def key(s):
        parts = s.split('_')
        try:
            return (0, tuple(int(x) for x in parts))
        except ValueError:
            return (1, s)
    return sorted(out, key=key)
 
 
def read_centres_csv(path):
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                out[str(row['stem']).strip()] = (float(row['cx']), float(row['cy']),
                                                 float(row.get('r') or 'nan'))
            except (KeyError, TypeError, ValueError):
                continue
    return out
 
 
# =============================================================================
# the picker
# =============================================================================
class Picker:
    def __init__(self, img_dir, stems, mask_suffix, output_csv,
                 strand_width_mm, strand_gap_mm, px_per_mm,
                 init_mode, prior):
        self.img_dir  = img_dir
        self.stems    = stems
        self.suffix   = mask_suffix
        self.out_csv  = output_csv
        self.sw       = strand_width_mm
        self.sg       = strand_gap_mm
        self.pxmm     = px_per_mm
        self.pxmm0    = px_per_mm
        self.init     = init_mode
        self.prior    = prior
 
        self.i        = 0
        self.accepted = {}          # stem -> (cx, cy, r)
        self.quit     = False
 
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._load(self.i)
 
    # ------------------------------------------------------------------ state
    def _initial_centre(self, stem, h, w):
        if self.init == 'csv' and stem in self.prior:
            cx, cy, _r = self.prior[stem]
            return float(cx), float(cy), 'prior CSV'
        if self.init == 'print' and self.pred is not None and self.pred.any():
            ys, xs = np.nonzero(self.pred)
            return float(xs.mean()), float(ys.mean()), 'print centroid'
        if stem in self.prior:
            cx, cy, _r = self.prior[stem]
            return float(cx), float(cy), 'prior CSV'
        return w / 2.0, h / 2.0, 'image centre'
 
    def _load(self, i):
        stem = self.stems[i]
        ip = find_image(self.img_dir, stem)
        if ip is None:
            print(f'[SKIP] no image file for {stem}')
            self.img = None
            return
        self.stem = stem
        self.img  = load_rgb(ip)
        h, w = self.img.shape[:2]
 
        mp = self.img_dir / f'{stem}{self.suffix}'
        self.pred = load_binary_mask(mp) if mp.exists() else None
        if self.pred is not None and self.pred.shape != (h, w):
            print(f'[WARN] {stem}: mask {self.pred.shape} != image {(h, w)}, ignored')
            self.pred = None
 
        if stem in self.accepted:
            self.cx, self.cy, _ = self.accepted[stem]
            self.src = 'accepted'
        else:
            self.cx, self.cy, self.src = self._initial_centre(stem, h, w)
        self.draw()
 
    # ----------------------------------------------------------------- render
    def draw(self):
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        tgt = make_target_mask(h, w, self.cx, self.cy, self.sw, self.sg, self.pxmm)
        iou = compute_iou(self.pred, tgt) if self.pred is not None else float('nan')
 
        vis = self.img.astype(np.float32).copy()
        if self.pred is not None:
            vis[self.pred > 0] = 0.55 * vis[self.pred > 0] + \
                0.45 * np.array([255, 60, 60], np.float32)
        vis[tgt > 0] = 0.35 * vis[tgt > 0] + \
            0.65 * np.array([40, 235, 40], np.float32)
 
        self.ax.clear()
        self.ax.imshow(np.clip(vis, 0, 255).astype(np.uint8))
        self.ax.plot([self.cx], [self.cy], marker='+', ms=22, mew=2, color='magenta')
        self.ax.set_xticks([]); self.ax.set_yticks([])
 
        iou_str = f'{iou:.3f}' if np.isfinite(iou) else 'n/a (no mask)'
        done = len(self.accepted)
        pitch_px = self.sg * self.pxmm
        self.ax.set_title(
            f'[{self.i + 1}/{len(self.stems)}]  {self.stem}   '
            f'centre=({self.cx:.0f}, {self.cy:.0f}) [{self.src}]   IoU={iou_str}\n'
            f'px_per_mm={self.pxmm:.1f}   strand={self.sw * self.pxmm:.0f}px   '
            f'pitch={pitch_px:.0f}px\n'
            f'click = place   arrows = nudge (shift x10)   [ ] = scale +/-1 '
            f'({{ }} = +/-5)\n'
            f'enter = accept   s = skip   b = back   r = reset   q = save+quit'
            f'      accepted: {done}',
            fontsize=9)
        self.fig.canvas.draw_idle()
 
    # ----------------------------------------------------------------- events
    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        self.cx, self.cy = float(ev.xdata), float(ev.ydata)
        self.src = 'clicked'
        self.draw()
 
    def on_key(self, ev):
        k = ev.key or ''
        step = 10.0 if k.startswith('shift+') else 1.0
        base = k.replace('shift+', '')
 
        if base in ('left', 'right', 'up', 'down'):
            if base == 'left':
                self.cx -= step
            elif base == 'right':
                self.cx += step
            elif base == 'up':
                self.cy -= step
            else:
                self.cy += step
            self.src = 'clicked'
            self.draw()
            return
 
        # Scale calibration. The G-code strand pitch is a KNOWN physical
        # distance, so matching the target's pitch to the printed lattice is a
        # calibration of the camera, not a fit of the metric. Adjust until the
        # green pitch lines up with the printed pore lattice, then use the
        # resulting number as --px_per_mm everywhere.
        if k in ('[', ']', '{', '}'):
            d = {'[': -1.0, ']': +1.0, '{': -5.0, '}': +5.0}[k]
            self.pxmm = max(1.0, self.pxmm + d)
            self.draw()
            return
 
        if k in ('enter', ' '):
            self.accepted[self.stem] = (self.cx, self.cy, float('nan'))
            print(f'  accepted {self.stem}: ({self.cx:.2f}, {self.cy:.2f})')
            self._advance(+1)
        elif k == 's':
            print(f'  skipped {self.stem}')
            self._advance(+1)
        elif k == 'b':
            self._advance(-1)
        elif k == 'r':
            h, w = self.img.shape[:2]
            self.cx, self.cy, self.src = self._initial_centre(self.stem, h, w)
            self.draw()
        elif k == 'q':
            self.quit = True
            plt.close(self.fig)
 
    def _advance(self, d):
        j = self.i + d
        if j < 0:
            j = 0
        if j >= len(self.stems):
            print('\nLast well reached.')
            plt.close(self.fig)
            return
        self.i = j
        self._load(self.i)
 
    # ------------------------------------------------------------------ write
    def save(self):
        if not self.accepted:
            print('\nNothing accepted, no file written.')
            return None
        with open(self.out_csv, 'w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=FIELDNAMES)
            wtr.writeheader()
            for stem in self.stems:
                if stem not in self.accepted:
                    continue
                cx, cy, r = self.accepted[stem]
                wtr.writerow({'stem': stem, 'cx': f'{cx:.2f}', 'cy': f'{cy:.2f}',
                              'r': '' if not np.isfinite(r) else f'{r:.2f}',
                              'score': '1.000', 'source': 'manual', 'iou': '',
                              'status': 'ok', 'reason': 'set by hand'})
        print(f'\nWrote {len(self.accepted)} centre(s) -> {self.out_csv}')
        print(f'Final scale used in the preview: px_per_mm = {self.pxmm:.1f}')
        if abs(self.pxmm - self.pxmm0) > 0.5:
            print(f'  You changed it from {self.pxmm0:.1f}. If the green pitch '
                  f'now matches the printed\n  lattice, {self.pxmm:.1f} is your '
                  f'calibrated camera scale. Pass it as --px_per_mm to\n'
                  f'  draw_target_geometry and use the TRUE design values for '
                  f'--strand_width_mm and\n  --strand_gap_mm, rather than '
                  f'inflating those.')
        return self.out_csv
 
 
def main(args):
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        sys.exit(f'Not a directory: {img_dir}')
 
    suffix = '-pred-mask.png' if args.pred_masks else args.mask_suffix
    stems  = enumerate_stems(img_dir, suffix)
    if args.only:
        want = [s.strip() for s in args.only.split(',') if s.strip()]
        missing = [s for s in want if s not in stems]
        if missing:
            sys.exit(f'--only names stems that are not in {img_dir}: {missing}')
        stems = want
    if not stems:
        sys.exit(f'No images found in {img_dir}')
 
    out_csv = Path(args.output_csv) if args.output_csv else img_dir / 'well_centres_manual.csv'
    prior   = read_centres_csv(args.centres_csv) if args.centres_csv else \
        read_centres_csv(out_csv)
 
    n_masks = sum(1 for s in stems if (img_dir / f'{s}{suffix}').exists())
    print(f'{len(stems)} well(s) in {img_dir}')
    print(f'{n_masks} have {suffix}; without one the IoU readout is disabled')
    print(f'target preview: strand_width={args.strand_width_mm}mm  '
          f'strand_gap={args.strand_gap_mm}mm  px_per_mm={args.px_per_mm}')
    print('these three MUST match what you pass to draw_target_geometry, '
          'or the preview\nyou are aligning against is not the mask that will '
          'be scored.\n')
    if prior:
        print(f'{len(prior)} prior centre(s) loaded as the starting guess\n')
 
    p = Picker(img_dir, stems, suffix, out_csv,
               args.strand_width_mm, args.strand_gap_mm, args.px_per_mm,
               args.init, prior)
    plt.show()
    saved = p.save()
    if saved:
        print('\nNext:')
        print(f'  python draw_target_geometry_gelma.py "{img_dir}" '
              f'{"--pred_masks " if args.pred_masks else ""}'
              f'--centres_csv "{saved.name}"')
        print(f'  (a bare filename is resolved inside the image folder, so the '
              f'same\n   argument works for every folder in a for /D loop)')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Set well centres by hand and write well_centres.csv.')
    ap.add_argument('img_dir')
    ap.add_argument('--output_csv', default=None)
    ap.add_argument('--centres_csv', default=None,
                    help='Existing centres to start from (default: the '
                         'well_centres.csv already in img_dir, if any)')
    ap.add_argument('--pred_masks', action='store_true',
                    help='Show -pred-mask.png (U-Net++ output) as the printed shape')
    ap.add_argument('--mask_suffix', default='-mask.png')
    ap.add_argument('--strand_width_mm', type=float, default=DEFAULT_STRAND_WIDTH_MM)
    ap.add_argument('--strand_gap_mm', type=float, default=DEFAULT_STRAND_GAP_MM)
    ap.add_argument('--px_per_mm', type=float, default=DEFAULT_PX_PER_MM)
    ap.add_argument('--init', choices=['csv', 'image', 'print'], default='csv',
                    help="Initial guess. 'csv' uses a prior centre if there is "
                         "one, else the image centre. 'print' starts at the "
                         "printed-mask centroid, which is faster but biases you "
                         "towards the print rather than the well, so only use it "
                         "if you are going to judge each one anyway.")
    ap.add_argument('--only', default=None,
                    help='Comma-separated stems to work on, e.g. 0_1,0_3')
    main(ap.parse_args())