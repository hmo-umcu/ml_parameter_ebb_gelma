"""
STEP 2 — Solid Geometry Region Extraction + Visualization
===========================================================
Reads Frangi masks from Step 1 (.npz), extracts the solid filled
geometry region (the printed 2×2 grid as one solid shape), and
saves clean visualizations.

No metrics computed here — this is purely segmentation output
ready for: (a) human annotation comparison, (b) deep learning training.

Pipeline:
    Frangi mask (Step 1)
         ↓
    binary_closing(close_r)   — seals gaps within and between strands
         ↓
    binary_fill_holes()       — fills pores → solid geometry block
         ↓
    keep largest component    — removes noise outside geometry
         ↓
    (optional) remove pores   — subtract enclosed holes to show
                                strand region only vs solid block
         ↓
    Visualization + saved masks (PNG + NPZ)

Usage:
    # Preview on first image only:
    python step2_geometry_mask.py \
        --mask_dir  ./step1_out \
        --raw_dir   ./images \
        --output_dir ./step2_out \
        --close_r 20 \
        --preview_only

    # Full batch:
    python step2_geometry_mask.py \
        --mask_dir  ./step1_out \
        --raw_dir   ./images \
        --output_dir ./step2_out \
        --close_r 20

Parameters to tune:
  --close_r         Closing disk radius in px (default: 20)
                    MOST IMPORTANT.
                    Too small → gaps in geometry, pores not sealed
                    Too large → corners rounded, geometry over-expanded
                    Recommended: start at 16, increase by 4 until solid
                    Try: 12, 16, 20, 24, 28, 32

  --min_geo_area    Min area px² to keep as geometry component (default: 10000)
                    Increase if noise blobs are kept; decrease if geometry missed

  --fill_pores      If set, fills the 4 pores → outputs one solid filled block
                    If not set, outputs strand region (pores visible as holes)
                    Default: fill_pores = True (solid block)

  --n_pores         Number of pores to subtract if fill_pores=False (default: 4)

  --min_pore_area   Min pore area px² (default: 3000)  used only if fill_pores=False

  --contour_thick   Contour line thickness px for visualization (default: 4)

  --overlay_alpha   Transparency of mask overlay on raw image 0.0-1.0 (default: 0.45)

  --mask_color      RGB color of geometry mask overlay (default: 80 220 80 = green)

  --contour_color   RGB color of contour line (default: 255 60 60 = red)
"""

import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, measure
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
import warnings
warnings.filterwarnings('ignore')

# ── Arguments ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--mask_dir',       required=True,
               help='Directory with _mask.npz files from step1')
p.add_argument('--raw_dir',        required=True,
               help='Directory with original raw images (.tif)')
p.add_argument('--output_dir',     required=True)

# Core geometry parameters
p.add_argument('--close_r',        type=int,   default=20,
               help='Closing disk radius px — TUNE THIS FIRST (default: 20)')
p.add_argument('--min_geo_area',   type=int,   default=5000,
               help='Min geometry component area px² (default: 5000)')

# Pore handling
p.add_argument('--fill_pores',     action='store_true', default=True,
               help='Output solid filled block (pores filled). Default: True')
p.add_argument('--no_fill_pores',  action='store_true',
               help='Output strand region with visible pores')
p.add_argument('--n_pores',        type=int,   default=4,
               help='Expected pore count (default: 4)')
p.add_argument('--min_pore_area',  type=int,   default=2000,
               help='Min pore area px² (default: 2000)')
p.add_argument('--max_pore_frac',  type=float, default=0.15,
               help='Max pore area as fraction of image (default: 0.15)')

# Visualization
p.add_argument('--contour_thick',  type=int,   default=4,
               help='Contour line thickness px (default: 4)')
p.add_argument('--overlay_alpha',  type=float, default=0.45,
               help='Mask overlay transparency 0-1 (default: 0.45)')
p.add_argument('--mask_color',     nargs=3, type=int, default=[80, 220, 80],
               metavar=('R','G','B'),
               help='Mask overlay color RGB (default: 80 220 80 = green)')
p.add_argument('--contour_color',  nargs=3, type=int, default=[255, 60, 60],
               metavar=('R','G','B'),
               help='Contour color RGB (default: 255 60 60 = red)')

# Run control
p.add_argument('--preview_only',   action='store_true',
               help='Process first image only and exit')
p.add_argument('--save_overlay',   action='store_true', default=True,
               help='Save overlay PNG (default: True)')
p.add_argument('--save_mask_png',  action='store_true', default=True,
               help='Save binary mask PNG (default: True)')
p.add_argument('--dpi',            type=int,   default=120)
args = p.parse_args()

# Handle fill_pores logic
fill_pores = not args.no_fill_pores

os.makedirs(args.output_dir, exist_ok=True)

print("=" * 55)
print(f"close_r         = {args.close_r}  ← tune first")
print(f"fill_pores      = {fill_pores}")
print(f"min_geo_area    = {args.min_geo_area} px²")
print(f"min_pore_area   = {args.min_pore_area} px²  (if fill_pores=False)")
print(f"contour_thick   = {args.contour_thick} px")
print(f"overlay_alpha   = {args.overlay_alpha}")
print(f"mask_color      = {args.mask_color}")
print(f"contour_color   = {args.contour_color}")
print("=" * 55)

# ── Core extraction ───────────────────────────────────────────────────────────
def extract_geometry(frangi_mask, args, fill_pores):
    """
    Returns:
        geo_solid   : solid geometry mask (filled block or strand region)
        geo_filled  : always fully filled (used for contour)
        pore_mask   : detected pores (empty if fill_pores=True)
        n_pores     : number of pores detected
        status      : info string
    """
    # Step 1: Close to seal intra-strand gaps
    closed = morphology.binary_closing(frangi_mask, disk(args.close_r))

    # Step 2: Fill holes → solid filled geometry block
    geo_filled = binary_fill_holes(closed)

    # Step 3: Keep largest component only
    labeled = measure.label(geo_filled)
    props   = sorted(measure.regionprops(labeled),
                     key=lambda p: p.area, reverse=True)

    # Filter by minimum area
    valid = [p for p in props if p.area >= args.min_geo_area]
    if not valid:
        return None, None, None, 0, "No component above min_geo_area"

    geo_filled = labeled == valid[0].label

    if fill_pores:
        # Output = solid filled block (pores filled in)
        geo_solid  = geo_filled.copy()
        pore_mask  = np.zeros_like(geo_filled)
        n_pores    = 0
        status     = f"solid block  area={geo_filled.sum()}"
    else:
        # Find enclosed pores = filled − closed lines
        closed_geo = morphology.binary_closing(frangi_mask, disk(args.close_r))
        # Restrict to the main geometry region
        closed_geo = closed_geo & geo_filled
        pores_raw  = geo_filled & ~closed_geo
        lp         = measure.label(pores_raw)
        pp         = measure.regionprops(lp)
        img_area   = frangi_mask.size
        valid_p    = sorted(
            [p for p in pp
             if args.min_pore_area <= p.area <= args.max_pore_frac * img_area],
            key=lambda p: p.area, reverse=True
        )[:args.n_pores]
        pore_mask = np.zeros_like(geo_filled)
        for p in valid_p:
            pore_mask[lp == p.label] = True
        # Strand mask = filled − pores
        geo_solid = geo_filled & ~pore_mask
        geo_solid = morphology.remove_small_objects(geo_solid, min_size=500)
        n_pores   = len(valid_p)
        status    = f"strand region  pores={n_pores}  area={geo_solid.sum()}"

    return geo_solid, geo_filled, pore_mask, n_pores, status

# ── Overlay helpers ───────────────────────────────────────────────────────────
def make_overlay(raw_img, geo_solid, pore_mask, contour, args):
    """Blend mask color onto raw image with given alpha."""
    ov  = raw_img.copy().astype(float)
    mc  = np.array(args.mask_color, float)
    # Blend strand region
    for ch in range(3):
        ov[geo_solid, ch] = (
            ov[geo_solid, ch] * (1 - args.overlay_alpha)
            + mc[ch] * args.overlay_alpha
        )
    # Pores in blue (if visible)
    if pore_mask.any():
        ov[pore_mask, 0] = ov[pore_mask,0]*0.3 + 60*0.7
        ov[pore_mask, 1] = ov[pore_mask,1]*0.3 + 80*0.7
        ov[pore_mask, 2] = ov[pore_mask,2]*0.3 + 200*0.7
    # Contour on top (full opacity)
    cc = np.array(args.contour_color, float)
    for ch in range(3):
        ov[contour, ch] = cc[ch]
    return np.clip(ov, 0, 255).astype(np.uint8)

def make_mask_png(geo_solid, mask_color):
    """Binary mask as colored PNG (mask_color on black)."""
    out = np.zeros((*geo_solid.shape, 3), np.uint8)
    out[geo_solid] = mask_color
    return out

# ── Process files ─────────────────────────────────────────────────────────────
mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*_mask.npz')))
if not mask_files:
    print(f"No *_mask.npz found in {args.mask_dir}"); exit(1)

if args.preview_only:
    mask_files = mask_files[:1]
    print("PREVIEW MODE — first file only\n")

print(f"Found {len(mask_files)} mask files.\n")

for mpath in mask_files:
    mname = os.path.basename(mpath)
    base  = mname.replace('_mask.npz', '')

    # Load Frangi mask
    try:
        data        = np.load(mpath)
        frangi_mask = data['mask'].astype(bool)
    except Exception as e:
        print(f"  SKIP {mname}: {e}"); continue

    # Load raw image (apply crop if saved)
    raw_path = os.path.join(args.raw_dir, f'{base}.tif')
    try:
        raw_full = np.array(Image.open(raw_path))
        if 'crop_box' in data:
            x0,y0,x1,y1 = data['crop_box']
            raw_img = raw_full[y0:y1, x0:x1]
        else:
            raw_img = raw_full
    except:
        raw_img = np.zeros((*frangi_mask.shape, 3), np.uint8)
        print(f"  WARNING: raw not found for {base}")

    # Extract geometry
    geo_solid, geo_filled, pore_mask, n_pores, status = \
        extract_geometry(frangi_mask, args, fill_pores)

    if geo_solid is None:
        print(f"  FAIL {base}: {status}"); continue

    # Outer contour of geo_filled (thickened for visibility)
    contour_line  = find_boundaries(geo_filled, mode='outer')
    contour_thick = morphology.binary_dilation(
        contour_line, disk(args.contour_thick))

    print(f"  OK: {base}  {status}")

    # ── Build overlay and mask images ─────────────────────────────────────────
    overlay_img  = make_overlay(raw_img, geo_solid, pore_mask,
                                contour_thick, args)
    mask_img     = make_mask_png(geo_solid, args.mask_color)

    # ── Save individual PNGs ──────────────────────────────────────────────────
    if args.save_overlay:
        Image.fromarray(overlay_img).save(
            os.path.join(args.output_dir, f'{base}_overlay.png'))

    if args.save_mask_png:
        Image.fromarray(mask_img).save(
            os.path.join(args.output_dir, f'{base}_geo_mask.png'))

    # ── Save NPZ for downstream use ───────────────────────────────────────────
    np.savez_compressed(
        os.path.join(args.output_dir, f'{base}_geometry.npz'),
        geo_solid=geo_solid,
        geo_filled=geo_filled,
        pore_mask=pore_mask,
        contour=contour_thick,
        close_r=np.array(args.close_r))

    # ── Summary figure: 4 panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    fig.patch.set_facecolor('#0a0a14')

    # Panel 1: Raw
    axes[0].imshow(raw_img)
    axes[0].set_title(f'① Raw image\n{base}',
                      color='white', fontsize=11, fontweight='bold')

    # Panel 2: Frangi mask input
    axes[1].imshow(frangi_mask, cmap='gray')
    axes[1].set_title(f'② Frangi mask (Step 1 input)',
                      color='#aaffaa', fontsize=11)

    # Panel 3: Geometry mask alone
    axes[2].imshow(mask_img)
    geo_area_px = geo_solid.sum()
    axes[2].set_title(
        f'③ Geometry mask\n'
        f'{"Solid filled" if fill_pores else "Strand region"}  '
        f'area={geo_area_px:,} px\n'
        f'close_r={args.close_r}',
        color='#aaffaa', fontsize=11)

    # Panel 4: Overlay on raw
    axes[3].imshow(overlay_img)
    axes[3].set_title(
        f'④ Overlay + contour\n'
        f'alpha={args.overlay_alpha}  '
        f'contour_thick={args.contour_thick}',
        color='white', fontsize=11)

    for ax in axes:
        ax.axis('off')

    plt.suptitle(
        f'{base}  |  close_r={args.close_r}  '
        f'{"fill_pores=True" if fill_pores else f"pores_detected={n_pores}"}',
        color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, f'{base}_step2.png'),
        dpi=args.dpi, bbox_inches='tight', facecolor='#0a0a14')
    plt.close(fig)

print(f"\nDone. Outputs in: {args.output_dir}")
print(f"  *_geo_mask.png  — binary mask (colored)")
print(f"  *_overlay.png   — mask overlaid on raw image")
print(f"  *_geometry.npz  — arrays: geo_solid, geo_filled, pore_mask, contour")
print(f"  *_step2.png     — 4-panel summary figure")
