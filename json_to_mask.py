import json
import numpy as np
import base64
import io
from PIL import Image, ImageDraw
from pathlib import Path

# ── label map ── adjust here if label names ever change
STRAND_LABELS = {'strands', 'strand'}
PORE_LABELS   = {'pores', 'pore'}


def decode_polygon_mask(shape, h, w):
    """Rasterize a polygon shape into a binary HxW mask."""
    pts = [tuple(p) for p in shape['points']]
    canvas = Image.new('L', (w, h), 0)
    ImageDraw.Draw(canvas).polygon(pts, fill=1)
    return np.array(canvas, dtype=np.uint8)


def decode_labelme_mask(shape, h, w):
    """Decode a SAM2 base64-PNG mask shape into a binary HxW mask."""
    x1, y1 = shape['points'][0]   # top-left of bounding box
    x2, y2 = shape['points'][1]   # bottom-right of bounding box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    png_bytes  = base64.b64decode(shape['mask'])
    crop_img   = Image.open(io.BytesIO(png_bytes)).convert('L')
    crop_arr   = (np.array(crop_img) > 0).astype(np.uint8)

    full = np.zeros((h, w), dtype=np.uint8)
    ch, cw = crop_arr.shape

    # clamp to image bounds (safety)
    r1, r2 = y1, min(y1 + ch, h)
    c1, c2 = x1, min(x1 + cw, w)
    full[r1:r2, c1:c2] = crop_arr[:r2-r1, :c2-c1]
    return full


def json_to_mask(json_path, img_h=1024, img_w=1280):
    """
    Convert a labelme JSON (polygon or mask shapes) to a semantic HxW mask.

    Pixel values
    ------------
    0 : background
    1 : strand_clean  (strands with pore pixels removed)
    2 : pore          (kept for reference/validation)

    Works correctly when:
    - strands annotated as polygon OR SAM2 mask
    - pores annotated as polygon (or absent → no-op subtraction)
    - label names are 'strands'/'strand' and 'pores'/'pore'
    """
    with open(json_path) as f:
        data = json.load(f)

    h = data.get('imageHeight', img_h)
    w = data.get('imageWidth',  img_w)

    strand_mask = np.zeros((h, w), dtype=np.uint8)
    pore_mask   = np.zeros((h, w), dtype=np.uint8)

    for shape in data['shapes']:
        label      = shape['label']
        shape_type = shape['shape_type']

        if shape_type == 'polygon':
            region = decode_polygon_mask(shape, h, w)
        elif shape_type == 'mask':
            region = decode_labelme_mask(shape, h, w)
        else:
            print(f'  [SKIP] unknown shape_type: {shape_type}')
            continue

        if label in STRAND_LABELS:
            strand_mask[region > 0] = 1
        elif label in PORE_LABELS:
            pore_mask[region > 0] = 1
        else:
            print(f'  [SKIP] unknown label: {label}')

    # subtract pores from strands → clean strand
    strand_clean = strand_mask.copy()
    strand_clean[pore_mask > 0] = 0

    final_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask[strand_clean > 0] = 1   # strand_clean
    final_mask[pore_mask    > 0] = 2   # pore (reference)

    return final_mask


STRAND_COLOUR = np.array([255, 60, 60], dtype=np.float32)   # red


def process_folder(input_dir, output_dir=None, alpha=0.5):
    """
    Find all .tif/.tiff files in input_dir that have a matching .json file,
    run json_to_mask() on each pair, and save:
      <stem>-mask.png         — binary mask (0=background, 1=strand_clean)
      <stem>-mask-visible.png — original image with strand overlay at alpha

    Parameters
    ----------
    input_dir  : str or Path
    output_dir : str or Path  (defaults to input_dir)
    alpha      : float 0–1    overlay opacity (default 0.5)
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {'.tif', '.tiff'}
    )

    pairs = []
    for tif in tif_files:
        json_path = tif.with_suffix('.json')
        if json_path.exists():
            pairs.append((tif, json_path))
        else:
            print(f'[SKIP] no matching JSON for {tif.name}')

    if not pairs:
        print('No .tif/.json pairs found.')
        return

    print(f'Found {len(pairs)} pair(s) in {input_dir}  |  alpha={alpha}\n')

    for tif_path, json_path in pairs:
        stem         = tif_path.stem
        mask_path    = output_dir / f'{stem}-mask.png'
        visible_path = output_dir / f'{stem}-mask-visible.png'

        try:
            full_mask   = json_to_mask(json_path)
            binary_mask = (full_mask == 1).astype(np.uint8)

            # ── binary mask (0/1 pixel values, for model training)
            Image.fromarray(binary_mask).save(mask_path)

            # ── visible overlay with alpha blending
            orig    = np.array(Image.open(tif_path).convert('RGB'), dtype=np.float32)
            overlay = orig.copy()
            strand_pixels = binary_mask == 1
            overlay[strand_pixels] = (
                (1 - alpha) * orig[strand_pixels] + alpha * STRAND_COLOUR
            )
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(visible_path)

            strand_px = int(binary_mask.sum())
            print(f'[OK] {tif_path.name}'
                  f'\n     mask    → {mask_path.name}  (strand={strand_px}px)'
                  f'\n     visible → {visible_path.name}')

        except Exception as e:
            print(f'[ERROR] {tif_path.name}: {e}')

    print(f'\nDone. Masks saved to: {output_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert labelme JSON+TIF pairs to semantic mask PNGs.'
    )
    parser.add_argument(
        'input_dir',
        help='Folder containing .tif and .json annotation pairs'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default=None,
        help='Where to save mask PNGs (default: same as input_dir)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Overlay opacity for visible output, 0.0–1.0 (default: 0.5)'
    )
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.alpha)