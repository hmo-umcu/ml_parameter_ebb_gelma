"""
evaluate_gelma_iou.py
---------------------
Compute IoU between U-Net++ predictions and manual annotations
for cell-free and cell-laden GelMA segmentation.

Directory structure expected
----------------------------
    <base_dir>/
        data_cell-laden_annot/       annotation masks from labelme + json_to_mask.py
            {stem}-mask.png          binary, values 0/1
        data_cell-laden_pred/        U-Net++ predictions
            {stem}-pred-mask.png     binary, values 0/1
        data_cell-free_annot/
            {stem}-mask.png
        data_cell-free_pred/
            {stem}-pred-mask.png

Mask format
-----------
    Both masks: uint8, 0 = background, >0 = strand
    (annotation values are 0/1 from json_to_mask.py;
     prediction values are 0/1 from unetplusplus_test.py)

Usage
-----
    python evaluate_gelma_iou.py
        --base_dir  /home/hmo/.../data_procedure/data
        [--annot_suffix  -mask.png]
        [--pred_suffix   -pred-mask.png]
        [--output_csv    gelma_iou_results.csv]

Output
------
    Prints per-image IoU, Dice, pixel accuracy.
    Prints per-group and overall summary.
    Saves CSV with all per-image metrics.
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path
from PIL import Image


# ── metrics ───────────────────────────────────────────────────────────────────
def binary_iou(pred, gt):
    p, g   = pred > 0, gt > 0
    inter  = np.logical_and(p, g).sum()
    union  = np.logical_or(p, g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def binary_dice(pred, gt):
    p, g   = pred > 0, gt > 0
    inter  = np.logical_and(p, g).sum()
    denom  = p.sum() + g.sum()
    return float(2 * inter) / float(denom) if denom > 0 else 0.0


def pixel_accuracy(pred, gt):
    p = (pred > 0).astype(np.uint8)
    g = (gt   > 0).astype(np.uint8)
    return float((p == g).sum()) / float(g.size)


def load_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


# ── evaluate one group (cell-laden or cell-free) ──────────────────────────────
def evaluate_group(annot_dir, pred_dir, annot_suffix, pred_suffix, group_name):
    """
    Match annotation and prediction files in their respective directories.
    Returns list of result dicts.
    """
    annot_dir = Path(annot_dir)
    pred_dir  = Path(pred_dir)

    if not annot_dir.exists():
        print(f'  [ERROR] annotation dir not found: {annot_dir}')
        return []
    if not pred_dir.exists():
        print(f'  [ERROR] prediction dir not found: {pred_dir}')
        return []

    # find all annotation masks
    annot_files = sorted(annot_dir.glob(f'*{annot_suffix}'))
    if not annot_files:
        print(f'  [WARN] no files matching *{annot_suffix} in {annot_dir}')
        return []

    results = []
    for annot_path in annot_files:
        # derive stem: remove the annot_suffix to get the image stem
        stem      = annot_path.name[: -len(annot_suffix)]
        pred_path = pred_dir / f'{stem}{pred_suffix}'

        if not pred_path.exists():
            print(f'  [SKIP] no prediction found for {stem} '
                  f'(expected: {pred_path.name})')
            continue

        gt   = load_mask(annot_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            print(f'  [WARN] shape mismatch for {stem}: '
                  f'annot={gt.shape} pred={pred.shape} — skipping')
            continue

        iou  = binary_iou(pred, gt)
        dice = binary_dice(pred, gt)
        acc  = pixel_accuracy(pred, gt)

        results.append({
            'group':      group_name,
            'stem':       stem,
            'annot_file': annot_path.name,
            'pred_file':  pred_path.name,
            'iou':        iou,
            'dice':       dice,
            'pixel_acc':  acc,
            'pred_px':    int((pred > 0).sum()),
            'annot_px':   int((gt   > 0).sum()),
        })

        print(f'  [{group_name}] {stem:<20}  '
              f'IoU={iou:.4f}  Dice={dice:.4f}  Acc={acc:.4f}')

    return results


# ── summary stats ─────────────────────────────────────────────────────────────
def print_summary(label, results):
    if not results:
        print(f'\n  [{label}] No results.')
        return
    ious  = [r['iou']       for r in results]
    dices = [r['dice']      for r in results]
    accs  = [r['pixel_acc'] for r in results]
    n     = len(results)
    print(f'\n  {"─"*55}')
    print(f'  {label}  (n={n} images)')
    print(f'  {"─"*55}')
    print(f'  IoU         : {np.mean(ious):.4f} ± {np.std(ious):.4f}  '
          f'[{np.min(ious):.4f} – {np.max(ious):.4f}]')
    print(f'  Dice        : {np.mean(dices):.4f} ± {np.std(dices):.4f}  '
          f'[{np.min(dices):.4f} – {np.max(dices):.4f}]')
    print(f'  Pixel acc   : {np.mean(accs):.4f} ± {np.std(accs):.4f}  '
          f'[{np.min(accs):.4f} – {np.max(accs):.4f}]')


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    base = Path(args.base_dir)

    groups = [
        ('cell-laden',
         base / 'data_cell-laden_annot',
         base / 'data_cell-laden_pred'),
        ('cell-free',
         base / 'data_cell-free_annot',
         base / 'data_cell-free_pred'),
    ]

    all_results = []

    for group_name, annot_dir, pred_dir in groups:
        print(f'\n{"="*60}')
        print(f'  Group: {group_name}')
        print(f'  Annot : {annot_dir}')
        print(f'  Pred  : {pred_dir}')
        print(f'{"="*60}')

        results = evaluate_group(
            annot_dir, pred_dir,
            args.annot_suffix, args.pred_suffix,
            group_name
        )
        all_results.extend(results)

        print_summary(f'{group_name.upper()} summary', results)

    # overall summary
    print_summary('OVERALL (cell-laden + cell-free)', all_results)

    if not all_results:
        print('\n[ERROR] No results computed. Check directory paths and file names.')
        sys.exit(1)

    # save CSV
    csv_path = Path(args.output_csv) if args.output_csv \
               else base / 'gelma_iou_results.csv'
    fieldnames = ['group', 'stem', 'annot_file', 'pred_file',
                  'iou', 'dice', 'pixel_acc', 'pred_px', 'annot_px']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for r in all_results:
            row = {k: f'{v:.6f}' if isinstance(v, float) else v
                   for k, v in r.items()}
            writer.writerow(row)

    print(f'\n✓ Per-image results saved → {csv_path}')
    print(f'  Total images evaluated: {len(all_results)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute IoU between U-Net++ predictions and annotations '
                    'for cell-free and cell-laden GelMA segmentation.'
    )
    parser.add_argument('--base_dir', required=True,
        help='Parent directory containing data_cell-laden_annot/, '
             'data_cell-laden_pred/, data_cell-free_annot/, data_cell-free_pred/')
    parser.add_argument('--annot_suffix', default='-mask.png',
        help='Suffix of annotation mask files (default: -mask.png)')
    parser.add_argument('--pred_suffix', default='-pred-mask.png',
        help='Suffix of prediction mask files (default: -pred-mask.png)')
    parser.add_argument('--output_csv', default=None,
        help='Path for output CSV (default: <base_dir>/gelma_iou_results.csv)')
    args = parser.parse_args()

    main(args)