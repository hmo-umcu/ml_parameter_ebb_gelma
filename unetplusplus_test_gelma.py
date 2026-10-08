"""
unetplusplus_test.py
--------------------
Load a trained U-Net++ model and run inference on a test folder, or
recursively over a whole tree of folders.
 
Two modes
---------
1. SINGLE FOLDER (original behaviour, unchanged)
       --data_dir <folder> --output_dir <folder>
 
2. RECURSIVE (deployment)
       --data_dir <parent> --recursive
   Walks the parent, finds every folder that contains .tif images, and runs
   inference on each one. By default predictions are written NEXT TO THE
   INPUT IMAGES (in place). Pass --mirror_output <root> to write them into a
   mirrored tree instead, leaving the raw data untouched.
 
   The model is loaded ONCE and reused across all folders, not reloaded per
   folder.
 
Annotations are optional
------------------------
Deployment images have no labelme JSON and no *-mask.png. That is fine:
the annotation-dependent outputs are simply skipped. Only
{stem}-pred-mask.png and {stem}-pred-visible.png are produced. Nothing here
requires an annotation.
 
Usage
-----
    python unetplusplus_test.py
        --model_path  .../models/unetplusplus/run_01/best_model.pth
        --data_dir    .../data/gelma_deployment
        --recursive
        [--mirror_output <root>]   write to a mirrored tree instead of in place
        [--output_dir <dir>]       single-folder mode only
        [--img_size   512]
        [--threshold  0.5]
        [--skip_existing]          resume: skip images already predicted
        [--dry_run]                list the folders that would be processed
        [--no_gpu]
 
Output (per folder)
-------------------
    {stem}-pred-mask.png        predicted binary mask (0/1)
    {stem}-pred-visible.png     original + prediction overlay (green)
    {stem}-pred-vs-annot.png    only when an annotation mask exists
    test_info.csv               per-image metadata table (;-separated)
and, in recursive mode, one combined test_info_all.csv at the top.
 
NOTE on mask values: -pred-mask.png is written with values 0 and 1, not
0 and 255, so it looks black in an image viewer. That is the existing
convention and downstream code reads it as (arr > 0). Left unchanged on
purpose so nothing already written against it breaks.
"""
 
import argparse
import csv
import shutil
import sys
import numpy as np
from pathlib import Path
from PIL import Image
 
 
# -- colour constants ---------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
 
PRED_COLOUR  = np.array([60,  220, 60], dtype=np.float32)   # green
ANNOT_COLOUR = np.array([220, 60,  60], dtype=np.float32)   # red
ALPHA = 0.45
 
IMAGE_EXTS   = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
EXCLUDE_TAGS = ('visible', 'overlay', 'target', 'pred', 'mask')
 
 
def preprocess_image(img_rgb, img_size):
    """Resize + ImageNet normalise -> [3,H,W] float32 numpy."""
    import cv2
    img   = cv2.resize(img_rgb, (img_size, img_size),
                       interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    return img_f.transpose(2, 0, 1)   # CHW
 
 
def overlay_masks(img_rgb, pred_mask=None, annot_mask=None):
    """green = prediction only, red = annotation only, yellow = overlap."""
    out = img_rgb.astype(np.float32).copy()
    if pred_mask is not None:
        pb = pred_mask > 0
        if annot_mask is not None:
            ab         = annot_mask > 0
            overlap    = pb & ab
            pred_only  = pb & ~ab
            annot_only = ab & ~pb
            out[pred_only]  = (1 - ALPHA) * out[pred_only]  + ALPHA * PRED_COLOUR
            out[annot_only] = (1 - ALPHA) * out[annot_only] + ALPHA * ANNOT_COLOUR
            out[overlap]    = (1 - ALPHA) * out[overlap] + \
                ALPHA * np.array([255, 220, 0], dtype=np.float32)
        else:
            out[pb] = (1 - ALPHA) * out[pb] + ALPHA * PRED_COLOUR
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
# =============================================================================
# discovery
# =============================================================================
def list_images(folder):
    """Original .tif images in a folder, excluding anything we generated."""
    out = []
    for pat in IMAGE_EXTS:
        for p in folder.glob(pat):
            if any(tag in p.stem.lower() for tag in EXCLUDE_TAGS):
                continue
            out.append(p)
    return sorted(set(out))
 
 
def find_image_folders(root, mirror_root=None):
    """Every folder at or under `root` that holds original .tif images."""
    root = Path(root)
    folders = []
    if list_images(root):
        folders.append(root)
    for d in sorted(p for p in root.rglob('*') if p.is_dir()):
        if mirror_root is not None:
            try:                      # never walk into our own output tree
                d.relative_to(Path(mirror_root))
                continue
            except ValueError:
                pass
        if list_images(d):
            folders.append(d)
    return folders
 
 
# =============================================================================
# model loading (once, reused for every folder)
# =============================================================================
def load_model(model_path, device):
    import torch
    import segmentation_models_pytorch as smp
 
    print(f'Loading model: {model_path}')
    checkpoint = torch.load(str(model_path), map_location=device)
    arch    = checkpoint.get('arch',    'unetplusplus')
    encoder = checkpoint.get('encoder', 'resnet34')
 
    arch_map = {
        'unetplusplus': smp.UnetPlusPlus,
        'unet':         smp.Unet,
        'fpn':          smp.FPN,
    }
    model = arch_map[arch](
        encoder_name=encoder,
        encoder_weights=None,     # weights come from the checkpoint
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f'  Architecture : {arch} ({encoder})')
    print(f'  Trained epoch: {checkpoint.get("epoch", "?")}')
    if 'val_iou' in checkpoint:
        print(f'  Best val IoU : {checkpoint["val_iou"]:.4f}')
    return model
 
 
# =============================================================================
# inference on one folder
# =============================================================================
def run_folder(model, device, data_dir, output_dir, img_size, threshold,
               skip_existing=False, rel_label=''):
    import torch
    import cv2
 
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_place   = output_dir.resolve() == data_dir.resolve()
 
    tif_files = list_images(data_dir)
    if not tif_files:
        return [], 0, 0
 
    # copy companions only when writing somewhere else; copying a file onto
    # itself raises SameFileError, which is exactly what in-place output does
    if not in_place:
        COPY_SUFFIXES = ['-mask.png', '-mask-visible.png',
                         '-target-overlay.png', '-target-mask.png', '.json']
        for tif_path in tif_files:
            shutil.copy2(tif_path, output_dir / tif_path.name)
            for suf in COPY_SUFFIXES:
                src = data_dir / f'{tif_path.stem}{suf}'
                if src.exists():
                    shutil.copy2(src, output_dir / src.name)
 
    rows, done, skipped = [], 0, 0
    with torch.no_grad():
        for tif_path in tif_files:
            stem      = tif_path.stem
            pred_path = output_dir / f'{stem}-pred-mask.png'
            if skip_existing and pred_path.exists():
                skipped += 1
                continue
 
            img_rgb = np.array(Image.open(tif_path).convert('RGB'))
            h_orig, w_orig = img_rgb.shape[:2]
 
            img_t = preprocess_image(img_rgb, img_size)
            img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)
 
            logits = model(img_t)                       # [1,1,H,W]
            prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
 
            prob_full   = cv2.resize(prob, (w_orig, h_orig),
                                     interpolation=cv2.INTER_LINEAR)
            pred_binary = (prob_full > threshold).astype(np.uint8)
 
            Image.fromarray(pred_binary).save(pred_path)
            Image.fromarray(overlay_masks(img_rgb, pred_mask=pred_binary)).save(
                output_dir / f'{stem}-pred-visible.png')
 
            # annotation is optional; in deployment there is none
            annot_path = data_dir / f'{stem}-mask.png'
            annot_binary = None
            if annot_path.exists():
                annot_arr = np.array(Image.open(annot_path))
                if annot_arr.ndim == 3:
                    annot_arr = annot_arr[:, :, 0]
                annot_binary = (annot_arr > 0).astype(np.uint8)
                Image.fromarray(
                    overlay_masks(img_rgb, pred_binary, annot_binary)
                ).save(output_dir / f'{stem}-pred-vs-annot.png')
 
            parts = stem.split('_')
            rows.append({
                'folder':          rel_label,
                'stem':            stem,
                'Sample_ID':       parts[0] if len(parts) >= 2 else stem,
                'row':             parts[1] if len(parts) >= 2 else '',
                'tif':             tif_path.name,
                'pred_mask':       f'{stem}-pred-mask.png',
                'pred_visible':    f'{stem}-pred-visible.png',
                'pred_vs_annot':   f'{stem}-pred-vs-annot.png'
                                   if annot_binary is not None else '',
                'annot_mask':      f'{stem}-mask.png'
                                   if annot_binary is not None else '',
                'pred_px':         int(pred_binary.sum()),
                'annot_px':        int(annot_binary.sum())
                                   if annot_binary is not None else '',
                'iou_pred_annot':  '',
                'iou_pred_target': '',
                'dice_pred_annot': '',
                'pixel_acc':       '',
            })
            done += 1
 
    if rows:
        write_csv(output_dir / 'test_info.csv', rows)
    return rows, done, skipped
 
 
FIELDNAMES = [
    'folder', 'stem', 'Sample_ID', 'row', 'tif',
    'pred_mask', 'pred_visible', 'pred_vs_annot', 'annot_mask',
    'pred_px', 'annot_px',
    'iou_pred_annot', 'iou_pred_target', 'dice_pred_annot', 'pixel_acc',
]
 
 
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
        w.writeheader()
        w.writerows(rows)
 
 
# =============================================================================
# driver
# =============================================================================
def main(args):
    data_root = Path(args.data_dir)
    if not data_root.exists():
        sys.exit(f'data_dir does not exist: {data_root}')
 
    if not Path(args.model_path).exists():
        sys.exit(f'Checkpoint not found: {args.model_path}')
 
    if args.recursive:
        folders = find_image_folders(data_root, args.mirror_output)
        if args.output_dir:
            print('[WARN] --output_dir is ignored with --recursive. '
                  'Use --mirror_output to write into a separate tree, or omit '
                  'both to write next to the inputs.')
    else:
        folders = [data_root] if list_images(data_root) else []
    if not folders:
        sys.exit(f'No .tif images found under {data_root}')
 
    plan = []
    for folder in folders:
        rel = folder.relative_to(data_root) if folder != data_root else Path('.')
        if args.recursive:
            out = Path(args.mirror_output) / rel if args.mirror_output else folder
        else:
            out = Path(args.output_dir) if args.output_dir else folder
        plan.append((folder, out, rel, len(list_images(folder))))
 
    total_imgs = sum(p[3] for p in plan)
    print(f'Folders : {len(plan)}')
    print(f'Images  : {total_imgs}')
    print(f'Output  : {"in place, next to the inputs" if not args.mirror_output and args.recursive else "separate folder"}\n')
    for folder, out, rel, n in plan:
        print(f'  {str(rel):<62} {n:>5} img')
 
    if args.dry_run:
        print('\nDry run: nothing was written.')
        return
 
    import torch
    device = torch.device('cuda' if not args.no_gpu and torch.cuda.is_available()
                          else 'cpu')
    print(f'\nDevice: {device}\n')
    model = load_model(args.model_path, device)
    print()
 
    all_rows, n_done, n_skip, n_fail = [], 0, 0, 0
    for i, (folder, out, rel, n) in enumerate(plan, 1):
        print(f'[{i}/{len(plan)}] {rel}  ({n} images)')
        try:
            rows, done, skipped = run_folder(
                model, device, folder, out, args.img_size, args.threshold,
                skip_existing=args.skip_existing, rel_label=str(rel))
        except Exception as exc:
            print(f'    [FAIL] {exc}')
            n_fail += 1
            continue
        all_rows.extend(rows)
        n_done += done
        n_skip += skipped
        print(f'    predicted {done}' + (f', skipped {skipped} already done'
                                         if skipped else ''))
 
    if args.recursive and all_rows:
        root_out = Path(args.mirror_output) if args.mirror_output else data_root
        root_out.mkdir(parents=True, exist_ok=True)
        write_csv(root_out / 'test_info_all.csv', all_rows)
        print(f'\nCombined table -> {root_out / "test_info_all.csv"}')
 
    print(f'\nDone. {n_done} predicted, {n_skip} skipped, '
          f'{n_fail} folder(s) failed.')
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run U-Net++ inference on a folder, or recursively on a tree.')
    parser.add_argument('--model_path', required=True,
                        help='Checkpoint to use for every folder '
                             '(best_model.pth / final_model.pth)')
    parser.add_argument('--data_dir', required=True,
                        help='Folder with *.tif, or a parent tree with --recursive')
    parser.add_argument('--output_dir', default=None,
                        help='Single-folder mode: where to save predictions '
                             '(default: next to the inputs)')
    parser.add_argument('--recursive', action='store_true',
                        help='Treat --data_dir as a parent and process every '
                             'subfolder containing .tif images')
    parser.add_argument('--mirror_output', default=None,
                        help='Recursive mode: write predictions into a mirrored '
                             'tree under this root instead of next to the inputs')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Inference resize (must match training, default: 512)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Sigmoid threshold for binary prediction (default: 0.5)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip images that already have a -pred-mask.png')
    parser.add_argument('--dry_run', action='store_true',
                        help='List folders and image counts, write nothing')
    parser.add_argument('--no_gpu', action='store_true')
    main(parser.parse_args())