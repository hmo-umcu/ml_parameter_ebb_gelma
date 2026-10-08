"""
unetplusplus_cross_validate.py
-------------------------------
Run k-fold cross-validation for U-Net++ scaffold segmentation.

Splitting strategy
------------------
Splits at SAMPLE level (by Sample_ID), not at image level.
All 6 replicate images of a sample stay together in train or test.
This prevents data leakage between replicates of the same parameter set.

With 32 samples and k=4 folds:
    fold 0 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 1 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 2 : test=8 samples (48 images),  train=24 samples (144 images)
    fold 3 : test=8 samples (48 images),  train=24 samples (144 images)

Input
-----
Single flat folder containing all annotated images:
    {sid}_{row}.tif
    {sid}_{row}-mask.png
    {sid}_{row}-mask-visible.png     (optional, copied)
    {sid}_{row}.json                 (optional, copied)
    {sid}_{row}-target-overlay.png   (required for iou_pred_target metric)

Usage
-----
    python unetplusplus_cross_validate.py
        --data_dir       /home/hmo/.../data/dev_images/dev_annot_all
        --output_dir     /home/hmo/.../data/dev_images/cv_unetpp
        [--k              4]
        [--architecture   unetplusplus]
        [--encoder        resnet34]
        [--n_epochs       100]
        [--batch_size     4]
        [--learning_rate  1e-4]
        [--weight_decay   1e-4]
        [--val_frac       0.0]     set 0 for CV (test fold IS the val set)
        [--patience       0]       set 0 to disable early stopping in CV
        [--img_size       512]
        [--seed           42]
        [--no_gpu]

Output
------
    <output_dir>/
        samples_discovered.json
        cv_splits.json
        fold_0/
            train/                   copies of train images+masks
            test/                    copies of test images+masks
            model/
                best_model.pth       best weights (highest val IoU)
                final_model.pth      final epoch weights
                training_log.txt     per-epoch train loss + val IoU
                training_curves.png  loss and IoU curves for this fold
            predictions/
                {stem}-pred-mask.png
                {stem}-pred-visible.png
                {stem}-pred-vs-annot.png
                test_info.csv        with all metrics filled
        fold_1/ ...
        fold_2/ ...
        fold_3/ ...
        cv_summary.csv              per-fold metrics
        cv_training_curves.png      all fold train loss curves overlaid
        cv_val_iou_curves.png       all fold val IoU curves overlaid
"""

import argparse
import csv
import json
import re
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ── discover sample IDs ───────────────────────────────────────────────────────
def discover_samples(data_dir):
    data_dir = Path(data_dir)
    samples  = defaultdict(list)
    for tif in data_dir.glob('*.tif'):
        if any(x in tif.name for x in ['visible', 'overlay', 'target']):
            continue
        m = re.match(r'^(\d+)_(\d+)$', tif.stem)
        if m:
            samples[int(m.group(1))].append(int(m.group(2)))
    return dict(samples)


# ── k-fold split at sample level ─────────────────────────────────────────────
def kfold_sample_split(sample_ids, k=4, seed=42):
    rng   = np.random.default_rng(seed)
    sids  = [int(s) for s in rng.permutation(sorted(sample_ids))]
    folds = [sids[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        test_ids  = folds[i]
        train_ids = [s for j, fold in enumerate(folds)
                     if j != i for s in fold]
        splits.append((sorted(train_ids), sorted(test_ids)))
    return splits


# ── copy files ────────────────────────────────────────────────────────────────
SUFFIXES = ['.tif', '.tiff', '-mask.png', '-mask-visible.png',
            '.json', '-target-overlay.png', '-target-mask.png']


def copy_split(data_dir, dest_dir, sample_ids, rows_map):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    for sid in sample_ids:
        for row in rows_map.get(sid, []):
            stem = f'{sid}_{row}'
            for suf in SUFFIXES:
                src = data_dir / f'{stem}{suf}'
                if src.exists():
                    shutil.copy2(src, dest_dir / src.name)


# ── call unetplusplus scripts as functions ────────────────────────────────────
def run_train(train_dir, model_dir, arch, encoder, n_epochs, batch_size,
              learning_rate, weight_decay, val_frac, patience, img_size,
              use_gpu):
    sys.argv = ['unetplusplus_train.py']
    from unetplusplus_train import train as _train
    _train(
        data_dir=str(train_dir),
        model_dir=str(model_dir),
        arch=arch,
        encoder=encoder,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_frac=val_frac,
        patience=patience,
        img_size=img_size,
        use_gpu=use_gpu,
    )


def run_test(model_path, test_dir, pred_dir, img_size, threshold, use_gpu):
    from unetplusplus_test import run_test as _test
    _test(
        model_path=str(model_path),
        data_dir=str(test_dir),
        output_dir=str(pred_dir),
        img_size=img_size,
        threshold=threshold,
        use_gpu=use_gpu,
    )


def run_evaluate(pred_dir):
    from unetplusplus_evaluate import evaluate as _evaluate
    _evaluate(pred_dir=str(pred_dir))


# ── read fold metrics from test_info.csv ─────────────────────────────────────
def read_fold_metrics(csv_path):
    """
    Read per-image metrics from test_info.csv.
    Returns dict with both per-fold summary AND all individual image values.
    """
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))

    def collect(key):
        vals = []
        for r in rows:
            v = r.get(key, '').strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    iou_pa = collect('iou_pred_annot')
    iou_pt = collect('iou_pred_target')
    dice_  = collect('dice_pred_annot')
    acc_   = collect('pixel_acc')

    return {
        'n_test_images':        len(rows),
        # per-fold summary (mean of images in this fold)
        'mean_iou_pred_annot':  float(np.mean(iou_pa))  if iou_pa  else '',
        'std_iou_pred_annot':   float(np.std(iou_pa))   if iou_pa  else '',
        'mean_iou_pred_target': float(np.mean(iou_pt))  if iou_pt  else '',
        'std_iou_pred_target':  float(np.std(iou_pt))   if iou_pt  else '',
        'mean_dice_pred_annot': float(np.mean(dice_))   if dice_   else '',
        'mean_pixel_acc':       float(np.mean(acc_))    if acc_    else '',
        # all per-image values — used for pooled CV aggregation
        '_all_iou_pa': iou_pa,
        '_all_iou_pt': iou_pt,
        '_all_dice':   dice_,
        '_all_acc':    acc_,
    }


# ── read training log for curve plotting ─────────────────────────────────────
def read_training_log(log_path):
    """
    Parse training_log.txt from unetplusplus_train.py.
    Returns (train_losses, val_ious) as lists.
    """
    train_losses, val_ious = [], []
    in_table = False

    for line in Path(log_path).read_text().splitlines():
        if line.strip().startswith('epoch') and 'train_loss' in line:
            in_table = True
            continue
        if line.startswith('─'):
            if in_table and train_losses:
                in_table = False
            continue
        if not in_table:
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            in_table = False
            continue
        try:
            train_losses.append(float(parts[1]))
        except (IndexError, ValueError):
            continue
        try:
            vi = parts[2] if len(parts) > 2 else 'n/a'
            vi = vi.replace('←', '').replace('best', '').strip()
            val_ious.append(float(vi) if vi not in ('n/a', '') else None)
        except (IndexError, ValueError):
            val_ious.append(None)

    return train_losses, val_ious


# ── cross-fold curve plots ────────────────────────────────────────────────────
def plot_cv_curves(output_dir, k):
    """
    Two plots:
        cv_training_curves.png  — train loss per fold + mean ± std
        cv_val_iou_curves.png   — val IoU per fold + mean ± std
    """
    COLOURS = plt.cm.tab10.colors
    fold_train_losses = {}
    fold_val_ious     = {}

    for fold_idx in range(k):
        log_path = output_dir / f'fold_{fold_idx}' / 'model' / 'training_log.txt'
        if not log_path.exists():
            print(f'  [WARNING] no training_log.txt for fold {fold_idx}')
            continue
        tl, vi = read_training_log(log_path)
        if tl:
            fold_train_losses[fold_idx] = tl
        real_vi = [v for v in vi if v is not None]
        if len(real_vi) == len(tl):
            fold_val_ious[fold_idx] = real_vi

    def _save_panel_plot(fold_data, ylabel, title, out_path, ymin=None, ymax=None):
        if not fold_data:
            return
        min_len = min(len(v) for v in fold_data.values())
        matrix  = np.array([v[:min_len] for v in fold_data.values()])
        epochs  = np.arange(1, min_len + 1)
        mean_   = matrix.mean(axis=0)
        std_    = matrix.std(axis=0)

        fig, ax = plt.subplots(figsize=(9, 5))
        for fold_idx, vals in fold_data.items():
            ax.plot(epochs, vals[:min_len],
                    color=COLOURS[fold_idx % len(COLOURS)],
                    linewidth=1.2, alpha=0.75,
                    label=f'fold {fold_idx}')
        ax.fill_between(epochs, mean_-std_, mean_+std_,
                        color='grey', alpha=0.18, label='mean ± std')
        ax.plot(epochs, mean_, color='black', linewidth=2.0,
                linestyle='--', label='mean')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=12)
        if ymin is not None:
            ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'✓ {out_path.name} → {out_path}')

    _save_panel_plot(
        fold_train_losses,
        ylabel='Train loss (Dice+BCE)',
        title=f'CV train loss — {k} folds',
        out_path=output_dir / 'cv_training_curves.png',
    )
    _save_panel_plot(
        fold_val_ious,
        ylabel='Val IoU',
        title=f'CV val IoU — {k} folds',
        out_path=output_dir / 'cv_val_iou_curves.png',
        ymin=0.0, ymax=1.0,
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover samples ──────────────────────────────────────────────────────
    print(f'\n[0] Scanning {data_dir} ...')
    rows_map = discover_samples(data_dir)
    all_sids = sorted(rows_map.keys())
    print(f'    Found {len(all_sids)} unique Sample_IDs: {all_sids}')
    print(f'    Total images: {sum(len(v) for v in rows_map.values())}')

    (output_dir / 'samples_discovered.json').write_text(
        json.dumps({str(sid): rows_map[sid] for sid in all_sids}, indent=2)
    )

    # ── k-fold splits ─────────────────────────────────────────────────────────
    splits = kfold_sample_split(all_sids, k=args.k, seed=args.seed)

    splits_log = {}
    for i, (train_ids, test_ids) in enumerate(splits):
        splits_log[f'fold_{i}'] = {
            'train_sample_ids': [int(s) for s in train_ids],
            'test_sample_ids':  [int(s) for s in test_ids],
            'n_train_samples':  len(train_ids),
            'n_test_samples':   len(test_ids),
        }
    (output_dir / 'cv_splits.json').write_text(
        json.dumps(splits_log, indent=2)
    )
    print(f'\n    {args.k}-fold splits → {output_dir}/cv_splits.json\n')

    # ── run each fold ─────────────────────────────────────────────────────────
    fold_summaries = []

    for fold_idx, (train_ids, test_ids) in enumerate(splits):
        fold_dir  = output_dir / f'fold_{fold_idx}'
        train_dir = fold_dir / 'train'
        test_dir  = fold_dir / 'test'
        model_dir = fold_dir / 'model'
        pred_dir  = fold_dir / 'predictions'

        n_train_imgs = sum(len(rows_map[s]) for s in train_ids)
        n_test_imgs  = sum(len(rows_map[s]) for s in test_ids)

        print(f'\n{"="*60}')
        print(f'  FOLD {fold_idx}  '
              f'train={len(train_ids)} samples ({n_train_imgs} imgs)  '
              f'test={len(test_ids)} samples ({n_test_imgs} imgs)')
        print(f'  train: {train_ids}')
        print(f'  test : {test_ids}')
        print(f'{"="*60}')

        # ── copy files ────────────────────────────────────────────────────────
        print(f'\n  [1/3] Copying train/test files...')
        copy_split(data_dir, train_dir, train_ids, rows_map)
        copy_split(data_dir, test_dir,  test_ids,  rows_map)
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        # ── train ─────────────────────────────────────────────────────────────
        print(f'\n  [2/3] Training fold {fold_idx}...')
        run_train(
            train_dir=train_dir,
            model_dir=model_dir,
            arch=args.architecture,
            encoder=args.encoder,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_frac=args.val_frac,
            patience=args.patience,
            img_size=args.img_size,
            use_gpu=not args.no_gpu,
        )

        # use best_model if it exists, else final_model
        model_path = model_dir / 'best_model.pth'
        if not model_path.exists():
            model_path = model_dir / 'final_model.pth'
        print(f'  Using model: {model_path.name}')

        # ── test ──────────────────────────────────────────────────────────────
        print(f'\n  [3a/3] Predicting test set...')
        run_test(
            model_path=model_path,
            test_dir=test_dir,
            pred_dir=pred_dir,
            img_size=args.img_size,
            threshold=args.threshold,
            use_gpu=not args.no_gpu,
        )

        # ── evaluate ──────────────────────────────────────────────────────────
        print(f'\n  [3b/3] Evaluating...')
        run_evaluate(pred_dir=pred_dir)

        # ── collect metrics ───────────────────────────────────────────────────
        csv_path = pred_dir / 'test_info.csv'
        metrics  = read_fold_metrics(csv_path)
        metrics['fold']             = fold_idx
        metrics['train_sample_ids'] = str(train_ids)
        metrics['test_sample_ids']  = str(test_ids)
        fold_summaries.append(metrics)

        print(f'\n  Fold {fold_idx} complete.')
        print(f'    IoU (pred|annot) : {metrics["mean_iou_pred_annot"]}')
        print(f'    IoU (pred|target): {metrics["mean_iou_pred_target"]}')

    # ── cv_summary.csv — per-fold numbers ────────────────────────────────────
    summary_path = output_dir / 'cv_summary.csv'
    fieldnames   = [
        'fold', 'n_test_images',
        'mean_iou_pred_annot',  'std_iou_pred_annot',
        'mean_iou_pred_target', 'std_iou_pred_target',
        'mean_dice_pred_annot', 'mean_pixel_acc',
        'train_sample_ids', 'test_sample_ids',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                delimiter=';', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(fold_summaries)

    # ── pooled aggregation across ALL folds ───────────────────────────────────
    # Collect every per-image value from every test fold.
    # Since each sample appears in exactly one test fold, this gives
    # one IoU value per image across all 32 samples — the correct
    # estimate of generalisation performance.
    all_iou_pa = [v for m in fold_summaries for v in m.get('_all_iou_pa', [])]
    all_iou_pt = [v for m in fold_summaries for v in m.get('_all_iou_pt', [])]
    all_dice   = [v for m in fold_summaries for v in m.get('_all_dice',   [])]
    all_acc    = [v for m in fold_summaries for v in m.get('_all_acc',    [])]

    def stats(vals):
        if not vals:
            return 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'
        a = np.array(vals)
        return (f'{np.mean(a):.4f}', f'{np.std(a):.4f}',
                f'{np.median(a):.4f}',
                f'{np.percentile(a,25):.4f}', f'{np.percentile(a,75):.4f}')

    iou_pa_mean, iou_pa_std, iou_pa_med, iou_pa_q1, iou_pa_q3 = stats(all_iou_pa)
    iou_pt_mean, iou_pt_std, iou_pt_med, iou_pt_q1, iou_pt_q3 = stats(all_iou_pt)
    dice_mean,   dice_std,   *_                                  = stats(all_dice)
    acc_mean,    acc_std,    *_                                  = stats(all_acc)

    final_lines = [
        f'Cross-validation final results',
        f'==============================',
        f'architecture : {args.architecture} ({args.encoder})',
        f'k            : {args.k} folds',
        f'n_epochs     : {args.n_epochs}',
        f'learning_rate: {args.learning_rate}',
        f'weight_decay : {args.weight_decay}',
        f'img_size     : {args.img_size}',
        f'total images evaluated: {len(all_iou_pa)} '
        f'(all samples seen exactly once)',
        f'',
        f'Pooled across all {args.k} test folds (image-level):',
        f'',
        f'  IoU (pred vs annot):',
        f'    mean ± std   : {iou_pa_mean} ± {iou_pa_std}',
        f'    median [IQR] : {iou_pa_med} [{iou_pa_q1} – {iou_pa_q3}]',
        f'',
        f'  IoU (pred vs target):',
        f'    mean ± std   : {iou_pt_mean} ± {iou_pt_std}',
        f'    median [IQR] : {iou_pt_med} [{iou_pt_q1} – {iou_pt_q3}]',
        f'',
        f'  Dice (pred vs annot) : {dice_mean} ± {dice_std}',
        f'  Pixel accuracy       : {acc_mean}  ± {acc_std}',
        f'',
        f'Per-fold breakdown:',
    ]
    for m in fold_summaries:
        final_lines.append(
            f'  fold {m["fold"]}: '
            f'IoU(annot)={m["mean_iou_pred_annot"]:.4f}  '
            f'IoU(target)={m["mean_iou_pred_target"]:.4f}  '
            f'n={m["n_test_images"]}'
        )

    final_path = output_dir / 'cv_final_results.txt'
    final_path.write_text('\n'.join(final_lines))

    # ── plot training curves ──────────────────────────────────────────────────
    print(f'\nPlotting cross-fold training curves...')
    plot_cv_curves(output_dir, args.k)

    print(f'\n{"="*60}')
    print(f'  CROSS-VALIDATION COMPLETE  ({args.k} folds)')
    print(f'  Pooled across all test folds ({len(all_iou_pa)} images):')
    print(f'  IoU (pred|annot)  : {iou_pa_mean} ± {iou_pa_std}  '
          f'median={iou_pa_med}')
    print(f'  IoU (pred|target) : {iou_pt_mean} ± {iou_pt_std}  '
          f'median={iou_pt_med}')
    print(f'  Dice (pred|annot) : {dice_mean} ± {dice_std}')
    print(f'  Pixel accuracy    : {acc_mean} ± {acc_std}')
    print(f'  Summary CSV  → {summary_path}')
    print(f'  Final results→ {final_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='k-fold cross-validation for U-Net++ scaffold segmentation.'
    )
    parser.add_argument('--data_dir',      required=True,
        help='Flat folder with all annotated *.tif + *-mask.png files')
    parser.add_argument('--output_dir',    required=True,
        help='Where to save all fold outputs')
    parser.add_argument('--k',             type=int,   default=4)
    parser.add_argument('--architecture',  default='unetplusplus',
        choices=['unetplusplus', 'unet', 'fpn'])
    parser.add_argument('--encoder',       default='resnet34')
    parser.add_argument('--n_epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--val_frac',      type=float, default=0.0,
        help='Internal val fraction during training. '
             'Set 0.0 for CV (test fold serves as validation). (default: 0.0)')
    parser.add_argument('--patience',      type=int,   default=0,
        help='Early stopping patience. 0=disabled for CV. (default: 0)')
    parser.add_argument('--img_size',      type=int,   default=512)
    parser.add_argument('--threshold',     type=float, default=0.5)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--no_gpu',        action='store_true')
    args = parser.parse_args()

    main(args)