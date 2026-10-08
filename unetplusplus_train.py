"""
unetplusplus_train.py
---------------------
Train U-Net++ (segmentation-models-pytorch) on scaffold images for
binary strand segmentation.

Key differences from Cellpose-SAM:
  - Binary semantic segmentation: pixel = strand (1) or background (0)
  - No instance mask conversion, no min_size filtering — masks used directly
  - ImageNet-pretrained ResNet34 encoder — strong initialisation for 2D RGB
  - Per-epoch val IoU monitored natively (no Cellpose API limitations)
  - Early stopping on val IoU with best model saved automatically
  - Dice + BCE combined loss — robust to class imbalance

Input folder must contain pairs of:
    {sid}_{row}.tif          original scaffold image (RGB)
    {sid}_{row}-mask.png     strand_clean mask from json_to_mask.py
                             pixel values: 0=background, >0=strand_clean
                             pores already subtracted inside json_to_mask.py

Usage
-----
    python unetplusplus_train.py
        --data_dir    /home/hmo/.../data/dev_images/dev_annot_train
        --model_dir   /home/hmo/.../models/unetplusplus/run_01
        [--architecture  unetplusplus]   unetplusplus | unet | fpn
        [--encoder       resnet34]
        [--n_epochs      100]
        [--batch_size    4]
        [--learning_rate 1e-4]
        [--weight_decay  1e-4]
        [--val_frac      0.15]
        [--patience      20]
        [--img_size      512]
        [--no_gpu]

Output
------
    <model_dir>/
        best_model.pth           best weights (highest val IoU)
        final_model.pth          weights at last epoch
        training_log.txt         per-epoch train loss + val IoU
        training_curves.png      loss and IoU curves
"""

import argparse
import csv
import random
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


# ── ImageNet normalisation constants ─────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── mask and image loading ────────────────────────────────────────────────────
def load_mask(mask_path):
    """Load *-mask.png as binary float32 array (0.0 or 1.0)."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.float32)


def load_image(tif_path):
    """Load TIF as uint8 RGB numpy array."""
    return np.array(Image.open(tif_path).convert('RGB'))


def collect_pairs(data_dir):
    """
    Scan data_dir for (tif, mask) pairs.
    Returns (images, masks, stems) as lists of numpy arrays.
    No min_size filtering — U-Net uses masks directly as binary targets.
    """
    data_dir = Path(data_dir)
    images, masks, stems = [], [], []

    mask_files = sorted(
        p for p in data_dir.glob('*-mask.png')
        if 'visible' not in p.name
        and 'target'  not in p.name
        and 'pred'    not in p.name
    )

    for mask_path in mask_files:
        stem = mask_path.stem.replace('-mask', '')
        tif_path = next(
            (data_dir / f'{stem}{ext}'
             for ext in ('.tif', '.tiff', '.TIF', '.TIFF')
             if (data_dir / f'{stem}{ext}').exists()),
            None
        )
        if tif_path is None:
            print(f'  [SKIP] no TIF for {stem}')
            continue

        images.append(load_image(tif_path))
        masks.append(load_mask(mask_path))
        stems.append(stem)
        strand_px = int((masks[-1] > 0).sum())
        print(f'  [OK] {stem}  strand_px={strand_px}')

    return images, masks, stems


# ── augmentation + preprocessing ─────────────────────────────────────────────
def preprocess(img_rgb, mask, img_size, augment=False):
    """
    Resize, optionally augment, normalise image and mask.

    Returns (img_tensor [3,H,W] float32, mask_tensor [1,H,W] float32).
    """
    import cv2

    h, w = img_rgb.shape[:2]

    # resize
    img  = cv2.resize(img_rgb, (img_size, img_size),
                      interpolation=cv2.INTER_LINEAR)
    msk  = cv2.resize(mask,    (img_size, img_size),
                      interpolation=cv2.INTER_NEAREST)

    if augment:
        # horizontal flip
        if random.random() > 0.5:
            img = img[:, ::-1, :].copy()
            msk = msk[:, ::-1   ].copy()
        # vertical flip
        if random.random() > 0.5:
            img = img[::-1, :, :].copy()
            msk = msk[::-1, :   ].copy()
        # 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()
        # brightness / contrast jitter
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # ImageNet normalise
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    img_t = img_f.transpose(2, 0, 1)
    msk_t = msk[np.newaxis, :, :]        # [1, H, W]

    return img_t.astype(np.float32), msk_t.astype(np.float32)


# ── metrics ───────────────────────────────────────────────────────────────────
def binary_iou_np(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or (pred_bin, gt_bin).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def binary_dice_np(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter) / float(denom) if denom > 0 else 0.0


# ── save training outputs ─────────────────────────────────────────────────────
def save_training_outputs(model_dir, train_losses, val_ious,
                          best_epoch, best_val_iou,
                          arch, encoder, n_epochs, batch_size,
                          learning_rate, weight_decay, img_size,
                          data_dir, train_stems, val_stems, patience):
    """Save training_log.txt and training_curves.png."""
    model_dir = Path(model_dir)
    has_val   = bool(val_ious)

    # ── text log ──────────────────────────────────────────────────────────────
    lines = [
        f'architecture:   {arch}',
        f'encoder:        {encoder}',
        f'data_dir:       {data_dir}',
        f'n_train_images: {len(train_stems)}',
        f'n_val_images:   {len(val_stems)}',
        f'n_epochs_run:   {len(train_losses)} / {n_epochs}',
        f'batch_size:     {batch_size}',
        f'learning_rate:  {learning_rate}',
        f'weight_decay:   {weight_decay}',
        f'img_size:       {img_size}',
        f'patience:       {patience}',
        f'train_stems:    {train_stems}',
        f'val_stems:      {val_stems}',
        f'',
        f'{"epoch":<8}{"train_loss":<16}{"val_IoU":<16}',
        f'{"─"*40}',
    ]

    for i, tl in enumerate(train_losses):
        vi     = f'{val_ious[i]:.6f}' if has_val and i < len(val_ious) else 'n/a'
        marker = ' ← best' if has_val and (i + 1) == best_epoch else ''
        lines.append(f'{i+1:<8}{tl:.6f}        {vi}{marker}')

    if train_losses:
        lines += [
            f'{"─"*40}',
            f'final train loss : {train_losses[-1]:.6f}',
            f'min   train loss : {min(train_losses):.6f}'
            f'  (epoch {np.argmin(train_losses)+1})',
        ]
    if has_val and best_epoch > 0:
        lines += [
            f'best  val IoU    : {best_val_iou:.6f}  (epoch {best_epoch})',
            f'final val IoU    : {val_ious[-1]:.6f}',
        ]

    log_path = model_dir / 'training_log.txt'
    log_path.write_text('\n'.join(lines))
    print(f'✓ Training log    → {log_path}')

    # ── curves PNG ────────────────────────────────────────────────────────────
    if not train_losses:
        return

    epochs   = list(range(1, len(train_losses) + 1))
    n_panels = 2 if has_val else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                             squeeze=False)

    # left: train loss
    ax = axes[0][0]
    ax.plot(epochs, train_losses, color='steelblue', linewidth=1.5,
            label='Train loss (Dice+BCE)')
    min_idx = int(np.argmin(train_losses))
    rng = max(train_losses) - min(train_losses) + 1e-9
    ax.annotate(
        f'min {train_losses[min_idx]:.4f}\n(ep {min_idx+1})',
        xy=(min_idx+1, train_losses[min_idx]),
        xytext=(min_idx+1+max(1, len(epochs)*0.05),
                train_losses[min_idx]+rng*0.08),
        fontsize=8, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0),
    )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss',  fontsize=12)
    ax.set_title(f'{arch} ({encoder})  |  {len(train_stems)} train imgs  |'
                 f' lr={learning_rate}', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # right: val IoU
    if has_val:
        ax2 = axes[0][1]
        ax2.plot(epochs[:len(val_ious)], val_ious,
                 color='tomato', linewidth=1.5, marker='o', markersize=3,
                 label='Val IoU')
        if best_epoch > 0:
            ax2.axvline(best_epoch, color='tomato', linestyle='--',
                        alpha=0.5, label=f'best ep {best_epoch}')
            rng2 = max(val_ious) - min(val_ious) + 1e-9
            ax2.annotate(
                f'best {best_val_iou:.4f}\n(ep {best_epoch})',
                xy=(best_epoch, best_val_iou),
                xytext=(best_epoch+max(1, len(epochs)*0.05),
                        best_val_iou-rng2*0.12),
                fontsize=8, color='tomato',
                arrowprops=dict(arrowstyle='->', color='tomato', lw=1.0),
            )
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('IoU',   fontsize=12)
        ax2.set_title(f'Val IoU per epoch  |  {len(val_stems)} val imgs',
                      fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{arch} scaffold segmentation training', fontsize=13)
    fig.tight_layout()
    curve_path = model_dir / 'training_curves.png'
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f'✓ Training curves → {curve_path}')


# ── main training function ────────────────────────────────────────────────────
def train(data_dir, model_dir, arch, encoder, n_epochs, batch_size,
          learning_rate, weight_decay, val_frac, patience, img_size,
          use_gpu, pretrained_model=None):

    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    # ── load data ─────────────────────────────────────────────────────────────
    print(f'\n[1/3] Loading data from: {data_dir}')
    images, masks, stems = collect_pairs(data_dir)

    if not images:
        data_dir_p = Path(data_dir)
        tifs  = list(data_dir_p.glob('*.tif')) + list(data_dir_p.glob('*.tiff'))
        mpngs = [p for p in data_dir_p.glob('*-mask.png')
                 if 'visible' not in p.name and 'target' not in p.name]
        print('[ERROR] No (tif, mask) pairs found.')
        print(f'  data_dir : {data_dir_p.resolve()}')
        print(f'  .tif files   : {len(tifs)}  {[p.name for p in tifs[:5]]}')
        print(f'  *-mask.png   : {len(mpngs)}  {[p.name for p in mpngs[:5]]}')
        sys.exit(1)

    print(f'      {len(images)} images from '
          f'{len(set(s.split("_")[0] for s in stems))} samples.')

    # ── sample-level val split ─────────────────────────────────────────────────
    if val_frac > 0.0 and len(images) >= 6:
        sample_ids    = sorted(set(s.split('_')[0] for s in stems))
        n_val_samples = max(1, round(len(sample_ids) * val_frac))
        rng           = random.Random(42)
        val_sids      = set(rng.sample(sample_ids, n_val_samples))
        train_sids    = set(sample_ids) - val_sids

        tr_idx  = [i for i, s in enumerate(stems) if s.split('_')[0] in train_sids]
        val_idx = [i for i, s in enumerate(stems) if s.split('_')[0] in val_sids]

        train_imgs   = [images[i] for i in tr_idx]
        train_msks   = [masks[i]  for i in tr_idx]
        train_stems  = [stems[i]  for i in tr_idx]
        val_imgs     = [images[i] for i in val_idx]
        val_msks     = [masks[i]  for i in val_idx]
        val_stems    = [stems[i]  for i in val_idx]

        print(f'      Val : {n_val_samples} samples ({len(val_idx)} images) — '
              f'IDs: {sorted(val_sids)}')
        print(f'      Train: {len(train_imgs)}  Val: {len(val_imgs)}')
    else:
        print('      val_frac=0 or too few images — training on all data.')
        train_imgs,  train_msks,  train_stems = images, masks, stems
        val_imgs,    val_msks,    val_stems   = [], [], []

    # ── PyTorch Dataset ───────────────────────────────────────────────────────
    class ScaffoldDataset(Dataset):
        def __init__(self, imgs, msks, img_size, augment):
            self.imgs     = imgs
            self.msks     = msks
            self.img_size = img_size
            self.augment  = augment

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img_t, msk_t = preprocess(self.imgs[idx], self.msks[idx],
                                      self.img_size, self.augment)
            return torch.from_numpy(img_t), torch.from_numpy(msk_t)

    train_ds = ScaffoldDataset(train_imgs, train_msks, img_size, augment=True)
    val_ds   = ScaffoldDataset(val_imgs,   val_msks,   img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── build model ───────────────────────────────────────────────────────────
    print(f'\n[2/3] Building {arch} with {encoder} encoder...')
    arch_map = {
        'unetplusplus': smp.UnetPlusPlus,
        'unet':         smp.Unet,
        'fpn':          smp.FPN,
    }
    if arch not in arch_map:
        print(f'[ERROR] Unknown architecture: {arch}. '
              f'Choose from: {list(arch_map.keys())}')
        sys.exit(1)

    if pretrained_model and Path(pretrained_model).exists():
        # load from a previously trained checkpoint (e.g. Pluronic model)
        # encoder_weights=None since we load full state dict
        print(f'      Loading weights from: {pretrained_model}')
        model = arch_map[arch](
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(device)
        ckpt = torch.load(str(pretrained_model), map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'      Loaded checkpoint from epoch {ckpt.get("epoch","?")}  '
              f'val_IoU={ckpt.get("val_iou","?")}')
    else:
        # standard ImageNet-pretrained encoder
        model = arch_map[arch](
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'      Parameters: {n_params:.1f}M')

    # ── loss: Dice + BCE ──────────────────────────────────────────────────────
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

    def combined_loss(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=learning_rate * 0.01
    )

    # ── training loop ─────────────────────────────────────────────────────────
    print(f'\n[3/3] Training: {n_epochs} epochs  batch={batch_size}  '
          f'lr={learning_rate}  wd={weight_decay}  '
          f'patience={patience if patience>0 else "off"}')

    train_losses, val_ious = [], []
    best_val_iou = -1.0
    best_epoch   = 0
    no_improve   = 0
    best_path    = model_dir / 'best_model.pth'
    final_path   = model_dir / 'final_model.pth'

    for epoch in range(n_epochs):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for img_b, msk_b in train_loader:
            img_b = img_b.to(device)
            msk_b = msk_b.to(device)
            optimizer.zero_grad()
            pred  = model(img_b)
            loss  = combined_loss(pred, msk_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * img_b.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_ds)
        train_losses.append(avg_loss)

        # ── val IoU ───────────────────────────────────────────────────────────
        if val_loader.dataset and len(val_loader.dataset) > 0:
            model.eval()
            ious = []
            with torch.no_grad():
                for img_b, msk_b in val_loader:
                    img_b  = img_b.to(device)
                    pred   = torch.sigmoid(model(img_b))
                    pred_b = (pred.cpu().numpy() > 0.5)
                    gt_b   = (msk_b.numpy()      > 0.5)
                    for p, g in zip(pred_b, gt_b):
                        ious.append(binary_iou_np(p[0], g[0]))

            val_iou = float(np.mean(ious))
            val_ious.append(val_iou)

            improved = val_iou > best_val_iou
            if improved:
                best_val_iou = val_iou
                best_epoch   = epoch + 1
                no_improve   = 0
                torch.save({'epoch':      epoch + 1,
                            'model_state': model.state_dict(),
                            'val_iou':    val_iou,
                            'arch':       arch,
                            'encoder':    encoder}, str(best_path))
            else:
                no_improve += 1

            flag = ' ← best' if improved else ''
            print(f'  Epoch {epoch+1:4d}/{n_epochs}  '
                  f'loss={avg_loss:.4f}  val_IoU={val_iou:.4f}{flag}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

            if patience > 0 and no_improve >= patience:
                print(f'\n  Early stopping at epoch {epoch+1} '
                      f'(no val improvement for {patience} epochs)')
                break
        else:
            print(f'  Epoch {epoch+1:4d}/{n_epochs}  loss={avg_loss:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

    # ── save final model ──────────────────────────────────────────────────────
    torch.save({'epoch':       len(train_losses),
                'model_state': model.state_dict(),
                'arch':        arch,
                'encoder':     encoder}, str(final_path))

    print(f'\n✓ Final model → {final_path}')
    if best_epoch > 0:
        print(f'✓ Best model  → {best_path}  '
              f'(epoch {best_epoch}, val IoU={best_val_iou:.4f})')

    save_training_outputs(
        model_dir=model_dir,
        train_losses=train_losses,
        val_ious=val_ious,
        best_epoch=best_epoch,
        best_val_iou=best_val_iou,
        arch=arch,
        encoder=encoder,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        img_size=img_size,
        data_dir=data_dir,
        train_stems=train_stems,
        val_stems=val_stems,
        patience=patience,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train U-Net++ for scaffold binary segmentation.'
    )
    parser.add_argument('--data_dir',    required=True,
        help='Folder with *.tif + *-mask.png pairs')
    parser.add_argument('--model_dir',   required=True,
        help='Where to save model weights and logs')
    parser.add_argument('--architecture', default='unetplusplus',
        choices=['unetplusplus', 'unet', 'fpn'],
        help='Model architecture (default: unetplusplus)')
    parser.add_argument('--encoder',     default='resnet34',
        help='Encoder backbone (default: resnet34)')
    parser.add_argument('--n_epochs',    type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=4,
        help='Batch size (default: 4). Reduce to 2 if GPU OOM.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
        help='Weight decay (default: 1e-4)')
    parser.add_argument('--val_frac',    type=float, default=0.15,
        help='Val fraction at sample level (default: 0.15). 0=disable.')
    parser.add_argument('--patience',    type=int,   default=20,
        help='Early stopping patience (default: 20). 0=disable.')
    parser.add_argument('--img_size',    type=int,   default=512,
        help='Resize images to this square size (default: 512)')
    parser.add_argument('--pretrained_model', default=None,
        help='Path to existing checkpoint to fine-tune from '
             '(e.g. Pluronic best_model.pth). '
             'If not set, uses ImageNet encoder weights.')
    parser.add_argument('--no_gpu',      action='store_true')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
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
        pretrained_model=args.pretrained_model,
    )
