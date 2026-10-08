"""
compare_cv_results.py
---------------------
Compare cross-validation results across architectures (unetplusplus, unet, fpn).

Reads cv_final_results.txt from:
    <base_dir>/cv_unetplusplus/cv_final_results.txt
    <base_dir>/cv_unet/cv_final_results.txt
    <base_dir>/cv_fpn/cv_final_results.txt

Usage
-----
    python compare_cv_results.py
        --base_dir  /home/hmo/BioRT/Rheology-informed-optimization/data_procedure/data/dev_images
        [--output_dir  <same as base_dir>]   # where to save comparison outputs

Output
------
    architecture_comparison.txt   side-by-side text table
    architecture_comparison.png   bar chart with error bars
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ── architecture directories ──────────────────────────────────────────────────
ARCH_DIRS = {
    'U-Net++': 'cv_unetpp',
    'U-Net':   'cv_unet',
    'FPN':     'cv_fpn',
}

# metrics to extract and display
METRICS = [
    ('iou_pred_annot',  'IoU (pred vs annot)'),
    ('iou_pred_target', 'IoU (pred vs target)'),
    ('dice_pred_annot', 'Dice (pred vs annot)'),
    ('pixel_acc',       'Pixel accuracy'),
]


# ── parse cv_final_results.txt ────────────────────────────────────────────────
def parse_cv_final_results(txt_path):
    """
    Parse cv_final_results.txt produced by unetplusplus_cross_validate.py.

    Returns dict with keys:
        architecture, n_epochs, learning_rate, n_images
        iou_pred_annot_mean, iou_pred_annot_std,
        iou_pred_annot_median, iou_pred_annot_q1, iou_pred_annot_q3,
        iou_pred_target_mean, iou_pred_target_std,
        iou_pred_target_median, iou_pred_target_q1, iou_pred_target_q3,
        dice_pred_annot_mean, dice_pred_annot_std,
        pixel_acc_mean, pixel_acc_std,
        fold_iou_annot  [list of per-fold means]
        fold_iou_target [list of per-fold means]
    """
    txt = Path(txt_path).read_text()

    def find_float(pattern, text, default=None):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    def find_str(pattern, text, default=''):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    result = {}

    result['architecture']  = find_str(r'architecture\s*:\s*(\S.*)', txt)
    result['n_epochs']      = find_str(r'n_epochs\s*:\s*(\d+)', txt)
    result['learning_rate'] = find_str(r'learning_rate\s*:\s*(\S+)', txt)
    result['n_images']      = find_float(r'total images evaluated:\s*(\d+)', txt, 0)

    # IoU pred vs annot
    m_pa = re.search(
        r'IoU \(pred vs annot\):.*?mean ± std\s*:\s*([\d.]+)\s*±\s*([\d.]+)'
        r'.*?median \[IQR\]\s*:\s*([\d.]+)\s*\[([\d.]+)\s*[–-]\s*([\d.]+)\]',
        txt, re.DOTALL)
    if m_pa:
        result['iou_pred_annot_mean']   = float(m_pa.group(1))
        result['iou_pred_annot_std']    = float(m_pa.group(2))
        result['iou_pred_annot_median'] = float(m_pa.group(3))
        result['iou_pred_annot_q1']     = float(m_pa.group(4))
        result['iou_pred_annot_q3']     = float(m_pa.group(5))
    else:
        for key in ['mean', 'std', 'median', 'q1', 'q3']:
            result[f'iou_pred_annot_{key}'] = None

    # IoU pred vs target
    m_pt = re.search(
        r'IoU \(pred vs target\):.*?mean ± std\s*:\s*([\d.]+)\s*±\s*([\d.]+)'
        r'.*?median \[IQR\]\s*:\s*([\d.]+)\s*\[([\d.]+)\s*[–-]\s*([\d.]+)\]',
        txt, re.DOTALL)
    if m_pt:
        result['iou_pred_target_mean']   = float(m_pt.group(1))
        result['iou_pred_target_std']    = float(m_pt.group(2))
        result['iou_pred_target_median'] = float(m_pt.group(3))
        result['iou_pred_target_q1']     = float(m_pt.group(4))
        result['iou_pred_target_q3']     = float(m_pt.group(5))
    else:
        for key in ['mean', 'std', 'median', 'q1', 'q3']:
            result[f'iou_pred_target_{key}'] = None

    # Dice
    m_d = re.search(r'Dice \(pred vs annot\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)', txt)
    result['dice_pred_annot_mean'] = float(m_d.group(1)) if m_d else None
    result['dice_pred_annot_std']  = float(m_d.group(2)) if m_d else None

    # Pixel accuracy
    m_a = re.search(r'Pixel accuracy\s*:\s*([\d.]+)\s*±\s*([\d.]+)', txt)
    result['pixel_acc_mean'] = float(m_a.group(1)) if m_a else None
    result['pixel_acc_std']  = float(m_a.group(2)) if m_a else None

    # Per-fold breakdown
    fold_pa, fold_pt = [], []
    for m in re.finditer(
            r'fold\s+\d+:.*?IoU\(annot\)=([\d.]+).*?IoU\(target\)=([\d.]+)', txt):
        fold_pa.append(float(m.group(1)))
        fold_pt.append(float(m.group(2)))
    result['fold_iou_annot']  = fold_pa
    result['fold_iou_target'] = fold_pt

    return result


# ── text comparison table ─────────────────────────────────────────────────────
def print_and_save_table(arch_results, output_path):
    arch_names = list(arch_results.keys())
    available  = [a for a in arch_names if arch_results[a] is not None]

    lines = [
        'Architecture Comparison — Cross-Validation Results',
        '===================================================',
        '',
    ]

    # header
    col_w = 24
    header = f'{"Metric":<32}' + ''.join(f'{a:>{col_w}}' for a in available)
    lines.append(header)
    lines.append('─' * len(header))

    def fmt(val, err=None):
        if val is None:
            return 'n/a'
        if err is not None:
            return f'{val:.4f} ± {err:.4f}'
        return f'{val:.4f}'

    metric_rows = [
        ('IoU (pred vs annot)  mean±std',
         'iou_pred_annot_mean', 'iou_pred_annot_std'),
        ('IoU (pred vs annot)  median[IQR]',
         'iou_pred_annot_median', None),
        ('IoU (pred vs target) mean±std',
         'iou_pred_target_mean', 'iou_pred_target_std'),
        ('IoU (pred vs target) median[IQR]',
         'iou_pred_target_median', None),
        ('Dice (pred vs annot) mean±std',
         'dice_pred_annot_mean', 'dice_pred_annot_std'),
        ('Pixel accuracy       mean±std',
         'pixel_acc_mean', 'pixel_acc_std'),
    ]

    for label, key_mean, key_err in metric_rows:
        if 'median' in key_mean:
            # format as median [Q1–Q3]
            row = f'{label:<32}'
            base = key_mean.replace('_median', '')
            for a in available:
                r = arch_results[a]
                med = r.get(f'{base}_median')
                q1  = r.get(f'{base}_q1')
                q3  = r.get(f'{base}_q3')
                if med is not None and q1 is not None and q3 is not None:
                    cell = f'{med:.4f} [{q1:.4f}–{q3:.4f}]'
                else:
                    cell = 'n/a'
                row += f'{cell:>{col_w}}'
            lines.append(row)
        else:
            row = f'{label:<32}'
            for a in available:
                r   = arch_results[a]
                val = r.get(key_mean)
                err = r.get(key_err) if key_err else None
                row += f'{fmt(val, err):>{col_w}}'
            lines.append(row)

    lines += ['', 'Per-fold IoU (pred vs annot):']
    max_folds = max(len(arch_results[a].get('fold_iou_annot', []))
                    for a in available)
    for fi in range(max_folds):
        row = f'  fold {fi:<28}'
        for a in available:
            vals = arch_results[a].get('fold_iou_annot', [])
            cell = f'{vals[fi]:.4f}' if fi < len(vals) else 'n/a'
            row += f'{cell:>{col_w}}'
        lines.append(row)

    lines += ['', 'Per-fold IoU (pred vs target):']
    for fi in range(max_folds):
        row = f'  fold {fi:<28}'
        for a in available:
            vals = arch_results[a].get('fold_iou_target', [])
            cell = f'{vals[fi]:.4f}' if fi < len(vals) else 'n/a'
            row += f'{cell:>{col_w}}'
        lines.append(row)

    # training config
    lines += ['', 'Training configuration:']
    for cfg_key, cfg_label in [('architecture', 'architecture'),
                                ('n_epochs', 'n_epochs'),
                                ('learning_rate', 'learning_rate'),
                                ('n_images', 'n_images evaluated')]:
        row = f'  {cfg_label:<30}'
        for a in available:
            val = arch_results[a].get(cfg_key, 'n/a')
            row += f'{str(val):>{col_w}}'
        lines.append(row)

    text = '\n'.join(lines)
    print(text)
    Path(output_path).write_text(text)
    print(f'\n✓ Comparison table → {output_path}')


# ── bar chart ─────────────────────────────────────────────────────────────────
def plot_comparison(arch_results, output_path):
    available = [a for a, r in arch_results.items() if r is not None]

    plot_metrics = [
        ('iou_pred_annot_mean',  'iou_pred_annot_std',  'IoU (pred vs annot)'),
        ('iou_pred_target_mean', 'iou_pred_target_std', 'IoU (pred vs target)'),
        ('dice_pred_annot_mean', 'dice_pred_annot_std', 'Dice (pred vs annot)'),
        ('pixel_acc_mean',       'pixel_acc_std',       'Pixel accuracy'),
    ]

    n_metrics = len(plot_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5),
                             sharey=False)

    colours = ['#2196F3', '#FF9800', '#4CAF50']   # blue, orange, green

    for ax, (key_mean, key_std, label) in zip(axes, plot_metrics):
        means = [arch_results[a].get(key_mean) for a in available]
        stds  = [arch_results[a].get(key_std,  0) for a in available]

        # replace None with 0 for plotting
        means = [v if v is not None else 0 for v in means]
        stds  = [v if v is not None else 0 for v in stds]

        x = np.arange(len(available))
        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color=colours[:len(available)],
                      alpha=0.82, width=0.55,
                      error_kw=dict(elinewidth=1.5, ecolor='#333'))

        # annotate bar tops with mean ± std
        for xi, (m, s) in enumerate(zip(means, stds)):
            if m > 0:
                ax.text(xi, m + s + 0.008, f'{m:.3f}±{s:.3f}',
                        ha='center', va='bottom', fontsize=8.5)

        # per-fold scatter
        for xi, a in enumerate(available):
            fold_vals_key = ('fold_iou_annot' if 'annot' in key_mean
                             else 'fold_iou_target'
                             if 'target' in key_mean else None)
            if fold_vals_key:
                fv = arch_results[a].get(fold_vals_key, [])
                if fv:
                    jitter = np.random.default_rng(42).uniform(
                        -0.12, 0.12, size=len(fv))
                    ax.scatter(np.full(len(fv), xi) + jitter, fv,
                               color='#222', alpha=0.55, s=22, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(available, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Architecture Comparison — 4-Fold Cross-Validation (ResNet34 encoder)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Comparison plot  → {output_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compare CV results across U-Net++, U-Net, FPN architectures.'
    )
    parser.add_argument('--base_dir', required=True,
        help='Parent folder containing cv_unetplusplus/, cv_unet/, cv_fpn/')
    parser.add_argument('--output_dir', default=None,
        help='Where to save outputs (default: same as base_dir)')
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load results ──────────────────────────────────────────────────────────
    arch_results = {}
    for display_name, subdir in ARCH_DIRS.items():
        txt_path = base_dir / subdir / 'cv_final_results.txt'
        if not txt_path.exists():
            print(f'[WARN] Not found: {txt_path}  — skipping {display_name}')
            arch_results[display_name] = None
            continue
        print(f'Reading: {txt_path}')
        arch_results[display_name] = parse_cv_final_results(txt_path)

    available = [a for a, r in arch_results.items() if r is not None]
    if not available:
        print('[ERROR] No cv_final_results.txt files found. '
              'Check --base_dir and that CV jobs have completed.')
        return

    print(f'\nArchitectures found: {available}\n')

    # ── outputs ───────────────────────────────────────────────────────────────
    print_and_save_table(
        {a: arch_results[a] for a in available},
        output_dir / 'architecture_comparison.txt',
    )

    plot_comparison(
        {a: arch_results[a] for a in available},
        output_dir / 'architecture_comparison.png',
    )


if __name__ == '__main__':
    main()