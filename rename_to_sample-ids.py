"""
rename_to_sample_ids.py
-----------------------
Renames image files across all subfolders to use true Sample_IDs
instead of column indices (0-7 per NC file).

Background
----------
NC files are generated with stride-based interleaving:
  NC file k assigns samples: k, k+n_files, k+2*n_files, ...
  to columns 0, 1, 2, ... in that 48-well plate.

Image filenames follow the pattern:
  {col_idx}_{row}.tif          col_idx: 0-7, row: 0-5
  {col_idx}_{row}-mask.png
  {col_idx}_{row}-target-overlay.png

After renaming:
  {Sample_ID}_{row}.tif
  {Sample_ID}_{row}-mask.png
  {Sample_ID}_{row}-target-overlay.png

Folder names must contain 's{first_sample}' to identify which NC file
they belong to — e.g. 'dev_annot_s0-49_trans_260601' → first_sample=0.

Usage
-----
    python rename_to_sample_ids.py <root_dir> <csv_path>
        [--dry_run]     print what would be renamed without doing it
        [--n_cols 8]    columns per NC file (default: 8)

Output
------
  - Files renamed in-place inside each subfolder
  - <root_dir>/rename_conversion_table.csv  — full conversion table
    with original name, new name, and all LHS parameters
"""

import argparse
import math
import re
import shutil
import csv
from pathlib import Path


# ── file suffixes to rename ───────────────────────────────────────────────────
# All files whose stem matches {col_idx}_{row}{optional_suffix} are renamed.
# Extensions handled:
HANDLED_PATTERNS = [
    # stem pattern                     example stem
    r'^(\d+)_(\d+)$',                 # 3_2
    r'^(\d+)_(\d+)-mask$',            # 3_2-mask
    r'^(\d+)_(\d+)-mask-visible$',    # 3_2-mask-visible
    r'^(\d+)_(\d+)-target-overlay$',  # 3_2-target-overlay
]
COMPILED = [(re.compile(p), p) for p, _ in zip(HANDLED_PATTERNS, HANDLED_PATTERNS)]


def parse_stem(stem):
    """
    Try to match stem against known patterns.
    Returns (col_idx, row, suffix) or None.
    suffix is the part after '{col}_{row}', e.g. '-mask' or ''.
    """
    # general pattern: starts with digits_digits, optionally followed by -something
    m = re.match(r'^(\d+)_(\d+)(-.+)?$', stem)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3) or ''
    return None


def build_col_to_sample_map(first_sample, n_total, n_cols, n_files):
    """
    Given the first sample in this NC file batch (= NC file index k),
    return a dict {col_idx: Sample_ID} for col_idx 0..n_cols-1.
    """
    k = first_sample  # NC file index
    mapping = {}
    for j in range(n_cols):
        sid = k + j * n_files
        if sid < n_total:
            mapping[j] = sid
    return mapping


def extract_first_sample_from_folder(folder_name):
    """
    Extract first sample number from folder name.
    Looks for pattern like 's0-' or 's12-' at start of a token.
    e.g. 'dev_annot_s0-49_trans_260601' -> 0
         'dev_annot_s14-63_trans_260601' -> 14
    """
    m = re.search(r's(\d+)-', folder_name)
    return int(m.group(1)) if m else None


def load_lhs_csv(csv_path):
    """Load LHS CSV (semicolon-delimited) into dict {Sample_ID: row_dict}."""
    samples = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            sid = int(row['Sample_ID'])
            samples[sid] = {k: v.strip() for k, v in row.items()}
    return samples


def process_root(root_dir, csv_path, n_cols=8, dry_run=False):
    root_dir = Path(root_dir)
    lhs      = load_lhs_csv(csv_path)
    n_total  = len(lhs)
    n_files  = math.ceil(n_total / n_cols)

    # all renamed files go into a single flat output folder next to root_dir
    out_dir  = root_dir.parent / (root_dir.name + '_renamed')
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Root:      {root_dir}')
    print(f'Output:    {out_dir}')
    print(f'Samples:   {n_total}  →  {n_files} NC files × {n_cols} cols')
    print(f'Dry run:   {dry_run}\n')

    conversion_rows = []
    subfolders = sorted(p for p in root_dir.iterdir() if p.is_dir())

    if not subfolders:
        print('[WARN] No subfolders found.')
        return

    for folder in subfolders:
        first_sample = extract_first_sample_from_folder(folder.name)
        if first_sample is None:
            print(f'[SKIP] Cannot parse first_sample from folder: {folder.name}')
            continue

        col_to_sid = build_col_to_sample_map(first_sample, n_total, n_cols, n_files)
        print(f'Folder: {folder.name}  (first_sample={first_sample})')
        print(f'  col→SampleID: { {k:v for k,v in col_to_sid.items()} }')

        all_files = sorted(f for f in folder.iterdir() if f.is_file())
        copied = 0

        for fpath in all_files:
            stem   = fpath.stem
            ext    = fpath.suffix
            parsed = parse_stem(stem)
            if parsed is None:
                continue

            col_idx, row, suffix = parsed
            if col_idx not in col_to_sid:
                print(f'  [SKIP] col_idx={col_idx} not in mapping: {fpath.name}')
                continue

            sid      = col_to_sid[col_idx]
            new_name = f'{sid}_{row}{suffix}{ext}'
            dst      = out_dir / new_name

            lhs_row = lhs.get(sid, {})
            conversion_rows.append({
                'source_folder':    folder.name,
                'old_filename':     fpath.name,
                'new_filename':     new_name,
                'col_idx':          col_idx,
                'row':              row,
                'Sample_ID':        sid,
                'Pressure_kPa':     lhs_row.get('Pressure_kPa', ''),
                'NozzleSpeed_mms':  lhs_row.get('NozzleSpeed_mms', ''),
                'Zoffset_mm':       lhs_row.get('Zoffset_mm', ''),
            })

            if dry_run:
                print(f'  [DRY] {fpath.name}  →  {new_name}')
            else:
                shutil.copy2(fpath, dst)
                copied += 1

        if not dry_run:
            print(f'  Copied {copied} file(s).')
        print()

    # ── write conversion table CSV ────────────────────────────────────────────
    out_csv = (out_dir if not dry_run else root_dir) / 'rename_conversion_table.csv'
    if conversion_rows:
        fieldnames = ['source_folder','old_filename','new_filename','col_idx','row',
                      'Sample_ID','Pressure_kPa','NozzleSpeed_mms','Zoffset_mm']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            writer.writerows(conversion_rows)
        print(f'Conversion table saved → {out_csv}')
        print(f'Total entries: {len(conversion_rows)}')
    else:
        print('No files were matched for renaming.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename image files from col_idx to Sample_ID across subfolders.'
    )
    parser.add_argument('root_dir',
        help='Root folder containing subfolders like dev_annot_s0-49_trans_...')
    parser.add_argument('csv_path',
        help='Path to lhs_bioprint_samples_semicolon.csv')
    parser.add_argument('--dry_run', action='store_true',
        help='Print what would be renamed without actually renaming')
    parser.add_argument('--n_cols', type=int, default=8,
        help='Number of columns (samples) per NC file (default: 8)')
    args = parser.parse_args()

    process_root(args.root_dir, args.csv_path,
                 n_cols=args.n_cols, dry_run=args.dry_run)