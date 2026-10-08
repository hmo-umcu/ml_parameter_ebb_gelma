"""
build_conversion_table_48well.py
---------------------------------
Build rename_conversion_table.csv for a 48-well LHS deployment folder whose
images have ALREADY been renamed to {Sample_ID}_{row} by rename_to_sampleids.py.
 
Why this exists
---------------
rename_to_sampleids.py writes a conversion table as a side effect of copying
files, with old_filename/new_filename/col_idx columns that only make sense
during the rename. Once the folder is renamed, what build_sample_sf_table.py
actually needs is much simpler: Sample_ID -> print parameters. This script
produces exactly that, from the LHS CSV, and (optionally) checks it against
the images really present on disk.
 
Usage
-----
    python build_conversion_table_48well.py \
        --lhs_csv    lhs_bioprint_samples_semicolon.csv \
        --output_csv rename_conversion_table.csv \
        [--data_dir  .../cell_gelma_7_60_renamed] \
        [--n_rows    6]
 
    --data_dir : if given, one row is written per image stem actually found,
                 and any missing sample or replicate is reported. If omitted,
                 one row per Sample_ID is written from the LHS CSV alone.
 
Output columns (;-separated)
----------------------------
    Sample_ID;row;stem;Pressure_kPa;NozzleSpeed_mms;Zoffset_mm
 
build_sample_sf_table.py auto-detects these headers.
"""
 
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
 
 
STEM_RE = re.compile(r'^(\d+)_(\d+)$')
FIELDNAMES = ['Sample_ID', 'row', 'stem',
              'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
 
 
def sniff_delimiter(path):
    with open(path, newline='') as f:
        head = f.readline()
    return ';' if head.count(';') >= head.count(',') else ','
 
 
def load_lhs(csv_path):
    """Load the LHS CSV into {Sample_ID(str): {param: value}}."""
    delim = sniff_delimiter(csv_path)
    out = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f, delimiter=delim):
            row = {k.strip(): (v or '').strip() for k, v in row.items() if k}
            sid = row.get('Sample_ID', '').strip()
            if sid == '':
                continue
            out[sid] = {
                'Pressure_kPa':    row.get('Pressure_kPa', ''),
                'NozzleSpeed_mms': row.get('NozzleSpeed_mms', ''),
                'Zoffset_mm':      row.get('Zoffset_mm', ''),
            }
    return out
 
 
def scan_stems(data_dir):
    """Every {Sample_ID}_{row} stem that has a real .tif in data_dir."""
    data_dir = Path(data_dir)
    found = defaultdict(set)
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in data_dir.glob(ext):
            # skip anything this pipeline generated
            if any(t in p.stem.lower() for t in
                   ('mask', 'overlay', 'visible', 'pred', 'target')):
                continue
            m = STEM_RE.match(p.stem)
            if m:
                found[m.group(1)].add(int(m.group(2)))
    return found
 
 
def main(args):
    lhs = load_lhs(args.lhs_csv)
    print(f'LHS CSV: {len(lhs)} sample(s) -> IDs '
          f'{sorted(lhs, key=int)[:5]}...{sorted(lhs, key=int)[-3:]}')
 
    rows = []
    if args.data_dir:
        found = scan_stems(args.data_dir)
        n_imgs = sum(len(v) for v in found.values())
        print(f'Folder : {args.data_dir}')
        print(f'         {len(found)} sample(s), {n_imgs} image(s)\n')
 
        for sid in sorted(found, key=int):
            for r in sorted(found[sid]):
                p = lhs.get(sid)
                if p is None:
                    p = {'Pressure_kPa': '', 'NozzleSpeed_mms': '', 'Zoffset_mm': ''}
                rows.append({'Sample_ID': sid, 'row': r, 'stem': f'{sid}_{r}', **p})
 
        # report anything that does not line up, rather than silently proceeding
        no_params = [s for s in found if s not in lhs]
        if no_params:
            print(f'[WARN] {len(no_params)} sample(s) on disk have no row in the '
                  f'LHS CSV, so their parameters are blank: '
                  f'{sorted(no_params, key=int)}')
        no_images = [s for s in lhs if s not in found]
        if no_images:
            print(f'[WARN] {len(no_images)} sample(s) in the LHS CSV have no '
                  f'images in this folder: {sorted(no_images, key=int)}')
        short = {s: sorted(v) for s, v in found.items() if len(v) != args.n_rows}
        if short:
            print(f'[WARN] {len(short)} sample(s) do not have exactly '
                  f'{args.n_rows} replicate rows:')
            for s in sorted(short, key=int):
                print(f'         Sample {s}: rows {short[s]}')
    else:
        print('No --data_dir given: writing one row per Sample_ID from the '
              'LHS CSV, with no cross-check against the images.\n')
        for sid in sorted(lhs, key=int):
            rows.append({'Sample_ID': sid, 'row': '', 'stem': '', **lhs[sid]})
 
    with open(args.output_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
        wtr.writeheader()
        wtr.writerows(rows)
 
    n_samples = len({r['Sample_ID'] for r in rows})
    print(f'\nWrote {len(rows)} row(s) covering {n_samples} sample(s)')
    print(f'-> {args.output_csv}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build rename_conversion_table.csv for a renamed 48-well folder.')
    ap.add_argument('--lhs_csv', required=True,
                    help='lhs_bioprint_samples_semicolon.csv')
    ap.add_argument('--output_csv', required=True,
                    help='Where to write the conversion table')
    ap.add_argument('--data_dir', default=None,
                    help='Renamed image folder; if given, rows follow the images '
                         'actually present and mismatches are reported')
    ap.add_argument('--n_rows', type=int, default=6,
                    help='Expected replicate rows per sample (default: 6)')
    main(ap.parse_args())