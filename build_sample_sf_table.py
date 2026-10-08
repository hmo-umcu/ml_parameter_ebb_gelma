"""
build_sample_sf_table.py
-------------------------
Merge pore_scores_all_folds.csv (per-image SF scores from pore_analysis.py)
with rename_conversion_table.csv (maps {Sample_ID}_{row} stems to their
print parameters) into one final per-sample summary table.
 
For each Sample_ID (typically 6 replicate images, rows 0-5), reports:
    Sample_ID, Pressure_kPa, NozzleSpeed_mms, Zoffset_mm,
    fold, n_images, SF_mean, SF_std
 
"fold" is read from pore_scores_all_folds.csv's own fold column (present
when that file came from pore_analysis.py's --cv_parent_dir multi-fold
mode). Since cross-validation splits at the SAMPLE level, all replicates
of a sample should land in the same fold — if a sample's images are found
split across more than one fold, that's flagged as a warning (it would
indicate a CV leakage issue worth investigating).
 
Sample_ID and replicate row are parsed directly from the "stem" column in
pore_scores_all_folds.csv (stems follow the {Sample_ID}_{row} convention
used throughout the pipeline), so this works regardless of which fold an
image ended up in during cross-validation.
 
Print parameters (Pressure_kPa, NozzleSpeed_mms, Zoffset_mm) are looked up
per Sample_ID from rename_conversion_table.csv. Column names in that table
are auto-detected (case-insensitive, matching "sample_id", "pressure",
"speed"/"nozzlespeed", "zoffset"/"z_offset") — override with the
--sample_id_col/--pressure_col/--speed_col/--zoffset_col flags if your
table uses different headers; the script will print the columns it found
if auto-detection fails, so you can copy the exact name from there.
 
SF_std uses population std (ddof=0), matching the convention used in
unetplusplus_aggregate.py elsewhere in this pipeline.
 
Usage
-----
    python build_sample_sf_table.py \
        --pore_scores_csv     /path/to/pore_scores_all_folds.csv \
        --rename_table_csv    /path/to/rename_conversion_table.csv \
        --output_csv          /path/to/sample_sf_summary.csv \
        [--sample_id_col Sample_ID] \
        [--pressure_col  Pressure_kPa] \
        [--speed_col     NozzleSpeed_mms] \
        [--zoffset_col   Zoffset_mm]
 
Output
------
    <output_csv>   one row per Sample_ID, ;-separated
"""
 
import argparse
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
 
 
STEM_RE  = re.compile(r'^(\d+)_(\d+)$')                       # 48-well: 23_4
SWEEP_RE = re.compile(r'^([A-Za-z]+)[_-](\d+)$')              # sweep:   sweep_14
 
 
def parse_stem(stem):
    """
    Map an image stem to (Sample_ID, replicate_row, kind).
 
      '23_4'      -> ('23', '4',  '48well')   Sample 23, replicate row 4
      'sweep_14'  -> ('14', '0',  'sweep')    sweep condition 14, single image
 
    The 48-well plates print 6 replicates of each LHS sample, so Sample_ID is
    the first field and the second is the replicate. A pressure sweep prints
    every well at a DIFFERENT pressure, so each sweep index is its own sample
    with exactly one image -- there are no replicates to average over.
 
    Returns (None, None, None) if the stem matches neither form.
    """
    stem = stem.strip()
    m = STEM_RE.match(stem)
    if m:
        return m.group(1), m.group(2), '48well'
    m = SWEEP_RE.match(stem)
    if m:
        return m.group(2), '0', 'sweep'
    return None, None, None
 
 
def sniff_delimiter(path):
    """Detect ; vs , delimiter from the header line."""
    with open(path, newline='') as f:
        first_line = f.readline()
    return ';' if first_line.count(';') >= first_line.count(',') else ','
 
 
def load_csv(path):
    delim = sniff_delimiter(path)
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames, delim
 
 
def auto_detect_column(fieldnames, candidates, label):
    """Case-insensitive substring match against a list of candidate names."""
    lower_map = {fn.lower(): fn for fn in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for fn_lower, fn in lower_map.items():
        for cand in candidates:
            if cand.lower() in fn_lower:
                return fn
    print(f'[ERROR] Could not auto-detect the "{label}" column.')
    print(f'        Columns found in rename_conversion_table.csv: {fieldnames}')
    print(f'        Pass it explicitly, e.g. --{label}_col <exact_column_name>')
    return None
 
 
def build_table(pore_scores_csv, rename_table_csv, output_csv,
                sample_id_col, pressure_col, speed_col, zoffset_col):
 
    # ── load pore_scores_all_folds.csv ────────────────────────────────────────
    pore_rows, pore_fields, _ = load_csv(pore_scores_csv)
    if 'stem' not in pore_fields or 'SF' not in pore_fields:
        print(f'[ERROR] {pore_scores_csv} is missing "stem" or "SF" columns. '
              f'Found: {pore_fields}')
        return
    has_fold_col = 'fold' in pore_fields
    if not has_fold_col:
        print(f'[WARN] {pore_scores_csv} has no "fold" column — '
              f'fold info will be left blank in the output. '
              f'(Expected if this came from single-folder mode rather than '
              f'--cv_parent_dir.)')
 
    print(f'Loaded {len(pore_rows)} image rows from {pore_scores_csv}')
 
    # ── load rename_conversion_table.csv and auto-detect columns ─────────────
    rename_rows, rename_fields, _ = load_csv(rename_table_csv)
    print(f'Loaded {len(rename_rows)} rows from {rename_table_csv}')
    print(f'  Columns: {rename_fields}\n')
 
    sid_col = sample_id_col or auto_detect_column(
        rename_fields, ['sample_id', 'sampleid', 'sample'], 'sample_id')
    p_col = pressure_col or auto_detect_column(
        rename_fields, ['pressure_kpa', 'pressure'], 'pressure')
    s_col = speed_col or auto_detect_column(
        rename_fields, ['nozzlespeed_mms', 'nozzlespeed', 'speed'], 'speed')
    z_col = zoffset_col or auto_detect_column(
        rename_fields, ['zoffset_mm', 'zoffset', 'z_offset'], 'zoffset')
    stem_col = next((fn for fn in rename_fields if fn.strip().lower() == 'stem'),
                    None)
 
    if sid_col is None:
        print('[ERROR] Cannot proceed without identifying the Sample_ID '
              'column in rename_conversion_table.csv.')
        return
 
    print(f'Using columns -> Sample_ID: "{sid_col}", Pressure: "{p_col}", '
          f'Speed: "{s_col}", Zoffset: "{z_col}", stem: "{stem_col}"\n')
 
    def row_params(row):
        return {
            'Pressure_kPa':    row.get(p_col, '') if p_col else '',
            'NozzleSpeed_mms': row.get(s_col, '') if s_col else '',
            'Zoffset_mm':      row.get(z_col, '') if z_col else '',
        }
 
    # Direct stem -> Sample_ID lookup. This is the reliable join, because it
    # does not care what the images are called. Sweep folders are not named
    # consistently: some hold sweep_0..sweep_18, others hold col_row names.
    # Re-parsing the stem guesses wrong on the second form (it reads '0_5' as
    # sample 0 replicate 5 rather than sweep well 5), which is how an empty or
    # wrong summary table gets produced.
    sid_by_stem, params_by_stem = {}, {}
    if stem_col:
        for row in rename_rows:
            st = str(row.get(stem_col, '')).strip()
            if st:
                sid_by_stem[st] = str(row.get(sid_col, '')).strip()
                params_by_stem[st] = row_params(row)
        print(f'Stem lookup: {len(sid_by_stem)} stem(s) from the conversion table')
 
    # Fallback for tables with no stem column: Sample_ID -> params
    params_by_sample = {}
    for row in rename_rows:
        sid_raw = str(row.get(sid_col, '')).strip()
        if not sid_raw:
            continue
        parsed_sid, _r, _k = parse_stem(sid_raw)
        sid = parsed_sid if parsed_sid is not None else sid_raw
        params = row_params(row)
        if sid in params_by_sample and params_by_sample[sid] != params:
            print(f'[WARN] Sample_ID {sid} has conflicting parameter rows '
                  f'in {rename_table_csv} - keeping the first one seen.')
            continue
        params_by_sample[sid] = params
 
    # ── group SF (and fold) by Sample_ID ──────────────────────────────────────
    sf_by_sample   = defaultdict(list)
    fold_by_sample = defaultdict(set)
    unparsed, kinds, n_by_stem = [], set(), 0
 
    for row in pore_rows:
        stem = row['stem'].strip()
        if stem in sid_by_stem:
            sid = sid_by_stem[stem]
            kinds.add('table')
            n_by_stem += 1
        elif stem_col:
            # The conversion table HAS a stem column, so it is authoritative.
            # Do NOT quietly fall back to parsing the stem here: with a sweep
            # folder named col_row, parsing reads '0_0'..'0_5' as six replicates
            # of sample 0 and yields a small, plausible-looking, WRONG table.
            # An unmatched stem is a mismatch to report, not a shape to guess.
            unparsed.append(stem)
            continue
        else:
            sid, _rep, kind = parse_stem(stem)
            if sid is None:
                unparsed.append(stem)
                continue
            kinds.add(kind)
        sf_str = row.get('SF', '').strip()
        if sf_str:
            try:
                sf_by_sample[sid].append(float(sf_str))
            except ValueError:
                pass
        if has_fold_col:
            fold_val = row.get('fold', '').strip()
            if fold_val:
                fold_by_sample[sid].add(fold_val)
 
    if sid_by_stem:
        print(f'Matched {n_by_stem}/{len(pore_rows)} image(s) by stem')
        if n_by_stem == 0:
            print()
            print('[ERROR] The conversion table has a "stem" column but NOT ONE '
                  'image stem matched it.')
            print('        These two files describe different folders. Nothing '
                  'was written.')
            print(f'        Stems in {pore_scores_csv}:')
            print(f'          {[r["stem"] for r in pore_rows][:10]}')
            print(f'        Stems in {rename_table_csv}:')
            print(f'          {list(sid_by_stem)[:10]}')
            print('        Re-run the matching build_conversion_table_*.py with '
                  '--data_dir pointing')
            print('        at the SAME folder pore_analysis.py scored.')
            return
        if n_by_stem < len(pore_rows):
            print(f'[WARN] {len(pore_rows) - n_by_stem} image(s) are not in the '
                  f'conversion table and were dropped. The two files may be out '
                  f'of sync.')
    if unparsed:
        print(f'[WARN] {len(unparsed)} stem(s) were not in the conversion table '
              f'and did not match {{Sample_ID}}_{{row}} or sweep_N, so they were '
              f'skipped: {unparsed[:8]}{"..." if len(unparsed) > 8 else ""}')
 
    def _key(x):
        xs = str(x)
        return (0, int(xs)) if xs.lstrip('-').isdigit() else (1, 0)
    sample_ids = sorted(sf_by_sample.keys(), key=_key)
 
    if not sample_ids:
        # This is the "empty output CSV" case. Say why, instead of writing a
        # header-only file and leaving you to work it out.
        print()
        print('[ERROR] No image could be matched, so the output table would be '
              'empty. Nothing was written.')
        print(f'        Stems in {pore_scores_csv}:')
        print(f'          {[r["stem"] for r in pore_rows][:10]}')
        if stem_col:
            print(f'        Stems in {rename_table_csv}:')
            print(f'          {list(sid_by_stem)[:10]}')
            print('        These two lists must overlap. Re-run the matching '
                  'build_conversion_table_*.py')
            print('        with --data_dir pointing at the SAME folder that '
                  'pore_analysis.py scored,')
            print('        so the stems are taken from the images themselves.')
        else:
            print(f'        {rename_table_csv} has no "stem" column, so matching '
                  f'fell back to parsing')
            print('        the stems. Regenerate it with '
                  'build_conversion_table_*.py.')
        return
 
    kind = ('sweep' if kinds == {'sweep'} else
            '48well' if kinds == {'48well'} else
            'conversion table (by stem)' if kinds == {'table'} else 'mixed')
    if kind == 'mixed':
        print('[WARN] This scores file mixes stem types, whose Sample_ID spaces '
              'overlap. Unrelated conditions could be merged into one row. '
              'Score them separately.')
    print(f'Matched via: {kind}')
    print(f'Found {len(sample_ids)} unique Sample_IDs with SF scores: '
          f'{sample_ids}\n')
 
    # ── build final table ──────────────────────────────────────────────────────
    out_rows = []
    missing_params = []
    multi_fold_samples = []
    params_by_sid_from_stem = {}
    for st, sid_ in sid_by_stem.items():
        params_by_sid_from_stem.setdefault(sid_, params_by_stem[st])
 
    for sid in sample_ids:
        sf_vals = sf_by_sample[sid]
        params  = params_by_sid_from_stem.get(sid) or params_by_sample.get(sid)
        if params is None:
            missing_params.append(sid)
            params = {'Pressure_kPa': '', 'NozzleSpeed_mms': '', 'Zoffset_mm': ''}
 
        folds_seen = sorted(fold_by_sample.get(sid, set()))
        if len(folds_seen) > 1:
            multi_fold_samples.append(sid)
        fold_str = ','.join(folds_seen)
 
        sf_arr = np.array(sf_vals)
        out_rows.append({
            'Sample_ID':       sid,
            'Pressure_kPa':    params['Pressure_kPa'],
            'NozzleSpeed_mms': params['NozzleSpeed_mms'],
            'Zoffset_mm':      params['Zoffset_mm'],
            'fold':            fold_str,
            'n_images':        len(sf_vals),
            'SF_mean':         f'{np.mean(sf_arr):.4f}' if len(sf_vals) else '',
            # population std (ddof=0), matching unetplusplus_aggregate.py.
            # A single image gives 0.0000 by definition, not a missing value.
            'SF_std':          f'{np.std(sf_arr):.4f}'  if len(sf_vals) else '',
        })
 
    if missing_params:
        print(f'[WARN] {len(missing_params)} Sample_ID(s) had SF scores but '
              f'no matching row in rename_conversion_table.csv: '
              f'{missing_params}')
    if multi_fold_samples:
        print(f'[WARN] {len(multi_fold_samples)} Sample_ID(s) had replicate '
              f'images split across MORE THAN ONE fold — this should not '
              f'happen with sample-level CV splitting and may indicate a '
              f'data leakage issue: {multi_fold_samples}')
 
    fieldnames = ['Sample_ID', 'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm',
                  'fold', 'n_images', 'SF_mean', 'SF_std']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(out_rows)
 
    print(f'\n── Per-sample SF summary ──')
    header = f'{"Sample":>8}  {"Pressure":>10}  {"Speed":>10}  {"Zoffset":>10}  '\
             f'{"fold":>10}  {"n":>3}  {"SF_mean":>9}  {"SF_std":>9}'
    print(header)
    print('─' * len(header))
    for r in out_rows:
        print(f'{r["Sample_ID"]:>8}  {str(r["Pressure_kPa"]):>10}  '
              f'{str(r["NozzleSpeed_mms"]):>10}  {str(r["Zoffset_mm"]):>10}  '
              f'{r["fold"]:>10}  {r["n_images"]:>3}  {r["SF_mean"]:>9}  '
              f'{r["SF_std"]:>9}')
 
    print(f'\n✓ Final table → {output_csv}')
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build per-sample SF summary table from '
                    'pore_scores_all_folds.csv + rename_conversion_table.csv.'
    )
    parser.add_argument('--pore_scores_csv', required=True)
    parser.add_argument('--rename_table_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--sample_id_col', default=None,
        help='Column name for Sample_ID in rename_conversion_table.csv '
             '(auto-detected if omitted)')
    parser.add_argument('--pressure_col', default=None,
        help='Column name for Pressure_kPa (auto-detected if omitted)')
    parser.add_argument('--speed_col', default=None,
        help='Column name for NozzleSpeed_mms (auto-detected if omitted)')
    parser.add_argument('--zoffset_col', default=None,
        help='Column name for Zoffset_mm (auto-detected if omitted)')
    args = parser.parse_args()
 
    build_table(
        pore_scores_csv=args.pore_scores_csv,
        rename_table_csv=args.rename_table_csv,
        output_csv=args.output_csv,
        sample_id_col=args.sample_id_col,
        pressure_col=args.pressure_col,
        speed_col=args.speed_col,
        zoffset_col=args.zoffset_col,
    )