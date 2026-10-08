"""
run_pipeline.py
---------------
Run the whole SF pipeline for one deployment folder (or all of them) with a
single command, so you only ever type the image folder path.
 
What it runs, in order
----------------------
  48-well branch, on <folder>:
    1. rename_to_sampleids.py       -> <folder>_renamed
    2. pore_analysis.py             -> <folder>_renamed/pore_scores.csv
    3. build_conversion_table_48well.py
                                    -> <folder>_renamed/rename_conversion_table.csv
    4. build_sample_sf_table.py     -> <parent>/<name>_sf_summary_48well.csv
 
  Pressure-sweep branch, on the pressure_sweep* subfolder inside <folder>:
    5. pore_analysis.py             -> <sweep>/pore_scores.csv
    6. build_conversion_table_sweep.py
                                    -> <sweep>/rename_conversion_table_sweep.csv
    7. build_sample_sf_table.py     -> <parent>/<name>_sf_summary_sweep.csv
 
The sweep subfolder is FOUND, not assumed. Its name is not consistent across
your folders (pressure_sweep_gelma_7_60_cells, pressure_sweep_gelma_10_60,
pressure_sweep_gelma_10_80_cells_tri..., and so on), so hardcoding it in a
batch file would silently break on some folders. Anything matching
`pressure_sweep*` is used; 0 matches skips the sweep branch with a warning,
2+ matches is an error rather than a guess.
 
Preconditions are checked before each pore_analysis call
--------------------------------------------------------
pore_analysis needs {stem}.tif, {stem}-target-mask.png and {stem}-pred-mask.png.
If the target masks or predictions are missing, this stops and names the
upstream script to run (draw_target_geometry*.py / unetplusplus_test.py)
instead of failing three steps later with a confusing message.
 
Usage (Windows cmd, from your project root)
--------------------------------------------
  One folder:
    python run_pipeline.py data\\dev_images\\ml_gelma_bioprinting\\gelma_deployment\\cell_gelma_7_80 ^
        --lhs_csv data\\lhs_gelma\\lhs_bioprint_samples_semicolon.csv ^
        --nc_file data\\lhs_gelma\\pressure_sweap_30-120_step-5.nc ^
        --w 0.2
 
  All six folders under gelma_deployment:
    python run_pipeline.py data\\dev_images\\ml_gelma_bioprinting\\gelma_deployment ^
        --all ^
        --lhs_csv data\\lhs_gelma\\lhs_bioprint_samples_semicolon.csv ^
        --nc_file data\\lhs_gelma\\pressure_sweap_30-120_step-5.nc ^
        --w 0.2
 
  See what would run without running it:  add --dry_run
  Re-run scoring without re-copying files: add --skip_rename
 
On Linux/HPC the same command works with forward slashes.
"""
 
import argparse
import subprocess
import sys
from pathlib import Path
 
 
RENAME_CANDIDATES = ['rename_to_sampleids.py', 'rename_to_sample_ids.py',
                     'rename_to_sample-ids.py']
NEEDED_SCRIPTS = ['pore_analysis.py', 'build_conversion_table_48well.py',
                  'build_conversion_table_sweep.py', 'build_sample_sf_table.py']
 
 
def find_rename_script(scripts_dir):
    for name in RENAME_CANDIDATES:
        p = scripts_dir / name
        if p.exists():
            return p
    return None
 
 
def run(cmd, dry_run, label):
    printable = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
    print(f'\n--- {label} ---')
    print(f'    {printable}')
    if dry_run:
        return True
    res = subprocess.run([str(c) for c in cmd])
    if res.returncode != 0:
        print(f'[FAIL] {label} exited with code {res.returncode}')
        return False
    return True
 
 
def count_triplets(folder):
    """(n_tif, n_target_masks, n_pred_masks) of real image stems in a folder."""
    folder = Path(folder)
    stems = set()
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in folder.glob(ext):
            if any(t in p.stem.lower() for t in
                   ('mask', 'overlay', 'visible', 'pred', 'target')):
                continue
            stems.add(p.stem)
    n_tgt = sum(1 for s in stems if (folder / f'{s}-target-mask.png').exists())
    n_prd = sum(1 for s in stems if (folder / f'{s}-pred-mask.png').exists())
    return len(stems), n_tgt, n_prd
 
 
def check_inputs(folder, label, sweep):
    """Fail early and say which upstream script is missing, not 'no files found'."""
    n_tif, n_tgt, n_prd = count_triplets(folder)
    print(f'    {label}: {n_tif} image(s), {n_tgt} target mask(s), '
          f'{n_prd} pred mask(s)')
    if n_tif == 0:
        print(f'[STOP] No .tif images in {folder}')
        return False
    ok = True
    if n_tgt < n_tif:
        script = ('draw_target_geometry_pressure-sweep.py' if sweep
                  else 'draw_target_geometry.py')
        print(f'[STOP] {n_tif - n_tgt} image(s) have no -target-mask.png. '
              f'Run {script} on this folder first.')
        ok = False
    if n_prd < n_tif:
        print(f'[STOP] {n_tif - n_prd} image(s) have no -pred-mask.png. '
              f'Run unetplusplus_test.py on this folder first.')
        ok = False
    return ok
 
 
def find_sweep_dir(folder):
    cands = sorted(p for p in folder.glob('pressure_sweep*') if p.is_dir())
    if len(cands) == 1:
        return cands[0], None
    if not cands:
        return None, f'no pressure_sweep* subfolder inside {folder.name}'
    return None, (f'{len(cands)} pressure_sweep* subfolders inside '
                  f'{folder.name}: {[c.name for c in cands]}. Expected one.')
 
 
def process_folder(folder, args, scripts, py):
    folder = Path(folder).resolve()
    parent = folder.parent
    name = folder.name
    renamed = parent / f'{name}_renamed'
 
    print('=' * 72)
    print(f'  {name}')
    print('=' * 72)
 
    out48 = parent / f'{name}_sf_summary_48well.csv'
    outsw = parent / f'{name}_sf_summary_sweep.csv'
    made = []
 
    # ---------------------------------------------------------- 48-well branch
    if not args.skip_48well:
        if args.skip_rename and renamed.exists():
            print(f'\n--- 1/4 rename (skipped, {renamed.name} exists) ---')
        else:
            if not run([py, scripts['rename'], folder, args.lhs_csv],
                       args.dry_run, '1/4 rename to Sample_IDs'):
                return made, False
 
        if not args.dry_run:
            if not renamed.exists():
                print(f'[STOP] {renamed} was not created by the rename step.')
                return made, False
            print(f'\n    checking {renamed.name} ...')
            if not check_inputs(renamed, 'renamed', sweep=False):
                return made, False
 
        conv48 = renamed / 'rename_conversion_table.csv'
        scores48 = renamed / 'pore_scores.csv'
 
        steps = [
            ([py, scripts['pore'], '--data_dir', renamed,
              '--output_dir', renamed, '--w', args.w], '2/4 pore analysis'),
            ([py, scripts['conv48'], '--lhs_csv', args.lhs_csv,
              '--data_dir', renamed, '--output_csv', conv48],
             '3/4 conversion table'),
            ([py, scripts['sf'], '--pore_scores_csv', scores48,
              '--rename_table_csv', conv48, '--output_csv', out48],
             '4/4 SF summary (48-well)'),
        ]
        for cmd, label in steps:
            if not run(cmd, args.dry_run, label):
                return made, False
        made.append(out48)
 
    # ------------------------------------------------------------ sweep branch
    if not args.skip_sweep:
        sweep_dir, err = find_sweep_dir(folder)
        if sweep_dir is None:
            print(f'\n[WARN] Sweep branch skipped: {err}')
        else:
            print(f'\n    sweep folder: {sweep_dir.name}')
            if not args.dry_run and not check_inputs(sweep_dir, 'sweep', sweep=True):
                return made, False
 
            convsw = sweep_dir / 'rename_conversion_table_sweep.csv'
            scoressw = sweep_dir / 'pore_scores.csv'
            steps = [
                ([py, scripts['pore'], '--data_dir', sweep_dir,
                  '--output_dir', sweep_dir, '--w', args.w], '1/3 pore analysis (sweep)'),
                ([py, scripts['convsw'], '--nc_file', args.nc_file,
                  '--data_dir', sweep_dir, '--output_csv', convsw],
                 '2/3 conversion table (sweep)'),
                ([py, scripts['sf'], '--pore_scores_csv', scoressw,
                  '--rename_table_csv', convsw, '--output_csv', outsw],
                 '3/3 SF summary (sweep)'),
            ]
            for cmd, label in steps:
                if not run(cmd, args.dry_run, label):
                    return made, False
            made.append(outsw)
 
    return made, True
 
 
def main(args):
    root = Path(args.target).resolve()
    if not root.is_dir():
        sys.exit(f'Not a directory: {root}')
 
    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir \
        else Path(__file__).resolve().parent
 
    rename_path = find_rename_script(scripts_dir)
    if rename_path is None and not args.skip_48well:
        sys.exit(f'Could not find the rename script in {scripts_dir}. '
                 f'Looked for: {", ".join(RENAME_CANDIDATES)}')
    missing = [s for s in NEEDED_SCRIPTS if not (scripts_dir / s).exists()]
    if missing:
        sys.exit(f'Missing script(s) in {scripts_dir}: {missing}\n'
                 f'Pass --scripts_dir if they live elsewhere.')
 
    scripts = {
        'rename': rename_path,
        'pore':   scripts_dir / 'pore_analysis.py',
        'conv48': scripts_dir / 'build_conversion_table_48well.py',
        'convsw': scripts_dir / 'build_conversion_table_sweep.py',
        'sf':     scripts_dir / 'build_sample_sf_table.py',
    }
    for p in (args.lhs_csv, args.nc_file):
        if p and not Path(p).exists():
            sys.exit(f'File not found: {p}')
 
    if args.all:
        folders = sorted(d for d in root.iterdir()
                         if d.is_dir() and not d.name.endswith('_renamed'))
        if not folders:
            sys.exit(f'No subfolders to process under {root}')
    else:
        folders = [root]
 
    print(f'Scripts : {scripts_dir}')
    print(f'Python  : {sys.executable}')
    print(f'Folders : {len(folders)}  -> {[f.name for f in folders]}')
    print(f'w       : {args.w}')
    if args.dry_run:
        print('DRY RUN: commands are printed, nothing is executed.')
 
    py = sys.executable
    results = []
    for folder in folders:
        made, ok = process_folder(folder, args, scripts, py)
        results.append((folder.name, made, ok))
 
    print('\n' + '=' * 72)
    print('  SUMMARY')
    print('=' * 72)
    n_ok = 0
    for name, made, ok in results:
        status = 'OK  ' if ok else 'FAIL'
        if ok:
            n_ok += 1
        print(f'  [{status}] {name}')
        for m in made:
            print(f'           {m.name}')
    print(f'\n{n_ok}/{len(results)} folder(s) completed.')
    if n_ok != len(results):
        sys.exit(1)
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Run the full SF pipeline for a deployment folder.')
    ap.add_argument('target',
                    help='Image folder (e.g. ...\\gelma_deployment\\cell_gelma_7_80), '
                         'or the parent folder together with --all')
    ap.add_argument('--all', action='store_true',
                    help='Treat `target` as the parent and process every subfolder')
    ap.add_argument('--lhs_csv', required=True,
                    help='lhs_bioprint_samples_semicolon.csv')
    ap.add_argument('--nc_file', required=True,
                    help='Pressure-sweep .nc G-code file')
    ap.add_argument('--w', default='0.2',
                    help='Pore-bonus weight passed to pore_analysis.py (default: 0.2)')
    ap.add_argument('--scripts_dir', default=None,
                    help='Where the pipeline scripts live (default: next to this file)')
    ap.add_argument('--skip_rename', action='store_true',
                    help='Skip the rename step when <folder>_renamed already exists')
    ap.add_argument('--skip_48well', action='store_true',
                    help='Run only the pressure-sweep branch')
    ap.add_argument('--skip_sweep', action='store_true',
                    help='Run only the 48-well branch')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print every command without running it')
    main(ap.parse_args())