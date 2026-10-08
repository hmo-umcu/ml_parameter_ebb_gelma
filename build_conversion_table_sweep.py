"""
build_conversion_table_sweep.py
--------------------------------
Build rename_conversion_table.csv for a pressure-sweep folder (sweep_0.tif ...
sweep_18.tif) by reading the print parameters straight out of the .nc G-code.
 
Why parse the G-code instead of assuming a formula
---------------------------------------------------
The sweep is nominally "30-120 kPa in 5 kPa steps", and for
pressure_sweap_30120_step5.nc that formula does hold exactly (P = 30 + 5*N,
verified against all 19 wells). But the human-readable comments in that same
file disagree with the machine codes:
 
    ; -- Col 1 | Sample 0  | P=10-20kPa  ...   <- wrong, wells are 30-55 kPa
    ; -- Col 2 | Sample 7  | P=55kPa     ...   <- wrong, wells are 60-85 kPa
    ; -- Col 3 | Sample 14 | P=85kPa     ...   <- wrong, wells are 90-115 kPa
    ; -- Col 4 | Sample 21 | P=115kPa    ...   <- wrong, well is 120 kPa
 
Those comments are stale generator output. The M200 codes the printer actually
executes are the ground truth, so this script reads those (M200=300 -> 30.0 kPa,
i.e. value/10), plus the F feedrate and the print-height Z, per well. If you
ever run a sweep with a different range or step, this keeps working without
edits.
 
Well order is column-major (A1..F1, A2..F2, A3..F3, A4), which is the order the
images are numbered: sweep_N is the Nth well printed.
 
Usage
-----
    python build_conversion_table_sweep.py \
        --nc_file    pressure_sweap_30120_step5.nc \
        --output_csv rename_conversion_table_sweep.csv \
        [--data_dir  .../pressure_sweep_cell_gelma_7_60] \
        [--stem_prefix sweep]
 
Several sweeps, several NC files
--------------------------------
The sweep range is not the same for every category: some plates were printed
30-120 kPa (19 wells) and later ones 30-140 kPa (23 wells). One hardcoded
--nc_file therefore fails on half the folders, with the misleading message
"23 image(s) on disk but 19 well(s) in the NC".
 
So --nc_file may be given MORE THAN ONCE, and --nc_dir takes a folder of them.
Every candidate is parsed, and the one whose well count matches the number of
images in --data_dir is selected. If none match, or several do, the script
lists what it found and refuses rather than guessing.
 
Do not trust the filename. The file shipped as
`pressure_sweap_30120_step5.nc` actually contains 23 wells at 30-140 kPa: its
name is stale. Selection is by parsed well count and the printed range, never
by name.
 
Output columns (;-separated)
----------------------------
    Sample_ID;row;stem;well;Pressure_kPa;NozzleSpeed_mms;Zoffset_mm
 
Sample_ID is the sweep index N (each sweep well is its own condition, so unlike
the 48-well plates there is one image per sample, not six replicates).
"""
 
import argparse
import csv
import re
from pathlib import Path
 
 
def natural_key(stem):
    """Sort 'sweep_2' before 'sweep_10', and '0_5' before '1_0'."""
    parts = re.findall(r'\d+', stem)
    return ([int(x) for x in parts], stem)
 
 
def scan_stems(data_dir):
    """Real image stems in a folder, excluding anything the pipeline generated."""
    data_dir = Path(data_dir)
    stems = set()
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in data_dir.glob(ext):
            if any(t in p.stem.lower() for t in
                   ('mask', 'overlay', 'visible', 'pred', 'target')):
                continue
            stems.add(p.stem)
    return sorted(stems, key=natural_key)
 
 
FIELDNAMES = ['Sample_ID', 'row', 'stem', 'well',
              'Pressure_kPa', 'NozzleSpeed_mms', 'Zoffset_mm']
 
ORIGIN_RE = re.compile(r'G55\s+origin:\s*([A-H]\d+)', re.IGNORECASE)
M200_RE   = re.compile(r'^\s*M200\s*=\s*(\d+)')
F_RE      = re.compile(r'^\s*F(\d+(?:\.\d+)?)\s*$')
Z_RE      = re.compile(r'^\s*(?:G0?0\s+)?Z(\d+\.\d+)\s*(?:;.*)?$')
 
# the imaging block at the end of the file also moves in Z; stop before it
END_MARKER = 'CAMERA IMAGING POSITIONS'
 
# Z moves are also used for lifts between wells (Z1.2, Z18.4, Z20, Z40).
# The print height is the small one, so ignore anything above this.
MAX_PRINT_Z_MM = 1.0
 
 
def parse_nc(nc_path):
    """Return a list of dicts, one per printed well, in print order."""
    text = Path(nc_path).read_text(errors='replace')
    body = text.split(END_MARKER)[0]
 
    wells, cur = [], None
    for line in body.splitlines():
        m = ORIGIN_RE.search(line)
        if m:
            if cur is not None:
                wells.append(cur)
            cur = {'well': m.group(1).upper(), 'P': None, 'F': None, 'Z': None}
            continue
        if cur is None:
            continue
 
        m = M200_RE.match(line)
        if m and cur['P'] is None:
            cur['P'] = int(m.group(1)) / 10.0        # M200=300 -> 30.0 kPa
            continue
        m = F_RE.match(line)
        if m and cur['F'] is None:
            cur['F'] = float(m.group(1))
            continue
        m = Z_RE.match(line)
        if m and cur['Z'] is None:
            z = float(m.group(1))
            if z <= MAX_PRINT_Z_MM:
                cur['Z'] = z
    if cur is not None:
        wells.append(cur)
    return wells
 
 
def fmt(v):
    """Trim trailing .0 so 30.0 prints as 30, matching the LHS CSV style."""
    if v is None:
        return ''
    return str(int(v)) if float(v).is_integer() else f'{v:g}'
 
 
def collect_nc_files(nc_files, nc_dir, no_recursive=False):
    """Every candidate NC path, de-duplicated, in a stable order."""
    out = []
    for f in (nc_files or []):
        p = Path(f)
        if not p.exists():
            raise SystemExit(f'NC file not found: {p}')
        out.append(p)
    if nc_dir:
        d = Path(nc_dir)
        if not d.is_dir():
            raise SystemExit(f'--nc_dir is not a directory: {d}')
        # RECURSIVE by default. Asking a person to locate and copy the right
        # .nc into one blessed folder is a step that can be got wrong, and was:
        # two files with near-identical names (pressure_sweap_30-120_step-5.nc
        # with 19 wells, pressure_sweap_30120_step5.nc with 23) sat in
        # different folders and the wrong one was the one being found. Point
        # --nc_dir at the project root and every sweep file under it is a
        # candidate; the right one is still chosen by well count.
        it = d.iterdir() if no_recursive else d.rglob('*')
        # case-insensitive, so a file saved as .NC is not silently invisible
        found = sorted({q for q in it
                        if q.is_file() and q.suffix.lower() == '.nc'})
        where = 'in' if no_recursive else 'under'
        print(f'--nc_dir {where} {d}: {len(found)} .nc file(s) found')
        for q in found:
            try:
                rel = q.relative_to(d)
            except ValueError:
                rel = q
            print(f'    {rel}')
        out.extend(found)
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    if not uniq:
        raise SystemExit('No NC file given. Pass --nc_file (repeatable) '
                         'or --nc_dir.')
    return uniq
 
 
ROW_LETTERS = 'ABCDEFGH'
IDX_RE     = re.compile(r'^(.*?)(\d+)$')
COLROW_RE  = re.compile(r'^(\d+)_(\d+)$')
 
 
def well_to_colrow(label):
    """'C2' -> (col 2, row index 2). Column-major print order uses this."""
    m = re.match(r'^([A-H])(\d+)$', label.upper())
    if not m:
        return None
    return int(m.group(2)), ROW_LETTERS.index(m.group(1))
 
 
def check_image_coverage(stems, wells):
    """
    Verify the images cover the printed wells with NO GAPS.
 
    The image-to-well mapping is by ORDER, so a missing image in the middle
    shifts every well after it onto the wrong pressure, silently. Equal counts
    are not enough on their own: a folder can hold the right NUMBER of images
    and still have a hole plus a stray, and order-based mapping would then be
    wrong from the hole onwards.
 
    Two naming conventions appear in these folders:
      sweep_N   the Nth well printed, so indices must be a gapless 0..len-1
      col_row   0-based (column, row), checked against the NC's own well
                labels, which is exact rather than inferred
 
    Returns (ok: bool, lines: list[str]).
    """
    n = len(wells)
 
    # --- col_row, checked against the NC's actual well labels ---------------
    if stems and all(COLROW_RE.match(s) for s in stems):
        want = set()
        for w in wells:
            cr = well_to_colrow(w['well'])
            if cr is None:
                return True, ['[NOTE] well labels are not A1-style; '
                              'coverage not verified']
            want.add((cr[0] - 1, cr[1]))          # NC cols are 1-based
        have = {tuple(int(x) for x in COLROW_RE.match(s).groups()) for s in stems}
        missing = sorted(want - have)
        extra   = sorted(have - want)
        lines = []
        if missing:
            lbl = [f'{c}_{r} ({ROW_LETTERS[r]}{c + 1})' for c, r in missing]
            lines.append(f'[ERROR] {len(missing)} printed well(s) have no '
                         f'image: {lbl}')
        if extra:
            lines.append(f'[ERROR] {len(extra)} image(s) match no printed '
                         f'well: {[f"{c}_{r}" for c, r in extra]}')
        return (not lines), lines
 
    # --- <prefix>N -----------------------------------------------------------
    ms = [IDX_RE.match(s) for s in stems]
    if stems and all(ms):
        prefixes = {m.group(1) for m in ms}
        if len(prefixes) == 1:
            idx = sorted(int(m.group(2)) for m in ms)
            want = set(range(n))
            have = set(idx)
            missing = sorted(want - have)
            extra   = sorted(have - want)
            lines = []
            if missing:
                lines.append(f'[ERROR] index gap: {len(missing)} of the {n} '
                             f'printed wells have no image. Missing '
                             f'{list(prefixes)[0]}N for N = {missing}')
            if extra:
                lines.append(f'[ERROR] {len(extra)} image(s) index outside '
                             f'0..{n - 1}: {extra}')
            return (not lines), lines
 
    return True, [f'[NOTE] stem naming not recognised '
                  f'({stems[:3]}...), so gap checking was skipped. '
                  f'The order-based mapping is only as good as the sort order.']
 
 
def filename_disagrees(path, wells):
    """
    Warn when the digits in the filename do not match the parsed content.
 
    Filenames in this project have gone stale in both directions: a file named
    `..._30-120_...` can hold a 30-140 sweep. Nothing in this script depends on
    the name, but a log line saying "selected pressure_sweap_30-120_step-5.nc"
    reads as though a 30-120 sweep was used, so the disagreement is called out
    explicitly rather than left to be misread.
 
    Returns a message, or None when the name is consistent or carries no
    plausible pressure at all.
    """
    ps = [w['P'] for w in wells if w['P'] is not None]
    if not ps:
        return None
    lo, hi = int(min(ps)), int(max(ps))
    nums = [int(n) for n in re.findall(r'\d+', path.stem)]
    if hi in nums:
        return None
    # only complain about numbers that could plausibly BE an upper pressure
    stale = [n for n in nums if 40 <= n <= 400 and n != lo]
    if not stale:
        return None
    return (f'filename mentions {stale}, but the file actually contains '
            f'{fmt(lo)}-{fmt(hi)} kPa. The name is stale; the content is what '
            f'is used.')
 
 
def describe(path, wells):
    ps = [w['P'] for w in wells if w['P'] is not None]
    rng = f'{fmt(ps[0])}-{fmt(ps[-1])} kPa' if ps else 'no M200 codes'
    return f'{path.name}: {len(wells)} well(s), {rng}'
 
 
def select_nc(candidates, n_images):
    """
    Pick the NC whose well count equals the image count.
 
    Selection is by PARSED WELL COUNT, never by filename. Filenames in this
    project have gone stale (a file named 30120 holding a 30-140 sweep), and a
    name-based choice would silently attach the wrong pressures to every well.
    """
    parsed = [(p, parse_nc(p)) for p in candidates]
    parsed = [(p, w) for p, w in parsed if w]
    if not parsed:
        raise SystemExit('None of the candidate NC files contained printed '
                         'wells. Expected lines like '
                         '"G805[...] ; G55 origin: A1".')
 
    if len(parsed) == 1 and n_images is None:
        return parsed[0]
 
    if n_images is None:
        raise SystemExit(
            f'{len(parsed)} candidate NC file(s) but no --data_dir, so there '
            f'is no image count to match against:\n  ' +
            '\n  '.join(describe(p, w) for p, w in parsed))
 
    hits = [(p, w) for p, w in parsed if len(w) == n_images]
    if len(hits) == 1:
        if len(parsed) > 1:
            print(f'Selected {hits[0][0].name}: its {len(hits[0][1])} wells '
                  f'match the {n_images} image(s) on disk.')
            w = filename_disagrees(*hits[0])
            if w:
                print(f'  [WARN] {w}')
            for p, w in parsed:
                if p is not hits[0][0]:
                    print(f'  (not {describe(p, w)})')
            print()
        return hits[0]
 
    print(f'\n[ERROR] {n_images} image(s) on disk, and NONE of the '
          f'{len(parsed)} candidate NC file(s) has that many wells:')
    for p, w in parsed:
        mark = '  <-- matches' if len(w) == n_images else ''
        print(f'    {describe(p, w)}{mark}')
        d = filename_disagrees(p, w)
        if d:
            print(f'      [WARN] {d}')
    if not hits:
        if len(parsed) == 1:
            # By far the most common case, and the one the previous wording
            # obscured: only one NC was offered at all, so there was never a
            # choice to make. The file for THIS sweep range is simply not in
            # the folder that was searched.
            raise SystemExit(
                f'Only ONE candidate was offered, so there was nothing to '
                f'choose between.\n'
                f'The .nc for a {n_images}-well sweep is not in the folder '
                f'searched above.\n\n'
                f'Fix it by putting that file into your --nc_dir folder, or by '
                f'adding\n'
                f'--nc_file <path to it>. Selection is by parsed well count, so '
                f'the\nfilename does not matter, but a name that states the '
                f'real range helps.\n\n'
                f'If you believe the right .nc IS in that folder, check that it '
                f'ends in .nc\nand that it really contains {n_images} '
                f'"G55 origin:" wells.')
        raise SystemExit(
            f'None of the {len(parsed)} candidates has {n_images} wells. '
            f'Either the image folder\nand the NC files belong to different '
            f'runs, or some sweep images are missing.')
    # Several files with the right well count are only ambiguous if they
    # actually differ. Duplicated copies of the same sweep are common once the
    # search is recursive, and refusing on those would be pure obstruction.
    def signature(wells):
        return tuple((w['P'], w['F'], w['Z']) for w in wells)
 
    sigs = {signature(w) for _p, w in hits}
    if len(sigs) == 1:
        print(f'{len(hits)} copies of the same sweep found; they are identical '
              f'in pressure, feedrate and height, so it does not matter which '
              f'is used.')
        print(f'Using {hits[0][0]}')
        return hits[0]
    raise SystemExit(
        f'{len(hits)} NC files have {n_images} wells but they are NOT the '
        f'same sweep.\nPass exactly one --nc_file to say which you mean.')
 
 
def main(args):
    allow_gaps = args.allow_gaps
    candidates = collect_nc_files(args.nc_file, args.nc_dir,
                                  args.no_recursive)
 
    # The image count decides which NC is used, so it has to be known first.
    stems = None
    if args.data_dir:
        stems = scan_stems(args.data_dir)
        print(f'Folder : {args.data_dir}  ({len(stems)} image(s))')
        if not stems:
            raise SystemExit(f'No images found in {args.data_dir}. Expected .tif '
                             f'files; check the path.')
    if len(candidates) > 1:
        print(f'{len(candidates)} candidate NC file(s) offered')
 
    nc_path, wells = select_nc(candidates, len(stems) if stems is not None else None)
    args.nc_file = str(nc_path)
 
    incomplete = [w for w in wells if w['P'] is None]
    if incomplete:
        print(f'[WARN] {len(incomplete)} well(s) had no M200 pressure code and '
              f'will have a blank Pressure_kPa: {[w["well"] for w in incomplete]}')
 
    print(f'NC file: {nc_path}')
    print(f'         {len(wells)} printed well(s), column-major order')
    warn = filename_disagrees(nc_path, wells)
    if warn:
        print(f'[WARN]   {warn}')
    # sanity: is it a clean arithmetic sweep? report, do not assume
    ps = [w['P'] for w in wells if w['P'] is not None]
    if len(ps) >= 2:
        steps = {round(b - a, 6) for a, b in zip(ps, ps[1:])}
        if len(steps) == 1:
            print(f'\nUniform pressure step of {steps.pop():g} kPa, '
                  f'{fmt(ps[0])} to {fmt(ps[-1])} kPa.')
        else:
            print(f'\n[NOTE] Pressure step is not uniform: {sorted(steps)}. '
                  f'Values are taken from the M200 codes regardless.')
 
    # ---------------------------------------------------------------- stems
    # The image stems are taken from the folder and matched to wells BY ORDER,
    # not by assuming they are called sweep_0..sweep_N. Sweep folders are not
    # named consistently across runs (sweep_0.tif in some, col_row.tif in
    # others), and a name-based match silently produced an empty result table
    # when the guess was wrong. Print order is column-major, which is also
    # natural-sort order for both conventions, so the i-th image is the i-th
    # well in this NC.
    if stems is not None:
        if len(stems) != len(wells):
            # select_nc already refused when several candidates were offered;
            # this is the single-candidate case.
            print(f'\n[ERROR] {len(stems)} image(s) on disk but {len(wells)} '
                  f'well(s) in {nc_path.name}. They must match one-to-one for '
                  f'the order-based mapping to be valid.')
            print(f'        images: {stems}')
            print(f'        wells : {[w["well"] for w in wells]}')
            raise SystemExit(
                'Refusing to guess the mapping.\n'
                'If this category was printed with a different sweep range, '
                'pass every candidate\n'
                'with repeated --nc_file, or point --nc_dir at the folder '
                'holding them, and the\n'
                'right one is chosen by well count.')
        ok, lines = check_image_coverage(stems, wells)
        for ln in lines:
            print(f'  {ln}')
        if not ok and not allow_gaps:
            raise SystemExit(
                '\nRefusing to write a conversion table for a folder with '
                'gaps.\n'
                'The mapping is by print order, so a missing image shifts '
                'every well after it\n'
                'onto the wrong pressure, and the resulting sweep curve would '
                'be wrong without\n'
                'looking wrong. Re-image the missing wells, or pass '
                '--allow_gaps if you\n'
                'accept that the pressures after the first gap may be '
                'misattributed.')
        if not ok:
            print('  [WARN] --allow_gaps given: proceeding despite the gap '
                  'above. Pressures after\n         the first gap may be '
                  'attached to the wrong wells.')
        print(f'\n{"image stem":<16} {"well":<5} {"P(kPa)":>7} {"F(mm/s)":>8} {"Z(mm)":>6}')
        for st, wl in zip(stems, wells):
            print(f'{st:<16} {wl["well"]:<5} {fmt(wl["P"]):>7} '
                  f'{fmt(wl["F"]):>8} {fmt(wl["Z"]):>6}')
        print('\nCheck this mapping. If the image order does not match the print '
              'order,\nthe pressures are attached to the wrong wells.')
    else:
        stems = [f'{args.stem_prefix}_{i}' for i in range(len(wells))]
        print(f'\nNo --data_dir given: assuming stems {stems[0]} .. {stems[-1]}. '
              f'Pass --data_dir to\nmap against the images actually present.')
 
    rows = []
    for i, (st, wl) in enumerate(zip(stems, wells)):
        rows.append({
            'Sample_ID':       i,
            'row':             0,      # one image per sweep condition
            'stem':            st,
            'well':            wl['well'],
            'Pressure_kPa':    fmt(wl['P']),
            'NozzleSpeed_mms': fmt(wl['F']),
            'Zoffset_mm':      fmt(wl['Z']),
        })
 
    with open(args.output_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
        wtr.writeheader()
        wtr.writerows(rows)
 
    print(f'\nWrote {len(rows)} row(s) -> {args.output_csv}')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build a conversion table for a pressure-sweep folder from its .nc file.')
    ap.add_argument('--nc_file', action='append', default=None,
                    help='A pressure-sweep .nc G-code file. May be given more '
                         'than once; the one whose well count matches the '
                         'number of images in --data_dir is used.')
    ap.add_argument('--nc_dir', default=None,
                    help='Folder of candidate .nc files, selected the same way. '
                         'Use this when different categories were printed with '
                         'different sweep ranges.')
    ap.add_argument('--output_csv', required=True,
                    help='Where to write the conversion table')
    ap.add_argument('--data_dir', default=None,
                    help='Sweep image folder; if given, cross-checks that every '
                         'well in the NC has an image and vice versa')
    ap.add_argument('--no_recursive', action='store_true',
                    help='Only look directly inside --nc_dir instead of '
                         'searching it recursively')
    ap.add_argument('--allow_gaps', action='store_true',
                    help='Write the table even when some printed wells have no '
                         'image. The order-based mapping is then unreliable '
                         'after the first gap.')
    ap.add_argument('--stem_prefix', default='sweep',
                    help='Filename stem prefix (default: sweep, i.e. sweep_0.tif)')
    main(ap.parse_args())