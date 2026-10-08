"""
s08_make_validation_nc.py
-------------------------
Read recommendation_*.csv and emit an .nc that prints the recommended condition
in whichever plate column(s) you choose, then images exactly those wells.
 
NOTHING IS CALCULATED. Every coordinate, every move and every comment is copied
verbatim out of the template you supply. If a requested well is not in the
template, the run stops and tells you which ones are missing. It will not
extrapolate a position, fit a grid, or infer a step size.
 
What is copied verbatim
-----------------------
  * the INITIALIZATION block
  * `G805[x, y, z] ; G55 origin: <well>`  the print origin line for each well
  * the whole print block for each well: G55, the approach, the strand pattern,
    the tail-equalisation moves, the inter-strand lift, the end-of-well lift
  * `; --- Well <w> ---` and its four imaging lines, including the camera XY
  * the imaging preamble and the return-home block
 
What is substituted, and only this
----------------------------------
  M200      = round(P_kPa * 10)
  F         = Speed_mms
  Z<print>  = Z_star                     (whatever the template used, e.g. 0.300)
  Z<lift>   = Z_star + --strand_lift     (the between-strand lift only)
 
The end-of-well lift (Z18.400) and the imaging heights (Z40 safe, Z20 imaging)
are left exactly as the template has them.
 
The print height and the strand lift are read FROM THE TEMPLATE rather than
assumed: whatever Z the template's first well prints at is the one replaced, and
whatever lift follows it is the one shifted. A template printing at Z0.200 and
one printing at Z0.300 both work with no flags.
 
Choosing where to print
-----------------------
  --column       plate column for V1, the ML recommendation
  --hist_column  plate column for V2, the historical best, with --also_historical
                 (this used to be called --column2, which read like "column
                 number 2" rather than "the column for the second condition";
                 the old name still works)
  --start_row    first row of the column, default A
 
A plate map is printed before anything is written, and two conditions are never
allowed into the same wells.
 
Usage
-----
    python s08_make_validation_nc.py \\
        --recommendation_csv results/05_recommendation/recommendation_G.csv \\
        --template data_collection_48well_8cols_s0s28.nc \\
        --out validation_G_col3.nc \\
        --column 3 [--start_row A] [--n_wells 6] \\
        [--also_historical --hist_column 6]
 
Use the FULL 48-well template here, not a pressure-sweep file. A sweep .nc only
contains the wells it swept, so anything outside those columns is simply not in
it and this script will say so rather than invent it.
"""
 
import argparse
import re
from pathlib import Path
 
import pandas as pd
 
ROWS = ['A', 'B', 'C', 'D', 'E', 'F']
 
ORIGIN_RE = re.compile(r'^G805\[.*\]\s*;\s*G55 origin:\s*([A-H]\d+)', re.I)
IMG_WELL_RE = re.compile(r'^;\s*---\s*Well\s+([A-H]\d+)\s*---', re.I)
LIFT_RE = re.compile(r'^G00\s+Z[\d.]+\s*;\s*lift after well\s+([A-H]\d+)', re.I)
M200_RE = re.compile(r'^M200\s*=\s*\d+(.*)$')
F_RE = re.compile(r'^F\d+(?:\.\d+)?\s*$')
CAMERA = 'CAMERA IMAGING POSITIONS'
RETURN_HOME = '; --- Return home ---'
 
 
# =============================================================================
# template parsing: capture blocks, never coordinates
# =============================================================================
def parse_template(path):
    """
    Split the template into reusable verbatim blocks.
 
    Returns a dict of line lists and per-well line lists. No number in the
    template is ever parsed into a float here, because nothing is recomputed
    from it. The lines are moved around, not rebuilt.
    """
    lines = Path(path).read_text(errors='replace').splitlines()
    split = next((i for i, l in enumerate(lines) if CAMERA in l), None)
    if split is None:
        raise SystemExit(f'{path} has no "{CAMERA}" section, so there are no '
                         f'imaging positions to copy.')
    pp, ip = lines[:split], lines[split:]
 
    # ---- INITIALIZATION, verbatim ------------------------------------------
    try:
        i0 = next(i for i, l in enumerate(pp) if l.strip() == '; INITIALIZATION')
        i1 = next(i for i, l in enumerate(pp) if l.strip().startswith('M312'))
    except StopIteration:
        raise SystemExit(f'{path}: could not find the INITIALIZATION block '
                         f'(a "; INITIALIZATION" line and an "M312" line).')
    init = pp[i0:i1 + 1]
 
    # ---- print block per well ----------------------------------------------
    # A block runs from its G805 origin line to its own "lift after well X".
    starts = [(i, ORIGIN_RE.match(l.strip()).group(1).upper())
              for i, l in enumerate(pp) if ORIGIN_RE.match(l.strip())]
    blocks = {}
    for n, (i, w) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(pp)
        end = None
        for j in range(i, stop):
            m = LIFT_RE.match(pp[j].strip())
            if m:
                end = j
                break
        blocks.setdefault(w, pp[i:(end + 1) if end is not None else stop])
 
    # ---- imaging block per well --------------------------------------------
    img_starts = [(i, IMG_WELL_RE.match(l.strip()).group(1).upper())
                  for i, l in enumerate(ip) if IMG_WELL_RE.match(l.strip())]
    if not img_starts:
        raise SystemExit(f'{path}: no "; --- Well X ---" imaging blocks found.')
    home = next((i for i, l in enumerate(ip)
                 if l.strip() == RETURN_HOME.strip()), len(ip))
    img_header = ip[:img_starts[0][0]]
    img_tail = ip[home:]
    img_blocks = {}
    for n, (i, w) in enumerate(img_starts):
        stop = img_starts[n + 1][0] if n + 1 < len(img_starts) else home
        img_blocks.setdefault(w, ip[i:stop])
 
    # ---- the two canonical block shapes, taken from the template ------------
    # A well block is identical for every well except two lines: its G805 origin
    # and its "lift after well X" comment. Everything else (the approach, the
    # strand pattern, the tail-equalisation moves) is written in the G55 frame
    # and so is the same wherever the well is.
    #
    # There are two shapes. The first well of a job carries the tool-change
    # preamble and M151; the rest do not. Both are lifted verbatim from the
    # template, and the per-well origin and lift lines are swapped in from that
    # well's own block. Nothing is composed by hand.
    #
    # This matters because the 48-well template is an LHS plate: every column
    # prints at its own P, F and Z. Taking each well's own block would mean the
    # print height being replaced differs per column, and the substitution
    # silently misses. Using one canonical pair fixes the Z that is being
    # replaced to a single known value.
    first_w = starts[0][1]
    canon_first = blocks[first_w]
    rest_w = next((w for _i, w in starts[1:]
                   if not any('M151' in l for l in blocks[w])), None)
    if rest_w is None:
        raise SystemExit(f'{path}: could not find a well block without the '
                         f'tool-change preamble to use as the repeat shape.')
 
    origin_line = {w: next(l for l in blocks[w] if ORIGIN_RE.match(l.strip()))
                   for w in blocks}
    lift_line = {}
    for w, blk in blocks.items():
        m = next((l for l in blk if LIFT_RE.match(l.strip())), None)
        if m is not None:
            lift_line[w] = m
 
    return {'init': init, 'blocks': blocks, 'img_header': img_header,
            'img_blocks': img_blocks, 'img_tail': img_tail,
            'canon_first': canon_first, 'canon_rest': blocks[rest_w],
            'origin_line': origin_line, 'lift_line': lift_line,
            'first_well': first_w, 'rest_well': rest_w}
 
 
def build_well(T, well, is_first):
    """
    One well's block: a canonical shape with THAT well's own origin and lift
    lines swapped in. Every line comes from the template.
    """
    shape = T['canon_first'] if is_first else T['canon_rest']
    out = []
    for l in shape:
        if ORIGIN_RE.match(l.strip()):
            out.append(T['origin_line'][well])
        elif LIFT_RE.match(l.strip()):
            out.append(T['lift_line'].get(well, l))
        else:
            out.append(l)
    return out
 
 
def template_z(block):
    """
    The print height and the between-strand lift AS WRITTEN in the template.
 
    Read rather than assumed, so a template printing at Z0.300 works the same as
    one printing at Z0.200. The print Z is the first bare Z after the approach;
    the strand lift is the next larger Z that is not the end-of-well lift.
    """
    zs = []
    for l in block:
        s = l.strip()
        m = re.match(r'^(?:G0?0\s+)?Z(\d+\.\d+)', s)
        if m and 'lift after well' not in s:
            zs.append((float(m.group(1)), s))
    if not zs:
        raise SystemExit('No Z move found in the template well block.')
    zp = min(v for v, _ in zs)
    highs = sorted({v for v, _ in zs if v > zp})
    zl = highs[0] if highs else None
    return zp, zl
 
 
# =============================================================================
# substitution: pressure, feedrate, print height. Nothing else.
# =============================================================================
def apply_condition(block, P, Fs, Z, z_print_tpl, z_lift_tpl, lift_new):
    out = []
    m200 = int(round(P * 10))
    for l in block:
        s = l.strip()
        m = M200_RE.match(s)
        if m:
            tail = ' ; pressure %gkPa' % P if m.group(1).strip() else ''
            out.append(f'M200={m200}{tail}')
            continue
        if F_RE.match(s):
            out.append(f'F{Fs:.3f}')
            continue
        if 'lift after well' not in s:
            mz = re.match(r'^((?:G0?0\s+)?)Z(\d+\.\d+)(.*)$', s)
            if mz:
                v = float(mz.group(2))
                if abs(v - z_print_tpl) < 1e-9:
                    out.append(f'{mz.group(1)}Z{Z:.3f}{mz.group(3)}')
                    continue
                if z_lift_tpl is not None and abs(v - z_lift_tpl) < 1e-9:
                    out.append(f'{mz.group(1)}Z{lift_new:.3f}{mz.group(3)}')
                    continue
        out.append(l)
    return out
 
 
def plate_map(assignments, n_cols, title):
    marks = {}
    for sym, wells in assignments:
        for w in wells:
            marks[w] = sym
    print(f'\n  {title}')
    print('      ' + '  '.join(f'{c:>2}' for c in range(1, n_cols + 1)))
    for r in ROWS:
        print(f'   {r}  ' + '  '.join(
            f'{marks.get(f"{r}{c}", "."):>2}' for c in range(1, n_cols + 1)))
    for sym, wells in assignments:
        print(f'      {sym} = {wells[0]}-{wells[-1]}  ({len(wells)} wells)')
 
 
def main(a):
    rec = pd.read_csv(a.recommendation_csv, sep=';').iloc[0]
    P, Fs, Z = float(rec['P_star']), float(rec['Speed_star']), float(rec['Z_star'])
    T = parse_template(a.template)
    print(f'Template : {Path(a.template).name}')
    print(f'  {len(T["blocks"])} print block(s), {len(T["img_blocks"])} imaging '
          f'block(s)')
    print(f'  block shapes copied from {T["first_well"]} (with tool change) and '
          f'{T["rest_well"]} (repeat)')
    have_cols = sorted({int(w[1:]) for w in T['blocks']})
    print(f'  columns available for printing: {have_cols}')
 
    # ---- where on the plate -------------------------------------------------
    hist_col = a.hist_column if a.hist_column is not None else a.column2
    start = a.start_row.strip().upper()
    if start not in ROWS:
        raise SystemExit(f'--start_row must be one of {ROWS}, got {a.start_row!r}')
    r0 = ROWS.index(start)
    if r0 + a.n_wells > len(ROWS):
        raise SystemExit(f'--start_row {start} with --n_wells {a.n_wells} runs '
                         f'past row {ROWS[-1]}: the most you can fit from '
                         f'{start} is {len(ROWS) - r0}.')
 
    def wells_of(col):
        return [f'{ROWS[r0 + k]}{col}' for k in range(a.n_wells)]
 
    conds = [(a.column, P, Fs, Z, f'V1 ML recommendation ({rec["category"]})')]
    if a.also_historical:
        if hist_col == a.column:
            raise SystemExit(
                f'--column and --hist_column are both {a.column}, so V1 and V2 '
                f'would go into the\nsame wells. --hist_column defaults to 2, so '
                f'this happens if you set --column 2\nand forget it. Give them '
                f'different columns.')
        hp = str(rec['historical_best_params']).split('/')
        conds.append((hist_col, float(hp[0]), float(hp[1]), float(hp[2]),
                      'V2 historical best LHS'))
 
    # ---- every requested well must EXIST in the template --------------------
    wanted = [w for col, *_ in conds for w in wells_of(col)]
    miss_p = [w for w in wanted if w not in T['blocks']]
    miss_i = [w for w in wanted if w not in T['img_blocks']]
    if miss_p or miss_i:
        print(f'\n[ERROR] the template does not contain every requested well, '
              f'and this script\n        does not invent positions.')
        if miss_p:
            print(f'        missing print block(s)  : {miss_p}')
        if miss_i:
            print(f'        missing imaging block(s): {miss_i}')
        print(f'\n  Printable columns in {Path(a.template).name}: {have_cols}')
        print(f'  Imaging columns   : '
              f'{sorted({int(w[1:]) for w in T["img_blocks"]})}')
        print('\n  Use a template that covers the whole plate, or choose a '
              'column it contains.')
        raise SystemExit(1)
 
    plate_map([(f'{i + 1}', wells_of(c)) for i, (c, *_) in enumerate(conds)],
              a.plate_cols,
              f'plate layout ({a.n_wells} well(s) per column, rows '
              f'{start}-{ROWS[r0 + a.n_wells - 1]})')
 
    z_print_tpl, z_lift_tpl = template_z(T['canon_first'])
    print(f'\n  template prints at Z{z_print_tpl:.3f} with a strand lift of '
          f'Z{z_lift_tpl:.3f}' if z_lift_tpl else
          f'\n  template prints at Z{z_print_tpl:.3f}')
    print(f'  substituting Z{Z:.3f} and lift Z{Z + a.strand_lift:.3f}')
 
    # ---- build the print body from the template's own blocks ----------------
    body, all_wells = [], []
    for i, (col, p, f, z, lab) in enumerate(conds):
        body.append(f'; -- {lab}: col {col}  P={p:g} kPa  F={f:g} mm/s  '
                    f'Z={z:.3f} mm --')
        for j, w in enumerate(wells_of(col)):
            blk = build_well(T, w, is_first=(i == 0 and j == 0))
            body += apply_condition(blk, p, f, z, z_print_tpl, z_lift_tpl,
                                    z + a.strand_lift)
            all_wells.append(w)
        body.append('')
        print(f'  column {col} ({wells_of(col)[0]}-{wells_of(col)[-1]}): '
              f'P={p:g} kPa  F={f:g} mm/s  Z={z:g} mm  -> M200={int(round(p * 10))}')
 
    # ---- imaging, verbatim --------------------------------------------------
    img = []
    for l in T['img_header']:
        if l.strip().startswith('; Wells:'):
            img.append('; Wells: ' + ', '.join(all_wells))
        else:
            img.append(l)
    for w in all_wells:
        img += T['img_blocks'][w]
    img += T['img_tail']
 
    head = [f'% {Path(a.out).name}',
            '; Generated by s08_make_validation_nc.py',
            f'; Template: {Path(a.template).name}  (all geometry copied verbatim)',
            f'; Recommendation: {Path(a.recommendation_csv).name}', ';']
    for col, p, f, z, lab in conds:
        ws = wells_of(col)
        head.append(f'; Col {col}  {ws[0]}-{ws[-1]}   P={p:g} kPa  F={f:g} mm/s  '
                    f'Z={z:g} mm   {lab}')
    head += [';', f'; Model: {rec.get("model", "unrecorded")}',
             f'; Predicted SF_mean: {rec.get("pred_SF_mean", "")} '
             f'+/- {rec.get("pred_std", "")} ({rec.get("uncertainty_source", "")})',
             f'; Trained on: {rec.get("train_categories", "")}',
             '; Target SF values were NOT used to choose this condition.', ';']
 
    init = apply_condition(T['init'], conds[0][1], conds[0][2], Z,
                           z_print_tpl, z_lift_tpl, Z + a.strand_lift)
 
    nc = head + init + [''] + body + img
    Path(a.out).write_text('\n'.join(nc) + '\n')
    print(f'\nWrote {len(nc)} lines, {len(all_wells)} well(s) -> {a.out}')
    print(f'  print  : {len(conds)} column(s) x {a.n_wells} replicate(s)')
    print(f'  imaging: {len(all_wells)} well(s), blocks copied from the template')
    print('\nEvery coordinate came from the template. Only M200, F and the print '
          'Z were changed.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build a validation .nc by reusing a full-plate template.')
    ap.add_argument('--recommendation_csv', required=True)
    ap.add_argument('--template', required=True,
                    help='A FULL 48-well .nc. Every well you select must exist '
                         'in it; nothing is extrapolated.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--column', type=int, default=1,
                    help='Plate column for V1, the ML recommendation (default: 1)')
    ap.add_argument('--start_row', default='A',
                    help='First row of the column, A..F (default: A)')
    ap.add_argument('--n_wells', type=int, default=6,
                    help='Replicates, i.e. wells down the column (default: 6)')
    ap.add_argument('--also_historical', action='store_true',
                    help='Add a second column at the historical-best condition')
    ap.add_argument('--hist_column', type=int, default=None,
                    help='Plate column for V2. Must differ from --column.')
    ap.add_argument('--column2', type=int, default=2,
                    help='Deprecated name for --hist_column, still honoured')
    ap.add_argument('--plate_cols', type=int, default=8,
                    help='Columns on the plate, for the map only (default: 8)')
    ap.add_argument('--strand_lift', type=float, default=1.0,
                    help='Lift above print height between H and V strands (mm)')
    main(ap.parse_args())