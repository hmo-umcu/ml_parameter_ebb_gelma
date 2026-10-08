"""
s09_make_d5_transfer_nc.py
--------------------------
Turn the D5 output of s07 into one .nc file per condition to be printed.
 
d5_optimum_movement.csv holds ONE ROW PER PAIR, and each row carries TWO
conditions:
 
    x_free_*   the optimum found from the CELL-FREE half of the pair
    x_laden_*  the optimum found from the CELL-LADEN half of the pair
 
BOTH ARE PRINTED IN THE CELL-LADEN INK. The words "free" and "laden" say where
the numbers came from, not what goes in the cartridge. Nothing here is printed
cell-free. That is the point of the test: hold the ink fixed, vary only which
dataset the setting was derived from, and see whether it shows up in SF.
 
    2 pairs  ->  4 .nc files, one condition each, 6 wells (replicates) per file.
 
File naming
-----------
    d5_<pair>_<ink>_c<1|2>_<freeopt|ladenopt>_P<..>_F<..>_Z<..>_col<..>_n<..>.nc
 
e.g.  d5_A-E_cell_gelma_10_60_c1_freeopt_P120_F6_Z0p1_col1_n6.nc
 
Every parameter that distinguishes one run from another is in the filename, so a
plate photo plus a filename is a complete record.
 
How the template is used
------------------------
Nothing geometric is recomputed. The script reads your 48-well template and:
 
  * finds every `G805[x, y, z]` print origin and names the wells from the x/y
    grid (smallest x = column 1, largest y = row A), then CROSS-CHECKS that
    naming against the labelled `; --- Well XX ---` imaging blocks. If the two
    disagree it stops rather than printing in the wrong wells.
  * copies the FIRST well block of the template verbatim as the first-well
    pattern (it carries the tool change, G807, M151) and the SECOND well block
    verbatim as the pattern for every later well.
  * copies the imaging preamble, the per-well imaging blocks for exactly the
    wells being printed, and the return-home block, all verbatim.
 
Only four things are substituted:
 
    M200      = round(P_kPa * 10)
    F         = Speed_mms
    Z(print)  = Z_star            -> the LOWEST Z in a well block
    Z(lift)   = Z_star + delta    -> any Z within +2 mm of the print height,
                                     keeping the template's own delta
Z values more than 2 mm above the print height (18.4 mm between wells, the
imaging heights) are absolute travel clearances and are left untouched.
 
Usage
-----
    python s09_make_d5_transfer_nc.py \
        --d5_csv results/07_aim2_transfer/d5_optimum_movement.csv \
        --template ai_poc_48_template.nc \
        --outdir nc_d5_transfer
 
Use a FULL 48-well template. A partial one will not contain the plate columns
this script needs.
"""
 
import argparse
import re
from pathlib import Path
 
import pandas as pd
 
ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
 
G805_RE = re.compile(r'^G805\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')
WELLHDR_RE = re.compile(r'^;\s*---\s*Well\s+([A-H]\d+)\s*---', re.IGNORECASE)
RETHOME_RE = re.compile(r'^;\s*---\s*Return home', re.IGNORECASE)
M200_RE = re.compile(r'^(\s*M200\s*=\s*)(\d+)(.*)$')
F_RE = re.compile(r'^(\s*)F(\d+(?:\.\d+)?)\s*$')
M110_RE = re.compile(r'^(\s*M110\s*=\s*)(\d+)(.*)$')
Z_RE = re.compile(r'(?<![A-Za-z0-9])Z(\d+(?:\.\d+)?)')
 
# Comment-only lines in the template that describe the TEMPLATE's own
# conditions ("; -- Col 1 | Sample 0 | P=10-20kPa F=10.0mm/s Z=0.200mm --",
# "; Pressure sweep: 30-120 kPa ..."). Copying them into a file printed at a
# different pressure would put a lie in the header, so they are dropped.
STALE_RE = re.compile(r'(kpa|mm/s|sample|col\s|\bP\s*=|\bF\s*=|\bZ\s*=|sweep)',
                      re.IGNORECASE)
 
 
def drop_stale(lines, all_comments=False):
    out = []
    for l in lines:
        s = l.strip()
        if s.startswith('%'):
            continue
        if s.startswith(';'):
            if all_comments or STALE_RE.search(s):
                continue
        out.append(l)
    return out
 
# a condition is identified in the CSV by its column prefix
COND_SPEC = [
    ('x_free_',  'freeopt',
     'optimum derived from the CELL-FREE dataset, printed in cell-laden ink'),
    ('x_laden_', 'ladenopt',
     'optimum derived from the CELL-LADEN dataset, printed in cell-laden ink'),
]
 
 
# --------------------------------------------------------------------------
# template parsing
# --------------------------------------------------------------------------
 
def grid_names(coords, tol=1.0):
    """Name wells from a list of (x, y): smallest x = col 1, largest y = row A."""
    def cluster(vals):
        out = []
        for v in sorted(vals):
            if not out or abs(v - out[-1][-1]) > tol:
                out.append([v])
            else:
                out[-1].append(v)
        return [sum(g) / len(g) for g in out]
 
    xs = cluster({x for x, _ in coords})
    ys = cluster({y for _, y in coords})[::-1]        # descending: A at the top
    if len(ys) > len(ROWS):
        raise SystemExit(f'Template has {len(ys)} distinct Y positions; '
                         f'only {len(ROWS)} row letters are defined.')
 
    names = []
    for x, y in coords:
        ci = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
        ri = min(range(len(ys)), key=lambda i: abs(ys[i] - y))
        names.append(f'{ROWS[ri]}{ci + 1}')
    return names, len(xs), len(ys)
 
 
def parse_template(path):
    raw = Path(path).read_text(errors='replace')
    crlf = '\r\n' in raw
    lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')
 
    cam = next((i for i, l in enumerate(lines)
                if 'CAMERA IMAGING POSITIONS' in l.upper()), None)
    if cam is None:
        raise SystemExit(f'{path}: no "CAMERA IMAGING POSITIONS" section found. '
                         f'This script needs a template that both prints and images.')
    print_part, image_part = lines[:cam], lines[cam:]
 
    # ---- print origins and their well blocks
    idx = [i for i, l in enumerate(print_part) if G805_RE.match(l.strip())]
    if len(idx) < 2:
        raise SystemExit(f'{path}: found {len(idx)} G805 print origin(s). '
                         f'At least 2 are needed (first-well and later-well patterns).')
 
    coords, zs = [], []
    for i in idx:
        m = G805_RE.match(print_part[i].strip())
        coords.append((float(m.group(1)), float(m.group(2))))
        zs.append(float(m.group(3)))
    names, ncol, nrow = grid_names(coords)
    if len(set(names)) != len(names):
        raise SystemExit(f'{path}: two print origins mapped to the same well. '
                         f'The origins are not on a regular grid.')
    origins = {n: (c[0], c[1], z) for n, c, z in zip(names, coords, zs)}
 
    # trailer: the tail of the print section that is only blanks/comments/flush
    end = cam
    while end > idx[-1] + 1:
        s = print_part[end - 1].strip()
        if s == '' or s.startswith(';') or s.upper().startswith('#FLUSH'):
            end -= 1
        else:
            break
    trailer = print_part[end:cam]
 
    prologue = print_part[:idx[0]]
    bounds = idx + [end]
    blocks = [print_part[bounds[j] + 1:bounds[j + 1]] for j in range(len(idx))]
    # +1 strips the G805 line itself; the G55 line right after it is kept,
    # because it belongs to the block and is identical in every well.
 
    # ---- imaging
    img_pre, img_wells, img_home, cur, buf = [], {}, [], None, []
    started = False
    for l in image_part:
        s = l.strip()
        if RETHOME_RE.match(s):
            if cur:
                img_wells[cur] = buf
            cur, buf = None, []
            img_home = [l]
            continue
        if img_home:
            img_home.append(l)
            continue
        m = WELLHDR_RE.match(s)
        if m:
            if cur:
                img_wells[cur] = buf
            cur, buf, started = m.group(1).upper(), [l], True
            continue
        if cur is not None:
            buf.append(l)
        elif not started:
            img_pre.append(l)
    if cur:
        img_wells[cur] = buf
 
    if not img_wells:
        raise SystemExit(f'{path}: no "; --- Well XX ---" imaging blocks found.')
    if not img_home:
        raise SystemExit(f'{path}: no "; --- Return home ---" block found.')
 
    # ---- cross-check the two independent well namings
    a, b = set(origins), set(img_wells)
    if a != b:
        raise SystemExit(
            f'{path}: the print origins and the imaging blocks describe '
            f'different wells.\n'
            f'  only in print   : {sorted(a - b)}\n'
            f'  only in imaging : {sorted(b - a)}\n'
            f'Refusing to guess. Use the full 48-well template.')
 
    # drop the leading "; ====" banner lines from the imaging preamble; the
    # header written by this script replaces them.
    while img_pre and (img_pre[0].strip() == ''
                       or img_pre[0].strip().startswith(';')):
        img_pre.pop(0)
 
    return {'crlf': crlf, 'prologue': prologue, 'origins': origins,
            'blocks': blocks, 'names': names, 'trailer': trailer,
            'img_pre': img_pre, 'img_wells': img_wells, 'img_home': img_home,
            'ncol': ncol, 'nrow': nrow}
 
 
# --------------------------------------------------------------------------
# substitution
# --------------------------------------------------------------------------
 
def block_print_z(block):
    """Lowest Z in a well block = the print height."""
    vals = [float(v) for l in block for v in Z_RE.findall(l)]
    if not vals:
        raise SystemExit('A template well block contains no Z move. '
                         'Cannot identify the print height.')
    return min(vals)
 
 
WELLREF_RE = re.compile(r'(well\s+)([A-H]\d+)', re.IGNORECASE)
 
 
def subst(block, P, Fs, Z, z_print_tpl, m110, near=2.0, well=None):
    """Rewrite M200 / F / M110 / print-height Z lines. Everything else verbatim."""
    m200 = int(round(P * 10))
    out = []
    for l in block:
        if well and ';' in l:
            head, _, tail = l.partition(';')
            l = head + ';' + WELLREF_RE.sub(lambda m: m.group(1) + well, tail)
        m = M200_RE.match(l)
        if m:
            out.append(f'{m.group(1)}{m200} ; {P:g} kPa')
            continue
        m = F_RE.match(l)
        if m:
            out.append(f'{m.group(1)}F{Fs:.3f}')
            continue
        m = M110_RE.match(l)
        if m:
            out.append(f'{m.group(1)}{m110}{m.group(3)}')
            continue
 
        def rep(mm):
            v = float(mm.group(1))
            d = v - z_print_tpl
            if -1e-9 <= d <= near:          # print height, or a small lift above it
                return f'Z{Z + d:.3f}'
            return mm.group(0)              # absolute travel clearance: untouched
 
        out.append(Z_RE.sub(rep, l))
    return out
 
 
# --------------------------------------------------------------------------
# csv
# --------------------------------------------------------------------------
 
def numtag(v):
    return f'{float(v):g}'.replace('.', 'p').replace('-', 'm')
 
 
def read_conditions(csv_path, want_pairs):
    df = pd.read_csv(csv_path, sep=';')
    need = ['pair', 'cell_free', 'cell_laden',
            'x_free_P', 'x_free_Speed', 'x_free_Z',
            'x_laden_P', 'x_laden_Speed', 'x_laden_Z']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f'{csv_path} is missing column(s): {missing}\n'
            f'Found: {list(df.columns)}\n'
            f'This reads the d5_optimum_movement.csv written by the CURRENT '
            f's07_aim2_transfer_and_optimum.py. The older s07 wrote '
            f'free_Pressure_kPa / laden_Pressure_kPa instead; re-run s07.')
    if want_pairs:
        keep = {p.strip().upper().replace('-', '/') for p in want_pairs.split(',')}
        df = df[df['pair'].str.upper().isin(keep)]
        if df.empty:
            raise SystemExit(f'No rows left after --pairs {want_pairs}')
 
    out = []
    for _, r in df.iterrows():
        for i, (pre, tag, label) in enumerate(COND_SPEC, start=1):
            out.append({
                'pair': str(r['pair']),
                'cell_free_category': str(r['cell_free']),
                'ink': str(r['cell_laden']),
                'cond': i, 'tag': tag, 'label': label,
                'P_kPa': float(r[f'{pre}P']),
                'Speed_mms': float(r[f'{pre}Speed']),
                'Zoffset_mm': float(r[f'{pre}Z']),
                'pred_SF': (r.get('SF_laden_at_x_free') if i == 1
                            else r.get('SF_laden_at_x_laden')),
                'pred_sd': r.get('SF_laden_at_x_free_sd') if i == 1 else '',
                'transfer_regret': r.get('transfer_regret', ''),
                'noise_floor': r.get('noise_floor', ''),
                'regret_over_noise': r.get('regret_over_noise', ''),
                'verdict': r.get('verdict', ''),
                'edge': r.get('optimum_on_domain_edge', ''),
            })
    return out
 
 
# --------------------------------------------------------------------------
 
def header(c, wells, column, template, d5_csv, out_name):
    h = [f'% {Path(out_name).stem}',
         '; Generated by s09_make_d5_transfer_nc.py',
         f'; Template : {Path(template).name}',
         f'; Source   : {Path(d5_csv).name}   (Part D, step D5)',
         ';',
         '; =============== WHAT TO LOAD ===============',
         f'; INK IN THE CARTRIDGE : {c["ink"]}    <-- CELL-LADEN',
         '; Both conditions of this pair use this same cell-laden ink.',
         '; Nothing in this experiment is printed cell-free.',
         '; ============================================',
         ';',
         f'; Pair         : {c["pair"]}  ({c["cell_free_category"]} -> {c["ink"]})',
         f'; Condition    : {c["cond"]} of 2   [{c["tag"]}]',
         f'; Meaning      : {c["label"]}',
         f'; Derived from : '
         + (c['cell_free_category'] if c['cond'] == 1 else c['ink'])
         + ' 48-well data',
         ';',
         f'; Pressure     : {c["P_kPa"]:g} kPa  -> M200={int(round(c["P_kPa"] * 10))}',
         f'; Nozzle speed : {c["Speed_mms"]:g} mm/s -> F',
         f'; Z offset     : {c["Zoffset_mm"]:g} mm',
         f'; Wells        : column {column}, {wells[0]}-{wells[-1]} '
         f'({len(wells)} replicates)',
         ';',
         f'; Predicted SF on the cell-laden surface : {c["pred_SF"]}'
         + (f' +/- {c["pred_sd"]}' if str(c['pred_sd']).strip() else ''),
         f'; Transfer regret for this pair          : {c["transfer_regret"]} SF '
         f'({c["regret_over_noise"]}x noise floor {c["noise_floor"]})',
         f'; D5 verdict                             : {c["verdict"]}',
         ';']
    edge = str(c['edge'] or '').strip()
    if edge and edge.lower() != 'nan':
        h += [f'; [CAUTION] an optimum for this pair sits on the design-box edge '
              f'in {edge}.',
              ';           The true optimum may lie outside the swept range, so a',
              ';           tie between the two conditions is partly forced by the box.',
              ';']
    h += ['; Temperature: set manually in Architect UI (not in G-code)',
          '; Target SF was NOT used to choose the cond-1 setting.',
          ';']
    return h
 
 
def main(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
 
    cols = [int(x) for x in a.columns.split(',')]
    if len(cols) != len(COND_SPEC):
        raise SystemExit(f'--columns needs {len(COND_SPEC)} values, got {cols}')
 
    conds = read_conditions(a.d5_csv, a.pairs)
    T = parse_template(a.template)
    z_tpl = block_print_z(T['blocks'][0])
    print(f'Template: {len(T["origins"])} wells ({T["ncol"]} col x {T["nrow"]} row), '
          f'print height {z_tpl:g} mm, {len(T["blocks"][0])}/{len(T["blocks"][1])} '
          f'lines in the first/later well pattern')
 
    for col in set(cols):
        if col > T['ncol']:
            raise SystemExit(f'Template only covers {T["ncol"]} plate column(s); '
                             f'column {col} was requested.')
    if a.n_wells > T['nrow']:
        raise SystemExit(f'Template only covers {T["nrow"]} row(s); '
                         f'--n_wells {a.n_wells} was requested.')
 
    by_pair = {}
    for c in conds:
        by_pair.setdefault(c['pair'], []).append(c)
    for p, cs in by_pair.items():
        if len(cs) == 2 and all(cs[0][k] == cs[1][k] for k in
                                ('P_kPa', 'Speed_mms', 'Zoffset_mm')):
            print(f'  [WARN] pair {p}: cond 1 and cond 2 are the SAME setting, so '
                  f'the two files\n         print identical columns. That is 12 '
                  f'replicates of one condition,\n         not a comparison.')
 
    nl = '\r\n' if T['crlf'] else '\n'
    manifest = []
 
    for c in conds:
        column = cols[c['cond'] - 1]
        wells = [f'{ROWS[k]}{column}' for k in range(a.n_wells)]
        P, Fs, Z = c['P_kPa'], c['Speed_mms'], c['Zoffset_mm']
 
        name = (f'd5_{c["pair"].replace("/", "-")}_{c["ink"]}'
                f'_c{c["cond"]}_{c["tag"]}'
                f'_P{numtag(P)}_F{numtag(Fs)}_Z{numtag(Z)}'
                f'_col{column}_n{a.n_wells}.nc')
 
        body = []
        for j, w in enumerate(wells):
            x, y, z = T['origins'][w]
            src = drop_stale(T['blocks'][0] if j == 0 else T['blocks'][1])
            m110 = int(round(1000 * (j + 1) / (a.n_wells + 1)))
            body.append(f'; -- well {w} | replicate {j + 1}/{a.n_wells} | '
                        f'P={P:g} kPa  F={Fs:g} mm/s  Z={Z:.3f} mm --')
            body.append(f'G805[{x:.3f}, {y:.3f}, {z:.3f}] ; G55 origin: {w}')
            body += subst(src, P, Fs, Z, z_tpl, m110, a.lift_window, well=w)
 
        img = list(T['img_pre'])
        for w in wells:
            img += T['img_wells'][w]
        img += T['img_home']
 
        prologue = subst(drop_stale(T['prologue'], all_comments=True),
                         P, Fs, Z, z_tpl, 0, a.lift_window)
 
        nc = (header(c, wells, column, a.template, a.d5_csv, name)
              + prologue + body + list(T['trailer'])
              + [f'; CAMERA IMAGING POSITIONS',
                 f'; Wells: {", ".join(wells)}',
                 f'; {len(wells)} replicates of P={P:g} kPa F={Fs:g} mm/s '
                 f'Z={Z:g} mm in {c["ink"]}',
                 ';']
              + img)
 
        (outdir / name).write_text(nl.join(nc) + nl)
        print(f'  {name}')
        print(f'      ink {c["ink"]} | P={P:g} F={Fs:g} Z={Z:g} '
              f'| M200={int(round(P * 10))} | wells {wells[0]}-{wells[-1]} '
              f'| {len(nc)} lines')
 
        manifest.append({
            'nc_file': name, 'pair': c['pair'], 'ink_printed': c['ink'],
            'ink_is_cell_laden': 'yes', 'cond': c['cond'],
            'derivation': c['tag'],
            'derived_from': (c['cell_free_category'] if c['cond'] == 1
                             else c['ink']),
            'Pressure_kPa': P, 'NozzleSpeed_mms': Fs, 'Zoffset_mm': Z,
            'M200': int(round(P * 10)), 'column': column,
            'wells': ','.join(wells), 'n_replicates': a.n_wells,
            'pred_SF_on_laden_surface': c['pred_SF'],
            'transfer_regret': c['transfer_regret'],
            'noise_floor': c['noise_floor'],
            'regret_over_noise': c['regret_over_noise'],
            'verdict': c['verdict'],
            'printed_date': '', 'plate_id': '', 'ink_batch': '', 'operator': '',
            'measured_SF_mean': '', 'measured_SF_std': '', 'notes': '',
        })
 
    mf = outdir / 'd5_print_manifest.csv'
    pd.DataFrame(manifest).to_csv(mf, sep=';', index=False)
    print(f'\nWrote {len(manifest)} .nc file(s) -> {outdir}')
    print(f'Manifest with blank wet-lab log columns -> {mf}')
 
    print('\nRun order:')
    for p, cs in by_pair.items():
        print(f'  pair {p}: load {cs[0]["ink"]} (CELL-LADEN). '
              f'Same plate, same ink batch, same session.')
        for c in cs:
            print(f'      cond {c["cond"]}  col {cols[c["cond"] - 1]}  '
                  f'P={c["P_kPa"]:g} F={c["Speed_mms"]:g} Z={c["Zoffset_mm"]:g}  '
                  f'[{c["tag"]}]')
    print('\nBoth conditions of a pair must come from the SAME ink batch on the')
    print('SAME plate. Split across plates or days, batch effects and the transfer')
    print('effect are confounded and the test is void.')
    print('\nBefore running on the machine, diff one generated file against the')
    print('template and confirm only M200, F, M110 and the print-height Z changed.')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='One .nc per D5 condition, all printed in cell-laden ink.')
    ap.add_argument('--d5_csv', required=True,
                    help='results/07_aim2_transfer/d5_optimum_movement.csv')
    ap.add_argument('--template', required=True,
                    help='A full 48-well .nc that both prints and images')
    ap.add_argument('--outdir', default='nc_d5_transfer')
    ap.add_argument('--pairs', default=None,
                    help='Restrict to some pairs, e.g. "A/E" or "A/E,B/F"')
    ap.add_argument('--n_wells', type=int, default=6,
                    help='Replicates = wells down the column (default: 6)')
    ap.add_argument('--columns', default='1,2',
                    help='Plate columns for cond 1 and cond 2 (default: 1,2)')
    ap.add_argument('--lift_window', type=float, default=2.0,
                    help='A Z within this many mm above the print height is '
                         'treated as a strand lift and shifted with it; '
                         'anything higher is an absolute clearance (default: 2.0)')
    main(ap.parse_args())