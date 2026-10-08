#!/usr/bin/env python3
"""
nc_imaging.py
-------------
Loads a baseline R-GEN 100 NC print file and appends G-code camera
imaging positions for selected wells on a 48-well plate.

Camera is mounted at slot 5. Positions are recorded with slot 1 selected.
Z=2.24 is confirmed imaging height. Z=65.34 is confirmed safe travel height.

Plate layout: 8 columns (1-8), 6 rows (A-F)
  Corning 48-well, center-to-center 13.08 mm

Usage examples:
  python nc_imaging.py file.nc A1 B3 C4        # specific wells
  python nc_imaging.py file.nc row1             # full row A (A1-A8)
  python nc_imaging.py file.nc col1             # full column 1 (A1-F1)
  python nc_imaging.py file.nc col1 col2 col3   # multiple columns
  python nc_imaging.py file.nc all_cols         # all 8 columns in order (col1->col8)
  python nc_imaging.py file.nc all_rows         # all 6 rows in order (row1->row6)
  python nc_imaging.py file.nc all              # all 48 wells row by row
"""

import sys
import os
import re

# -----------------------------------------------------------------------------
# WELL COORDINATE MAP
# Derived by interpolating from 4 confirmed corner positions (slot 1 selected,
# camera physically at slot 5 over the well):
#   A1: X=-234.090, Y= 95.780
#   A8: X=-143.150, Y= 95.780
#   F1: X=-234.090, Y= 30.630
#   F8: X=-143.150, Y= 30.640
#
# X step per column = (-143.150 - (-234.090)) / 7 = +12.991 mm
# Y step per row    = ( 30.630  -   95.780  ) / 5 = -13.030 mm
# Z imaging height  = 2.24 mm   (confirmed)
# Z safe travel     = 65.34 mm  (confirmed)
# -----------------------------------------------------------------------------

ROWS = ['A', 'B', 'C', 'D', 'E', 'F']
COLS = [1, 2, 3, 4, 5, 6, 7, 8]

# Confirmed corner anchors (slot 1 selected, camera at slot 5 over well)
# X_A1, Y_A1 = -234.090,  95.780
# X_A8, Y_A8 = -143.150,  95.780
# X_F1, Y_F1 = -234.090,  30.630
# X_F8, Y_F8 = -143.150,  30.640

# X_A1, Y_A1 = -251.080,  50.950
# X_A8, Y_A8 = -159.220,  50.950
# X_F1, Y_F1 = -251.080,  -13.990
# X_F8, Y_F8 = -159.220,  -13.990

X_A1, Y_A1 = -250.160,  53.250
X_A8, Y_A8 = -158.200,  53.250
X_F1, Y_F1 = -250.160,  -11.790
X_F8, Y_F8 = -158.200,  -11.790

X_STEP = (X_A8 - X_A1) / 7   # per column: +12.991 mm
Y_STEP = (Y_F1 - Y_A1) / 5   # per row:    -13.030 mm

# Z_IMAGING = 2.24
# Z_SAFE    = 65.34

# Z_IMAGING = 10.00
# Z_SAFE    = 30.00

Z_IMAGING = 20.00
Z_SAFE    = 40.00

N_ROWS = len(ROWS)   # 6
N_COLS = len(COLS)   # 8


def build_well_map():
    """Build dict of well_name -> (X, Y, Z) for all 48 wells."""
    wells = {}
    for r_idx, row in enumerate(ROWS):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells


WELL_MAP = build_well_map()


def parse_targets(args):
    """
    Parse input arguments into an ordered list of well names, no duplicates.
    Accepts:
      - Well addresses : A1, B3, F8       (case-insensitive)
      - Row selectors  : row1 .. row6     (row1=A, ..., row6=F)
      - Col selectors  : col1 .. col8     (iterates A->F within that column)
      - 'all_cols'     : col1 then col2 ... col8  (column-major order)
      - 'all_rows'     : row1 then row2 ... row6  (row-major order)
      - 'all'          : all 48 wells in row-major order
    """
    selected = []
    seen = set()

    def add(well):
        w = well.upper()
        if w in WELL_MAP and w not in seen:
            selected.append(w)
            seen.add(w)

    def add_col(col_num):
        """Add all wells in a column, top to bottom (A->F)."""
        for r in ROWS:
            add(f"{r}{col_num}")

    def add_row(row_idx):
        """Add all wells in a row, left to right (col1->col8)."""
        row_letter = ROWS[row_idx]
        for c in COLS:
            add(f"{row_letter}{c}")

    for arg in args:
        arg_lower = arg.lower().strip()

        if arg_lower == 'all_cols':
            # Column-major: col1 (A1->F1), col2 (A2->F2), ..., col8 (A8->F8)
            for c in COLS:
                add_col(c)

        elif arg_lower == 'all_rows':
            # Row-major: row1 (A1->A8), row2 (B1->B8), ..., row6 (F1->F8)
            for r_idx in range(N_ROWS):
                add_row(r_idx)

        elif arg_lower == 'all':
            # Row-major order
            for r_idx in range(N_ROWS):
                add_row(r_idx)

        elif re.match(r'^row[1-6]$', arg_lower):
            row_idx = int(arg_lower[3]) - 1      # row1 -> 0 -> 'A'
            add_row(row_idx)

        elif re.match(r'^col[1-8]$', arg_lower):
            col_num = int(arg_lower[3])
            add_col(col_num)

        elif re.match(r'^[a-fA-F][1-8]$', arg):
            add(arg)

        else:
            print(f"  [WARNING] Unrecognised argument '{arg}' -- skipped.")
            print(f"            Valid: A1-F8, row1-row6, col1-col8, all_cols, all_rows, all")

    return selected


def generate_imaging_gcode(well_names):
    """Generate the G-code block for camera imaging of given wells."""
    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Slot 1 selected (T1) to match coordinate frame of recorded")
    lines.append("; positions. Camera physically mounted at slot 5.")
    lines.append(f"; Z safe travel = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1 (matches recorded coordinate frame)")
    lines.append("G803                       ; Move to system safe height")
    lines.append("")

    for well in well_names:
        x, y, z = WELL_MAP[well]
        lines.append(f"; --- Well {well} ---")
        lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Raise to safe Z first")
        lines.append(f"G00 X{x:.3f} Y{y:.3f}              ; Move XY to {well} camera position")
        lines.append(f"G00 Z{z:.3f}                ; Lower to imaging height")
        lines.append(f'V.E.UserInteraction.Message = "Camera at {well} - trigger imaging, then click OK"')
        lines.append(f"M121                       ; Pause for manual camera trigger")
        lines.append("")

    lines.append("; --- Return home ---")
    lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Safe Z before going home")
    lines.append("G800                       ; Go home")
    lines.append("M110=1000")
    lines.append("M30")

    return "\n".join(lines)


def strip_end_commands(nc_content):
    """
    Remove trailing G800 / M110=1000 / M30 from the print file
    so we can append the imaging block with its own ending.
    """
    lines = nc_content.splitlines()
    remove_patterns = re.compile(r'^\s*(G800|M110=1000|M30)\s*(;.*)?$')
    while lines and (lines[-1].strip() == '' or remove_patterns.match(lines[-1])):
        lines.pop()
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    nc_arg = sys.argv[1]
    target_args = sys.argv[2:]

    # Always resolve input file from nc_files/ subfolder
    NC_DIR = "nc_files_examples"
    nc_file = os.path.join(NC_DIR, os.path.basename(nc_arg))

    if not os.path.isfile(nc_file):
        print(f"[ERROR] File not found: {nc_file}")
        print(f"  Make sure the file exists in the '{NC_DIR}/' subfolder.")
        sys.exit(1)

    with open(nc_file, 'r') as f:
        nc_content = f.read()

    # Parse targets
    well_names = parse_targets(target_args)

    if not well_names:
        print("[ERROR] No valid wells selected. Check your input arguments.")
        print("  Valid wells    : A1-F8")
        print("  Row selectors  : row1 (=A) .. row6 (=F)")
        print("  Col selectors  : col1 .. col8")
        print("  Shortcuts      : all_cols, all_rows, all")
        sys.exit(1)

    print(f"\n  Source file   : {nc_file}")
    print(f"  Wells to image: {', '.join(well_names)}")

    # Strip end commands from print file
    nc_stripped = strip_end_commands(nc_content)

    # Generate imaging G-code
    imaging_block = generate_imaging_gcode(well_names)

    # Combine
    output_content = nc_stripped + "\n" + imaging_block + "\n"

    # Build output filename -- saved in same nc_files/ subfolder
    basename = os.path.basename(nc_file)
    base, ext = os.path.splitext(basename)
    output_file = os.path.join(NC_DIR, f"{base}_imaging{ext}")

    with open(output_file, 'w') as f:
        f.write(output_content)

    print(f"  Output file   : {output_file}")
    print(f"  Done. {len(well_names)} well(s) appended.\n")


if __name__ == "__main__":
    main()
