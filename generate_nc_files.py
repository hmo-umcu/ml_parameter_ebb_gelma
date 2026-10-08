#!/usr/bin/env python3
"""
generate_nc_files.py
--------------------
Reads the LHS sample CSV and generates NC files for a 48-well plate.
Each NC file contains 8 columns (= 8 samples), each column has 6 constructs (rows A-F).

Grouping strategy:
  - Samples are processed in CSV order (sequential batches of N_COLS)
  - Each NC file contains a mixed set of pressure/speed/z-offset values
  - Temperature is excluded from the optimization space; set manually in Architect UI

NC file naming:
  data_collection_48well_8cols_s{first}-s{last}.nc

48-well plate layout:
  Columns 1-8  → 8 samples (one per column)
  Rows    A-F  → 6 constructs per sample

Parameters updated per column:
  M200  : pressure    (kPa  → value * 10, unit 0.1 kPa)
  F     : nozzle speed (mm/s)
  Z     : z-offset    (mm)

Usage:
  python generate_nc_files.py

Expects:
  - Template NC file : (same folder as this script) ai_poc_48_template.nc
  - CSV file         : data/lhs/lhs_bioprint_samples_semicolon.csv

Output:
  data/nc_files/   (created automatically if not existing)
"""

import os
import re
import math
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent                              # script folder
TEMPLATE_FILE = BASE_DIR / "ai_poc_48_template.nc"                # template NC
CSV_FILE      = BASE_DIR / "data" / "lhs" / "lhs_bioprint_samples_semicolon.csv"
OUTPUT_DIR    = BASE_DIR / "data" / "nc_files"                    # output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                     # create if not existing

# ── 48-well plate geometry ────────────────────────────────────────────────────
ROWS_ALPHA = ['A', 'B', 'C', 'D', 'E', 'F']
COLS       = [1, 2, 3, 4, 5, 6, 7, 8]
N_COLS     = 8   # samples per NC file = columns per plate

Z_IMAGING  = 2.24
Z_SAFE     = 22.340

X_A1, Y_A1 = -234.090,  95.780
X_A8        = -143.150
X_F1        = -234.090
Y_F1        =  30.630
X_STEP = (X_A8 - X_A1) / 7   # +12.991 mm per column
Y_STEP = (Y_F1 - Y_A1) / 5   # -13.030 mm per row

def build_well_map():
    wells = {}
    for r_idx, row in enumerate(ROWS_ALPHA):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells

WELL_MAP = build_well_map()

# ── Column G805 X-values in the 48-well template ─────────────────────────────
COL_X = {
    1: -45.785,
    2: -32.705,
    3: -19.625,
    4:  -6.545,
    5:   6.535,
    6:  19.615,
    7:  32.695,
    8:  45.775,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Parse template into 8 column blocks + header
# ─────────────────────────────────────────────────────────────────────────────

def parse_template(template_path):
    """
    Split the template NC file into:
      - header_lines : everything before the first column block
      - col_blocks   : dict {col_num (1-8): [lines]}
    """
    with open(template_path, 'r') as f:
        raw = f.read()

    lines = raw.splitlines()

    # Find the first G805 line for each column (identified by X value)
    col_start = {}
    for col_num, x_val in COL_X.items():
        pattern = re.compile(
            r'G805\[\s*' + re.escape(f"{x_val:.3f}") + r'\s*,.*?\]'
        )
        for i, line in enumerate(lines):
            if pattern.search(line):
                col_start[col_num] = i
                break

    # Verify all 8 columns found
    missing = [c for c in COLS if c not in col_start]
    if missing:
        raise ValueError(
            f"Could not find column(s) {missing} in template.\n"
            f"Expected G805 X values: { {c: COL_X[c] for c in missing} }\n"
            f"Check that TEMPLATE_FILE='{template_path}' is the correct 48-well NC file."
        )

    sorted_starts = sorted(col_start.items(), key=lambda x: x[1])

    # Header = everything before column 1's first G805
    header_end   = sorted_starts[0][1]
    header_lines = lines[:header_end]

    # Column blocks
    col_blocks = {}
    end_cmds = re.compile(r'^\s*(G800|M110=1000|M30|#FLUSH WAIT)\s*(;.*)?$')
    for idx, (col_num, start) in enumerate(sorted_starts):
        if idx + 1 < len(sorted_starts):
            end = sorted_starts[idx + 1][1]
        else:
            end = len(lines)
        block = lines[start:end]
        # Strip trailing blanks and end commands from last column
        while block and (block[-1].strip() == '' or end_cmds.match(block[-1])):
            block.pop()
        col_blocks[col_num] = block

    return header_lines, col_blocks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Update a column block with new parameters
# ─────────────────────────────────────────────────────────────────────────────

def update_col_block(block_lines, pressure_kpa, speed_mms, z_offset_mm):
    """
    Return updated block lines with new parameter values.

    Strategy: always INJECT M200 and F at correct positions regardless of
    whether they exist in the template block, because columns 2-8 in the
    template do not contain M200 or F lines — they inherit from column 1.
    We must explicitly set all parameters at the start of every column block.

    Temperature (M300/M302) is intentionally omitted — set manually in UI.

    G-code encoding:
      M200 = pressure_kpa * 10   (0.1 kPa units)
      F    = speed_mms           (mm/s directly)
      Z    = z_offset_mm         (only printing Z < 5mm, not travel Z ~18.4)
    """
    m200_val = int(round(pressure_kpa * 10))
    f_val    = f"{speed_mms:.3f}"
    z_val    = f"{z_offset_mm:.3f}"

    # Pass 1: process existing lines
    # - Replace any existing M200, F if present
    # - Strip any M300/M302 lines (temperature not set via G-code)
    # - Update Z values
    new_lines  = []
    found_m200 = False
    found_f    = False

    for line in block_lines:
        stripped = line.strip()

        # Replace existing M200
        if re.match(r'M200=\d+', stripped):
            new_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
            found_m200 = True
            continue

        # Replace existing F
        if re.match(r'F\d+\.\d+', stripped):
            new_lines.append(f"F{f_val}")
            found_f = True
            continue

        # Strip M300/M302 — temperature controlled manually, not via G-code
        if re.match(r'M300=\d+', stripped) or re.match(r'M302', stripped):
            continue

        # Update Z offset (printing only — skip travel Z >= 5mm e.g. Z18.400)
        if re.match(r'Z\d+\.\d+', stripped):
            z_num = float(re.match(r'Z(\d+\.\d+)', stripped).group(1))
            if z_num < 5.0:
                new_lines.append(f"Z{z_val}")
                continue

        new_lines.append(line.rstrip())

    # Pass 2: inject missing M200 and F at the very start of the block
    # (after the G805 origin line, before the first motion command)
    final_lines = []
    params_injected = False   # M200 + F injected flag

    for line in new_lines:
        stripped = line.strip()

        # Inject M200 + F right after the G805 line that opens this column block
        if not params_injected and re.match(r'G805\[', stripped):
            final_lines.append(line.rstrip())
            if not found_m200:
                final_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
            if not found_f:
                final_lines.append(f"F{f_val}")
            params_injected = True
            continue

        final_lines.append(line.rstrip())

    return final_lines



def update_header(header_lines, pressure_kpa):
    """Update M200 in the initialization header."""
    m200_val = int(round(pressure_kpa * 10))
    new_lines = []
    for line in header_lines:
        if re.match(r'\s*M200=\d+', line):
            new_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
        else:
            new_lines.append(line.rstrip())
    return new_lines


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate imaging G-code block (column-major: col1 -> col8)
# ─────────────────────────────────────────────────────────────────────────────

def generate_imaging_block(n_active_cols=8):
    """
    Generate imaging G-code for columns 1 through n_active_cols (A->F per col).
    """
    well_names = []
    for c in range(1, n_active_cols + 1):
        for r in ROWS_ALPHA:
            well_names.append(f"{r}{c}")

    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Column-major order: col1 (A1-F1) -> col2 (A2-F2) -> ...")
    lines.append("; Slot 1 selected (T1) — camera physically at slot 5.")
    lines.append(f"; Z safe travel = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1 (coordinate frame reference)")
    lines.append("G803                       ; Move to system safe height")
    lines.append("")

    for well in well_names:
        x, y, z = WELL_MAP[well]
        lines.append(f"; --- Well {well} ---")
        lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Raise to safe Z first")
        lines.append(f"G00 X{x:.3f} Y{y:.3f}              ; Move XY to {well}")
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Sequential batching (mixed pressure per file)
# ─────────────────────────────────────────────────────────────────────────────

def batch_samples(df):
    """
    Split all samples into batches of N_COLS using stride-based interleaving,
    so that each NC file contains samples spread across the full pressure range.

    Strategy: treat the CSV (sorted by pressure ascending) as a
    n_files × N_COLS grid read column-major. File k gets every n_files-th
    sample starting at index k.

    Example with 96 samples → 12 files of 8:
      File  1: indices  0, 12, 24, 36, 48, 60, 72, 84  (low→high pressure)
      File  2: indices  1, 13, 25, 37, 49, 61, 73, 85
      ...
      File 12: indices 11, 23, 35, 47, 59, 71, 83, 95

    Any remainder (len(df) % N_COLS != 0) lands in a shorter final file.
    """
    n = len(df)
    n_files = math.ceil(n / N_COLS)
    all_samples = [df.iloc[i] for i in range(n)]
    batches = []
    for k in range(n_files):
        batch = [all_samples[k + j * n_files]
                 for j in range(N_COLS)
                 if k + j * n_files < n]
        batches.append(batch)
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not CSV_FILE.is_file():
        print(f"[ERROR] CSV not found: {CSV_FILE}")
        print("  Run lhs_sampling.py first.")
        return

    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from:")
    print(f"  {CSV_FILE}")

    # ── Parse template ────────────────────────────────────────────────────────
    if not TEMPLATE_FILE.is_file():
        print(f"\n[ERROR] Template not found: {TEMPLATE_FILE}")
        print(f"  Place your 48-well template NC file at that path.")
        return

    header_lines, col_blocks = parse_template(TEMPLATE_FILE)
    print(f"\nTemplate parsed: {TEMPLATE_FILE.name}")
    print(f"  Header: {len(header_lines)} lines")
    print(f"  Column blocks: { {c: len(b) for c, b in col_blocks.items()} }")

    # ── Batch samples ─────────────────────────────────────────────────────────
    batches = batch_samples(df)
    print(f"\nBatching into {len(batches)} NC files:")
    for idx, batch in enumerate(batches):
        sids = [int(s.name) for s in batch]
        print(f"  File {idx+1:>2}: {len(batch):>2} cols | sample IDs={sids}")

    # ── Output directory ──────────────────────────────────────────────────────
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # ── Generate NC files ─────────────────────────────────────────────────────
    print()
    for file_idx, batch in enumerate(batches):
        samples   = batch
        n_samples = len(samples)

        sids  = [int(s.name) for s in samples]

        fname = (f"data_collection_48well_8cols"
                 f"_s{sids[0]}-s{sids[-1]}.nc")
        fpath = OUTPUT_DIR / fname

        # ── Header ────────────────────────────────────────────────────────────
        first_pressure  = int(samples[0]['Pressure_kPa'])
        updated_header  = update_header(header_lines, first_pressure)

        # File-level comment block
        file_comments = [f"% {fname}"]
        file_comments.append(f"; Generated by generate_nc_files.py")
        file_comments.append(f"; 48-well plate | {n_samples} samples (columns)")
        file_comments.append(f"; CSV: {CSV_FILE.name}")
        file_comments.append(f"; Temperature: set manually in Architect UI (not in G-code)")
        file_comments.append(f";")
        file_comments.append(f"; {'Col':<5} {'SampleID':<10} {'P(kPa)':<10} "
                              f"{'F(mm/s)':<10} {'Z(mm)':<8}")
        for col_num, s in enumerate(samples, start=1):
            sid = int(s.name)
            p   = int(s['Pressure_kPa'])
            f_s = float(s['NozzleSpeed_mms'])
            z   = float(s['Zoffset_mm'])
            file_comments.append(
                f"; {col_num:<5} {sid:<10} {p:<10} {f_s:<10.1f} {z:<8}"
            )
        file_comments.append(f";")

        # Replace first header line (% filename) with comment block
        header_body = updated_header[1:]
        all_lines   = file_comments + header_body

        # ── Column blocks ─────────────────────────────────────────────────────
        for col_num in COLS:
            if col_num <= n_samples:
                sample   = samples[col_num - 1]
                pressure = float(sample['Pressure_kPa'])
                speed    = float(sample['NozzleSpeed_mms'])
                z_off    = float(sample['Zoffset_mm'])
                sid      = int(sample.name)

                all_lines.append("")
                all_lines.append(
                    f"; ── Col {col_num} | Sample {sid} | "
                    f"P={int(pressure)}kPa  F={speed:.1f}mm/s  Z={z_off:.3f}mm ──"
                )
                updated_block = update_col_block(
                    list(col_blocks[col_num]),
                    pressure, speed, z_off
                )
                all_lines.extend(updated_block)

        # ── Imaging block (active columns only) ───────────────────────────────
        all_lines.append("")
        all_lines.append("#FLUSH WAIT")
        all_lines.append(generate_imaging_block(n_active_cols=n_samples))

        # ── Write file ────────────────────────────────────────────────────────
        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  [{file_idx+1:>2}] {fname}")

    print(f"\nDone. {len(batches)} NC files written to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()