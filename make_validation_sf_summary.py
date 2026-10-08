#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


# ============================================================
# General regex for floating-point/integer numbers
# ============================================================

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


# ============================================================
# Utility
# ============================================================

def unique_numeric(values, ndigits=8):
    """
    Convert values to floats and return unique values,
    preserving order.
    """
    result = []

    for value in values:
        value = round(float(value), ndigits)

        if value not in result:
            result.append(value)

    return result


# ============================================================
# NC PARAMETER EXTRACTION
# ============================================================

def extract_from_print_blocks(text):
    """
    Preferred method.

    Extract P/F/Z from the actual printing commands around M151.

    Typical structure:

        M200=1200
        F6.000
        ...
        M151
        Z0.100
        ...
        M160

    M200 uses units of 0.1 kPa:
        M200=1200 -> 120 kPa

    We inspect all M151 printing blocks.
    If every replicate uses the same P/F/Z combination,
    return that condition.
    """

    lines = text.splitlines()

    conditions = []

    for i, line in enumerate(lines):

        if not re.search(r"\bM151\b", line, flags=re.IGNORECASE):
            continue

        # ----------------------------------------------------
        # Search backwards for current pressure and feed rate
        # ----------------------------------------------------

        before = lines[max(0, i - 30): i + 1]

        pressure = None
        speed = None
        zoffset = None

        # -----------------------------
        # Pressure: nearest M200 before M151
        # -----------------------------

        for candidate in reversed(before):

            match = re.search(
                rf"\bM200\s*=\s*({NUM})",
                candidate,
                flags=re.IGNORECASE,
            )

            if match:
                pressure = float(match.group(1)) / 10.0
                break

        # -----------------------------
        # Speed: nearest F before M151
        #
        # Handles:
        # F5.000
        #
        # and also something like:
        # G01 F5.000 X...
        # -----------------------------

        for candidate in reversed(before):

            match = re.search(
                rf"(?<![A-Za-z0-9_])F\s*({NUM})(?![A-Za-z0-9_])",
                candidate,
                flags=re.IGNORECASE,
            )

            if match:
                speed = float(match.group(1))
                break

        # -----------------------------
        # Z-offset:
        #
        # Search after M151 but before M160.
        #
        # This avoids accidentally taking lifting commands
        # such as:
        #
        # G00 Z18.400
        #
        # Instead it gets the real print height:
        #
        # Z0.100
        # -----------------------------

        after = lines[i: min(len(lines), i + 25)]

        for candidate in after:

            # Once extrusion starts, stop searching
            if re.search(
                r"\bM160\b",
                candidate,
                flags=re.IGNORECASE,
            ):
                break

            match = re.search(
                rf"(?<![A-Za-z0-9_])Z\s*({NUM})(?![A-Za-z0-9_])",
                candidate,
                flags=re.IGNORECASE,
            )

            if match:
                zoffset = float(match.group(1))
                break

        if (
            pressure is not None
            and speed is not None
            and zoffset is not None
        ):
            condition = (
                round(pressure, 8),
                round(speed, 8),
                round(zoffset, 8),
            )

            if condition not in conditions:
                conditions.append(condition)

    # --------------------------------------------------------
    # One NC file should represent one printing condition
    # repeated over the wells.
    # --------------------------------------------------------

    if len(conditions) == 1:
        return conditions[0]

    if len(conditions) > 1:
        raise ValueError(
            "Multiple different printing conditions were "
            f"found in M151 printing blocks: {conditions}"
        )

    return None


def extract_from_unique_gcode(text):
    """
    Second method.

    If print-block detection was not possible, inspect
    executable commands globally.

    Works when there is exactly one unique:
        M200
        F
        standalone Z

    condition in the entire NC file.
    """

    # --------------------------------------------------------
    # Pressure
    # M200=1200 -> 120 kPa
    # --------------------------------------------------------

    pressure_raw = re.findall(
        rf"(?mi)^\s*M200\s*=\s*({NUM})",
        text,
    )

    pressures = unique_numeric(
        [float(x) / 10.0 for x in pressure_raw]
    )

    # --------------------------------------------------------
    # Speed
    # Example:
    # F6.000
    # --------------------------------------------------------

    speeds = unique_numeric(
        re.findall(
            rf"(?mi)^\s*F\s*({NUM})\s*(?:;.*)?$",
            text,
        )
    )

    # --------------------------------------------------------
    # Print Z
    #
    # Only standalone Z commands are considered here.
    #
    # Example:
    # Z0.100
    #
    # This deliberately excludes:
    # G00 Z18.400
    # --------------------------------------------------------

    zoffsets = unique_numeric(
        re.findall(
            rf"(?mi)^\s*Z\s*({NUM})\s*(?:;.*)?$",
            text,
        )
    )

    if (
        len(pressures) == 1
        and len(speeds) == 1
        and len(zoffsets) == 1
    ):
        return (
            pressures[0],
            speeds[0],
            zoffsets[0],
        )

    return None


def extract_from_pfz_comments(text):
    """
    Third method.

    Handles comment formats such as:

        ; Col 1 ... P=82 kPa F=15 mm/s Z=0.1 mm

    or:

        ; -- well A1 ... P=120 kPa F=6 mm/s Z=0.100 mm --
    """

    conditions = []

    for line in text.splitlines():

        p_match = re.search(
            rf"\bP\s*=\s*({NUM})\s*kPa\b",
            line,
            flags=re.IGNORECASE,
        )

        f_match = re.search(
            rf"\bF\s*=\s*({NUM})\s*mm/s\b",
            line,
            flags=re.IGNORECASE,
        )

        z_match = re.search(
            rf"\bZ\s*=\s*({NUM})\s*mm\b",
            line,
            flags=re.IGNORECASE,
        )

        if p_match and f_match and z_match:

            condition = (
                round(float(p_match.group(1)), 8),
                round(float(f_match.group(1)), 8),
                round(float(z_match.group(1)), 8),
            )

            if condition not in conditions:
                conditions.append(condition)

    if len(conditions) == 1:
        return conditions[0]

    if len(conditions) > 1:
        raise ValueError(
            "Multiple different P/F/Z conditions were found "
            f"in NC comments: {conditions}"
        )

    return None


def extract_from_labeled_comments(text):
    """
    Fourth method.

    Handles D5-type headers such as:

        ; Pressure     : 120 kPa
        ; Nozzle speed : 6 mm/s -> F
        ; Z offset     : 0.1 mm
    """

    pressures = unique_numeric(
        re.findall(
            rf"(?mi)^\s*;\s*Pressure\s*:\s*({NUM})\s*kPa\b",
            text,
        )
    )

    speeds = unique_numeric(
        re.findall(
            rf"(?mi)^\s*;\s*Nozzle\s*speed\s*:\s*({NUM})\s*mm/s\b",
            text,
        )
    )

    zoffsets = unique_numeric(
        re.findall(
            rf"(?mi)^\s*;\s*Z\s*offset\s*:\s*({NUM})\s*mm\b",
            text,
        )
    )

    if (
        len(pressures) == 1
        and len(speeds) == 1
        and len(zoffsets) == 1
    ):
        return (
            pressures[0],
            speeds[0],
            zoffsets[0],
        )

    return None


def extract_printing_parameters(nc_file):
    """
    Robust NC parser.

    Extraction priority:

    1. Actual printing block around M151
    2. Unique executable G-code
    3. P/F/Z comments
    4. Labeled Pressure / Nozzle speed / Z offset comments

    The first successful unambiguous method is used.
    """

    text = nc_file.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    # --------------------------------------------------------
    # 1. Actual printing blocks
    # --------------------------------------------------------

    result = extract_from_print_blocks(text)

    if result is not None:
        return result

    # --------------------------------------------------------
    # 2. Unique executable G-code
    # --------------------------------------------------------

    result = extract_from_unique_gcode(text)

    if result is not None:
        return result

    # --------------------------------------------------------
    # 3. P/F/Z comments
    # --------------------------------------------------------

    result = extract_from_pfz_comments(text)

    if result is not None:
        return result

    # --------------------------------------------------------
    # 4. Separate labeled comments
    # --------------------------------------------------------

    result = extract_from_labeled_comments(text)

    if result is not None:
        return result

    raise ValueError(
        "Could not determine one unambiguous "
        "Pressure / NozzleSpeed / Z-offset condition."
    )


# ============================================================
# NC FILE NAME MATCHING
# ============================================================

def remove_windows_duplicate_suffix(stem):
    """
    Handle copied filenames such as:

        experiment(1).nc

    as equivalent to:

        experiment.nc
    """

    return re.sub(
        r"\s*\(\d+\)$",
        "",
        stem,
    ).strip()


def validation_name_without_material(folder_name):
    """
    Handle cases where the experiment folder includes the
    material name but the NC filename does not.

    Examples:

    Folder:
        validation_excl_A_t_E_gelma_10_60_col1

    NC:
        validation_excl_A_t_E_col1.nc


    Folder:
        validation_excl_A_t_E_gelma_10_60_gpr_col1

    NC:
        validation_excl_A_t_E_gpr_col1.nc


    Folder:
        validation_excl_B_t_F_gelma_10_80_col1

    NC:
        validation_excl_B_t_F_col1.nc
    """

    cleaned = re.sub(
        r"_(?:cell_)?gelma_"
        r"[0-9]+(?:p[0-9]+)?_"
        r"[0-9]+"
        r"(?=_(?:gpr_)?col[0-9]+$)",
        "",
        folder_name,
        flags=re.IGNORECASE,
    )

    return cleaned


def find_matching_nc(folder_name, nc_files):
    """
    Match an experiment folder to its NC file.

    Matching priority:

    1. Exact filename stem
    2. Exact after removing Windows '(1)' duplicate suffix
    3. Validation naming convention where material name exists
       only in folder name
    """

    candidates = []

    for nc_file in nc_files:

        stem = remove_windows_duplicate_suffix(
            nc_file.stem
        )

        candidates.append(
            (nc_file, stem)
        )

    # --------------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------------

    exact_matches = [
        nc_file
        for nc_file, stem in candidates
        if stem.lower() == folder_name.lower()
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Multiple exact NC matches found for "
            f"'{folder_name}':\n"
            + "\n".join(str(x) for x in exact_matches)
        )

    # --------------------------------------------------------
    # 2. Validation-style alias
    # --------------------------------------------------------

    alias = validation_name_without_material(
        folder_name
    )

    alias_matches = [
        nc_file
        for nc_file, stem in candidates
        if stem.lower() == alias.lower()
    ]

    if len(alias_matches) == 1:
        return alias_matches[0]

    if len(alias_matches) > 1:
        raise RuntimeError(
            f"Multiple NC matches found for "
            f"'{folder_name}' using alias '{alias}'."
        )

    # --------------------------------------------------------
    # No safe match
    #
    # Deliberately do NOT use aggressive fuzzy matching.
    # Picking the wrong NC file would give incorrect P/F/Z.
    # --------------------------------------------------------

    return None


# ============================================================
# PORE SCORE PROCESSING
# ============================================================

def find_pore_scores(folder):
    """
    Search recursively for pore_scores.csv.

    Supports:
        folder/pore_scores.csv

    as well as:
        folder/something/pore_scores.csv
    """

    # Direct location first
    direct = folder / "pore_scores.csv"

    if direct.exists():
        return direct

    # Recursive, case-insensitive search
    matches = [
        p
        for p in folder.rglob("*")
        if (
            p.is_file()
            and p.name.lower() == "pore_scores.csv"
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple pore_scores.csv files found in:\n"
            f"{folder}\n\n"
            + "\n".join(str(x) for x in matches)
        )

    return None


def calculate_sf_statistics(pore_scores_file):
    """
    Calculate SF mean and sample standard deviation.

    Separator is automatically detected so both ';' and ','
    CSV formats can be handled.
    """

    df = pd.read_csv(
        pore_scores_file,
        sep=None,
        engine="python",
    )

    # Clean accidental spaces in headers
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Case-insensitive SF lookup
    sf_column = None

    for column in df.columns:

        if column.lower() == "sf":
            sf_column = column
            break

    if sf_column is None:
        raise ValueError(
            "'SF' column was not found in "
            f"{pore_scores_file}.\n"
            f"Available columns: {list(df.columns)}"
        )

    sf_values = pd.to_numeric(
        df[sf_column],
        errors="coerce",
    ).dropna()

    if len(sf_values) == 0:
        raise ValueError(
            f"No valid SF values found in {pore_scores_file}"
        )

    sf_mean = sf_values.mean()

    # Sample standard deviation, same convention as pandas
    if len(sf_values) > 1:
        sf_std = sf_values.std(ddof=1)
    else:
        sf_std = 0.0

    return (
        sf_mean,
        sf_std,
        len(sf_values),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a validation summary table by combining "
            "printing parameters from NC files with SF values "
            "from pore_scores.csv."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Validation parent directory containing experiment "
            "folders and the nc_opt folder."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. "
            "Default: <input_dir>/validation_sf_summary.csv"
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    # ========================================================
    # Validate input
    # ========================================================

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory:\n{input_dir}"
        )

    # ========================================================
    # Find nc_opt
    # ========================================================

    nc_dir = input_dir / "nc_opt"

    if not nc_dir.exists():
        raise FileNotFoundError(
            f"'nc_opt' folder was not found:\n{nc_dir}"
        )

    # Recursive in case NC files are placed inside
    # another subfolder later.
    nc_files = sorted(
        [
            p
            for p in nc_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".nc"
        ]
    )

    if len(nc_files) == 0:
        raise FileNotFoundError(
            f"No .nc files were found in:\n{nc_dir}"
        )

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY GENERATOR")
    print("=" * 70)

    print(f"\nInput directory:")
    print(input_dir)

    print(f"\nNC directory:")
    print(nc_dir)

    print(
        f"\nFound {len(nc_files)} NC file(s)."
    )

    # ========================================================
    # Experiment folders
    # ========================================================

    experiment_folders = sorted(
        [
            p
            for p in input_dir.iterdir()
            if (
                p.is_dir()
                and p.name.lower() != "nc_opt"
            )
        ],
        key=lambda x: x.name.lower(),
    )

    print(
        f"Found {len(experiment_folders)} "
        f"experiment folder(s).\n"
    )

    rows = []
    problems = []

    # ========================================================
    # Process each experiment
    # ========================================================

    for folder in experiment_folders:

        folder_name = folder.name

        print("-" * 70)
        print(f"Folder: {folder_name}")

        # ----------------------------------------------------
        # Find pore_scores.csv
        # ----------------------------------------------------

        try:
            pore_scores_file = find_pore_scores(folder)

        except Exception as exc:

            message = (
                f"{folder_name}: pore_scores search failed: "
                f"{exc}"
            )

            problems.append(message)

            print(f"  ERROR: {exc}")
            continue

        if pore_scores_file is None:

            message = (
                f"{folder_name}: pore_scores.csv not found"
            )

            problems.append(message)

            print(
                "  SKIPPED: pore_scores.csv not found"
            )
            continue

        print(
            f"  pore_scores: "
            f"{pore_scores_file.relative_to(folder)}"
        )

        # ----------------------------------------------------
        # Match NC
        # ----------------------------------------------------

        try:
            nc_file = find_matching_nc(
                folder_name,
                nc_files,
            )

        except Exception as exc:

            problems.append(
                f"{folder_name}: NC matching failed: {exc}"
            )

            print(f"  ERROR: {exc}")
            continue

        if nc_file is None:

            alias = validation_name_without_material(
                folder_name
            )

            message = (
                f"{folder_name}: matching NC file not found. "
                f"Tried exact='{folder_name}' and "
                f"alias='{alias}'."
            )

            problems.append(message)

            print(
                "  SKIPPED: matching NC file not found"
            )

            print(
                f"           Exact : {folder_name}.nc"
            )

            if alias != folder_name:
                print(
                    f"           Alias : {alias}.nc"
                )

            continue

        print(
            f"  NC file    : {nc_file.name}"
        )

        # ----------------------------------------------------
        # Extract P/F/Z
        # ----------------------------------------------------

        try:
            pressure, speed, zoffset = (
                extract_printing_parameters(nc_file)
            )

        except Exception as exc:

            problems.append(
                f"{folder_name}: NC parsing failed: {exc}"
            )

            print(
                f"  ERROR parsing NC: {exc}"
            )
            continue

        # ----------------------------------------------------
        # Calculate SF
        # ----------------------------------------------------

        try:
            sf_mean, sf_std, n_sf = (
                calculate_sf_statistics(
                    pore_scores_file
                )
            )

        except Exception as exc:

            problems.append(
                f"{folder_name}: SF calculation failed: {exc}"
            )

            print(
                f"  ERROR calculating SF: {exc}"
            )
            continue

        # ----------------------------------------------------
        # Add row
        # ----------------------------------------------------

        rows.append(
            {
                "Folder_Name": folder_name,
                "Pressure_kPa": pressure,
                "NozzleSpeed_mms": speed,
                "Zoffset_mm": zoffset,
                "SF_mean": sf_mean,
                "SF_std": sf_std,
            }
        )

        print(
            f"  P/F/Z      : "
            f"{pressure:g} kPa / "
            f"{speed:g} mm/s / "
            f"{zoffset:g} mm"
        )

        print(
            f"  SF         : "
            f"{sf_mean:.4f} ± {sf_std:.4f} "
            f"(n={n_sf})"
        )

    # ========================================================
    # Create final dataframe
    # ========================================================

    columns = [
        "Folder_Name",
        "Pressure_kPa",
        "NozzleSpeed_mms",
        "Zoffset_mm",
        "SF_mean",
        "SF_std",
    ]

    result = pd.DataFrame(
        rows,
        columns=columns,
    )

    if not result.empty:

        result["SF_mean"] = (
            result["SF_mean"].round(4)
        )

        result["SF_std"] = (
            result["SF_std"].round(4)
        )

    # ========================================================
    # Output location
    # ========================================================

    if args.output is None:

        output_file = (
            input_dir
            / "validation_sf_summary.csv"
        )

    else:

        output_file = args.output.resolve()

    # Semicolon to match your current tables
    result.to_csv(
        output_file,
        sep=";",
        index=False,
    )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"Successful folders : {len(result)}"
    )

    print(
        f"Problems           : {len(problems)}"
    )

    print(
        f"Output             : {output_file}"
    )

    if not result.empty:

        print("\nGenerated table:\n")

        print(
            result.to_string(index=False)
        )

    if problems:

        print()
        print("=" * 70)
        print("WARNINGS / SKIPPED FOLDERS")
        print("=" * 70)

        for problem in problems:
            print(f"- {problem}")


if __name__ == "__main__":
    main()