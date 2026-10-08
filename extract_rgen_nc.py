from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _to_number(value: str) -> Any:
    """Convert a numeric string to int or float when possible."""
    value = value.strip()
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d*\.\d+", value) or re.fullmatch(r"[+-]?\d+\.\d*", value):
        return float(value)
    return value


def _parse_bracket_args(arg_string: str) -> list[Any]:
    """Parse comma-separated arguments inside brackets."""
    parts = [p.strip() for p in arg_string.split(",")]
    return [_to_number(p) for p in parts if p != ""]


def parse_nc_file(file_path: Path) -> dict[str, Any]:
    """
    Parse a REGENHU-style .nc file and extract a practical subset of parameters.

    This focuses on commonly relevant manufacturing parameters:
    - tool selection (Tn)
    - pressure (M200)
    - valve opening/closing time (M210, M211)
    - linear density (M230)
    - flow rate (M231)
    - tool temperature (M300)
    - workzone temperature (M310)
    - start/stop delays (G807)
    - feed rate F found in motion lines
    - selected G55 origin from G805

    The script stores the last seen value for each parameter.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data: dict[str, Any] = {
        "source_filename": file_path.name,
        "source_filepath": str(file_path.resolve()),
        "file_type": file_path.suffix.lower(),
    }

    current_tool: int | None = None
    last_feed_rate: float | int | None = None

    # Keep a small audit trail too
    commands_found: list[str] = []

    for raw_line in lines:
        # Remove comments after ';'
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue

        # Program name line: % something
        if line.startswith("%"):
            program_name = line[1:].strip()
            if program_name:
                data["program_name"] = program_name
            continue

        # Tool selection: T1, T2, ...
        m_tool = re.fullmatch(r"T(\d+)", line, flags=re.IGNORECASE)
        if m_tool:
            current_tool = int(m_tool.group(1))
            data["selected_tool_last"] = current_tool
            commands_found.append(f"T{current_tool}")
            continue

        # M-codes with assignment: M200=1000
        m_assign = re.fullmatch(r"(M\d+)\s*=\s*([^\s]+)", line, flags=re.IGNORECASE)
        if m_assign:
            code = m_assign.group(1).upper()
            value = _to_number(m_assign.group(2))
            commands_found.append(f"{code}={value}")

            if code == "M200":
                data["pressure"] = value
                if current_tool is not None:
                    data["pressure_tool"] = current_tool
            elif code == "M210":
                data["valve_opening_time"] = value
                if current_tool is not None:
                    data["valve_opening_time_tool"] = current_tool
            elif code == "M211":
                data["valve_closing_time"] = value
                if current_tool is not None:
                    data["valve_closing_time_tool"] = current_tool
            elif code == "M230":
                data["linear_density"] = value
                if current_tool is not None:
                    data["linear_density_tool"] = current_tool
            elif code == "M231":
                data["flow_rate"] = value
                if current_tool is not None:
                    data["flow_rate_tool"] = current_tool
            elif code == "M300":
                data["tool_temperature"] = value
                if current_tool is not None:
                    data["tool_temperature_tool"] = current_tool
            elif code == "M310":
                data["workzone_temperature"] = value
            elif code == "M110":
                data["program_progress_last"] = value
            else:
                # Store other M-codes too, just in case
                data[f"{code.lower()}"] = value

            continue

        # G-code with bracket args: G807[2,0.1,0.2]
        m_g_brackets = re.fullmatch(r"(G\d+)\[(.*)\]", line, flags=re.IGNORECASE)
        if m_g_brackets:
            code = m_g_brackets.group(1).upper()
            args = _parse_bracket_args(m_g_brackets.group(2))
            commands_found.append(f"{code}{args}")

            if code == "G807":
                # Mode, start delay, stop delay
                if len(args) >= 1:
                    data["start_stop_delay_mode"] = args[0]
                if len(args) >= 2:
                    data["start_delay"] = args[1]
                if len(args) >= 3:
                    data["stop_delay"] = args[2]
            elif code == "G805":
                # G55 origin offsets
                if len(args) >= 1:
                    data["origin_x_offset"] = args[0]
                if len(args) >= 2:
                    data["origin_y_offset"] = args[1]
                if len(args) >= 3:
                    data["origin_z_offset"] = args[2]
                if len(args) >= 4:
                    data["origin_add_substrate_height"] = args[3]
            elif code == "G806":
                if len(args) >= 1:
                    data["shc_store_index"] = args[0]
                if len(args) >= 2:
                    data["substrate_max_height"] = args[1]
            else:
                data[f"{code.lower()}_args"] = args

            continue

        # Plain G/M/T tokens in motion blocks, extract F feed rate
        feed_match = re.search(r"\bF([+-]?\d+(?:\.\d+)?)\b", line, flags=re.IGNORECASE)
        if feed_match:
            last_feed_rate = _to_number(feed_match.group(1))
            data["feed_rate_last"] = last_feed_rate

        # Capture motion coordinates if present, last seen
        for axis in ("X", "Y", "Z"):
            axis_match = re.search(rf"\b{axis}([+-]?\d+(?:\.\d+)?)\b", line, flags=re.IGNORECASE)
            if axis_match:
                data[f"last_{axis.lower()}"] = _to_number(axis_match.group(1))

    data["commands_found_count"] = len(commands_found)
    data["commands_found_preview"] = commands_found[:20]


    # Proposal-aligned parameter names
    if "feed_rate_last" in data:
        data["nozzle_speed"] = data["feed_rate_last"]
        data["nozzle_velocity"] = data["feed_rate_last"]

    if "pressure" in data:
        data["extrusion_pressure"] = data["pressure"]

    if "tool_temperature" in data:
        data["printhead_temperature"] = data["tool_temperature"]

    # Derive extrusion speed if both ingredients are available
    if "flow_rate" in data and "linear_density" in data:
        try:
            if float(data["linear_density"]) != 0:
                data["extrusion_speed"] = float(data["flow_rate"]) / float(data["linear_density"])
        except (ValueError, TypeError, ZeroDivisionError):
            data["extrusion_speed"] = None

    # Very rough Z-offset estimate
    if "origin_z_offset" in data and "last_z" in data:
        try:
            data["z_offset_estimated"] = float(data["last_z"]) - float(data["origin_z_offset"])
        except (ValueError, TypeError):
            data["z_offset_estimated"] = None


    return data


def write_json(data: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_catalog(csv_path: Path, row: dict[str, Any]) -> None:
    """
    Create or update catalog.csv using pandas.
    Each JSON key becomes a column; each file becomes a row.
    List values are serialized to JSON strings for CSV compatibility.
    Existing rows are matched by source_filename + json_output_filename and replaced;
    new files are appended.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize any list/dict values so they store cleanly in CSV
    serialized = {
        k: json.dumps(v) if isinstance(v, (list, dict)) else v
        for k, v in row.items()
    }

    new_df = pd.DataFrame([serialized])

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, sep=";")

        # Identify matching row (upsert): same source file + json output name
        mask = (
            (existing_df.get("source_filename", pd.Series(dtype=str)) == str(row.get("source_filename", "")))
            & (existing_df.get("json_output_filename", pd.Series(dtype=str)) == str(row.get("json_output_filename", "")))
        )

        if mask.any():
            existing_df = existing_df[~mask]

        catalog_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        catalog_df = new_df

    catalog_df.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")


def process_file(input_path: Path, output_dir: Path, csv_path: Path, index: int) -> None:
    """Parse a single .nc file, save its JSON, and update the catalog."""
    if input_path.suffix.lower() not in {".nc", ".biofile"}:
        print(f"  Warning: unexpected extension, continuing anyway.")

    parsed = parse_nc_file(input_path)

    json_path = output_dir / f"{index}.json"
    parsed["json_output_filename"] = json_path.name

    write_json(parsed, json_path)
    update_catalog(csv_path, parsed)

    print(f"  [{index}] {input_path.name}")
    print(f"       JSON -> {json_path}")


def main() -> None:
    # Resolve nc_files folder relative to this script
    script_dir = Path(__file__).parent
    nc_folder = script_dir / "nc_files"
    output_dir = script_dir / "data"
    csv_path = output_dir / "catalog.csv"

    if not nc_folder.exists():
        print(f"Error: nc_files folder not found -> {nc_folder}")
        sys.exit(1)

    nc_files = sorted(nc_folder.glob("*.nc")) + sorted(nc_folder.glob("*.biofile"))

    if not nc_files:
        print(f"No .nc or .biofile files found in {nc_folder}")
        sys.exit(0)

    print(f"Found {len(nc_files)} file(s) in {nc_folder}\n")

    for index, nc_path in enumerate(nc_files, start=1):
        process_file(nc_path, output_dir, csv_path, index)

    print(f"\nDone. Catalog saved to: {csv_path}")


if __name__ == "__main__":
    main()