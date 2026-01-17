#!/usr/bin/env python3
import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gpxpy


@dataclass
class CompareResult:
    ok: bool
    soft: bool
    messages: List[str]


def load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_log_lines(lines: List[str]) -> List[str]:
    normalized = []
    export_pattern = re.compile(r"^\[INFO\] Exported .*? GPX file: (.*)$")
    export_cfg_pattern = re.compile(r"^\[INFO\] +export_gpx_file=(.*)$")
    export_unmatched_cfg_pattern = re.compile(r"^\[INFO\] +export_unmatched_gpx_file=(.*)$")
    export_unmatched_min_pattern = re.compile(r"^\[INFO\] +export_unmatched_min_m=.*$")
    export_unmatched_max_pattern = re.compile(r"^\[INFO\] +export_unmatched_max_m=.*$")
    recorded_cfg_pattern = re.compile(r"^\[INFO\] +recorded=(.*)$")
    recorded_load_pattern = re.compile(r"^\[INFO\] Loading recorded track: (.*)$")
    for line in lines:
        if not line.strip():
            continue
        line = re.sub(r"^\d{4}-\d{2}-\d{2} [0-9:,]+ ", "", line)
        stripped = line.strip()
        if export_unmatched_min_pattern.match(stripped) or export_unmatched_max_pattern.match(stripped):
            continue
        match = export_pattern.match(stripped)
        if match:
            line = "[INFO] Exported GPX file: <export>"
        else:
            match = export_cfg_pattern.match(stripped)
            if match:
                line = "[INFO] export_gpx_file=<export>"
            else:
                match = export_unmatched_cfg_pattern.match(stripped)
                if match:
                    line = "[INFO] export_unmatched_gpx_file=<unmatched>"
                else:
                    match = recorded_cfg_pattern.match(stripped)
                    if match:
                        line = "[INFO] recorded=<recorded>"
                    else:
                        match = recorded_load_pattern.match(stripped)
                        if match:
                            line = "[INFO] Loading recorded track: <recorded>"
        normalized.append(line.strip())
    return normalized


def parse_summary_table(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Match") or line.strip().startswith("Segment"):
            header_idx = idx
            break
    if header_idx is None:
        return []

    header_cols = re.split(r"\s{2,}", lines[header_idx].strip())
    data_start = None
    for idx in range(header_idx + 1, len(lines)):
        if lines[idx].strip() and set(lines[idx].strip()) <= {"-"}:
            data_start = idx + 1
            break
    if data_start is None:
        return []

    rows = []
    for line in lines[data_start:]:
        if not line.strip():
            break
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) < len(header_cols):
            continue
        row = dict(zip(header_cols, cols))
        rows.append(row)
    return rows


def haversine_m(lat1, lon1, lat2, lon2):
    import math

    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gpx_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)
    track_count = len(gpx.tracks)
    point_count = 0
    distance_m = 0.0
    for track in gpx.tracks:
        for segment in track.segments:
            pts = segment.points
            point_count += len(pts)
            for p1, p2 in zip(pts, pts[1:]):
                distance_m += haversine_m(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
    return {
        "tracks": track_count,
        "points": point_count,
        "distance_m": distance_m,
    }


def compare_gpx(expected: Path, actual: Path, tol_distance: float) -> CompareResult:
    if not expected.exists():
        return CompareResult(False, False, [f"Missing expected GPX: {expected}"])
    if not actual.exists():
        return CompareResult(False, False, [f"Missing actual GPX: {actual}"])

    exp = gpx_metrics(expected)
    act = gpx_metrics(actual)
    messages = []
    soft = False

    if exp["tracks"] != act["tracks"]:
        messages.append(f"Track count mismatch ({exp['tracks']} != {act['tracks']})")
        return CompareResult(False, False, messages)

    if exp["points"] != act["points"]:
        messages.append(f"Point count mismatch ({exp['points']} != {act['points']})")
        soft = True

    dist_delta = abs(exp["distance_m"] - act["distance_m"])
    if dist_delta > tol_distance:
        messages.append(f"Distance mismatch ({dist_delta:.2f} m > {tol_distance} m)")
        return CompareResult(False, False, messages)
    if dist_delta > 0:
        messages.append(f"Distance delta {dist_delta:.2f} m")
        soft = True

    return CompareResult(True, soft, messages)


def float_or_none(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def compare_rows(expected_rows: List[Dict[str, str]], actual_rows: List[Dict[str, str]], tol: Dict[str, float]) -> CompareResult:
    messages = []
    soft = False

    exp_map = {}
    for row in expected_rows:
        key = (row.get("Match", ""), row.get("Segment", ""))
        exp_map[key] = row

    act_map = {}
    for row in actual_rows:
        key = (row.get("Match", ""), row.get("Segment", ""))
        act_map[key] = row

    missing = set(exp_map) - set(act_map)
    extra = set(act_map) - set(exp_map)
    if missing:
        messages.append(f"Missing rows: {sorted(missing)}")
        return CompareResult(False, False, messages)
    if extra:
        messages.append(f"Extra rows: {sorted(extra)}")
        return CompareResult(False, False, messages)

    def idx_diff(field):
        exp = exp_row.get(field)
        act = act_row.get(field)
        if exp is None or act is None:
            return None
        try:
            return abs(int(exp) - int(act))
        except ValueError:
            return None

    def float_diff(field):
        exp = float_or_none(exp_row.get(field, ""))
        act = float_or_none(act_row.get(field, ""))
        if exp is None or act is None:
            return None
        return abs(exp - act)

    for key, exp_row in exp_map.items():
        act_row = act_map[key]
        for field in ("Start Idx", "End Idx", "Start Cross 0", "Start Cross 1", "End Cross 0", "End Cross 1"):
            diff = idx_diff(field)
            if diff is None:
                continue
            if diff > tol.get("index", 0):
                messages.append(f"{key} {field} delta {diff} exceeds {tol.get('index', 0)}")
                return CompareResult(False, False, messages)
            if diff > 0:
                soft = True

        for field, tkey in (
            ("Detected Dist (m)", "distance_m"),
            ("DTW Avg (m)", "dtw_avg"),
            ("Time (s)", "time_s"),
            ("Start Diff (m)", "start_end_m"),
            ("End Diff (m)", "start_end_m"),
        ):
            diff = float_diff(field)
            if diff is None:
                continue
            limit = tol.get(tkey, 0.0)
            if diff > limit:
                messages.append(f"{key} {field} delta {diff:.2f} exceeds {limit}")
                return CompareResult(False, False, messages)
            if diff > 0:
                soft = True

    return CompareResult(True, soft, messages)


def compare_synthetic(expected: Dict, actual_rows: List[Dict[str, str]], tol: Dict[str, float]) -> CompareResult:
    messages = []
    soft = False
    expected_matches = expected.get("matches", [])

    actual_used = set()
    for match in expected_matches:
        segment = match.get("segment")
        start = match.get("start_idx")
        end = match.get("end_idx")
        found = False
        for idx, row in enumerate(actual_rows):
            if row.get("Segment") != segment or idx in actual_used:
                continue
            try:
                start_diff = abs(int(row.get("Start Idx")) - start)
                end_diff = abs(int(row.get("End Idx")) - end)
            except (TypeError, ValueError):
                continue
            if start_diff <= tol.get("index", 0) and end_diff <= tol.get("index", 0):
                actual_used.add(idx)
                found = True
                if start_diff or end_diff:
                    soft = True
                break
        if not found:
            messages.append(f"Missing synthetic match for {segment} at {start}-{end}")
            return CompareResult(False, False, messages)

    if len(actual_rows) > len(expected_matches):
        messages.append("Extra matches found in synthetic output")
        return CompareResult(False, False, messages)

    return CompareResult(True, soft, messages)

def run_case(
    case: Dict,
    update_expected: bool,
    keep_temp: bool,
    strict_trace: bool,
    trace_diff: bool,
) -> Tuple[str, bool, bool, List[str], str | None]:
    case_id = case["id"]
    messages = []
    trace_diff_text = None
    temp_dir = Path("tests/.tmp") / case_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    recorded = Path(case["recorded"])
    if not recorded.exists():
        return case_id, True, True, [f"Skipped (missing recorded file): {recorded}"], trace_diff_text

    segments = Path(case["segments"])
    if not segments.exists():
        return case_id, False, False, [f"Missing segments dir: {segments}"], trace_diff_text

    args = list(case["args"])
    for idx, arg in enumerate(args):
        if arg in {"--export-gpx-file", "--export-unmatched-gpx-file", "--output-file"}:
            if idx + 1 < len(args):
                args[idx + 1] = str(temp_dir / args[idx + 1])

    cmd = [
        sys.executable,
        "gpx-segment-timer.py",
        "-v",
        "-r",
        str(recorded),
        "-f",
        str(segments),
        *args,
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_path = temp_dir / "stdout.txt"
    trace_path = temp_dir / "trace.txt"
    stdout_path.write_text(proc.stdout)
    trace_path.write_text(proc.stderr)

    expected_dir = Path(case["expected_dir"])
    expected_dir.mkdir(parents=True, exist_ok=True)

    if update_expected:
        shutil.copy2(stdout_path, expected_dir / "stdout.txt")
        shutil.copy2(trace_path, expected_dir / "trace.txt")
        matches_dir = expected_dir / "matches"
        matches_dir.mkdir(exist_ok=True)
        for gpx in temp_dir.glob("*_match*.gpx"):
            shutil.copy2(gpx, matches_dir / gpx.name)
        expected_unmatched = case.get("expected_unmatched")
        if expected_unmatched:
            unmatched_path = temp_dir / expected_unmatched
            if unmatched_path.exists():
                shutil.copy2(unmatched_path, expected_dir / expected_unmatched)
        if case.get("kind") == "synthetic":
            act_rows = parse_summary_table(stdout_path.read_text())
            payload = {
                "matches": [
                    {
                        "segment": row.get("Segment"),
                        "start_idx": int(row.get("Start Idx")),
                        "end_idx": int(row.get("End Idx")),
                    }
                    for row in act_rows
                    if row.get("Segment")
                ],
                "source": "baseline",
            }
            (expected_dir / "expected.json").write_text(json.dumps(payload, indent=2))
        messages.append("Updated expected outputs")
        if not keep_temp:
            shutil.rmtree(temp_dir)
        return case_id, True, False, messages, trace_diff_text

    # Compare stdout summary
    expected_stdout = expected_dir / "stdout.txt"
    act_rows = parse_summary_table(stdout_path.read_text())
    if expected_stdout.exists():
        exp_rows = parse_summary_table(expected_stdout.read_text())
        row_result = compare_rows(exp_rows, act_rows, case.get("tolerances", {}))
        messages.extend(row_result.messages)
        ok = row_result.ok
        soft = row_result.soft
    else:
        expected_json = expected_dir / "expected.json"
        if expected_json.exists() and case.get("kind") == "synthetic":
            exp_payload = json.loads(expected_json.read_text())
            row_result = compare_synthetic(exp_payload, act_rows, case.get("tolerances", {}))
            messages.extend(row_result.messages)
            ok = row_result.ok
            soft = row_result.soft
        else:
            ok = False
            soft = False
            messages.append("Missing expected outputs for summary comparison")

    # Compare GPX matches
    matches_dir = expected_dir / "matches"
    if matches_dir.exists():
        expected_files = sorted(p.name for p in matches_dir.glob("*.gpx"))
        actual_files = sorted(p.name for p in temp_dir.glob("*_match*.gpx"))
        if expected_files != actual_files:
            messages.append(f"Match GPX file set mismatch: expected={expected_files} actual={actual_files}")
            ok = False
        else:
            for name in expected_files:
                res = compare_gpx(matches_dir / name, temp_dir / name, case.get("tolerances", {}).get("gpx_distance_m", 0.0))
                messages.extend([f"{name}: {msg}" for msg in res.messages])
                ok = ok and res.ok
                soft = soft or res.soft
    elif case.get("kind") != "synthetic":
        messages.append("Missing expected matches directory")
        ok = False

    # Compare unmatched GPX
    expected_unmatched = case.get("expected_unmatched")
    if expected_unmatched:
        exp_unmatched = expected_dir / expected_unmatched
        act_unmatched = temp_dir / expected_unmatched
        res = compare_gpx(exp_unmatched, act_unmatched, case.get("tolerances", {}).get("gpx_distance_m", 0.0))
        messages.extend([f"unmatched: {msg}" for msg in res.messages])
        ok = ok and res.ok
        soft = soft or res.soft

    # Compare trace
    if case.get("trace_compare"):
        expected_trace = expected_dir / "trace.txt"
        if expected_trace.exists():
            exp_lines = normalize_log_lines(expected_trace.read_text().splitlines())
            act_lines = normalize_log_lines(trace_path.read_text().splitlines())
            if exp_lines != act_lines:
                diff_count = sum(1 for e, a in zip(exp_lines, act_lines) if e != a) + abs(len(exp_lines) - len(act_lines))
                messages.append(f"Trace mismatch: {diff_count} differing lines")
                if trace_diff:
                    diff = difflib.unified_diff(
                        exp_lines,
                        act_lines,
                        fromfile="expected(trace)",
                        tofile="actual(trace)",
                        lineterm="",
                    )
                    trace_diff_text = "\n".join(diff)
                if strict_trace:
                    ok = False
                else:
                    soft = True
        else:
            messages.append("Missing expected trace.txt; skipping trace comparison")

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return case_id, ok, soft, messages, trace_diff_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", choices=["all", "public", "private", "synthetic"])
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--update-expected", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--strict-trace", action="store_true")
    parser.add_argument("--trace-diff", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(Path("tests/manifest.json"))
    all_cases = manifest.get("cases", [])

    selected = []
    for case in all_cases:
        if args.cases and case["id"] not in args.cases:
            continue
        if args.suite == "public" and case.get("visibility") != "public":
            continue
        if args.suite == "private" and case.get("visibility") != "private":
            continue
        if args.suite == "synthetic" and case.get("kind") != "synthetic":
            continue
        selected.append(case)

    if not selected:
        print("No matching test cases found")
        return 1

    hard_fail = False
    soft_fail = False
    for case in selected:
        case_id, ok, soft, messages, trace_diff_text = run_case(
            case,
            args.update_expected,
            args.keep_temp,
            args.strict_trace,
            args.trace_diff,
        )
        status = "ok" if ok and not soft else "soft" if ok else "fail"
        print(f"{case_id}: {status}")
        for msg in messages:
            print(f"  - {msg}")
        if trace_diff_text:
            print("  trace diff (normalized):")
            for line in trace_diff_text.splitlines():
                print(f"    {line}")
        if not ok:
            hard_fail = True
        if soft:
            soft_fail = True

    if hard_fail:
        return 1
    if soft_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
