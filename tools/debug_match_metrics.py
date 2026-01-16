#!/usr/bin/env python3
"""
Debug tool for analyzing exported match GPX files against reference segments.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import gpxpy


def _load_main_module() -> Any:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    module_path = os.path.join(repo_root, "gpx-segment-timer.py")
    spec = importlib.util.spec_from_file_location("gpx_segment_timer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track_points(track: gpxpy.gpx.GPXTrack) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for seg in track.segments:
        for pt in seg.points:
            points.append({"lat": pt.latitude, "lon": pt.longitude, "time": pt.time})
    return points


def _load_tracks(path: str) -> List[gpxpy.gpx.GPXTrack]:
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    return list(gpx.tracks)


def _select_reference_track(tracks: List[gpxpy.gpx.GPXTrack]) -> gpxpy.gpx.GPXTrack:
    for track in tracks:
        if track.name and track.name.startswith("reference:"):
            return track
    raise ValueError("Reference track not found (expected track name starting with 'reference:').")


def _select_match_track(tracks: List[gpxpy.gpx.GPXTrack]) -> gpxpy.gpx.GPXTrack:
    ignore_prefixes = ("reference:", "start line:", "finish line:", "start crossing", "finish crossing")
    candidates: List[gpxpy.gpx.GPXTrack] = []
    for track in tracks:
        name = track.name or ""
        if any(name.startswith(prefix) for prefix in ignore_prefixes):
            continue
        candidates.append(track)
    if not candidates:
        raise ValueError("Matched track not found (expected non-reference track in GPX).")
    candidates.sort(key=lambda t: sum(len(s.points) for s in t.segments), reverse=True)
    return candidates[0]


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return values_sorted[idx]


def _compute_crossings(mod: Any,
                       points: List[Dict[str, Any]],
                       ref_points: List[Dict[str, Any]],
                       edge_window_pts: int,
                       line_length_m: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(ref_points) < 2 or len(points) < 2:
        return None, None, [], []
    ref_start = (ref_points[0]["lat"], ref_points[0]["lon"])
    ref_end = (ref_points[-1]["lat"], ref_points[-1]["lon"])
    start_normal, start_lat_scale = mod.compute_line_normal(ref_start, (ref_points[1]["lat"], ref_points[1]["lon"]))
    end_normal, end_lat_scale = mod.compute_line_normal(ref_end, (ref_points[-2]["lat"], ref_points[-2]["lon"]))
    line_half_len = max(0.0, line_length_m / 2.0) if line_length_m > 0 else None
    edge_window = max(1, min(edge_window_pts, len(points) - 1))
    s_lo = 0
    s_hi = min(len(points) - 1, edge_window)
    e_lo = max(0, len(points) - 1 - edge_window)
    e_hi = len(points) - 1
    start_crossings = mod.find_line_crossings(points, ref_start, start_normal, s_lo, s_hi + 1, start_lat_scale, line_half_len)
    end_crossings = mod.find_line_crossings(points, ref_end, end_normal, e_lo, e_hi + 1, end_lat_scale, line_half_len)
    if not start_crossings:
        start_crossings = mod.find_line_crossings(points, ref_start, start_normal, 0, len(points), start_lat_scale, line_half_len)
    if not end_crossings:
        end_crossings = mod.find_line_crossings(points, ref_end, end_normal, 0, len(points), end_lat_scale, line_half_len)
    start_cross = start_crossings[0] if start_crossings else None
    end_cross = end_crossings[-1] if end_crossings else None
    return start_cross, end_cross, start_crossings, end_crossings


def _find_min_window_for_crossings(mod: Any,
                                   recorded_points: List[Dict[str, Any]],
                                   ref_points: List[Dict[str, Any]],
                                   base_start: int,
                                   base_end: int,
                                   max_extra: int,
                                   line_length_m: float) -> Tuple[int, int, int, int]:
    if len(ref_points) < 2 or len(recorded_points) < 2:
        return base_start, base_end, 0, 0
    ref_start = (ref_points[0]["lat"], ref_points[0]["lon"])
    ref_end = (ref_points[-1]["lat"], ref_points[-1]["lon"])
    start_normal, start_lat_scale = mod.compute_line_normal(ref_start, (ref_points[1]["lat"], ref_points[1]["lon"]))
    end_normal, end_lat_scale = mod.compute_line_normal(ref_end, (ref_points[-2]["lat"], ref_points[-2]["lon"]))
    line_half_len = max(0.0, line_length_m / 2.0) if line_length_m > 0 else None

    def has_start_cross(start_idx: int) -> bool:
        crossings = mod.find_line_crossings(
            recorded_points, ref_start, start_normal, start_idx, base_end, start_lat_scale, line_half_len
        )
        return bool(crossings)

    def has_end_cross(end_idx: int) -> bool:
        crossings = mod.find_line_crossings(
            recorded_points, ref_end, end_normal, base_start, end_idx, end_lat_scale, line_half_len
        )
        return bool(crossings)

    start_extra = 0
    end_extra = 0
    if not has_start_cross(base_start):
        for extra in range(1, max_extra + 1):
            s_lo = max(0, base_start - extra)
            if has_start_cross(s_lo):
                start_extra = extra
                break
    if not has_end_cross(base_end):
        for extra in range(1, max_extra + 1):
            e_hi = min(len(recorded_points), base_end + extra)
            if has_end_cross(e_hi):
                end_extra = extra
                break
    return max(0, base_start - start_extra), min(len(recorded_points), base_end + end_extra), start_extra, end_extra


def _shape_cost(mod: Any,
                points: List[Dict[str, Any]],
                ref_shape: List[Tuple[float, ...]],
                start_idx: int,
                end_idx: int,
                shape_mode: str,
                lat_scale_ref: float,
                resample_count: int) -> Optional[float]:
    if end_idx - start_idx < 2:
        return None
    segment = points[start_idx:end_idx]
    candidate_resampled = mod.resample_points(segment, resample_count)
    cand_shape = mod.build_shape_sequence(candidate_resampled, shape_mode, lat_scale_ref)
    if not cand_shape or not ref_shape:
        return None
    dtw_distance, _ = mod.fastdtw(cand_shape, ref_shape, dist=mod._DTW_DISTANCE_FN)
    return dtw_distance / max(1, len(ref_shape))


def _compute_metrics(mod: Any,
                     ref_points: List[Dict[str, Any]],
                     match_points: List[Dict[str, Any]],
                     args: argparse.Namespace,
                     resample_override: Optional[int] = None,
                     crossing_edge_window_pts_override: Optional[int] = None,
                     crossing_L_override: Optional[int] = None) -> Dict[str, Any]:
    ref_distance = mod.compute_total_distance(ref_points)
    match_distance = mod.compute_total_distance(match_points)
    resample_count = args.resample_count
    if args.target_spacing_m > 0:
        resample_count = mod.compute_resample_count(ref_points, args.target_spacing_m, args.resample_max)
    if resample_override:
        resample_count = resample_override
    lat_scale_ref = ref_points[0]["lat"] if ref_points else 0.0
    ref_resampled = mod.resample_points(ref_points, resample_count)
    match_resampled = mod.resample_points(match_points, resample_count)
    ref_shape = mod.build_shape_sequence(ref_resampled, args.shape_mode, lat_scale_ref)
    match_shape = mod.build_shape_sequence(match_resampled, args.shape_mode, lat_scale_ref)
    dtw_dist = mod.make_dtw_distance(args.dtw_penalty, args.dtw_penalty_scale_m, args.dtw_penalty_huber_k)
    dtw_distance, path = mod.fastdtw(match_shape, ref_shape, dist=dtw_dist)
    dtw_avg = dtw_distance / max(1, len(ref_shape))
    dtw_window_max_avg = None
    if args.dtw_window_m > 0 and path:
        step_m = ref_distance / max(1, len(ref_shape) - 1)
        window_steps = max(1, int(round(args.dtw_window_m / max(step_m, 1e-6))))
        costs = [dtw_dist(match_shape[i], ref_shape[j]) for (i, j) in path]
        if costs:
            if len(costs) >= window_steps:
                window_sum = sum(costs[:window_steps])
                dtw_window_max_avg = window_sum / window_steps
                for k in range(window_steps, len(costs)):
                    window_sum += costs[k] - costs[k - window_steps]
                    dtw_window_max_avg = max(dtw_window_max_avg, window_sum / window_steps)
            else:
                dtw_window_max_avg = sum(costs) / max(1, len(costs))

    median_dt = mod.compute_median_time_step(match_points)
    if crossing_edge_window_pts_override is not None:
        edge_window_pts = crossing_edge_window_pts_override
    elif median_dt and args.crossing_edge_window_s > 0:
        edge_window_pts = max(1, int(round(args.crossing_edge_window_s / median_dt)))
    else:
        edge_window_pts = max(1, int(round(0.1 * resample_count)))
    edge_window_pts = min(args.crossing_window_max, edge_window_pts)

    start_cross, end_cross, start_crossings, end_crossings = _compute_crossings(
        mod, match_points, ref_points, edge_window_pts, args.line_length_m
    )

    ref_start = (ref_points[0]["lat"], ref_points[0]["lon"]) if ref_points else (0.0, 0.0)
    ref_end = (ref_points[-1]["lat"], ref_points[-1]["lon"]) if ref_points else (0.0, 0.0)
    start_diff = mod.haversine_distance((start_cross["lat"], start_cross["lon"]), ref_start) if start_cross else None
    end_diff = mod.haversine_distance((end_cross["lat"], end_cross["lon"]), ref_end) if end_cross else None

    if crossing_L_override is not None:
        crossing_L = crossing_L_override
    else:
        crossing_L = max(args.crossing_shape_window_min, int(args.crossing_shape_window_frac * resample_count))
    ref_start_shape = mod.build_shape_sequence(mod.resample_points(ref_points[0:crossing_L], crossing_L), args.shape_mode, lat_scale_ref)
    ref_end_shape = mod.build_shape_sequence(mod.resample_points(ref_points[-crossing_L:], crossing_L), args.shape_mode, lat_scale_ref)

    start_shape_cost = None
    end_shape_cost = None
    pair_shape_cost = None
    if start_cross:
        s_idx = max(0, start_cross["idx1"])
        start_shape_cost = _shape_cost(mod, match_points, ref_start_shape, s_idx, min(len(match_points), s_idx + crossing_L), args.shape_mode, lat_scale_ref, crossing_L)
    if end_cross:
        e_idx = min(len(match_points), end_cross["idx0"] + 1)
        end_shape_cost = _shape_cost(mod, match_points, ref_end_shape, max(0, e_idx - crossing_L), e_idx, args.shape_mode, lat_scale_ref, crossing_L)
    if start_cross and end_cross:
        s_idx = start_cross["idx1"]
        e_idx = end_cross["idx0"]
        pair_resample = max(3, int(0.6 * resample_count))
        pair_resample = min(pair_resample, resample_count)
        ref_pair_shape = mod.build_shape_sequence(mod.resample_points(ref_points, pair_resample), args.shape_mode, lat_scale_ref)
        pair_shape_cost = _shape_cost(mod, match_points, ref_pair_shape, s_idx, e_idx + 1, args.shape_mode, lat_scale_ref, pair_resample)

    xtrack_p95 = None
    xtrack_max = None
    if ref_points and match_points:
        ref_latlon = [(p["lat"], p["lon"]) for p in ref_points]
        ref_xy = mod._points_to_xy(ref_latlon, ref_latlon[0], lat_scale_ref)
        distances = []
        for pt in match_points:
            x, y = mod._point_to_xy(pt["lat"], pt["lon"], ref_latlon[0], lat_scale_ref)
            distances.append(mod.point_to_polyline_distance_xy(x, y, ref_xy))
        xtrack_p95 = _percentile(distances, 95.0)
        xtrack_max = max(distances) if distances else None

    return {
        "ref_distance": ref_distance,
        "match_distance": match_distance,
        "length_ratio": (match_distance / ref_distance) if ref_distance > 0 else None,
        "dtw_avg": dtw_avg,
        "dtw_window_max_avg": dtw_window_max_avg,
        "resample_count": resample_count,
        "edge_window_pts": edge_window_pts,
        "start_cross": start_cross,
        "end_cross": end_cross,
        "start_crossings": start_crossings,
        "end_crossings": end_crossings,
        "start_diff": start_diff,
        "end_diff": end_diff,
        "start_shape_cost": start_shape_cost,
        "end_shape_cost": end_shape_cost,
        "pair_shape_cost": pair_shape_cost,
        "xtrack_p95": xtrack_p95,
        "xtrack_max": xtrack_max,
    }


def _format_cross(cross: Optional[Dict[str, Any]]) -> str:
    if not cross:
        return "none"
    interp = "interp" if cross.get("interp") else "raw"
    tval = cross.get("time")
    tstr = tval.isoformat() if tval else "n/a"
    return f"idx0={cross.get('idx0')} idx1={cross.get('idx1')} {interp} time={tstr}"


def _print_metrics(label: str, metrics: Dict[str, Any]) -> None:
    print(f"{label}")
    print(f"  ref_distance_m={metrics['ref_distance']:.2f}")
    print(f"  match_distance_m={metrics['match_distance']:.2f}")
    if metrics["length_ratio"] is not None:
        print(f"  length_ratio={metrics['length_ratio']:.3f}")
    print(f"  dtw_avg={metrics['dtw_avg']:.3f}")
    if metrics.get("dtw_window_max_avg") is not None:
        print(f"  dtw_window_max_avg={metrics['dtw_window_max_avg']:.3f}")
    print(f"  resample_count={metrics['resample_count']}")
    print(f"  crossing_edge_window_pts={metrics['edge_window_pts']}")
    print(f"  start_cross={_format_cross(metrics['start_cross'])}")
    print(f"  end_cross={_format_cross(metrics['end_cross'])}")
    print(f"  start_crossings={len(metrics['start_crossings'])} end_crossings={len(metrics['end_crossings'])}")
    if metrics["start_diff"] is not None:
        print(f"  start_diff_m={metrics['start_diff']:.2f}")
    if metrics["end_diff"] is not None:
        print(f"  end_diff_m={metrics['end_diff']:.2f}")
    if metrics["start_shape_cost"] is not None:
        print(f"  start_shape_dtw_avg={metrics['start_shape_cost']:.3f}")
    if metrics["end_shape_cost"] is not None:
        print(f"  end_shape_dtw_avg={metrics['end_shape_cost']:.3f}")
    if metrics["pair_shape_cost"] is not None:
        print(f"  pair_shape_dtw_avg={metrics['pair_shape_cost']:.3f}")
    if metrics["xtrack_p95"] is not None:
        print(f"  xtrack_p95_m={metrics['xtrack_p95']:.2f}")
    if metrics["xtrack_max"] is not None:
        print(f"  xtrack_max_m={metrics['xtrack_max']:.2f}")


def _parse_log_config(lines: List[str], label: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    in_block = False
    for line in lines:
        if f"{label} config:" in line:
            in_block = True
            continue
        if in_block:
            if " [INFO]   " not in line:
                if line.strip() == "":
                    break
                continue
            try:
                _, kv = line.split(" [INFO]   ", 1)
            except ValueError:
                continue
            if "=" not in kv:
                continue
            key, val = kv.strip().split("=", 1)
            config[key.strip()] = val.strip()
    return config


def _parse_segment_configs(lines: List[str]) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    in_block = False
    current: Dict[str, str] = {}
    for line in lines:
        if "Segment config:" in line:
            if current:
                blocks.append(current)
            current = {}
            in_block = True
            continue
        if in_block:
            if " [INFO]   " not in line:
                if line.strip() == "":
                    in_block = False
                continue
            try:
                _, kv = line.split(" [INFO]   ", 1)
            except ValueError:
                continue
            if "=" not in kv:
                continue
            key, val = kv.strip().split("=", 1)
            current[key.strip()] = val.strip()
    if current:
        blocks.append(current)
    return blocks


def _apply_log_config(args: argparse.Namespace, config: Dict[str, str]) -> None:
    mapping = {
        "shape_mode": "shape_mode",
        "target_spacing_m": "target_spacing_m",
        "resample_max": "resample_max",
        "resample_count": "resample_count",
        "dtw_penalty": "dtw_penalty",
        "dtw_penalty_scale_m": "dtw_penalty_scale_m",
        "dtw_penalty_huber_k": "dtw_penalty_huber_k",
        "dtw_window_m": "dtw_window_m",
        "dtw_window_max_avg": "dtw_window_max_avg",
        "line_length_m": "line_length_m",
        "crossing_edge_window_s": "crossing_edge_window_s",
        "crossing_window_max": "crossing_window_max",
        "crossing_shape_window_frac": "crossing_shape_window_frac",
        "crossing_shape_window_min": "crossing_shape_window_min",
    }
    for key, attr in mapping.items():
        if key not in config or not hasattr(args, attr):
            continue
        val = config[key]
        current = getattr(args, attr)
        if isinstance(current, float):
            setattr(args, attr, float(val))
        elif isinstance(current, int):
            setattr(args, attr, int(float(val)))
        else:
            setattr(args, attr, val)


def _parse_log_metrics(lines: List[str], segment_name: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    seg_re = re.compile(rf"Segment {re.escape(segment_name)}: indices \((\d+), (\d+)\), ref_dist=([^ ]+) m, detected_dist=([^ ]+) m, dtw_avg=([^ ]+), time=([^,]+), start_diff=([^ ]+) m, end_diff=([^ ]+) m")
    ts_re = re.compile(rf"Segment timestamps for '{re.escape(segment_name)}': start_idx=(\d+) time=([^ ]+) end_idx=(\d+) time=([^ ]+) raw_delta=([^ ]+)")
    cross_re = re.compile(rf"Crossing timestamps for '{re.escape(segment_name)}': start=([^ ]+) end=([^ ]+) delta=([^ ]+)")
    start_interp_re = re.compile(rf"Start interpolation for '{re.escape(segment_name)}': idx0=(\d+) time0=([^ ]+) idx1=(\d+) time1=([^ ]+) interp_time=([^ ]+)")
    end_interp_re = re.compile(rf"Finish interpolation for '{re.escape(segment_name)}': idx0=(\d+) time0=([^ ]+) idx1=(\d+) time1=([^ ]+) interp_time=([^ ]+)")
    for line in lines:
        match = seg_re.search(line)
        if match:
            metrics["seg_indices"] = f"{match.group(1)}-{match.group(2)}"
            metrics["ref_dist_m"] = match.group(3)
            metrics["detected_dist_m"] = match.group(4)
            metrics["dtw_avg"] = match.group(5)
            metrics["time"] = match.group(6)
            metrics["start_diff_m"] = match.group(7)
            metrics["end_diff_m"] = match.group(8)
        match = ts_re.search(line)
        if match:
            metrics["ts_start_idx"] = match.group(1)
            metrics["ts_start_time"] = match.group(2)
            metrics["ts_end_idx"] = match.group(3)
            metrics["ts_end_time"] = match.group(4)
            metrics["ts_raw_delta"] = match.group(5)
        match = cross_re.search(line)
        if match:
            metrics["cross_start_time"] = match.group(1)
            metrics["cross_end_time"] = match.group(2)
            metrics["cross_delta"] = match.group(3)
        match = start_interp_re.search(line)
        if match:
            metrics["start_interp"] = f"{match.group(1)}:{match.group(2)} -> {match.group(3)}:{match.group(4)} = {match.group(5)}"
        match = end_interp_re.search(line)
        if match:
            metrics["end_interp"] = f"{match.group(1)}:{match.group(2)} -> {match.group(3)}:{match.group(4)} = {match.group(5)}"
    return metrics


def _print_log_metrics(metrics: Dict[str, str]) -> None:
    if not metrics:
        print("Log metrics: not found")
        return
    print("Log metrics:")
    for key in sorted(metrics.keys()):
        print(f"  {key}={metrics[key]}")


def _parse_log_metrics_for_export(lines: List[str], export_basename: str) -> Tuple[Dict[str, str], Optional[str]]:
    export_line = f"Exported matched segments to GPX file: {export_basename}"
    metrics: Dict[str, str] = {}
    seg_name: Optional[str] = None
    for idx, line in enumerate(lines):
        if export_line in line:
            for back in range(idx - 1, max(-1, idx - 200), -1):
                seg_line = lines[back]
                if "Segment " in seg_line and ": indices (" in seg_line:
                    seg_part = seg_line.split("Segment ", 1)[1]
                    seg_name = seg_part.split(":", 1)[0].strip()
                    metrics = _parse_log_metrics(lines[back:idx + 1], seg_name)
                    return metrics, seg_name
    return metrics, seg_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug metrics for a match export GPX vs reference segment.")
    parser.add_argument("match_gpx", help="Path to exported match GPX file.")
    parser.add_argument("--ref-segment", action="append", default=[],
                        help="Additional reference segment GPX file to compare.")
    parser.add_argument("--log-file", default=None,
                        help="Optional verbose log from gpx-segment-timer.py for extracting config and metrics.")
    parser.add_argument("--recorded-gpx", default=None,
                        help="Optional recorded GPX to evaluate crossings using original indices.")
    parser.add_argument("--extend-window-pts", type=int, default=1,
                        help="Extra points to include before/after the matched window when using --recorded-gpx.")
    parser.add_argument("--extend-window-s", type=float, default=0.0,
                        help="Extra seconds to include before/after the matched window when using --recorded-gpx.")
    parser.add_argument("--shape-mode", default="step_vectors",
                        choices=["step_vectors", "heading", "centered"],
                        help="Shape mode for DTW.")
    parser.add_argument("--target-spacing-m", type=float, default=8.0,
                        help="Target spacing for resampling.")
    parser.add_argument("--resample-max", type=int, default=400,
                        help="Max resample count when target spacing is used.")
    parser.add_argument("--resample-count", type=int, default=200,
                        help="Fixed resample count when target spacing is not used.")
    parser.add_argument("--dtw-penalty", default="linear",
                        choices=["linear", "quadratic", "huber"],
                        help="DTW distance penalty mode.")
    parser.add_argument("--dtw-penalty-scale-m", type=float, default=10.0,
                        help="DTW penalty scale in meters.")
    parser.add_argument("--dtw-penalty-huber-k", type=float, default=5.0,
                        help="Huber k for DTW penalty.")
    parser.add_argument("--dtw-window-m", type=float, default=0.0,
                        help="Sliding DTW window size in meters (0 disables).")
    parser.add_argument("--dtw-window-max-avg", type=float, default=0.0,
                        help="Reject window if max DTW avg exceeds this (0 disables).")
    parser.add_argument("--line-length-m", type=float, default=8.0,
                        help="Start/finish line total length in meters.")
    parser.add_argument("--crossing-edge-window-s", type=float, default=1.0,
                        help="Crossing edge window in seconds.")
    parser.add_argument("--crossing-window-max", type=int, default=200,
                        help="Max edge window in points.")
    parser.add_argument("--crossing-shape-window-frac", type=float, default=0.2,
                        help="Shape window fraction for crossing context.")
    parser.add_argument("--crossing-shape-window-min", type=int, default=3,
                        help="Minimum shape window for crossings.")
    args = parser.parse_args()

    mod = _load_main_module()
    tracks = _load_tracks(args.match_gpx)
    ref_track = _select_reference_track(tracks)
    match_track = _select_match_track(tracks)

    ref_points = _track_points(ref_track)
    match_points = _track_points(match_track)

    print(f"Match GPX: {args.match_gpx}")
    print(f"  reference_track={ref_track.name}")
    print(f"  match_track={match_track.name}")
    print(f"  ref_points={len(ref_points)} match_points={len(match_points)}")
    print("")

    resample_override = None
    crossing_edge_window_pts_override = None
    crossing_L_override = None
    log_metrics: Dict[str, str] = {}
    if args.log_file:
        with open(args.log_file, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        ref_name = (ref_track.name or "").replace("reference: ", "")
        initial_cfg = _parse_log_config(log_lines, "Initial")
        post_cfg = _parse_log_config(log_lines, "Post-load")
        seg_cfg = {}
        for block in _parse_segment_configs(log_lines):
            if block.get("ref_segment") == ref_name:
                seg_cfg = block
                break
        if initial_cfg:
            print("Log initial config:")
            for key in sorted(initial_cfg.keys()):
                print(f"  {key}={initial_cfg[key]}")
        if post_cfg:
            print("Log post-load config:")
            for key in sorted(post_cfg.keys()):
                print(f"  {key}={post_cfg[key]}")
        if seg_cfg:
            print("Log segment config:")
            for key in sorted(seg_cfg.keys()):
                print(f"  {key}={seg_cfg[key]}")
        export_basename = os.path.basename(args.match_gpx)
        export_metrics, export_seg_name = _parse_log_metrics_for_export(log_lines, export_basename)
        if export_metrics:
            log_metrics = export_metrics
            if export_seg_name and export_seg_name != ref_name:
                print(f"Log segment name for export: {export_seg_name}")
        else:
            log_metrics = _parse_log_metrics(log_lines, ref_name)
        _apply_log_config(args, post_cfg)
        _apply_log_config(args, seg_cfg)
        if seg_cfg.get("resample_count"):
            resample_override = int(float(seg_cfg["resample_count"]))
            args.target_spacing_m = 0.0
        if seg_cfg.get("crossing_edge_window_pts"):
            crossing_edge_window_pts_override = int(float(seg_cfg["crossing_edge_window_pts"]))
        if seg_cfg.get("crossing_L"):
            crossing_L_override = int(float(seg_cfg["crossing_L"]))
        _print_log_metrics(log_metrics)
        print("")

    recorded_points: Optional[List[Dict[str, Any]]] = None
    recorded_window_base: Optional[Tuple[int, int]] = None
    recorded_window_max_extra = 0
    if args.recorded_gpx:
        recorded_points = mod.load_gpx_points(args.recorded_gpx)
        if recorded_points:
            if log_metrics.get("seg_indices"):
                seg_start_str, seg_end_str = log_metrics["seg_indices"].split("-", 1)
                seg_start = int(seg_start_str)
                seg_end = int(seg_end_str)
                extra_pts = max(0, int(args.extend_window_pts))
                if args.extend_window_s > 0:
                    median_dt = mod.compute_median_time_step(recorded_points) or 0.0
                    if median_dt > 0:
                        extra_pts = max(extra_pts, int(round(args.extend_window_s / median_dt)))
                recorded_window_base = (seg_start, seg_end)
                recorded_window_max_extra = extra_pts
            else:
                print("Recorded GPX provided but no seg_indices in log metrics; skipping recorded window.")

    base_metrics = _compute_metrics(
        mod,
        ref_points,
        match_points,
        args,
        resample_override=resample_override,
        crossing_edge_window_pts_override=crossing_edge_window_pts_override,
        crossing_L_override=crossing_L_override,
    )
    _print_metrics("Reference (from GPX):", base_metrics)

    for ref_path in args.ref_segment:
        extra_ref = mod.load_gpx_points(ref_path)
        if not extra_ref:
            print(f"\nReference override ({ref_path}): no points loaded")
            continue
        override_match_points = match_points
        override_label = f"Reference override ({os.path.basename(ref_path)}):"
        if recorded_points is not None and recorded_window_base is not None:
            base_start, base_end = recorded_window_base
            window_start = base_start
            window_end = base_end
            start_extra = 0
            end_extra = 0
            if recorded_window_max_extra > 0:
                window_start, window_end, start_extra, end_extra = _find_min_window_for_crossings(
                    mod,
                    recorded_points,
                    extra_ref,
                    base_start,
                    base_end,
                    recorded_window_max_extra,
                    args.line_length_m,
                )
            if window_end > window_start:
                override_match_points = recorded_points[window_start:window_end]
                override_label = (
                    f"Reference override ({os.path.basename(ref_path)}) using recorded window "
                    f"{window_start}-{window_end} (extra start {start_extra}, end {end_extra}):"
                )
        metrics = _compute_metrics(
            mod,
            extra_ref,
            override_match_points,
            args,
            resample_override=resample_override,
            crossing_edge_window_pts_override=crossing_edge_window_pts_override,
            crossing_L_override=crossing_L_override,
        )
        print("")
        _print_metrics(override_label, metrics)
        print("  delta_vs_gpx_ref:")
        if metrics["dtw_avg"] is not None and base_metrics["dtw_avg"] is not None:
            print(f"    dtw_avg_delta={metrics['dtw_avg'] - base_metrics['dtw_avg']:.3f}")
        if metrics.get("dtw_window_max_avg") is not None and base_metrics.get("dtw_window_max_avg") is not None:
            print(f"    dtw_window_max_avg_delta={metrics['dtw_window_max_avg'] - base_metrics['dtw_window_max_avg']:.3f}")
        if metrics["length_ratio"] is not None and base_metrics["length_ratio"] is not None:
            print(f"    length_ratio_delta={metrics['length_ratio'] - base_metrics['length_ratio']:.3f}")
        if metrics["start_diff"] is not None and base_metrics["start_diff"] is not None:
            print(f"    start_diff_delta_m={metrics['start_diff'] - base_metrics['start_diff']:.2f}")
        if metrics["end_diff"] is not None and base_metrics["end_diff"] is not None:
            print(f"    end_diff_delta_m={metrics['end_diff'] - base_metrics['end_diff']:.2f}")


if __name__ == "__main__":
    main()
