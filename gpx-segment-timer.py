#!/usr/bin/env python3
"""
GPX Segment Time Measurement Script

This script compares a recorded GPX track against reference segments (GPX files)
to measure elapsed time on each segment. It is designed to be robust against varying
sampling frequencies, nonuniform point counts, and repeated segments (e.g., laps).

The matching process is performed in several stages:
  1. A coarse candidate window is identified based on cumulative distances.
  2. A DTW-based refinement (using uniformly resampled points) produces a preliminary candidate.
  3. An iterative grid search (with independent windows for start and end) adjusts the candidate
     boundaries so that the overall candidate segment length approximates that of the reference.
  4. A final local sliding-window refinement ("endpoint anchoring") adjusts each endpoint independently.
     For each endpoint, a short subsegment (length L, default 10% of resample_count, min 3) is compared
     against the corresponding reference subsegment, and the candidate boundary is slid within a local window.
     
By default, if the candidate’s start or end deviates beyond the specified --bbox-margin from the reference,
the segment is rejected. An additional flag (--no-rejection) can be provided to allow segments
with boundary deviations (only a warning is logged).

Detected segments (with boundary, DTW, and timing information) are output and optionally exported as GPX.

Output can be printed as a pretty table (stdout), CSV, or XLSX.

Author: Petr Holub
Date: 2025
"""

import os
import math
import argparse
import datetime
import csv
import logging
import bisect
from typing import List, Tuple, Dict, Optional, Any
import re

import gpxpy
from fastdtw import fastdtw

def log_gpx_input_stats(filepath: str, label: str) -> None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        num_tracks = len(gpx.tracks)
        details = []
        total_pts = 0
        for t_idx, trk in enumerate(gpx.tracks, 1):
            pts_in_track = sum(len(seg.points) for seg in trk.segments)
            total_pts += pts_in_track
            details.append(f"track{t_idx}={pts_in_track} pts in {len(trk.segments)} segs")
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("%s GPX stats: %d track(s), %d total points (%s)", label, num_tracks, total_pts, "; ".join(details))
    except Exception as ex:
        logging.warning("Failed to parse %s for stats: %s", filepath, ex)

def export_candidate_runs_to_gpx(recorded_points: List[Dict[str, Any]],
                                 runs_and_candidates: List[Tuple[int, Tuple[int,int], List[Tuple[int,int]]]],
                                 ref_name: str,
                                 pattern: str) -> None:
    import gpxpy.gpx
    safe_ref = re.sub(r'[^A-Za-z0-9_.-]+', '_', ref_name)
    for run_idx, (rs, re_), cands in runs_and_candidates:
        if not cands:
            continue
        gpx = gpxpy.gpx.GPX()
        for (s, e) in cands:
            trk = gpxpy.gpx.GPXTrack(name=f"{safe_ref}_s{s}_e{e}")
            gpx.tracks.append(trk)
            seg = gpxpy.gpx.GPXTrackSegment()
            trk.segments.append(seg)
            for i in range(s, e+1):
                pt = recorded_points[i]
                seg.points.append(gpxpy.gpx.GPXTrackPoint(pt['lat'], pt['lon'], time=pt.get('time')))
        out_path = pattern.format(ref=safe_ref, run=run_idx, rs=rs, re=re_, n=len(cands))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(gpx.to_xml())
        logging.info("Dumped %d candidate segments to GPX: %s", len(cands), out_path)

# Try to import openpyxl for XLSX output.
try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

# -------------------------------------
# Helper Functions
# -------------------------------------


def _count_runs(arr):
    runs = 0
    prev = False
    for v in arr:
        if v and not prev:
            runs += 1
        prev = v
    return runs

def enforce_single_passage(recorded_points, start_idx, end_idx, ref_start_coords, ref_end_coords, radius_m, edge_frac):
    """Return True if the path between [start_idx, end_idx) touches the start buffer once near the beginning
    and the end buffer once near the end, with no extra re-entries (useful for self-intersecting tracks)."""
    if end_idx <= start_idx or end_idx - start_idx < 2:
        return False
    seg = recorded_points[start_idx:end_idx]
    start_near = [haversine_distance((p['lat'], p['lon']), ref_start_coords) <= radius_m for p in seg]
    end_near = [haversine_distance((p['lat'], p['lon']), ref_end_coords) <= radius_m for p in seg]
    start_runs = _count_runs(start_near)
    end_runs = _count_runs(end_near)
    n = len(seg)
    # Where do we first/last touch?
    try:
        first_start_touch = start_near.index(True)
    except ValueError:
        first_start_touch = None
    try:
        last_end_touch = n - 1 - list(reversed(end_near)).index(True)
    except ValueError:
        last_end_touch = None
    if start_runs > 1 or end_runs > 1:
        return False
    if first_start_touch is None or last_end_touch is None:
        return False
    # Must be close to edges
    edge_window = max(1, int(edge_frac * n))
    if first_start_touch > edge_window:
        return False
    if (n - 1 - last_end_touch) > edge_window:
        return False
    # Also avoid that we keep touching start near the end or end near the beginning (overlap)
    if any(start_near[int(0.5*n):]):  # start buffer shouldn't reappear in latter half
        return False
    if any(end_near[:int(0.5*n)]):    # end buffer shouldn't appear in first half
        return False
    return True
def haversine_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute the haversine distance (in meters) between two (lat, lon) points."""
    radius = 6371000  # Earth's radius in meters
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return radius * c

def load_gpx_points(filepath: str) -> List[Dict[str, Any]]:
    """Parse a GPX file and return a list of point dictionaries with keys 'lat', 'lon', and 'time'."""
    points: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'time': point.time
                })
    logging.debug("Loaded %d points from %s.", len(points), filepath)
    return points

def load_reference_segments(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all GPX files from the specified folder as reference segments."""
    segments: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(folder):
        raise ValueError(f"Reference folder '{folder}' is not a valid directory.")
    for filename in os.listdir(folder):
        if filename.lower().endswith('.gpx'):
            filepath = os.path.join(folder, filename)
            log_gpx_input_stats(filepath, f"Reference {filename}")
            pts = load_gpx_points(filepath)
            if pts:
                segments[filename] = pts
                logging.debug("Loaded reference segment '%s' with %d points.", filename, len(pts))
    return segments

def compute_total_distance(points: List[Dict[str, Any]]) -> float:
    """Compute the total distance (in meters) of a track by summing successive point distances."""
    assert len(points) >= 1, "At least one point required."
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_distance((points[i-1]['lat'], points[i-1]['lon']),
                                    (points[i]['lat'], points[i]['lon']))
    logging.debug("Computed total distance: %.2f m for %d points.", total, len(points))
    return total

def resample_points(points: List[Dict[str, Any]], num_samples: int) -> List[Tuple[float, float]]:
    """
    Uniformly resample points along the cumulative distance.
    Returns a list of (lat, lon) tuples.
    """
    if not points:
        return []
    if len(points) < 2:
        return [(points[0]['lat'], points[0]['lon'])]
    cum_dists = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance((points[i-1]['lat'], points[i-1]['lon']),
                               (points[i]['lat'], points[i]['lon']))
        cum_dists.append(cum_dists[-1] + d)
    total_dist = cum_dists[-1]
    targets = [i * total_dist / (num_samples - 1) for i in range(num_samples)]
    new_pts: List[Tuple[float, float]] = []
    j = 0
    for t in targets:
        while j < len(cum_dists) - 1 and cum_dists[j+1] < t:
            j += 1
        if j >= len(points) - 1:
            new_pts.append((points[-1]['lat'], points[-1]['lon']))
        else:
            seg_len = cum_dists[j+1] - cum_dists[j]
            if seg_len == 0:
                new_pts.append((points[j]['lat'], points[j]['lon']))
            else:
                r = (t - cum_dists[j]) / seg_len
                lat = points[j]['lat'] + r * (points[j+1]['lat'] - points[j]['lat'])
                lon = points[j]['lon'] + r * (points[j+1]['lon'] - points[j]['lon'])
                new_pts.append((lat, lon))
    #logging.debug("Resampled from %d to %d points.", len(points), num_samples)
    return new_pts

def compute_bounding_box(points: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Compute the bounding box (min_lat, max_lat, min_lon, max_lon) for a set of points."""
    lats = [p['lat'] for p in points]
    lons = [p['lon'] for p in points]
    return (min(lats), max(lats), min(lons), max(lons))

def expand_bounding_box(bbox: Tuple[float, float, float, float], margin_m: float = 30) -> Tuple[float, float, float, float]:
    """Expand a bounding box by margin_m meters."""
    min_lat, max_lat, min_lon, max_lon = bbox
    margin_deg_lat = margin_m / 111000
    avg_lat = (min_lat + max_lat) / 2
    margin_deg_lon = margin_m / (111000 * math.cos(math.radians(avg_lat)))
    return (min_lat - margin_deg_lat, max_lat + margin_deg_lat, min_lon - margin_deg_lon, max_lon + margin_deg_lon)

def point_in_bbox(point: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> bool:
    """Return True if the point lies within the given bounding box."""
    lat, lon = point['lat'], point['lon']
    min_lat, max_lat, min_lon, max_lon = bbox
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

def refine_boundaries_using_warping_path(recorded_points: List[Dict[str, Any]],
                         candidate_start: int,
                         candidate_end: int,
                         ref_resampled: List[Tuple[float, float]],
                         resample_count: int) -> Tuple[int, int]:
    """
    Use the DTW warping path from the candidate segment (resampled) to determine indices in recorded_points
    that best align with the reference's start and end.
    """
    candidate_segment = recorded_points[candidate_start:candidate_end]
    candidate_resampled = resample_points(candidate_segment, resample_count)
    _, path = fastdtw(candidate_resampled, ref_resampled, dist=haversine_distance)
    start_indices = [i for (i, j) in path if j == 0]
    end_indices = [i for (i, j) in path if j == len(ref_resampled) - 1]
    min_i = min(start_indices) if start_indices else 0
    max_i = max(end_indices) if end_indices else len(candidate_resampled) - 1
    candidate_length = candidate_end - candidate_start
    refined_start = candidate_start + round(min_i * (candidate_length - 1) / (resample_count - 1))
    refined_end = candidate_start + round(max_i * (candidate_length - 1) / (resample_count - 1)) + 1
    return refined_start, refined_end

def adjust_boundaries(recorded_points: List[Dict[str, Any]],
                      current_index: int,
                      ref_coord: Tuple[float, float],
                      window: int = 20) -> int:
    """
    Adjust a boundary index within ±window points to locate the point in recorded_points closest to ref_coord.
    """
    best_idx = current_index
    best_dist = haversine_distance((recorded_points[current_index]['lat'], recorded_points[current_index]['lon']), ref_coord)
    start_search = max(0, current_index - window)
    end_search = min(len(recorded_points) - 1, current_index + window)
    for i in range(start_search, end_search + 1):
        d = haversine_distance((recorded_points[i]['lat'], recorded_points[i]['lon']), ref_coord)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx

def refine_endpoint_boundary(recorded_points: List[Dict[str, Any]],
                             candidate_index: int,
                             ref_sub: List[Tuple[float, float]],
                             L: int,
                             window: int,
                             is_start: bool) -> int:
    """
    Refine a single endpoint (start if is_start True, else end) via a local sliding window.
    For the start, candidate subsegments of length L starting at each candidate index (within ±window)
    are compared with ref_sub (the first L reference points). For the end, candidate subsegments ending at each
    candidate index are compared with the last L reference points.
    Returns the candidate index that minimizes the sum of pointwise haversine distances.
    """
    best_idx = candidate_index
    best_cost = float('inf')
    candidate_range = range(max(0, candidate_index - window), min(len(recorded_points), candidate_index + window + 1))
    for i in candidate_range:
        if is_start:
            candidate_sub = recorded_points[i:i+L]
            if len(candidate_sub) < L:
                continue
            candidate_resampled = resample_points(candidate_sub, L)
        else:
            start_i = i - L + 1
            if start_i < 0:
                continue
            candidate_sub = recorded_points[start_i:i+1]
            if len(candidate_sub) < L:
                continue
            candidate_resampled = resample_points(candidate_sub, L)
        cost = sum(haversine_distance(candidate_resampled[k], ref_sub[k]) for k in range(L))
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx

def refine_boundaries_iteratively(recorded_points: List[Dict[str, Any]],
                                  initial_start: int,
                                  initial_end: int,
                                  ref_start_coord: Tuple[float, float],
                                  ref_end_coord: Tuple[float, float],
                                  ref_total_distance: float,
                                  rec_cum_dists: List[float],
                                  iterative_window_start: int,
                                  iterative_window_end: int,
                                  penalty_weight: float = 1.0) -> Tuple[int, int]:
    """
    Iteratively refine candidate boundaries by grid search over start and end indices independently.
    
    The cost is:
      cost(i, j) = |(rec_cum_dists[j] - rec_cum_dists[i]) - ref_total_distance| +
                   penalty_weight * (
                       haversine_distance((recorded_points[i]['lat'], recorded_points[i]['lon']), ref_start_coord) +
                       haversine_distance((recorded_points[j-1]['lat'], recorded_points[j-1]['lon']), ref_end_coord)
                   )
    Returns candidate boundaries (i, j) with minimum cost.
    """
    best_i = initial_start
    best_j = initial_end
    candidate_distance = rec_cum_dists[initial_end] - rec_cum_dists[initial_start]
    best_cost = abs(candidate_distance - ref_total_distance) + penalty_weight * (
        haversine_distance((recorded_points[initial_start]['lat'], recorded_points[initial_start]['lon']), ref_start_coord) +
        haversine_distance((recorded_points[initial_end-1]['lat'], recorded_points[initial_end-1]['lon']), ref_end_coord)
    )
    i_min = max(0, initial_start - iterative_window_start)
    i_max = min(len(recorded_points) - 2, initial_start + iterative_window_start)
    j_min = max(initial_end - iterative_window_end, i_min + 2)
    j_max = min(len(recorded_points), initial_end + iterative_window_end)
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if j <= i + 1:
                continue
            seg_distance = rec_cum_dists[j] - rec_cum_dists[i]
            cost = abs(seg_distance - ref_total_distance) + penalty_weight * (
                haversine_distance((recorded_points[i]['lat'], recorded_points[i]['lon']), ref_start_coord) +
                haversine_distance((recorded_points[j-1]['lat'], recorded_points[j-1]['lon']), ref_end_coord)
            )
            if cost < best_cost:
                best_cost = cost
                best_i = i
                best_j = j
    return best_i, best_j

def refine_boundaries_with_endpoint_anchoring(recorded_points: List[Dict[str, Any]],
                                              initial_start: int,
                                              initial_end: int,
                                              ref_points: List[Dict[str, Any]],
                                              rec_cum_dists: List[float],
                                              iterative_window_start: int,
                                              iterative_window_end: int,
                                              anchor_alpha: float,
                                              anchor_beta1: float,
                                              anchor_beta2: float,
                                              resample_count: int) -> Tuple[int, int]:
    """
    Refine candidate boundaries using endpoint anchoring.
    
    The function computes the overall full cost (via DTW on the entire segment) and then performs
    a grid search (within specified windows) that adds:
      - a cost for the start subsegment (first L points) weighted by anchor_beta1,
      - a cost for the end subsegment (last L points) weighted by anchor_beta2,
      - a boundary penalty (Euclidean distances from candidate endpoints to reference endpoints) weighted by anchor_alpha.
    
    L is set to max(3, int(0.1 * resample_count)).
    Returns refined candidate boundaries (i, j).
    """
    ref_total_distance = compute_total_distance(ref_points)
    ref_resampled = resample_points(ref_points, resample_count)
    candidate_full = recorded_points[initial_start:initial_end]
    candidate_full_resampled = resample_points(candidate_full, resample_count)
    full_cost = fastdtw(candidate_full_resampled, ref_resampled, dist=haversine_distance)[0] / resample_count

    L = max(3, int(0.1 * resample_count))
    ref_start_sub = resample_points(ref_points[0:L], L)
    ref_end_sub = resample_points(ref_points[-L:], L)

    best_i = initial_start
    best_j = initial_end
    best_cost = full_cost + anchor_alpha * (
        haversine_distance((recorded_points[initial_start]['lat'], recorded_points[initial_start]['lon']),
                           (ref_points[0]['lat'], ref_points[0]['lon'])) +
        haversine_distance((recorded_points[initial_end-1]['lat'], recorded_points[initial_end-1]['lon']),
                           (ref_points[-1]['lat'], ref_points[-1]['lon']))
    )

    i_min = max(0, initial_start - iterative_window_start)
    i_max = min(len(recorded_points) - 2, initial_start + iterative_window_start)
    j_min = max(initial_end - iterative_window_end, i_min + 2)
    j_max = min(len(recorded_points), initial_end + iterative_window_end)
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if j <= i + 1:
                continue
            candidate_resampled = resample_points(recorded_points[i:j], resample_count)
            full_cost_candidate = fastdtw(candidate_resampled, ref_resampled, dist=haversine_distance)[0] / resample_count
            candidate_start_sub = resample_points(recorded_points[i:i+L], L) if len(recorded_points[i:i+L]) >= L else None
            candidate_end_sub = resample_points(recorded_points[j-L:j], L) if len(recorded_points[j-L:j]) >= L else None
            if candidate_start_sub is None or candidate_end_sub is None:
                continue
            start_cost = fastdtw(candidate_start_sub, ref_start_sub, dist=haversine_distance)[0] / L
            end_cost = fastdtw(candidate_end_sub, ref_end_sub, dist=haversine_distance)[0] / L
            boundary_penalty = anchor_alpha * (
                haversine_distance((recorded_points[i]['lat'], recorded_points[i]['lon']),
                                   (ref_points[0]['lat'], ref_points[0]['lon'])) +
                haversine_distance((recorded_points[j-1]['lat'], recorded_points[j-1]['lon']),
                                   (ref_points[-1]['lat'], ref_points[-1]['lon']))
            )
            cost = full_cost_candidate + anchor_beta1 * start_cost + anchor_beta2 * end_cost + boundary_penalty
            if cost < best_cost:
                best_cost = cost
                best_i = i
                best_j = j
    return best_i, best_j

def find_all_segment_matches(recorded_points: List[Dict[str, Any]],
                             ref_points: List[Dict[str, Any]],
                             candidate_margin: float,
                             dtw_threshold: float,
                             resample_count: int,
                             min_gap: int = 1,
                             bbox_margin_m: float = 30,
                             bbox_margin_overall_m: float | None = None,
                             dump_pattern: Optional[str] = None,
                             ref_name: Optional[str] = None) -> List[Tuple[int, int, float, int, int]]:
    """
    Three-bbox candidate search with logging and pre-counting of DTW candidates.
    See docstring in patch description for details.
    """
    assert recorded_points, "Recorded track must contain points."
    assert ref_points, "Reference segment must contain points."
    assert candidate_margin > 0, "candidate_margin must be positive."
    assert resample_count >= 2, "resample_count must be at least 2."

    ref_total_distance = compute_total_distance(ref_points)
    ref_resampled = resample_points(ref_points, resample_count)

    # Build bboxes
    ref_bbox = compute_bounding_box(ref_points)
    overall_margin = bbox_margin_overall_m if (bbox_margin_overall_m is not None) else bbox_margin_m
    bbox_overall = expand_bounding_box(ref_bbox, margin_m=overall_margin)

    ref_start = {'lat': ref_points[0]['lat'], 'lon': ref_points[0]['lon']}
    ref_end   = {'lat': ref_points[-1]['lat'], 'lon': ref_points[-1]['lon']}
    bbox_start = expand_bounding_box((ref_start['lat'], ref_start['lat'], ref_start['lon'], ref_start['lon']), margin_m=bbox_margin_m)
    bbox_end   = expand_bounding_box((ref_end['lat'],   ref_end['lat'],   ref_end['lon'],   ref_end['lon']),   margin_m=bbox_margin_m)

    # Precompute cumulative distances and "outside overall bbox" prefix sum
    rec_cum_dists: List[float] = [0.0]
    outside = [0] * len(recorded_points)
    for i in range(len(recorded_points)):
        if not point_in_bbox(recorded_points[i], bbox_overall):
            outside[i] = 1
        if i > 0:
            d = haversine_distance((recorded_points[i-1]['lat'], recorded_points[i-1]['lon']),
                                   (recorded_points[i]['lat'], recorded_points[i]['lon']))
            rec_cum_dists.append(rec_cum_dists[-1] + d)
    # prefix sum of "outside" to quickly check segments
    pref_out = [0] * (len(recorded_points) + 1)
    for i, v in enumerate(outside, start=1):
        pref_out[i] = pref_out[i-1] + v

    # Phase 1: enumerate candidate (s,e) pairs
    n = len(recorded_points)
    import bisect as _bis

    start_inside = [point_in_bbox(recorded_points[i], bbox_start) for i in range(n)]
    # Log contiguous runs of start_inside == True
    runs = []
    run_s = None
    for i, flag in enumerate(start_inside + [False]):  # sentinel to flush
        if flag and run_s is None:
            run_s = i
        elif not flag and run_s is not None:
            runs.append((run_s, i-1))
            run_s = None

    if logging.getLogger().isEnabledFor(logging.INFO):
        if runs:
            logging.info("Start-bbox runs (inclusive indices): %s", ", ".join(f"[{a},{b}]" for a,b in runs))
        else:
            logging.info("No points in recorded track lie within start-bbox; skipping.")

    candidates: List[Tuple[int,int]] = []
    runs_and_candidates: List[Tuple[int, Tuple[int,int], List[Tuple[int,int]]]] = []
    # For each start index s inside start-bbox, compute e-range by distance window and filter
    for (rs, re) in runs:
        run_candidates_before_filter = 0
        accepted_in_run = 0
        # We will log the union of [e_lo, e_hi) ranges for the run
        e_ranges = []
        s = rs
        run_cands: List[Tuple[int,int]] = []
        while s <= re:
            if not start_inside[s]:
                s += 1
                continue
            lower = rec_cum_dists[s] + ref_total_distance * (1 - candidate_margin)
            upper = rec_cum_dists[s] + ref_total_distance * (1 + candidate_margin)
            e_lo = _bis.bisect_left(rec_cum_dists, lower, lo=s+1, hi=n-1)
            e_hi = _bis.bisect_right(rec_cum_dists, upper, lo=s+1, hi=n-1)
            if e_lo < e_hi:
                e_ranges.append((e_lo, e_hi))
            run_candidates_before_filter += max(0, e_hi - e_lo)

            for e in range(e_lo, e_hi):
                # end must be in end-bbox
                if not point_in_bbox(recorded_points[e], bbox_end):
                    continue
                # All points between s..e must lie within overall bbox
                outside_cnt = pref_out[e+1] - pref_out[s]
                if outside_cnt != 0:
                    continue
                if e <= s:
                    continue
                candidates.append((s, e))
                run_cands.append((s, e))
                accepted_in_run += 1
            s += 1

        if logging.getLogger().isEnabledFor(logging.INFO):
            if e_ranges:
                # merge overlapping e_ranges for clearer logging
                e_ranges.sort()
                merged = []
                for a,b in e_ranges:
                    if not merged or a > merged[-1][1]:
                        merged.append([a,b])
                    else:
                        merged[-1][1] = max(merged[-1][1], b)
                e_ranges_str = ", ".join(f"[{a},{b})" for a,b in merged)
                logging.info("For start-run [%d,%d], end-index windows by distance: %s", rs, re, e_ranges_str)
            logging.info("Run [%d,%d]: %d distance-window pairs pre-filter; %d candidates after bbox filters.",
                         rs, re, run_candidates_before_filter, accepted_in_run)
            if accepted_in_run == 0:
                logging.info("Run [%d,%d] produced no valid end candidates; discarding this start-bbox excursion.", rs, re)
            else:
                runs_and_candidates.append((len(runs_and_candidates), (rs, re), run_cands))

    if dump_pattern and runs_and_candidates and ref_name:
        try:
            export_candidate_runs_to_gpx(recorded_points, runs_and_candidates, ref_name, dump_pattern)
        except Exception as ex:
            logging.warning("Failed to dump candidate runs to GPX: %s", ex)
    total_candidates = len(candidates)
    if logging.getLogger().isEnabledFor(logging.INFO):
        logging.info("Total DTW candidates to assess: %d", total_candidates)

    # Early exit
    if total_candidates == 0:
        return []

    # Phase 2: compute DTW best per run (not global), then return list of per-run winners.
    matches: List[Tuple[int, int, float, int, int]] = []
    for run_idx, (rs, re), run_cands in runs_and_candidates:
        best = None  # (avg_cost, s, e)
        for s, e in run_cands:
            cand_pts = recorded_points[s:e+1]
            if len(cand_pts) < 2:
                continue
            cand_resampled = resample_points(cand_pts, resample_count)
            dtw_distance, _ = fastdtw(cand_resampled, ref_resampled, dist=haversine_distance)
            avg_cost = dtw_distance / resample_count
            logging.debug("DTW for run=%d s=%d e=%d: total %f, avg %f.", run_idx, s, e, dtw_distance, avg_cost)
            if best is None or avg_cost < best[0]:
                best = (avg_cost, s, e)
        if best is not None and best[0] < dtw_threshold:
            avg_cost, s_idx, e_idx = best
            matches.append((s_idx, e_idx+1, avg_cost, s_idx, e_idx+1))
            logging.info("Run %d winner: s=%d e=%d (exclusive), dtw_avg=%.2f", run_idx, s_idx, e_idx+1, avg_cost)
        else:
            logging.info("Run %d had no DTW winner under threshold (%.2f).", run_idx, dtw_threshold)
    return matches



def measure_segment_time(recorded_points: List[Dict[str, Any]],
                         start_idx: int,
                         end_idx: int) -> Optional[float]:
    """
    Compute the elapsed time (in seconds) between the first and last point of a segment.
    """
    if not (0 <= start_idx < end_idx <= len(recorded_points)):
        logging.warning("Invalid indices for time measurement: start=%s end=%s len(points)=%s", start_idx, end_idx, len(recorded_points))
        return None
    start_time = recorded_points[start_idx]['time']
    end_time = recorded_points[end_idx - 1]['time']
    if start_time and end_time:
        seconds = (end_time - start_time).total_seconds()
        logging.debug("Measured time for segment (%d, %d): %.2f seconds", start_idx, end_idx, seconds)
        return seconds
    logging.warning("Missing timestamp for segment endpoints %d and %d.", start_idx, end_idx)
    return None

def export_matches_to_gpx(matches: List[Tuple[int, int, float]],
                          recorded_points: List[Dict[str, Any]],
                          recorded_filename: str,
                          ref_filename: str,
                          output_gpx: str) -> None:
    """
    Export matched segments as individual GPX tracks in a file.
    Each track is named using the recorded filename, reference filename, and a match number.
    """
    gpx = gpxpy.gpx.GPX()
    for i, (start_idx, end_idx, dtw_avg) in enumerate(matches, start=1):
        track = gpxpy.gpx.GPXTrack()
        track.name = f"{os.path.basename(recorded_filename)} - {ref_filename} (match {i})"
        segment = gpxpy.gpx.GPXTrackSegment()
        for pt in recorded_points[start_idx:end_idx]:
            segment.points.append(gpxpy.gpx.GPXTrackPoint(pt['lat'], pt['lon'], time=pt['time']))
        track.segments.append(segment)
        gpx.tracks.append(track)
    with open(output_gpx, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())
    logging.info("Exported matched segments to GPX file: %s", output_gpx)

def output_results(results: List[Dict[str, Any]],
                   output_mode: str,
                   output_file: Optional[str]) -> None:
    """
    Output results in stdout, csv, or xlsx.
    
    The output columns include:
      - "Ref Start": reference segment start coordinates.
      - "Ref End": reference segment end coordinates.
      - "Start Diff (m)": difference between detected and reference start.
      - "End Diff (m)": difference between detected and reference end.
      - "Ref Dist (m)": total distance of the reference segment.
      - "Detected Dist (m)": total distance of the detected segment.
    """
    header = ["Segment", "Start Idx", "End Idx", "Ref Dist (m)", "Detected Dist (m)",
              "DTW Avg (m)", "Time (s)", "Time (H:M:S)",
              "Ref Start", "Ref End", "Start Diff (m)", "End Diff (m)"]
    if output_mode == "stdout":
        print("{:<25} {:>10} {:>10} {:>15} {:>20} {:>15} {:>12} {:>15} {:>25} {:>25} {:>18} {:>15}".format(*header))
        print("-" * 200)
        for res in results:
            print("{:<25} {:>10} {:>10} {:>15.2f} {:>20.2f} {:>15.2f} {:>12.2f} {:>15} {:>25} {:>25} {:>18.2f} {:>15.2f}".format(
                res["segment"],
                res["start_index"],
                res["end_index"],
                res["ref_distance"],
                res["detected_distance"],
                res["dtw_avg"],
                res["time_seconds"] if res["time_seconds"] is not None else -1,
                res["time_str"] if res["time_str"] is not None else "N/A",
                res["ref_start"],
                res["ref_end"],
                res["start_diff"],
                res["end_diff"]
            ))
    elif output_mode == "csv":
        if output_file is None:
            raise ValueError("Output file must be provided for CSV output.")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for res in results:
                writer.writerow([
                    res["segment"],
                    res["start_index"],
                    res["end_index"],
                    f"{res['ref_distance']:.2f}",
                    f"{res['detected_distance']:.2f}",
                    f"{res['dtw_avg']:.2f}",
                    f"{res['time_seconds']:.2f}" if res["time_seconds"] is not None else "",
                    res["time_str"] if res["time_str"] is not None else "",
                    res["ref_start"],
                    res["ref_end"],
                    f"{res['start_diff']:.2f}",
                    f"{res['end_diff']:.2f}"
                ])
        logging.info("Results written to CSV file: %s", output_file)
    elif output_mode == "xlsx":
        if output_file is None:
            raise ValueError("Output file must be provided for XLSX output.")
        if Workbook is None:
            raise ImportError("openpyxl is required for XLSX output.")
        wb = Workbook()
        ws = wb.active
        ws.title = "Segment Times"
        ws.append(header)
        for res in results:
            ws.append([
                res["segment"],
                res["start_index"],
                res["end_index"],
                float(f"{res['ref_distance']:.2f}"),
                float(f"{res['detected_distance']:.2f}"),
                float(f"{res['dtw_avg']:.2f}"),
                float(f"{res['time_seconds']:.2f}") if res["time_seconds"] is not None else None,
                res["time_str"] if res["time_str"] is not None else "",
                res["ref_start"],
                res["ref_end"],
                float(f"{res['start_diff']:.2f}"),
                float(f"{res['end_diff']:.2f}")
            ])
        wb.save(output_file)
        logging.info("Results written to XLSX file: %s", output_file)
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

# -------------------------------------
# Main Routine
# -------------------------------------

def main() -> None:
    """
    Main routine: Parses command-line arguments, loads the recorded track and reference segments,
    finds candidate matches, and refines candidate boundaries in several stages:
      1. Coarse candidate selection (cumulative distance and DTW).
      2. Iterative grid refinement (with independent windows for start and end).
      3. Final local sliding-window endpoint anchoring.
    
    The detected segment’s boundaries are compared against the reference endpoints.
    By default, if the difference exceeds --bbox-margin, the segment is rejected.
    With the flag --no-rejection the segment is stored (with a warning).
    Optionally, matched segments are exported as GPX tracks.
    Output includes both the reference segment's length and the detected segment's length.
    """
    parser = argparse.ArgumentParser(
        description="Measure elapsed time on reference segments within a recorded GPX track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-r", "--recorded", required=True,
                        help="Path to the GPX file containing the recorded track.")
    parser.add_argument("-f", "--reference-folder", required=True,
                        help="Path to the folder containing reference GPX segment files.")
    parser.add_argument("-o", "--output-mode", choices=["stdout", "csv", "xlsx"], default="stdout",
                        help="Output format: pretty printed to STDOUT, CSV, or XLSX.")
    parser.add_argument("-O", "--output-file", default=None,
                        help="Output file path (required for CSV and XLSX outputs).")
    parser.add_argument("--candidate-margin", type=float, default=0.2,
                        help="Allowed variation (fraction) in candidate segment distance relative to reference.")
    parser.add_argument("--allow-length-mismatch", action="store_true",
                        help="Allow final detected segment length to differ from reference beyond candidate-margin without rejecting the match.")
    parser.add_argument("--dtw-threshold", type=float, default=50,
                        help="Maximum allowed average DTW distance (m per resampled point) for a match.")
    parser.add_argument("--resample-count", type=int, default=50,
                        help="Number of points for resampling segments.")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="Minimum number of recorded points to skip after a match.")
    parser.add_argument("--bbox-margin", type=float, default=30,
                        help="Endpoint bbox expansion (meters) for start/end checks.")
    parser.add_argument("--bbox-margin-overall", type=float, default=None,
                        help="Overall bbox expansion (meters) for containment check. If not set, defaults to --bbox-margin.")
    parser.add_argument("--single-passage", action="store_true",
                        help="Enforce that within the detected segment the trajectory enters the start buffer once at the beginning and the end buffer once at the end.")
    parser.add_argument("--passage-radius", type=float, default=30.0,
                        help="Radius in meters for start/end buffers used by --single-passage (defaults close to bbox-margin).")
    parser.add_argument("--passage-edge-frac", type=float, default=0.10,
                        help="How close to the segment edges the start/end buffer touches must occur (fraction of point count).")
    parser.add_argument("--refine-window", type=int, default=5,
                        help="Window size (in points) for initial shape-based boundary refinement.")
    parser.add_argument("--iterative-window-start", type=int, default=50,
                        help="Search window (in points) for adjusting the start boundary independently.")
    parser.add_argument("--iterative-window-end", type=int, default=50,
                        help="Search window (in points) for adjusting the end boundary independently.")
    parser.add_argument("--penalty-weight", type=float, default=2.0,
                        help="Weight for the Euclidean endpoint penalty in grid refinement (formerly lambda-weight).")
    parser.add_argument("--anchor-beta1", type=float, default=1.0,
                        help="Weight for the DTW cost on the start subsegment in endpoint anchoring.")
    parser.add_argument("--anchor-beta2", type=float, default=1.0,
                        help="Weight for the DTW cost on the end subsegment in endpoint anchoring.")
    parser.add_argument("--endpoint-window-start", type=int, default=1000,
                        help="Local sliding window (in points) for refining the start boundary.")
    parser.add_argument("--endpoint-window-end", type=int, default=1000,
                        help="Local sliding window (in points) for refining the end boundary.")
    parser.add_argument("--no-refinement", action="store_true",
                        help="Disable final boundary refinements; use DTW-selected candidate indices as the result.")
    # New flag: by default segments are rejected when boundaries deviate beyond --bbox-margin.
    parser.add_argument("--no-rejection", action="store_true",
                        help="Do not reject segments that deviate beyond --bbox-margin (only log a warning).")
    parser.add_argument("--skip-endpoint-checks", action="store_true",
                        help="Alias to disable endpoint deviation rejection (sets --no-rejection)."),
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging (INFO level).")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging (DEBUG level).")
    parser.add_argument("--export-gpx", action="store_true",
                        help="Export matched segments as individual GPX tracks.")
    parser.add_argument("--export-gpx-file", default="matched_segments.gpx",
                        help="Output GPX file for exporting matched segments.")
    parser.add_argument("--dump-candidates-gpx", default=None,
                        help="If set, dumps per start-bbox run the candidate segments (after bbox filters) into GPX files. Use placeholders {ref}, {run}, {rs}, {re}, {n}.")
    args = parser.parse_args()
    if getattr(args, 'skip_endpoint_checks', False):
        args.no_rejection = True

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.output_mode in ["csv", "xlsx"] and not args.output_file:
        parser.error("Output file must be provided for CSV or XLSX outputs.")

    logging.info("Loading recorded track: %s", args.recorded)
    log_gpx_input_stats(args.recorded, "Recorded")
    recorded_points = load_gpx_points(args.recorded)
    if not recorded_points:
        logging.error("No points found in recorded GPX file: %s", args.recorded)
        return

    logging.info("Loading reference segments from folder: %s", args.reference_folder)
    ref_segments = load_reference_segments(args.reference_folder)
    if not ref_segments:
        logging.error("No valid reference GPX files found in: %s", args.reference_folder)
        return

    # Precompute cumulative distances for the recorded track.
    rec_cum_dists: List[float] = [0.0]
    for i in range(1, len(recorded_points)):
        d = haversine_distance((recorded_points[i-1]['lat'], recorded_points[i-1]['lon']),
                               (recorded_points[i]['lat'], recorded_points[i]['lon']))
        rec_cum_dists.append(rec_cum_dists[-1] + d)

    results: List[Dict[str, Any]] = []
    for seg_filename, ref_points in ref_segments.items():
        logging.info("Processing reference segment: %s", seg_filename)
        if not ref_points:
            logging.warning("Reference segment %s has no points; skipping.", seg_filename)
            continue

        ref_start_coords = (ref_points[0]['lat'], ref_points[0]['lon'])
        ref_end_coords = (ref_points[-1]['lat'], ref_points[-1]['lon'])
        ref_distance = compute_total_distance(ref_points)
        logging.info("Reference '%s' start: (%.6f, %.6f), end: (%.6f, %.6f), distance: %.2f m",
                     seg_filename, ref_start_coords[0], ref_start_coords[1],
                     ref_end_coords[0], ref_end_coords[1], ref_distance)
        current_ref_resampled = resample_points(ref_points, args.resample_count)

        matches = find_all_segment_matches(
            recorded_points,
            ref_points,
            candidate_margin=args.candidate_margin,
            dtw_threshold=args.dtw_threshold,
            resample_count=args.resample_count,
            min_gap=args.min_gap,
            bbox_margin_m=args.bbox_margin,
            bbox_margin_overall_m=(args.bbox_margin_overall if hasattr(args, 'bbox_margin_overall') else None),
            dump_pattern=args.dump_candidates_gpx,
            ref_name=seg_filename
        )
        if not matches:
            logging.info("No matching segments found for reference '%s'.", seg_filename)
            continue

        L = max(3, int(0.1 * args.resample_count))
        ref_start_sub = resample_points(ref_points[0:L], L)
        ref_end_sub = resample_points(ref_points[-L:], L)

        if args.no_refinement:
            final_start, final_end = start_idx, end_idx
            used_fallback = False
        for (start_idx, end_idx, dtw_avg, orig_start, orig_end) in matches:
            if not args.no_refinement:
                refined_start, refined_end = refine_boundaries_using_warping_path(
                recorded_points, start_idx, end_idx, current_ref_resampled, args.resample_count)
            if not args.no_refinement:
                grid_start, grid_end = refine_boundaries_iteratively(
                recorded_points, refined_start, refined_end,
                ref_start_coords, ref_end_coords, ref_distance, rec_cum_dists,
                iterative_window_start=args.iterative_window_start,
                iterative_window_end=args.iterative_window_end,
                penalty_weight=args.penalty_weight)
            if not args.no_refinement:
                final_start = refine_endpoint_boundary(recorded_points, grid_start, ref_start_sub, L, args.endpoint_window_start, True)
            if not args.no_refinement:
                final_end = refine_endpoint_boundary(recorded_points, grid_end - 1, ref_end_sub, L, args.endpoint_window_end, False) + 1
            # Fallback if refinement breaks monotonicity/length
            used_fallback = False
            if final_end <= final_start:
                logging.warning("Refinement produced end<=start; falling back to original candidate [%d,%d).", orig_start, orig_end)
                final_start, final_end = orig_start, orig_end
                used_fallback = True

            detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
            if detected_distance <= 0 and not used_fallback:
                logging.warning("Refinement produced non-positive length; falling back to original candidate [%d,%d).", orig_start, orig_end)
                final_start, final_end = orig_start, orig_end
                detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
                used_fallback = True
            if final_end <= final_start:
                logging.warning("Rejected match for '%s': end index (%d) <= start index (%d).", seg_filename, final_end, final_start)
                continue
            if detected_distance <= 0:
                logging.warning("Rejected match for '%s': non-positive detected distance %.2f m.", seg_filename, detected_distance)
                continue
            length_ratio = detected_distance / ref_distance if ref_distance > 0 else float('inf')
            min_ratio = 1.0 - args.candidate_margin
            max_ratio = 1.0 + args.candidate_margin
            if not args.allow_length_mismatch and not (min_ratio <= length_ratio <= max_ratio):
                if not used_fallback:
                    logging.warning("Refined match violates length window; falling back to original candidate [%d,%d).", orig_start, orig_end)
                    final_start, final_end = orig_start, orig_end
                    detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
                    length_ratio = detected_distance / ref_distance if ref_distance > 0 else float('inf')
                    used_fallback = True
                if not (min_ratio <= length_ratio <= max_ratio):
                    logging.warning("Rejected match for '%s': length ratio %.3f outside [%.3f, %.3f] (detected %.2f m vs ref %.2f m).",
                                    seg_filename, length_ratio, min_ratio, max_ratio, detected_distance, ref_distance)
                    continue
            detected_start_coords = (recorded_points[final_start]['lat'], recorded_points[final_start]['lon'])
            detected_end_coords = (recorded_points[final_end - 1]['lat'], recorded_points[final_end - 1]['lon'])
            start_diff = haversine_distance(detected_start_coords, ref_start_coords)
            end_diff = haversine_distance(detected_end_coords, ref_end_coords)
            logging.info("Detected start for '%s' is %.2f m from ref start (limit: %.2f m).",
                         seg_filename, start_diff, args.bbox_margin)
            logging.info("Detected end for '%s' is %.2f m from ref end (limit: %.2f m).",
                         seg_filename, end_diff, args.bbox_margin)
            logging.info("Detected segment length for '%s' is %.2f m; Reference segment length is %.2f m.",
                         seg_filename, detected_distance, ref_distance)
            if (start_diff > args.bbox_margin or end_diff > args.bbox_margin) and not args.no_rejection:
                if not used_fallback:
                    logging.warning("Refined boundaries deviate; falling back to original candidate [%d,%d).", orig_start, orig_end)
                    final_start, final_end = orig_start, orig_end
                    detected_start_coords = (recorded_points[final_start]['lat'], recorded_points[final_start]['lon'])
                    detected_end_coords = (recorded_points[final_end - 1]['lat'], recorded_points[final_end - 1]['lon'])
                    start_diff = haversine_distance(detected_start_coords, ref_start_coords)
                    end_diff = haversine_distance(detected_end_coords, ref_end_coords)
                    used_fallback = True
                
                if args.single_passage:
                    ok = enforce_single_passage(recorded_points, final_start, final_end,
                                              ref_start_coords, ref_end_coords,
                                              radius_m=args.passage_radius,
                                              edge_frac=args.passage_edge_frac)
                    if not ok:
                        logging.warning("Rejected match for '%s' by single-passage check (radius=%.1f m, edge_frac=%.2f).",
                                        seg_filename, args.passage_radius, args.passage_edge_frac)
                        continue
                logging.warning("Detected boundaries for '%s' deviate: start_diff=%.2f m, end_diff=%.2f m (limit: %.2f m).",
                                seg_filename, start_diff, end_diff, args.bbox_margin)
                continue
            time_seconds = measure_segment_time(recorded_points, final_start, final_end)
            time_str = str(datetime.timedelta(seconds=int(time_seconds))) if time_seconds is not None else "N/A"
            result = {
                "segment": seg_filename,
                "start_index": final_start,
                "end_index": final_end,
                "ref_distance": ref_distance,
                "detected_distance": detected_distance,
                "dtw_avg": dtw_avg,
                "time_seconds": time_seconds,
                "time_str": time_str,
                "ref_start": f"({ref_start_coords[0]:.6f}, {ref_start_coords[1]:.6f})",
                "ref_end": f"({ref_end_coords[0]:.6f}, {ref_end_coords[1]:.6f})",
                "start_diff": start_diff,
                "end_diff": end_diff
            }
            results.append(result)
            logging.info("Segment %s: indices (%d, %d), ref_dist=%.2f m, detected_dist=%.2f m, dtw_avg=%.2f, time=%s, start_diff=%.2f m, end_diff=%.2f m",
                         seg_filename, final_start, final_end, ref_distance, detected_distance, dtw_avg, time_str, start_diff, end_diff)
            if args.export_gpx:
                export_filename = f"{os.path.splitext(args.export_gpx_file)[0]}_{seg_filename}_match{results.index(result)+1}.gpx"
                export_matches_to_gpx([(final_start, final_end, dtw_avg)],
                                        recorded_points,
                                        args.recorded,
                                        seg_filename,
                                        export_filename)
    if results:
        output_results(results, args.output_mode, args.output_file)
    else:
        logging.info("No matching segments detected in the recorded track.")

if __name__ == "__main__":
    main()
