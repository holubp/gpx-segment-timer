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
the segment is rejected. Use --skip-endpoint-checks to allow segments with boundary deviations
(only a warning is logged).

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

OVERALL_BBOX_FACTOR = 3.33

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

def compute_resample_count(points: List[Dict[str, Any]],
                           target_spacing_m: float,
                           max_points: int) -> int:
    if target_spacing_m <= 0:
        return max(2, min(max_points, len(points)))
    total_distance = compute_total_distance(points)
    count = max(2, int(total_distance / target_spacing_m) + 1)
    return max(2, min(max_points, count))

def _vector_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

_DTW_DISTANCE_FN = _vector_distance

def make_dtw_distance(penalty: str,
                      scale_m: float,
                      huber_k: float) -> Any:
    def dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        d = _vector_distance(a, b)
        if penalty == "quadratic":
            if scale_m <= 0:
                return d * d
            return (d * d) / scale_m
        if penalty == "huber":
            if huber_k <= 0:
                return d
            if d <= huber_k:
                return d
            return huber_k + ((d - huber_k) ** 2) / (2.0 * huber_k)
        return d
    return dist

def _point_to_xy(lat: float, lon: float, origin: Tuple[float, float], lat_scale_ref: float) -> Tuple[float, float]:
    meters_per_deg = 111320.0
    dlat = lat - origin[0]
    dlon = lon - origin[1]
    x = dlon * math.cos(math.radians(lat_scale_ref)) * meters_per_deg
    y = dlat * meters_per_deg
    return (x, y)

def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
    meters_per_deg = 111320.0
    dlat = north_m / meters_per_deg
    dlon = east_m / (meters_per_deg * math.cos(math.radians(lat))) if meters_per_deg != 0 else 0.0
    return (lat + dlat, lon + dlon)

def _points_to_xy(points: List[Tuple[float, float]],
                  origin: Tuple[float, float],
                  lat_scale_ref: float) -> List[Tuple[float, float]]:
    return [_point_to_xy(lat, lon, origin, lat_scale_ref) for lat, lon in points]

def point_to_polyline_distance_xy(px: float, py: float, poly_xy: List[Tuple[float, float]]) -> float:
    if len(poly_xy) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(poly_xy) - 1):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist < best:
            best = dist
    return best

def compute_xtrack_stats(points: List[Dict[str, Any]],
                         ref_points: List[Dict[str, Any]],
                         lat_scale_ref: float,
                         start_idx: int,
                         end_idx: int,
                         sample_max: int = 0) -> Tuple[Optional[float], Optional[float]]:
    if end_idx - start_idx < 2:
        return None, None
    ref_origin = (ref_points[0]["lat"], ref_points[0]["lon"])
    ref_poly_xy = _points_to_xy([(p["lat"], p["lon"]) for p in ref_points], ref_origin, lat_scale_ref)
    if not ref_poly_xy:
        return None, None
    total_len = end_idx - start_idx
    step = 1
    if sample_max > 0 and total_len > sample_max:
        step = max(1, total_len // sample_max)
    dists: List[float] = []
    for i in range(start_idx, end_idx, step):
        pt = points[i]
        px, py = _point_to_xy(pt["lat"], pt["lon"], ref_origin, lat_scale_ref)
        dists.append(point_to_polyline_distance_xy(px, py, ref_poly_xy))
    if not dists:
        return None, None
    dists.sort()
    p95_idx = int(math.ceil(0.95 * len(dists))) - 1
    p95 = dists[max(0, min(p95_idx, len(dists) - 1))]
    return p95, dists[-1]

def build_shape_sequence(points_latlon: List[Tuple[float, float]],
                         shape_mode: str,
                         lat_scale_ref: float) -> List[Tuple[float, ...]]:
    if not points_latlon:
        return []
    if shape_mode == "step_vectors":
        xy = _points_to_xy(points_latlon, points_latlon[0], lat_scale_ref)
        return [(xy[i][0] - xy[i-1][0], xy[i][1] - xy[i-1][1]) for i in range(1, len(xy))]
    if shape_mode == "heading":
        xy = _points_to_xy(points_latlon, points_latlon[0], lat_scale_ref)
        seq = []
        for i in range(1, len(xy)):
            dx = xy[i][0] - xy[i-1][0]
            dy = xy[i][1] - xy[i-1][1]
            length = math.hypot(dx, dy)
            if length == 0:
                seq.append((1.0, 0.0, 0.0))
            else:
                seq.append((dx / length, dy / length, length))
        return seq
    if shape_mode == "centered":
        xy = _points_to_xy(points_latlon, points_latlon[0], lat_scale_ref)
        cx = sum(p[0] for p in xy) / len(xy)
        cy = sum(p[1] for p in xy) / len(xy)
        return [(p[0] - cx, p[1] - cy) for p in xy]
    return build_shape_sequence(points_latlon, "step_vectors", lat_scale_ref)

def build_centered_xy(points_latlon: List[Tuple[float, float]],
                      lat_scale_ref: float) -> List[Tuple[float, float]]:
    if not points_latlon:
        return []
    xy = _points_to_xy(points_latlon, points_latlon[0], lat_scale_ref)
    cx = sum(p[0] for p in xy) / len(xy)
    cy = sum(p[1] for p in xy) / len(xy)
    return [(p[0] - cx, p[1] - cy) for p in xy]

def compute_line_normal(p0: Tuple[float, float],
                        p1: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    lat_scale_ref = p0[0]
    v = _point_to_xy(p1[0], p1[1], p0, lat_scale_ref)
    nx, ny = v[0], v[1]
    norm = math.hypot(nx, ny)
    if norm == 0:
        return (1.0, 0.0), lat_scale_ref
    return (nx / norm, ny / norm), lat_scale_ref

def compute_median_time_step(points: List[Dict[str, Any]]) -> Optional[float]:
    deltas: List[float] = []
    prev_time = None
    for pt in points:
        t = pt.get("time")
        if t is None:
            prev_time = None
            continue
        if prev_time is not None:
            delta = (t - prev_time).total_seconds()
            if delta > 0:
                deltas.append(delta)
        prev_time = t
    if not deltas:
        return None
    deltas.sort()
    mid = len(deltas) // 2
    if len(deltas) % 2 == 1:
        return deltas[mid]
    return (deltas[mid - 1] + deltas[mid]) / 2.0

def log_config(args: argparse.Namespace, label: str) -> None:
    logging.info("%s config:", label)
    for key in sorted(vars(args).keys()):
        logging.info("  %s=%s", key, getattr(args, key))

def find_line_crossing(recorded_points: List[Dict[str, Any]],
                       line_anchor: Tuple[float, float],
                       line_normal: Tuple[float, float],
                       start_idx: int,
                       end_idx: int,
                       lat_scale_ref: float,
                       pick_last: bool,
                       line_half_len: Optional[float] = None) -> Optional[Dict[str, Any]]:
    crossings = find_line_crossings(
        recorded_points, line_anchor, line_normal, start_idx, end_idx, lat_scale_ref, line_half_len
    )
    if not crossings:
        return None
    return crossings[-1] if pick_last else crossings[0]

def find_line_crossings(recorded_points: List[Dict[str, Any]],
                        line_anchor: Tuple[float, float],
                        line_normal: Tuple[float, float],
                        start_idx: int,
                        end_idx: int,
                        lat_scale_ref: float,
                        line_half_len: Optional[float] = None) -> List[Dict[str, Any]]:
    if end_idx - start_idx < 2:
        return []
    crossings: List[Dict[str, Any]] = []
    tangent = (-line_normal[1], line_normal[0])
    for i in range(start_idx, end_idx - 1):
        p0 = recorded_points[i]
        p1 = recorded_points[i + 1]
        x0, y0 = _point_to_xy(p0["lat"], p0["lon"], line_anchor, lat_scale_ref)
        x1, y1 = _point_to_xy(p1["lat"], p1["lon"], line_anchor, lat_scale_ref)
        d0 = x0 * line_normal[0] + y0 * line_normal[1]
        d1 = x1 * line_normal[0] + y1 * line_normal[1]
        t0 = x0 * tangent[0] + y0 * tangent[1]
        t1 = x1 * tangent[0] + y1 * tangent[1]
        if d0 == 0.0 and d1 == 0.0:
            continue
        if d0 == 0.0:
            if line_half_len is not None and abs(t0) > line_half_len:
                continue
            crossings.append({
                "idx0": i,
                "idx1": i,
                "lat": p0["lat"],
                "lon": p0["lon"],
                "time": p0.get("time"),
                "interp": False,
            })
            continue
        if d1 == 0.0 or d0 * d1 < 0:
            t = d0 / (d0 - d1)
            t_along = t0 + (t1 - t0) * t
            if line_half_len is not None and abs(t_along) > line_half_len:
                continue
            lat = p0["lat"] + (p1["lat"] - p0["lat"]) * t
            lon = p0["lon"] + (p1["lon"] - p0["lon"]) * t
            time0 = p0.get("time")
            time1 = p1.get("time")
            interp_time = None
            if time0 and time1:
                interp_time = time0 + (time1 - time0) * t
            crossings.append({
                "idx0": i,
                "idx1": i + 1,
                "lat": lat,
                "lon": lon,
                "time": interp_time,
                "interp": True,
            })
    return crossings

def compute_bounding_box(points: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Compute the bounding box (min_lat, max_lat, min_lon, max_lon) for a set of points."""
    lats = [p['lat'] for p in points]
    lons = [p['lon'] for p in points]
    return (min(lats), max(lats), min(lons), max(lons))

def compute_rhomboid_extents(points: List[Dict[str, Any]],
                             lat_scale_ref: float,
                             margin_m: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute diamond (rhomboid) extents using x+y and x-y slabs.
    Returns (minsum, maxsum, mindiff, maxdiff) in meters.
    """
    if not points:
        return None
    meters_per_deg = 111320.0
    cos_lat = math.cos(math.radians(lat_scale_ref))
    minsum = mindiff = float("inf")
    maxsum = maxdiff = float("-inf")
    for pt in points:
        x = pt["lon"] * cos_lat * meters_per_deg
        y = pt["lat"] * meters_per_deg
        s = x + y
        d = x - y
        minsum = min(minsum, s)
        maxsum = max(maxsum, s)
        mindiff = min(mindiff, d)
        maxdiff = max(maxdiff, d)
    if margin_m > 0:
        minsum -= margin_m
        maxsum += margin_m
        mindiff -= margin_m
        maxdiff += margin_m
    return (minsum, maxsum, mindiff, maxdiff)

def rhomboid_extents_overlap(a: Tuple[float, float, float, float],
                             b: Tuple[float, float, float, float]) -> bool:
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])

def compute_octagon_extents(points: List[Dict[str, Any]],
                            lat_scale_ref: float,
                            margin_m: float = 0.0) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    """
    Compute coarse octagon extents using x/y, x+y, x-y slabs.
    Returns (minx, maxx, miny, maxy, minsum, maxsum, mindiff, maxdiff).
    """
    if not points:
        return None
    meters_per_deg = 111320.0
    cos_lat = math.cos(math.radians(lat_scale_ref))
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    minsum = mindiff = float("inf")
    maxsum = maxdiff = float("-inf")
    for pt in points:
        x = pt["lon"] * cos_lat * meters_per_deg
        y = pt["lat"] * meters_per_deg
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)
        s = x + y
        d = x - y
        minsum = min(minsum, s)
        maxsum = max(maxsum, s)
        mindiff = min(mindiff, d)
        maxdiff = max(maxdiff, d)
    if margin_m > 0:
        minx -= margin_m
        maxx += margin_m
        miny -= margin_m
        maxy += margin_m
        minsum -= margin_m
        maxsum += margin_m
        mindiff -= margin_m
        maxdiff += margin_m
    return (minx, maxx, miny, maxy, minsum, maxsum, mindiff, maxdiff)

def octagon_extents_overlap(a: Tuple[float, float, float, float, float, float, float, float],
                            b: Tuple[float, float, float, float, float, float, float, float]) -> bool:
    return not (
        a[1] < b[0] or a[0] > b[1] or
        a[3] < b[2] or a[2] > b[3] or
        a[5] < b[4] or a[4] > b[5] or
        a[7] < b[6] or a[6] > b[7]
    )

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
                         resample_count: int,
                         lat_scale_ref: float) -> Tuple[int, int]:
    """
    Use the DTW warping path from the candidate segment (resampled) to determine indices in recorded_points
    that best align with the reference's start and end.
    """
    candidate_segment = recorded_points[candidate_start:candidate_end]
    candidate_resampled = resample_points(candidate_segment, resample_count)
    ref_xy = build_centered_xy(ref_resampled, lat_scale_ref)
    cand_xy = build_centered_xy(candidate_resampled, lat_scale_ref)
    _, path = fastdtw(cand_xy, ref_xy, dist=_DTW_DISTANCE_FN)
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
                             is_start: bool,
                             shape_mode: str,
                             lat_scale_ref: float,
                             ref_coord: Tuple[float, float],
                             spatial_weight: float) -> int:
    """
    Refine a single endpoint (start if is_start True, else end) via a local sliding window.
    For the start, candidate subsegments of length L starting at each candidate index (within ±window)
    are compared with ref_sub (the first L reference points). For the end, candidate subsegments ending at each
    candidate index are compared with the last L reference points.
    Returns the candidate index that minimizes the sum of pointwise haversine distances.
    """
    best_idx = candidate_index
    best_cost = float('inf')
    ref_shape = build_shape_sequence(ref_sub, shape_mode, lat_scale_ref)
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
        cand_shape = build_shape_sequence(candidate_resampled, shape_mode, lat_scale_ref)
        min_len = min(len(cand_shape), len(ref_shape))
        if min_len == 0:
            continue
        shape_cost = sum(_vector_distance(cand_shape[k], ref_shape[k]) for k in range(min_len))
        if is_start:
            spatial_dist = haversine_distance(
                (recorded_points[i]['lat'], recorded_points[i]['lon']),
                ref_coord
            )
        else:
            spatial_dist = haversine_distance(
                (recorded_points[i]['lat'], recorded_points[i]['lon']),
                ref_coord
            )
        cost = shape_cost + spatial_weight * spatial_dist
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx

def _endpoint_window_pts_from_m(rec_cum_dists: List[float], idx: int, window_m: float) -> int:
    if window_m <= 0:
        return 1
    n = len(rec_cum_dists)
    idx = max(0, min(idx, n - 1))
    lower = rec_cum_dists[idx] - window_m
    upper = rec_cum_dists[idx] + window_m
    left = bisect.bisect_left(rec_cum_dists, lower, 0, idx + 1)
    right = bisect.bisect_right(rec_cum_dists, upper, idx, n)
    return max(1, idx - left, right - idx)

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
    logging.debug("Iterative refinement: start in range [%d,%d], end in range [%d,%d].", i_min, i_max, j_min, j_max)
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
    logging.debug("Best match of iterative refinement: start %d, end %d, best cost %f.", best_i, best_j, best_cost)
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
                                              resample_count: int,
                                              shape_mode: str,
                                              lat_scale_ref: float) -> Tuple[int, int]:
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
    ref_shape = build_shape_sequence(ref_resampled, shape_mode, lat_scale_ref)
    candidate_full = recorded_points[initial_start:initial_end]
    candidate_full_resampled = resample_points(candidate_full, resample_count)
    candidate_full_shape = build_shape_sequence(candidate_full_resampled, shape_mode, lat_scale_ref)
    full_cost = fastdtw(candidate_full_shape, ref_shape, dist=_DTW_DISTANCE_FN)[0] / max(1, len(ref_shape))

    L = max(3, int(0.1 * resample_count))
    ref_start_sub = resample_points(ref_points[0:L], L)
    ref_end_sub = resample_points(ref_points[-L:], L)
    ref_start_shape = build_shape_sequence(ref_start_sub, shape_mode, lat_scale_ref)
    ref_end_shape = build_shape_sequence(ref_end_sub, shape_mode, lat_scale_ref)

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
            candidate_shape = build_shape_sequence(candidate_resampled, shape_mode, lat_scale_ref)
            full_cost_candidate = fastdtw(candidate_shape, ref_shape, dist=_DTW_DISTANCE_FN)[0] / max(1, len(ref_shape))
            candidate_start_sub = resample_points(recorded_points[i:i+L], L) if len(recorded_points[i:i+L]) >= L else None
            candidate_end_sub = resample_points(recorded_points[j-L:j], L) if len(recorded_points[j-L:j]) >= L else None
            if candidate_start_sub is None or candidate_end_sub is None:
                continue
            start_shape = build_shape_sequence(candidate_start_sub, shape_mode, lat_scale_ref)
            end_shape = build_shape_sequence(candidate_end_sub, shape_mode, lat_scale_ref)
            start_cost = fastdtw(start_shape, ref_start_shape, dist=_DTW_DISTANCE_FN)[0] / max(1, len(ref_start_shape))
            end_cost = fastdtw(end_shape, ref_end_shape, dist=_DTW_DISTANCE_FN)[0] / max(1, len(ref_end_shape))
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
                             dtw_window_m: float,
                             dtw_window_max_avg: float,
                             resample_count: int,
                             shape_mode: str,
                             lat_scale_ref: float,
                             min_gap: int = 1,
                             bbox_margin_m: float = 30,
                             gps_error_m: float = 12.0,
                             candidate_endpoint_margin_m: float = -1.0,
                             envelope_max_m: float = -1.0,
                             envelope_allow_off: Tuple[int, float] = (2, 100.0),
                             envelope_sample_max: int = 200,
                             strict_envelope_window_m: float = -1.0,
                             strict_envelope_off_pct: float = 0.0,
                             prefilter_xtrack_p95_m: float = -1.0,
                             prefilter_xtrack_samples: int = 80,
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
    ref_shape = build_shape_sequence(ref_resampled, shape_mode, lat_scale_ref)
    rh_margin = max(0.0, gps_error_m)
    ref_rhom = compute_rhomboid_extents(ref_points, lat_scale_ref, margin_m=rh_margin)
    rec_rhom = compute_rhomboid_extents(recorded_points, lat_scale_ref, margin_m=rh_margin)
    if ref_rhom and rec_rhom and not rhomboid_extents_overlap(ref_rhom, rec_rhom):
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("Skipping reference %s: no overlap with recorded track rhomboid bounds.", ref_name or "<ref>")
        return []
    ref_origin = (ref_points[0]["lat"], ref_points[0]["lon"])
    ref_poly_xy = _points_to_xy([(p["lat"], p["lon"]) for p in ref_points], ref_origin, lat_scale_ref)
    envelope_max_m = gps_error_m if envelope_max_m < 0 else envelope_max_m
    candidate_endpoint_margin_m = gps_error_m if candidate_endpoint_margin_m < 0 else candidate_endpoint_margin_m

    # Build bboxes
    ref_bbox = compute_bounding_box(ref_points)
    overall_margin = max(bbox_margin_m * OVERALL_BBOX_FACTOR, gps_error_m)
    bbox_overall = expand_bounding_box(ref_bbox, margin_m=overall_margin)

    ref_start = {'lat': ref_points[0]['lat'], 'lon': ref_points[0]['lon']}
    ref_end   = {'lat': ref_points[-1]['lat'], 'lon': ref_points[-1]['lon']}
    bbox_start = expand_bounding_box((ref_start['lat'], ref_start['lat'], ref_start['lon'], ref_start['lon']), margin_m=candidate_endpoint_margin_m)
    bbox_end   = expand_bounding_box((ref_end['lat'],   ref_end['lat'],   ref_end['lon'],   ref_end['lon']),   margin_m=candidate_endpoint_margin_m)

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

    # Precompute distances to reference polyline once per segment.
    dist_all: List[float] = []
    pref_off_all: List[int] = [0] * (len(recorded_points) + 1)
    if envelope_max_m > 0 or prefilter_xtrack_p95_m > 0 or strict_envelope_window_m > 0:
        off_all: List[int] = []
        for i, pt in enumerate(recorded_points):
            px, py = _point_to_xy(pt["lat"], pt["lon"], ref_origin, lat_scale_ref)
            dist = point_to_polyline_distance_xy(px, py, ref_poly_xy)
            dist_all.append(dist)
            off_all.append(1 if envelope_max_m > 0 and dist > envelope_max_m else 0)
            pref_off_all[i + 1] = pref_off_all[i] + off_all[-1]
    strict_max_bad_start: Optional[List[int]] = None
    strict_rejects_total = 0
    dtw_window_rejects_total = 0
    if strict_envelope_window_m > 0:
        if envelope_max_m <= 0:
            envelope_max_m = gps_error_m
        n_points = len(recorded_points)
        end_idx = [0] * n_points
        j = 0
        for i in range(n_points):
            if j < i:
                j = i
            while j < n_points and (rec_cum_dists[j] - rec_cum_dists[i]) <= strict_envelope_window_m:
                j += 1
            end_idx[i] = max(i, j - 1)
        starts_by_end: List[List[int]] = [[] for _ in range(n_points)]
        allow_pts, allow_m = envelope_allow_off
        allow_m = max(1.0, allow_m)
        allow_off_base = int(math.ceil(allow_pts * (strict_envelope_window_m / allow_m)))
        bad_windows = 0
        for i in range(n_points):
            e = end_idx[i]
            if e < i:
                continue
            window_pts = e - i + 1
            off_count = pref_off_all[e + 1] - pref_off_all[i]
            allowed_off = allow_off_base
            if strict_envelope_off_pct > 0:
                allowed_off = max(allowed_off, int(math.ceil(strict_envelope_off_pct * window_pts)))
            if off_count > allowed_off:
                starts_by_end[e].append(i)
                bad_windows += 1
        strict_max_bad_start = [-1] * n_points
        max_bad = -1
        for e in range(n_points):
            if starts_by_end[e]:
                max_bad = max(max_bad, max(starts_by_end[e]))
            strict_max_bad_start[e] = max_bad
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info(
                "Envelope prefilter (strict window): window=%.1fm envelope=%.2fm allow_off_base=%d off_pct=%.3f bad_windows=%d",
                strict_envelope_window_m,
                envelope_max_m,
                allow_off_base,
                strict_envelope_off_pct,
                bad_windows,
            )

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
        strict_rejects_run = 0
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
                if envelope_max_m > 0 or prefilter_xtrack_p95_m > 0:
                    edge_guard = max(1, int(0.05 * (e - s)))
                    inner_s = s + edge_guard
                    inner_e = e - edge_guard
                    if inner_e > inner_s:
                        inner_len_m = rec_cum_dists[inner_e] - rec_cum_dists[inner_s]
                        if envelope_max_m > 0:
                            allow_pts, allow_m = envelope_allow_off
                            allow_m = max(1.0, allow_m)
                            allowed_off = int(math.ceil(allow_pts * (inner_len_m / allow_m)))
                            if envelope_sample_max > 0 and (inner_e - inner_s + 1) > envelope_sample_max:
                                step = max(1, (inner_e - inner_s + 1) // envelope_sample_max)
                                off_count = 0
                                for idx in range(inner_s, inner_e + 1, step):
                                    if dist_all[idx] > envelope_max_m:
                                        off_count += 1
                                if off_count > allowed_off:
                                    continue
                            else:
                                off_count = pref_off_all[inner_e + 1] - pref_off_all[inner_s]
                                if off_count > allowed_off:
                                    continue
                        if prefilter_xtrack_p95_m > 0 and dist_all:
                            sample_len = max(1, prefilter_xtrack_samples)
                            step = max(1, (inner_e - inner_s + 1) // sample_len)
                            sampled = dist_all[inner_s:inner_e + 1:step]
                            sampled.sort()
                            p95_idx = int(math.ceil(0.95 * len(sampled))) - 1
                            p95 = sampled[max(0, min(p95_idx, len(sampled) - 1))]
                            if p95 > prefilter_xtrack_p95_m:
                                continue
                if strict_max_bad_start is not None and strict_max_bad_start[e] >= s:
                    strict_rejects_total += 1
                    strict_rejects_run += 1
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
            if strict_envelope_window_m > 0:
                logging.info("Run [%d,%d]: strict envelope rejects: %d", rs, re, strict_rejects_run)
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
        if strict_envelope_window_m > 0:
            logging.info("Strict envelope rejects total: %d", strict_rejects_total)
        if dtw_window_m > 0 and dtw_window_max_avg > 0:
            logging.info("DTW window rejects total: %d", dtw_window_rejects_total)

    # Early exit
    if total_candidates == 0:
        return []

    # Phase 2: compute DTW best per run (not global), then return list of per-run winners.
    matches: List[Tuple[int, int, float, int, int]] = []
    for run_idx, (rs, re), run_cands in runs_and_candidates:
        best = None  # (avg_cost, s, e, max_avg)
        dtw_window_rejects_run = 0
        for s, e in run_cands:
            cand_pts = recorded_points[s:e+1]
            if len(cand_pts) < 2:
                continue
            cand_resampled = resample_points(cand_pts, resample_count)
            cand_shape = build_shape_sequence(cand_resampled, shape_mode, lat_scale_ref)
            dtw_distance, path = fastdtw(cand_shape, ref_shape, dist=_DTW_DISTANCE_FN)
            denom = max(1, len(ref_shape))
            avg_cost = dtw_distance / denom
            max_avg = None
            if dtw_window_m > 0 and dtw_window_max_avg > 0 and path:
                step_m = ref_total_distance / max(1, len(ref_shape) - 1)
                window_steps = max(1, int(round(dtw_window_m / max(step_m, 1e-6))))
                costs = [_DTW_DISTANCE_FN(cand_shape[i], ref_shape[j]) for (i, j) in path]
                if len(costs) >= window_steps:
                    window_sum = sum(costs[:window_steps])
                    max_avg = window_sum / window_steps
                    for k in range(window_steps, len(costs)):
                        window_sum += costs[k] - costs[k - window_steps]
                        max_avg = max(max_avg, window_sum / window_steps)
                else:
                    max_avg = sum(costs) / max(1, len(costs))
                if max_avg > dtw_window_max_avg:
                    dtw_window_rejects_run += 1
                    dtw_window_rejects_total += 1
                    continue
            logging.debug("DTW for run=%d s=%d e=%d: total %f, avg %f.", run_idx, s, e, dtw_distance, avg_cost)
            if best is None or avg_cost < best[0]:
                best = (avg_cost, s, e, max_avg)
        if best is not None and best[0] < dtw_threshold:
            avg_cost, s_idx, e_idx, max_avg = best
            matches.append((s_idx, e_idx+1, avg_cost, s_idx, e_idx+1))
            if max_avg is not None:
                logging.info("Run %d winner: s=%d e=%d (exclusive), dtw_avg=%.2f, dtw_window_max_avg=%.2f", run_idx, s_idx, e_idx+1, avg_cost, max_avg)
            else:
                logging.info("Run %d winner: s=%d e=%d (exclusive), dtw_avg=%.2f", run_idx, s_idx, e_idx+1, avg_cost)
        else:
            logging.info("Run %d had no DTW winner under threshold (%.2f).", run_idx, dtw_threshold)
        if dtw_window_m > 0 and dtw_window_max_avg > 0 and dtw_window_rejects_run:
            logging.info("Run %d: DTW window rejects: %d", run_idx, dtw_window_rejects_run)
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

def export_match_bundle(match: Dict[str, Any],
                        recorded_points: List[Dict[str, Any]],
                        ref_points: List[Dict[str, Any]],
                        recorded_filename: str,
                        ref_filename: str,
                        output_gpx: str,
                        bbox_margin: float,
                        match_num: int,
                        line_length_m: float) -> None:
    """
    Export a single matched segment with reference, start/finish lines, and crossings.
    """
    gpx = gpxpy.gpx.GPX()
    start_idx = match["start_index"]
    end_idx = match["end_index"]

    def add_track(track_name: str, points: List[gpxpy.gpx.GPXTrackPoint]) -> None:
        track = gpxpy.gpx.GPXTrack()
        track.name = track_name
        segment = gpxpy.gpx.GPXTrackSegment()
        for pt in points:
            segment.points.append(pt)
        track.segments.append(segment)
        gpx.tracks.append(track)

    recorded_track_name = f"{os.path.basename(recorded_filename)} - {ref_filename} (match {match_num})"
    recorded_pts = [
        gpxpy.gpx.GPXTrackPoint(pt["lat"], pt["lon"], time=pt.get("time"))
        for pt in recorded_points[start_idx:end_idx]
    ]
    add_track(recorded_track_name, recorded_pts)

    ref_track_name = f"reference: {ref_filename}"
    ref_pts = [
        gpxpy.gpx.GPXTrackPoint(pt["lat"], pt["lon"], time=pt.get("time"))
        for pt in ref_points
    ]
    add_track(ref_track_name, ref_pts)

    if len(ref_points) >= 2:
        start_coords = (ref_points[0]["lat"], ref_points[0]["lon"])
        end_coords = (ref_points[-1]["lat"], ref_points[-1]["lon"])
        start_normal, _ = compute_line_normal(start_coords, (ref_points[1]["lat"], ref_points[1]["lon"]))
        end_normal, _ = compute_line_normal(end_coords, (ref_points[-2]["lat"], ref_points[-2]["lon"]))
        line_len = max(0.1, float(line_length_m))

        def line_points(anchor: Tuple[float, float], normal: Tuple[float, float]) -> List[gpxpy.gpx.GPXTrackPoint]:
            tangent = (-normal[1], normal[0])
            half = line_len / 2.0
            east = tangent[0] * half
            north = tangent[1] * half
            p1 = offset_latlon(anchor[0], anchor[1], north, east)
            p2 = offset_latlon(anchor[0], anchor[1], -north, -east)
            return [
                gpxpy.gpx.GPXTrackPoint(p1[0], p1[1]),
                gpxpy.gpx.GPXTrackPoint(p2[0], p2[1]),
            ]

        add_track(f"start line: {ref_filename}", line_points(start_coords, start_normal))
        add_track(f"finish line: {ref_filename}", line_points(end_coords, end_normal))

    def add_crossing_track(name: str, crossing: Optional[Dict[str, Any]]) -> None:
        if not crossing:
            return
        points: List[gpxpy.gpx.GPXTrackPoint] = []
        interp = crossing.get("interp", False)
        idx0 = crossing.get("idx0")
        idx1 = crossing.get("idx1")
        if interp and idx0 is not None and idx1 is not None and 0 <= idx0 < len(recorded_points) and 0 <= idx1 < len(recorded_points):
            before = recorded_points[idx0]
            after = recorded_points[idx1]
            before_pt = gpxpy.gpx.GPXTrackPoint(before["lat"], before["lon"], time=before.get("time"))
            before_pt.description = f"{name} bracket start"
            points.append(before_pt)
            interp_pt = gpxpy.gpx.GPXTrackPoint(crossing["lat"], crossing["lon"], time=crossing.get("time"))
            interp_pt.description = f"{name} interpolated"
            points.append(interp_pt)
            after_pt = gpxpy.gpx.GPXTrackPoint(after["lat"], after["lon"], time=after.get("time"))
            after_pt.description = f"{name} bracket end"
            points.append(after_pt)
        else:
            single = gpxpy.gpx.GPXTrackPoint(crossing["lat"], crossing["lon"], time=crossing.get("time"))
            single.description = name
            points.append(single)
        add_track(name, points)

    add_crossing_track("start crossing", match.get("start_crossing"))
    add_crossing_track("finish crossing", match.get("end_crossing"))

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
    header = ["Match", "Segment", "Start Idx", "End Idx", "Start Cross 0", "Start Cross 1",
              "End Cross 0", "End Cross 1", "Start Interp", "End Interp",
              "Ref Dist (m)", "Detected Dist (m)",
              "DTW Avg (m)", "Time (s)", "Time (H:M:S)",
              "Ref Start", "Ref End", "Start Diff (m)", "End Diff (m)"]
    if output_mode == "stdout":
        print("{:<8} {:<25} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10} {:>15} {:>20} {:>15} {:>12} {:>15} {:>25} {:>25} {:>18} {:>15}".format(*header))
        print("-" * 240)
        for res in results:
            print("{:<8} {:<25} {:>10} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10} {:>15.2f} {:>20.2f} {:>15.2f} {:>12.2f} {:>15} {:>25} {:>25} {:>18.2f} {:>15.2f}".format(
                f"match{res['match_num']}",
                res["segment"],
                res["start_index"],
                res["end_index"],
                res["start_cross_idx0"] if res["start_cross_idx0"] is not None else -1,
                res["start_cross_idx1"] if res["start_cross_idx1"] is not None else -1,
                res["end_cross_idx0"] if res["end_cross_idx0"] is not None else -1,
                res["end_cross_idx1"] if res["end_cross_idx1"] is not None else -1,
                "Y" if res["start_cross_interp"] else "N",
                "Y" if res["end_cross_interp"] else "N",
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
                    f"match{res['match_num']}",
                    res["segment"],
                    res["start_index"],
                    res["end_index"],
                    res["start_cross_idx0"] if res["start_cross_idx0"] is not None else "",
                    res["start_cross_idx1"] if res["start_cross_idx1"] is not None else "",
                    res["end_cross_idx0"] if res["end_cross_idx0"] is not None else "",
                    res["end_cross_idx1"] if res["end_cross_idx1"] is not None else "",
                    "Y" if res["start_cross_interp"] else "N",
                    "Y" if res["end_cross_interp"] else "N",
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
                f"match{res['match_num']}",
                res["segment"],
                res["start_index"],
                res["end_index"],
                res["start_cross_idx0"] if res["start_cross_idx0"] is not None else None,
                res["start_cross_idx1"] if res["start_cross_idx1"] is not None else None,
                res["end_cross_idx0"] if res["end_cross_idx0"] is not None else None,
                res["end_cross_idx1"] if res["end_cross_idx1"] is not None else None,
                "Y" if res["start_cross_interp"] else "N",
                "Y" if res["end_cross_interp"] else "N",
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
    With the flag --skip-endpoint-checks the segment is stored (with a warning).
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
    parser.add_argument("--candidate-endpoint-margin-m", type=float, default=-1.0,
                        help="Start/end bbox margin in meters for candidate selection; negative uses --gps-error-m.")
    parser.add_argument("--envelope-max-m", type=float, default=-1.0,
                        help="Max distance from reference polyline for envelope prefilter; negative uses --gps-error-m.")
    parser.add_argument("--envelope-allow-off", nargs=2, type=float, metavar=("POINTS", "METERS"),
                        default=(2, 100.0),
                        help="Allowed off-envelope samples per meters: <points> <meters>.")
    parser.add_argument("--envelope-sample-max", type=int, default=0,
                        help="Max number of samples per candidate for envelope prefilter; 0 uses all points.")
    parser.add_argument("--strict-envelope-window-m", type=float, default=-1.0,
                        help="Enable strict envelope prefilter with a sliding window (meters); negative disables.")
    parser.add_argument("--strict-envelope-off-pct", type=float, default=0.0,
                        help="Allowed off-envelope percent per strict window (0 disables).")
    parser.add_argument("--prefilter-xtrack-p95-m", type=float, default=-1.0,
                        help="Enable x-track p95 prefilter (meters); negative disables.")
    parser.add_argument("--prefilter-xtrack-samples", type=int, default=80,
                        help="Number of samples for x-track p95 prefilter.")
    parser.add_argument("--final-xtrack-p95-m", type=float, default=-1.0,
                        help="Reject final matches if x-track p95 exceeds this (meters); negative disables.")
    parser.add_argument("--final-xtrack-max-m", type=float, default=-1.0,
                        help="Reject final matches if x-track max exceeds this (meters); negative disables.")
    parser.add_argument("--allow-length-mismatch", action="store_true",
                        help="Allow final detected segment length to differ from reference beyond candidate-margin without rejecting the match.")
    parser.add_argument("--min-length-ratio", type=float, default=0.8,
                        help="Reject matches shorter than this fraction of reference length (0 disables).")
    parser.add_argument("--dtw-threshold", type=float, default=50,
                        help="Maximum allowed average DTW distance (m per resampled point) for a match.")
    parser.add_argument("--dtw-window-m", type=float, default=-1.0,
                        help="DTW window length in meters for local max-avg checks; negative disables.")
    parser.add_argument("--dtw-window-max-avg", type=float, default=-1.0,
                        help="Reject candidates whose max avg DTW within the window exceeds this; negative disables.")
    parser.add_argument("--dtw-penalty", choices=["linear", "quadratic", "huber"], default="linear",
                        help="Penalty function for DTW point distances.")
    parser.add_argument("--dtw-penalty-scale-m", type=float, default=10.0,
                        help="Scale for quadratic DTW penalty (meters).")
    parser.add_argument("--dtw-penalty-huber-k", type=float, default=5.0,
                        help="Huber k parameter for DTW penalty (meters).")
    parser.add_argument("--shape-mode", choices=["step_vectors", "heading", "centered", "auto"], default="step_vectors",
                        help="Shape representation used for DTW matching.")
    parser.add_argument("--gps-error-m", type=float, default=12.0,
                        help="Max expected GPS translation error (meters) for spatial gating.")
    parser.add_argument("--target-spacing-m", type=float, default=8.0,
                        help="Target spacing (meters) for adaptive resampling; <=0 disables.")
    parser.add_argument("--resample-max", type=int, default=400,
                        help="Maximum points for adaptive resampling.")
    parser.add_argument("--resample-count", type=int, default=200,
                        help="Fallback number of points for resampling segments when target-spacing-m <= 0.")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="Minimum number of recorded points to skip after a match.")
    parser.add_argument("--bbox-margin", type=float, default=30,
                        help="Endpoint bbox expansion (meters) for start/end checks.")
    parser.add_argument("--single-passage", action="store_true",
                        help="Enforce that within the detected segment the trajectory enters the start buffer once at the beginning and the end buffer once at the end.")
    parser.add_argument("--passage-radius", type=float, default=30.0,
                        help="Radius in meters for start/end buffers used by --single-passage (defaults close to bbox-margin).")
    parser.add_argument("--passage-edge-frac", type=float, default=0.10,
                        help="How close to the segment edges the start/end buffer touches must occur (fraction of point count).")
    parser.add_argument("--iterative-window-start", type=int, default=20,
                        help="Search window (in points) for adjusting the start boundary independently.")
    parser.add_argument("--iterative-window-end", type=int, default=20,
                        help="Search window (in points) for adjusting the end boundary independently.")
    parser.add_argument("--penalty-weight", type=float, default=2.0,
                        help="Weight for the Euclidean endpoint penalty in grid refinement (formerly lambda-weight).")
    parser.add_argument("--anchor-beta1", type=float, default=1.0,
                        help="Weight for the DTW cost on the start subsegment in endpoint anchoring.")
    parser.add_argument("--anchor-beta2", type=float, default=1.0,
                        help="Weight for the DTW cost on the end subsegment in endpoint anchoring.")
    parser.add_argument("--endpoint-spatial-weight", type=float, default=0.25,
                        help="Spatial penalty weight applied during endpoint refinement.")
    parser.add_argument("--crossing-endpoint-weight", type=float, default=1.0,
                        help="Weight for endpoint proximity when selecting crossing pairs.")
    parser.add_argument("--crossing-shape-weight", type=float, default=1.0,
                        help="Weight for local shape matching when selecting crossings.")
    parser.add_argument("--crossing-shape-window-frac", type=float, default=0.2,
                        help="Fraction of resample count to use for local crossing shape matching.")
    parser.add_argument("--crossing-shape-window-min", type=int, default=3,
                        help="Minimum number of points for local crossing shape matching.")
    parser.add_argument("--line-length-m", type=float, default=8.0,
                        help="Start/finish line length in meters (total length, not half).")
    parser.add_argument("--crossing-length-weight", type=float, default=-1.0,
                        help="Weight for length error when selecting crossing pairs; negative enables auto-tuning.")
    parser.add_argument("--crossing-window-max", type=int, default=200,
                        help="Max number of points to extend around candidate window when selecting crossings.")
    parser.add_argument("--crossing-edge-window-s", type=float, default=1.0,
                        help="Edge window in seconds for start/end line crossing search (converted using median sampling rate).")
    parser.add_argument("--crossing-expand-mode", default="fixed",
                        choices=["fixed", "ratio"],
                        help="How to expand crossing search windows when no crossings are found (fixed or ratio).")
    parser.add_argument("--crossing-expand-k", type=float, default=1.0,
                        help="Scale factor for ratio-based crossing window expansion.")
    parser.add_argument("--endpoint-window-start", type=float, default=10.0,
                        help="Local sliding window (meters) for refining the start boundary.")
    parser.add_argument("--endpoint-window-end", type=float, default=10.0,
                        help="Local sliding window (meters) for refining the end boundary.")
    parser.add_argument("--no-refinement", action="store_true",
                        help="Disable final boundary refinements; use DTW-selected candidate indices as the result.")
    # By default segments are rejected when boundaries deviate beyond --bbox-margin.
    parser.add_argument("--skip-endpoint-checks", action="store_true",
                        help="Do not reject segments that deviate beyond --bbox-margin (only log a warning).")
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
    parser.add_argument("--group-by-segment", action="store_true",
                        help="Group output by segment name; otherwise results are sorted by start index.")
    args = parser.parse_args()
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    global _DTW_DISTANCE_FN
    _DTW_DISTANCE_FN = make_dtw_distance(
        args.dtw_penalty,
        args.dtw_penalty_scale_m,
        args.dtw_penalty_huber_k,
    )

    if args.output_mode in ["csv", "xlsx"] and not args.output_file:
        parser.error("Output file must be provided for CSV or XLSX outputs.")

    if args.verbose:
        log_config(args, "Initial")
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
    median_dt = compute_median_time_step(recorded_points)
    if median_dt is not None:
        logging.info("Recorded GPX median sampling interval: %.3f s", median_dt)
    if args.verbose:
        adjusted = argparse.Namespace(**vars(args))
        adjusted.median_sampling_s = median_dt
        log_config(adjusted, "Post-load")

    results: List[Dict[str, Any]] = []
    for seg_filename, ref_points in ref_segments.items():
        logging.info("Processing reference segment: %s", seg_filename)
        if not ref_points:
            logging.warning("Reference segment %s has no points; skipping.", seg_filename)
            continue
        shape_mode = "step_vectors" if args.shape_mode == "auto" else args.shape_mode
        lat_scale_ref = sum(p["lat"] for p in ref_points) / len(ref_points)

        ref_start_coords = (ref_points[0]['lat'], ref_points[0]['lon'])
        ref_end_coords = (ref_points[-1]['lat'], ref_points[-1]['lon'])
        ref_distance = compute_total_distance(ref_points)
        logging.info("Reference '%s' start: (%.6f, %.6f), end: (%.6f, %.6f), distance: %.2f m",
                     seg_filename, ref_start_coords[0], ref_start_coords[1],
                     ref_end_coords[0], ref_end_coords[1], ref_distance)
        resample_count = args.resample_count
        if args.target_spacing_m > 0:
            resample_count = compute_resample_count(ref_points, args.target_spacing_m, args.resample_max)
        current_ref_resampled = resample_points(ref_points, resample_count)

        matches = find_all_segment_matches(
            recorded_points,
            ref_points,
            candidate_margin=args.candidate_margin,
            dtw_threshold=args.dtw_threshold,
            dtw_window_m=args.dtw_window_m,
            dtw_window_max_avg=args.dtw_window_max_avg,
            resample_count=resample_count,
            shape_mode=shape_mode,
            lat_scale_ref=lat_scale_ref,
            min_gap=args.min_gap,
            bbox_margin_m=args.bbox_margin,
            gps_error_m=args.gps_error_m,
            candidate_endpoint_margin_m=args.candidate_endpoint_margin_m,
            envelope_max_m=args.envelope_max_m,
            envelope_allow_off=(int(args.envelope_allow_off[0]), float(args.envelope_allow_off[1])),
            envelope_sample_max=args.envelope_sample_max,
            strict_envelope_window_m=args.strict_envelope_window_m,
            strict_envelope_off_pct=args.strict_envelope_off_pct,
            prefilter_xtrack_p95_m=args.prefilter_xtrack_p95_m,
            prefilter_xtrack_samples=args.prefilter_xtrack_samples,
            dump_pattern=args.dump_candidates_gpx,
            ref_name=seg_filename
        )
        if not matches:
            logging.info("No matching segments found for reference '%s'.", seg_filename)
            continue

        L = max(3, int(0.1 * resample_count))
        ref_start_sub = resample_points(ref_points[0:L], L)
        ref_end_sub = resample_points(ref_points[-L:], L)
        ref_start_shape = build_shape_sequence(ref_start_sub, shape_mode, lat_scale_ref)
        ref_end_shape = build_shape_sequence(ref_end_sub, shape_mode, lat_scale_ref)
        crossing_L = max(args.crossing_shape_window_min, int(args.crossing_shape_window_frac * resample_count))
        ref_start_sub_cross = resample_points(ref_points[0:crossing_L], crossing_L)
        ref_end_sub_cross = resample_points(ref_points[-crossing_L:], crossing_L)
        ref_start_shape_cross = build_shape_sequence(ref_start_sub_cross, shape_mode, lat_scale_ref)
        ref_end_shape_cross = build_shape_sequence(ref_end_sub_cross, shape_mode, lat_scale_ref)

        accepted_windows: List[Tuple[int, int]] = []
        crossing_edge_window_pts: Optional[int] = None
        if median_dt is not None and args.crossing_edge_window_s > 0:
            crossing_edge_window_pts = max(1, int(round(args.crossing_edge_window_s / median_dt)))
        crossing_edge_window_pts: Optional[int] = None
        if median_dt is not None and args.crossing_edge_window_s > 0:
            crossing_edge_window_pts = max(1, int(round(args.crossing_edge_window_s / median_dt)))
        if args.verbose:
            seg_cfg = argparse.Namespace(
                ref_segment=seg_filename,
                resample_count=resample_count,
                crossing_L=crossing_L,
                crossing_edge_window_pts=crossing_edge_window_pts,
                line_length_m=args.line_length_m,
                bbox_margin=args.bbox_margin,
                candidate_margin=args.candidate_margin,
            )
            log_config(seg_cfg, "Segment")
        for (start_idx, end_idx, dtw_avg, orig_start, orig_end) in matches:
            if args.no_refinement:
                logging.info("Skipping refinement", orig_start, orig_end)
                final_start, final_end = start_idx, end_idx
                used_fallback = False
            else:
                refined_start, refined_end = refine_boundaries_using_warping_path(
                    recorded_points, start_idx, end_idx, current_ref_resampled, resample_count, lat_scale_ref)
                grid_start, grid_end = refine_boundaries_iteratively(
                        recorded_points, refined_start, refined_end,
                        ref_start_coords, ref_end_coords, ref_distance, rec_cum_dists,
                        iterative_window_start=args.iterative_window_start,
                        iterative_window_end=args.iterative_window_end,
                        penalty_weight=args.penalty_weight)
                spatial_weight = args.endpoint_spatial_weight
                start_window_pts = _endpoint_window_pts_from_m(rec_cum_dists, grid_start, args.endpoint_window_start)
                end_window_pts = _endpoint_window_pts_from_m(rec_cum_dists, grid_end - 1, args.endpoint_window_end)
                final_start = refine_endpoint_boundary(
                    recorded_points, grid_start, ref_start_sub, L, start_window_pts, True,
                    shape_mode, lat_scale_ref, ref_start_coords, spatial_weight)
                final_end = refine_endpoint_boundary(
                    recorded_points, grid_end - 1, ref_end_sub, L, end_window_pts, False,
                    shape_mode, lat_scale_ref, ref_end_coords, spatial_weight) + 1
                # Fallback if refinement breaks monotonicity/length
                used_fallback = False
                if final_end <= final_start:
                    logging.warning("Refinement produced end<=start for '%s'; falling back to original candidate [%d,%d).", seg_filename, orig_start, orig_end)
                    final_start, final_end = orig_start, orig_end
                    used_fallback = True

            detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
            if detected_distance <= 0 and not used_fallback:
                logging.warning("Refinement produced non-positive length for '%s'; falling back to original candidate [%d,%d).", seg_filename, orig_start, orig_end)
                final_start, final_end = orig_start, orig_end
                detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
                used_fallback = True
            if final_end <= final_start:
                logging.warning("Rejected match for '%s': end index (%d) <= start index (%d).", seg_filename, final_end, final_start)
                logging.info("Rejected DTW winner for '%s' at [%d,%d): end<=start.", seg_filename, final_start, final_end)
                continue
            if detected_distance <= 0:
                logging.warning("Rejected match for '%s': non-positive detected distance %.2f m.", seg_filename, detected_distance)
                logging.info("Rejected DTW winner for '%s' at [%d,%d): non-positive length.", seg_filename, final_start, final_end)
                continue
            def compute_crossings_and_diffs(start_idx: int, end_idx: int) -> Tuple[Tuple[float, float], Tuple[float, float], float, float, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
                start_cross = None
                end_cross = None
                start_crossings: List[Dict[str, Any]] = []
                end_crossings: List[Dict[str, Any]] = []
                crossing_shape_weight = args.crossing_shape_weight
                line_half_len = max(0.0, args.line_length_m / 2.0) if args.line_length_m > 0 else None
                pair_resample = max(3, int(0.6 * resample_count))
                pair_resample = min(pair_resample, resample_count)
                ref_pair_resampled = resample_points(ref_points, pair_resample)
                ref_pair_shape = build_shape_sequence(ref_pair_resampled, shape_mode, lat_scale_ref)
                pair_shape_cache: Dict[Tuple[int, int], float] = {}
                def local_shape_cost(crossing: Dict[str, Any], is_start: bool) -> float:
                    idx0 = crossing.get("idx0")
                    idx1 = crossing.get("idx1")
                    if idx0 is None or idx1 is None:
                        return float("inf")
                    if is_start:
                        window_start = max(0, idx1)
                        window_end = min(len(recorded_points), window_start + crossing_L)
                        ref_shape = ref_start_shape_cross
                    else:
                        window_end = min(len(recorded_points), idx1 + 1)
                        window_start = max(0, window_end - crossing_L)
                        ref_shape = ref_end_shape_cross
                    if window_end - window_start < 2:
                        return float("inf")
                    candidate = resample_points(recorded_points[window_start:window_end], crossing_L)
                    cand_shape = build_shape_sequence(candidate, shape_mode, lat_scale_ref)
                    if not cand_shape or not ref_shape:
                        return float("inf")
                    dtw_distance, _ = fastdtw(cand_shape, ref_shape, dist=_DTW_DISTANCE_FN)
                    return dtw_distance / max(1, len(ref_shape))
                def pair_shape_cost(start_crossing: Dict[str, Any], end_crossing: Dict[str, Any]) -> float:
                    s_idx = start_crossing.get("idx1")
                    e_idx = end_crossing.get("idx0")
                    if s_idx is None or e_idx is None:
                        return float("inf")
                    if e_idx <= s_idx:
                        return float("inf")
                    key = (s_idx, e_idx)
                    cached = pair_shape_cache.get(key)
                    if cached is not None:
                        return cached
                    candidate = recorded_points[s_idx:e_idx + 1]
                    if len(candidate) < 2:
                        pair_shape_cache[key] = float("inf")
                        return pair_shape_cache[key]
                    candidate_resampled = resample_points(candidate, pair_resample)
                    cand_shape = build_shape_sequence(candidate_resampled, shape_mode, lat_scale_ref)
                    if not cand_shape or not ref_pair_shape:
                        pair_shape_cache[key] = float("inf")
                        return pair_shape_cache[key]
                    dtw_distance, _ = fastdtw(cand_shape, ref_pair_shape, dist=_DTW_DISTANCE_FN)
                    pair_shape_cache[key] = dtw_distance / max(1, len(ref_pair_shape))
                    return pair_shape_cache[key]
                if len(ref_points) >= 2:
                    start_normal, start_lat_scale = compute_line_normal(ref_start_coords, (ref_points[1]['lat'], ref_points[1]['lon']))
                    end_normal, end_lat_scale = compute_line_normal(ref_end_coords, (ref_points[-2]['lat'], ref_points[-2]['lon']))
                    if crossing_edge_window_pts is not None:
                        edge_window = crossing_edge_window_pts
                    else:
                        edge_window = max(30, int(0.1 * resample_count))
                    edge_window = min(args.crossing_window_max, edge_window)
                    endpoint_start_pts = _endpoint_window_pts_from_m(rec_cum_dists, start_idx, args.endpoint_window_start)
                    endpoint_end_pts = _endpoint_window_pts_from_m(rec_cum_dists, end_idx - 1, args.endpoint_window_end)
                    expanded_start_window = max(edge_window, endpoint_start_pts)
                    expanded_end_window = max(edge_window, endpoint_end_pts)
                    if args.crossing_expand_mode == "ratio" and ref_distance > 0:
                        expected_end_idx = bisect.bisect_left(
                            rec_cum_dists, rec_cum_dists[start_idx] + ref_distance,
                            lo=start_idx + 1, hi=len(rec_cum_dists))
                        expected_start_idx = bisect.bisect_right(
                            rec_cum_dists, rec_cum_dists[end_idx] - ref_distance,
                            lo=0, hi=end_idx)
                        expected_pts_start = max(1, expected_end_idx - start_idx)
                        expected_pts_end = max(1, end_idx - expected_start_idx)
                        local_detected = rec_cum_dists[end_idx] - rec_cum_dists[start_idx]
                        length_ratio = local_detected / ref_distance
                        ratio_factor = max(1.0, length_ratio)
                        dynamic_start = int(round(args.crossing_expand_k * expected_pts_start * ratio_factor))
                        dynamic_end = int(round(args.crossing_expand_k * expected_pts_end * ratio_factor))
                        expanded_start_window = max(expanded_start_window, dynamic_start)
                        expanded_end_window = max(expanded_end_window, dynamic_end)
                    expanded_start_window = min(args.crossing_window_max, expanded_start_window)
                    expanded_end_window = min(args.crossing_window_max, expanded_end_window)

                    s_lo = max(0, start_idx - edge_window)
                    s_hi = min(len(recorded_points) - 1, start_idx + edge_window)
                    e_lo = max(0, end_idx - edge_window)
                    e_hi = min(len(recorded_points) - 1, end_idx + edge_window)
                    start_cross = find_line_crossing(
                        recorded_points, ref_start_coords, start_normal, s_lo, s_hi, start_lat_scale, False, line_half_len)
                    end_cross = find_line_crossing(
                        recorded_points, ref_end_coords, end_normal, e_lo, e_hi, end_lat_scale, True, line_half_len)
                    start_crossings = find_line_crossings(
                        recorded_points, ref_start_coords, start_normal, s_lo, s_hi, start_lat_scale, line_half_len)
                    end_crossings = find_line_crossings(
                        recorded_points, ref_end_coords, end_normal, e_lo, e_hi, end_lat_scale, line_half_len)

                    if not start_crossings and expanded_start_window > edge_window:
                        s_lo2 = max(0, start_idx - expanded_start_window)
                        s_hi2 = min(len(recorded_points) - 1, start_idx + expanded_start_window)
                        start_cross = find_line_crossing(
                            recorded_points, ref_start_coords, start_normal, s_lo2, s_hi2, start_lat_scale, False, line_half_len)
                        start_crossings = find_line_crossings(
                            recorded_points, ref_start_coords, start_normal, s_lo2, s_hi2, start_lat_scale, line_half_len)
                        if args.verbose:
                            logging.info(
                                "Expanded start crossing window for '%s' from %d to %d points (mode=%s).",
                                seg_filename, edge_window, expanded_start_window, args.crossing_expand_mode)

                    if not end_crossings and expanded_end_window > edge_window:
                        e_lo2 = max(0, end_idx - expanded_end_window)
                        e_hi2 = min(len(recorded_points) - 1, end_idx + expanded_end_window)
                        end_cross = find_line_crossing(
                            recorded_points, ref_end_coords, end_normal, e_lo2, e_hi2, end_lat_scale, True, line_half_len)
                        end_crossings = find_line_crossings(
                            recorded_points, ref_end_coords, end_normal, e_lo2, e_hi2, end_lat_scale, line_half_len)
                        if args.verbose:
                            logging.info(
                                "Expanded end crossing window for '%s' from %d to %d points (mode=%s).",
                                seg_filename, edge_window, expanded_end_window, args.crossing_expand_mode)

                    def end_crossings_for_start(s_idx: int) -> List[Dict[str, Any]]:
                        extra = max(args.gps_error_m * 4.0, ref_distance * 0.02)
                        lower = rec_cum_dists[s_idx] + ref_distance * (1 - args.candidate_margin) - extra
                        upper = rec_cum_dists[s_idx] + ref_distance * (1 + args.candidate_margin) + extra
                        import bisect as _bis
                        e_lo = _bis.bisect_left(rec_cum_dists, lower, lo=s_idx + 1, hi=len(rec_cum_dists) - 1)
                        e_hi = _bis.bisect_right(rec_cum_dists, upper, lo=s_idx + 1, hi=len(rec_cum_dists) - 1)
                        return find_line_crossings(recorded_points, ref_end_coords, end_normal, e_lo, e_hi, end_lat_scale, line_half_len)

                    def start_crossings_for_end(e_idx: int) -> List[Dict[str, Any]]:
                        extra = max(args.gps_error_m * 4.0, ref_distance * 0.02)
                        lower = rec_cum_dists[e_idx] - ref_distance * (1 + args.candidate_margin) - extra
                        upper = rec_cum_dists[e_idx] - ref_distance * (1 - args.candidate_margin) + extra
                        import bisect as _bis
                        s_lo2 = _bis.bisect_left(rec_cum_dists, lower, lo=0, hi=e_idx)
                        s_hi2 = _bis.bisect_right(rec_cum_dists, upper, lo=0, hi=e_idx)
                        return find_line_crossings(recorded_points, ref_start_coords, start_normal, s_lo2, s_hi2, start_lat_scale, line_half_len)

                    if start_crossings and (end_cross is None or (end_cross and haversine_distance((end_cross["lat"], end_cross["lon"]), ref_end_coords) > args.bbox_margin)):
                        extended_end: List[Dict[str, Any]] = []
                        for sc in start_crossings:
                            extended_end.extend(end_crossings_for_start(sc["idx1"]))
                        if extended_end:
                            end_crossings = extended_end

                    if end_crossings and (start_cross is None or (start_cross and haversine_distance((start_cross["lat"], start_cross["lon"]), ref_start_coords) > args.bbox_margin)):
                        extended_start: List[Dict[str, Any]] = []
                        for ec in end_crossings:
                            extended_start.extend(start_crossings_for_end(ec["idx0"]))
                        if extended_start:
                            start_crossings = extended_start

                    if (start_cross is None or end_cross is None) or (
                        (start_cross is not None and haversine_distance((start_cross["lat"], start_cross["lon"]), ref_start_coords) > args.bbox_margin)
                        or (end_cross is not None and haversine_distance((end_cross["lat"], end_cross["lon"]), ref_end_coords) > args.bbox_margin)
                    ):
                        if args.crossing_length_weight >= 0:
                            length_weight = args.crossing_length_weight
                        else:
                            length_weight = max(0.05, min(0.5, args.gps_error_m / max(1.0, ref_distance) * 50.0))
                        endpoint_weight = args.crossing_endpoint_weight
                        start_shape_costs: Dict[Tuple[int, int], float] = {}
                        end_shape_costs: Dict[Tuple[int, int], float] = {}
                        for sc in start_crossings:
                            key = (sc["idx0"], sc["idx1"])
                            start_shape_costs[key] = local_shape_cost(sc, True)
                        for ec in end_crossings:
                            key = (ec["idx0"], ec["idx1"])
                            end_shape_costs[key] = local_shape_cost(ec, False)
                        best = None
                        best_in_bounds = None
                        import bisect as _bis
                        for sc in start_crossings:
                            s_idx = sc["idx1"]
                            lower = rec_cum_dists[s_idx] + ref_distance * (1 - args.candidate_margin)
                            upper = rec_cum_dists[s_idx] + ref_distance * (1 + args.candidate_margin)
                            e_lo = _bis.bisect_left(rec_cum_dists, lower, lo=s_idx + 1, hi=len(rec_cum_dists) - 1)
                            e_hi = _bis.bisect_right(rec_cum_dists, upper, lo=s_idx + 1, hi=len(rec_cum_dists) - 1)
                            for ec in end_crossings:
                                e_idx = ec["idx0"]
                                if e_idx <= s_idx:
                                    continue
                                in_window = e_lo <= e_idx <= e_hi
                                seg_len = rec_cum_dists[e_idx] - rec_cum_dists[s_idx]
                                length_err = abs(seg_len - ref_distance)
                                s_diff = haversine_distance((sc["lat"], sc["lon"]), ref_start_coords)
                                e_diff = haversine_distance((ec["lat"], ec["lon"]), ref_end_coords)
                                sc_key = (sc["idx0"], sc["idx1"])
                                ec_key = (ec["idx0"], ec["idx1"])
                                shape_cost = pair_shape_cost(sc, ec)
                                shape_cost += start_shape_costs.get(sc_key, float("inf"))
                                shape_cost += 2.0 * end_shape_costs.get(ec_key, float("inf"))
                                score = crossing_shape_weight * shape_cost + endpoint_weight * (s_diff + e_diff) + length_weight * length_err
                                if not in_window:
                                    score += endpoint_weight * 5.0
                                if s_diff <= args.bbox_margin and e_diff <= args.bbox_margin:
                                    in_bounds_score = crossing_shape_weight * shape_cost + length_weight * length_err * 0.2
                                    if best_in_bounds is None or in_bounds_score < best_in_bounds[0]:
                                        best_in_bounds = (in_bounds_score, sc, ec)
                                if best is None or score < best[0]:
                                    best = (score, sc, ec)
                        if best_in_bounds is not None:
                            start_cross, end_cross = best_in_bounds[1], best_in_bounds[2]
                        elif best is not None:
                            start_cross, end_cross = best[1], best[2]
                            # If endpoint proximity still exceeds threshold, try snapping to best end/start crossing.
                            s_idx = start_cross["idx1"]
                            e_idx = end_cross["idx0"]
                            if e_idx <= s_idx:
                                start_cross = None
                                end_cross = None
                            if start_cross and end_cross:
                                s_diff = haversine_distance((start_cross["lat"], start_cross["lon"]), ref_start_coords)
                                e_diff = haversine_distance((end_cross["lat"], end_cross["lon"]), ref_end_coords)
                                if e_diff > args.bbox_margin:
                                    best_end = None
                                    for ec in end_crossings:
                                        e_idx = ec["idx0"]
                                        if e_idx <= s_idx:
                                            continue
                                        seg_len = rec_cum_dists[e_idx] - rec_cum_dists[s_idx]
                                        length_err = abs(seg_len - ref_distance)
                                        e_diff = haversine_distance((ec["lat"], ec["lon"]), ref_end_coords)
                                        score = endpoint_weight * e_diff + length_weight * length_err
                                        if best_end is None or score < best_end[0]:
                                            best_end = (score, ec)
                                    if best_end is not None:
                                        end_cross = best_end[1]
                                if s_diff > args.bbox_margin:
                                    best_start = None
                                    e_idx = end_cross["idx0"]
                                    for sc in start_crossings:
                                        s_idx = sc["idx1"]
                                        if e_idx <= s_idx:
                                            continue
                                        seg_len = rec_cum_dists[e_idx] - rec_cum_dists[s_idx]
                                        length_err = abs(seg_len - ref_distance)
                                        s_diff = haversine_distance((sc["lat"], sc["lon"]), ref_start_coords)
                                        score = endpoint_weight * s_diff + length_weight * length_err
                                        if best_start is None or score < best_start[0]:
                                            best_start = (score, sc)
                                    if best_start is not None:
                                        start_cross = best_start[1]
                    # If still out of bounds, prioritize endpoint proximity over length.
                    if start_cross and end_cross:
                        s_diff = haversine_distance((start_cross["lat"], start_cross["lon"]), ref_start_coords)
                        e_diff = haversine_distance((end_cross["lat"], end_cross["lon"]), ref_end_coords)
                        if e_diff > args.bbox_margin and start_crossings:
                            s_idx = start_cross["idx1"]
                            best_end = None
                            for ec in end_crossings:
                                e_idx = ec["idx0"]
                                if e_idx <= s_idx:
                                    continue
                                e_diff = haversine_distance((ec["lat"], ec["lon"]), ref_end_coords)
                                if e_diff > args.bbox_margin:
                                    continue
                                score = endpoint_weight * e_diff
                                if best_end is None or score < best_end[0]:
                                    best_end = (score, ec)
                            if best_end is not None:
                                end_cross = best_end[1]
                        if s_diff > args.bbox_margin and end_crossings:
                            e_idx = end_cross["idx0"]
                            best_start = None
                            for sc in start_crossings:
                                s_idx = sc["idx1"]
                                if e_idx <= s_idx:
                                    continue
                                s_diff = haversine_distance((sc["lat"], sc["lon"]), ref_start_coords)
                                if s_diff > args.bbox_margin:
                                    continue
                                score = endpoint_weight * s_diff
                                if best_start is None or score < best_start[0]:
                                    best_start = (score, sc)
                            if best_start is not None:
                                start_cross = best_start[1]
                if start_cross:
                    start_coords = (start_cross["lat"], start_cross["lon"])
                else:
                    start_coords = (recorded_points[start_idx]['lat'], recorded_points[start_idx]['lon'])
                if end_cross:
                    end_coords = (end_cross["lat"], end_cross["lon"])
                else:
                    end_coords = (recorded_points[end_idx - 1]['lat'], recorded_points[end_idx - 1]['lon'])
                start_d = haversine_distance(start_coords, ref_start_coords)
                end_d = haversine_distance(end_coords, ref_end_coords)
                return start_coords, end_coords, start_d, end_d, start_cross, end_cross

            detected_start_coords, detected_end_coords, start_diff, end_diff, start_crossing, end_crossing = compute_crossings_and_diffs(final_start, final_end)
            if start_crossing and start_crossing["idx0"] is not None:
                adj_start = max(0, start_crossing["idx0"])
            else:
                adj_start = final_start
            if end_crossing and end_crossing["idx1"] is not None:
                adj_end = min(len(recorded_points), end_crossing["idx1"] + 1)
            else:
                adj_end = final_end
            if adj_end > adj_start:
                final_start = adj_start
                final_end = adj_end
                detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]

            length_ratio = detected_distance / ref_distance if ref_distance > 0 else float('inf')
            min_ratio = 1.0 - args.candidate_margin
            max_ratio = 1.0 + args.candidate_margin
            if args.min_length_ratio > 0 and length_ratio < args.min_length_ratio:
                logging.warning("Rejected match for '%s': length ratio %.3f below min %.3f (detected %.2f m vs ref %.2f m).",
                                seg_filename, length_ratio, args.min_length_ratio, detected_distance, ref_distance)
                logging.info("Rejected DTW winner for '%s' at [%d,%d): length ratio %.3f below min %.3f.",
                             seg_filename, final_start, final_end, length_ratio, args.min_length_ratio)
                continue
            if not args.allow_length_mismatch and not (min_ratio <= length_ratio <= max_ratio):
                if start_diff <= args.bbox_margin and end_diff <= args.bbox_margin:
                    logging.warning("Length mismatch tolerated for '%s' due to endpoint alignment (ratio %.3f).",
                                    seg_filename, length_ratio)
                else:
                    if not used_fallback:
                        logging.warning("Refined match violates length window; falling back to original candidate [%d,%d).", orig_start, orig_end)
                        final_start, final_end = orig_start, orig_end
                        detected_distance = rec_cum_dists[final_end] - rec_cum_dists[final_start]
                        detected_start_coords, detected_end_coords, start_diff, end_diff, start_crossing, end_crossing = compute_crossings_and_diffs(final_start, final_end)
                        length_ratio = detected_distance / ref_distance if ref_distance > 0 else float('inf')
                        used_fallback = True
                    if not (min_ratio <= length_ratio <= max_ratio) and (start_diff > args.bbox_margin or end_diff > args.bbox_margin):
                        logging.warning("Rejected match for '%s': length ratio %.3f outside [%.3f, %.3f] (detected %.2f m vs ref %.2f m).",
                                        seg_filename, length_ratio, min_ratio, max_ratio, detected_distance, ref_distance)
                        logging.info("Rejected DTW winner for '%s' at [%d,%d): length ratio %.3f outside [%.3f, %.3f].",
                                     seg_filename, final_start, final_end, length_ratio, min_ratio, max_ratio)
                        continue
            logging.info("Detected start for '%s' is %.2f m from ref start (limit: %.2f m).",
                         seg_filename, start_diff, args.bbox_margin)
            logging.info("Detected end for '%s' is %.2f m from ref end (limit: %.2f m).",
                         seg_filename, end_diff, args.bbox_margin)
            logging.info("Detected segment length for '%s' is %.2f m; Reference segment length is %.2f m.",
                         seg_filename, detected_distance, ref_distance)
            start_time = recorded_points[final_start].get("time")
            end_time = recorded_points[final_end - 1].get("time")
            if start_time and end_time:
                raw_delta = (end_time - start_time).total_seconds()
                logging.info("Segment timestamps for '%s': start_idx=%d time=%s end_idx=%d time=%s raw_delta=%.2f",
                             seg_filename, final_start, start_time, final_end - 1, end_time, raw_delta)
            if start_crossing and end_crossing and start_crossing.get("time") and end_crossing.get("time"):
                crossing_delta = (end_crossing["time"] - start_crossing["time"]).total_seconds()
                logging.info("Crossing timestamps for '%s': start=%s end=%s delta=%.2f",
                             seg_filename, start_crossing["time"], end_crossing["time"], crossing_delta)
            if start_crossing and start_crossing.get("interp"):
                s_idx0 = start_crossing.get("idx0")
                s_idx1 = start_crossing.get("idx1")
                if s_idx0 is not None and s_idx1 is not None:
                    s_t0 = recorded_points[s_idx0].get("time")
                    s_t1 = recorded_points[s_idx1].get("time")
                    logging.info("Start interpolation for '%s': idx0=%d time0=%s idx1=%d time1=%s interp_time=%s",
                                 seg_filename, s_idx0, s_t0, s_idx1, s_t1, start_crossing.get("time"))
            if end_crossing and end_crossing.get("interp"):
                e_idx0 = end_crossing.get("idx0")
                e_idx1 = end_crossing.get("idx1")
                if e_idx0 is not None and e_idx1 is not None:
                    e_t0 = recorded_points[e_idx0].get("time")
                    e_t1 = recorded_points[e_idx1].get("time")
                    logging.info("Finish interpolation for '%s': idx0=%d time0=%s idx1=%d time1=%s interp_time=%s",
                                 seg_filename, e_idx0, e_t0, e_idx1, e_t1, end_crossing.get("time"))
            if (start_diff > args.bbox_margin or end_diff > args.bbox_margin) and not args.skip_endpoint_checks:
                if not used_fallback:
                    logging.warning("Refined boundaries deviate; falling back to original candidate [%d,%d).", orig_start, orig_end)
                    final_start, final_end = orig_start, orig_end
                    detected_start_coords, detected_end_coords, start_diff, end_diff, start_crossing, end_crossing = compute_crossings_and_diffs(final_start, final_end)
                    used_fallback = True
                if (start_diff > args.bbox_margin or end_diff > args.bbox_margin) and not args.skip_endpoint_checks:
                    if args.single_passage:
                        ok = enforce_single_passage(recorded_points, final_start, final_end,
                                                  ref_start_coords, ref_end_coords,
                                                  radius_m=args.passage_radius,
                                                  edge_frac=args.passage_edge_frac)
                        if not ok:
                            logging.warning("Rejected match for '%s' by single-passage check (radius=%.1f m, edge_frac=%.2f).",
                                            seg_filename, args.passage_radius, args.passage_edge_frac)
                            logging.info("Rejected DTW winner for '%s' at [%d,%d): single-passage check failed.",
                                         seg_filename, final_start, final_end)
                            continue
                    logging.warning("Detected boundaries for '%s' deviate: start_diff=%.2f m, end_diff=%.2f m (limit: %.2f m).",
                                    seg_filename, start_diff, end_diff, args.bbox_margin)
                    logging.info("Rejected DTW winner for '%s' at [%d,%d): endpoint deviation.",
                                 seg_filename, final_start, final_end)
                    continue
            if args.final_xtrack_p95_m > 0 or args.final_xtrack_max_m > 0:
                p95, xmax = compute_xtrack_stats(
                    recorded_points, ref_points, lat_scale_ref, final_start, final_end, sample_max=0
                )
                if p95 is not None and xmax is not None:
                    logging.info("Final xtrack for '%s': p95=%.2f m, max=%.2f m.",
                                 seg_filename, p95, xmax)
                    if args.final_xtrack_p95_m > 0 and p95 > args.final_xtrack_p95_m:
                        logging.warning("Rejected match for '%s': xtrack p95 %.2f m exceeds %.2f m.",
                                        seg_filename, p95, args.final_xtrack_p95_m)
                        logging.info("Rejected DTW winner for '%s' at [%d,%d): xtrack p95 gate.",
                                     seg_filename, final_start, final_end)
                        continue
                    if args.final_xtrack_max_m > 0 and xmax > args.final_xtrack_max_m:
                        logging.warning("Rejected match for '%s': xtrack max %.2f m exceeds %.2f m.",
                                        seg_filename, xmax, args.final_xtrack_max_m)
                        logging.info("Rejected DTW winner for '%s' at [%d,%d): xtrack max gate.",
                                     seg_filename, final_start, final_end)
                        continue
            time_seconds = None
            if start_crossing and end_crossing and start_crossing.get("time") and end_crossing.get("time"):
                time_seconds = (end_crossing["time"] - start_crossing["time"]).total_seconds()
            if time_seconds is None:
                time_seconds = measure_segment_time(recorded_points, final_start, final_end)
            time_str = str(datetime.timedelta(seconds=int(time_seconds))) if time_seconds is not None else "N/A"
            for (prev_s, prev_e) in accepted_windows:
                overlap = max(0, min(final_end, prev_e) - max(final_start, prev_s))
                if overlap <= 0:
                    continue
                denom = min(prev_e - prev_s, final_end - final_start)
                if denom > 0 and overlap / denom > 0.9:
                    logging.info("Skipping near-duplicate match for '%s' at [%d,%d].", seg_filename, final_start, final_end)
                    logging.info("Rejected DTW winner for '%s' at [%d,%d): near-duplicate overlap.",
                                 seg_filename, final_start, final_end)
                    break
            else:
                accepted_windows.append((final_start, final_end))
                match_num = len(results) + 1
                result = {
                    "match_num": match_num,
                    "segment": seg_filename,
                    "start_index": final_start,
                    "end_index": final_end,
                    "start_cross_idx0": start_crossing.get("idx0") if start_crossing else None,
                    "start_cross_idx1": start_crossing.get("idx1") if start_crossing else None,
                    "end_cross_idx0": end_crossing.get("idx0") if end_crossing else None,
                    "end_cross_idx1": end_crossing.get("idx1") if end_crossing else None,
                    "start_cross_interp": bool(start_crossing.get("interp")) if start_crossing else False,
                    "end_cross_interp": bool(end_crossing.get("interp")) if end_crossing else False,
                    "ref_distance": ref_distance,
                    "detected_distance": detected_distance,
                    "dtw_avg": dtw_avg,
                    "time_seconds": time_seconds,
                    "time_str": time_str,
                    "ref_start": f"({ref_start_coords[0]:.6f}, {ref_start_coords[1]:.6f})",
                    "ref_end": f"({ref_end_coords[0]:.6f}, {ref_end_coords[1]:.6f})",
                    "start_diff": start_diff,
                    "end_diff": end_diff,
                    "start_crossing": start_crossing,
                    "end_crossing": end_crossing
                }
                results.append(result)
                logging.info("Segment %s: indices (%d, %d), ref_dist=%.2f m, detected_dist=%.2f m, dtw_avg=%.2f, time=%s, start_diff=%.2f m, end_diff=%.2f m",
                             seg_filename, final_start, final_end, ref_distance, detected_distance, dtw_avg, time_str, start_diff, end_diff)
                if args.export_gpx:
                    export_filename = f"{os.path.splitext(args.export_gpx_file)[0]}_{seg_filename}_match{match_num}.gpx"
                    export_match_bundle(
                        result,
                        recorded_points,
                        ref_points,
                        args.recorded,
                        seg_filename,
                        export_filename,
                        args.bbox_margin,
                        match_num,
                        args.line_length_m,
                    )
    if results:
        if args.group_by_segment:
            results.sort(key=lambda r: (r["segment"], r["start_index"], -(r["end_index"] - r["start_index"])))
        else:
            results.sort(key=lambda r: (r["start_index"], -(r["end_index"] - r["start_index"])))
        output_results(results, args.output_mode, args.output_file)
    else:
        logging.info("No matching segments detected in the recorded track.")

if __name__ == "__main__":
    main()
