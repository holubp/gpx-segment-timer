#!/usr/bin/env python3
"""
GPX Segment Time Measurement Script

This script compares a recorded GPX track against a set of reference segments (GPX files)
to determine the elapsed time on each segment. It is robust against varying sampling frequencies,
nonuniform point counts, and repeated segments (e.g. laps). The matching process uses a
distance-based candidate extraction combined with Dynamic Time Warping (DTW) on resampled points.

Additional features:
  - Refines match boundaries by locally searching for the best shape alignment of the candidate
    endpoints to the reference endpoints using a mini-DTW.
  - Optionally exports matched segments as individual GPX tracks.
  - The DTW search is restricted to candidate windows whose endpoints lie within an expanded 
    bounding box (default ±30 m) computed from the reference segment.
  - In the final output, the script prints the reference segment's start/end coordinates and
    ensures that the detected segment start/end are within the limit given by --bbox-margin.
    If not, it attempts to adjust the boundary within a limited search window and reports the distance.
  
Output can be printed to STDOUT (pretty-printed table), CSV, XLSX, and optionally exported as GPX.

Author: [Your Name]
Date: [Today's Date]
"""

import os
import math
import argparse
import datetime
import csv
import logging
import bisect
from typing import List, Tuple, Dict, Optional, Any

import gpxpy
from fastdtw import fastdtw

# Try to import openpyxl for XLSX output.
try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

# -------------------------------------
# Helper Functions
# -------------------------------------

def haversine_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the haversine distance (in meters) between two (latitude, longitude) points.

    Preconditions:
      - p1 and p2 are tuples of two floats (latitude, longitude).
    """
    radius = 6371000  # Earth's radius in meters
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return radius * c

def load_gpx_points(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a GPX file and return a list of point dictionaries.

    Each dictionary has:
      - 'lat': latitude (float)
      - 'lon': longitude (float)
      - 'time': timestamp (datetime) or None

    Preconditions: The file exists and is a valid GPX file.
    """
    points: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'time': point.time  # May be None
                })
    logging.debug("Loaded %d points from %s.", len(points), filepath)
    return points

def load_reference_segments(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all GPX files from the given folder as reference segments.

    Returns a dictionary mapping filename to list of point dictionaries.
    """
    segments: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(folder):
        raise ValueError(f"Reference folder '{folder}' is not a valid directory.")
    for filename in os.listdir(folder):
        if filename.lower().endswith('.gpx'):
            filepath = os.path.join(folder, filename)
            seg_points = load_gpx_points(filepath)
            if seg_points:
                segments[filename] = seg_points
                logging.debug("Loaded reference segment '%s' with %d points.", filename, len(seg_points))
    return segments

def compute_total_distance(points: List[Dict[str, Any]]) -> float:
    """
    Compute the total distance (in meters) of a track by summing the distances between successive points.
    """
    assert len(points) >= 1, "At least one point is required."
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_distance((points[i-1]['lat'], points[i-1]['lon']),
                                    (points[i]['lat'], points[i]['lon']))
    logging.debug("Computed total distance: %.2f meters for %d points.", total, len(points))
    return total

def resample_points(points: List[Dict[str, Any]], num_samples: int) -> List[Tuple[float, float]]:
    """
    Resample a list of points uniformly along its cumulative distance.

    Args:
      points: List of point dictionaries.
      num_samples: Target number of samples.
    
    Returns a list of (lat, lon) tuples.
    """
    if not points:
        return []
    if len(points) < 2:
        return [(points[0]['lat'], points[0]['lon'])]
    
    cum_dists = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            (points[i-1]['lat'], points[i-1]['lon']),
            (points[i]['lat'], points[i]['lon'])
        )
        cum_dists.append(cum_dists[-1] + d)
    total_dist = cum_dists[-1]
    target_dists = [i * total_dist / (num_samples - 1) for i in range(num_samples)]
    new_points: List[Tuple[float, float]] = []
    j = 0
    for t in target_dists:
        while j < len(cum_dists)-1 and cum_dists[j+1] < t:
            j += 1
        if j >= len(points)-1:
            new_points.append((points[-1]['lat'], points[-1]['lon']))
        else:
            seg_length = cum_dists[j+1] - cum_dists[j]
            if seg_length == 0:
                new_points.append((points[j]['lat'], points[j]['lon']))
            else:
                ratio = (t - cum_dists[j]) / seg_length
                lat = points[j]['lat'] + ratio * (points[j+1]['lat'] - points[j]['lat'])
                lon = points[j]['lon'] + ratio * (points[j+1]['lon'] - points[j]['lon'])
                new_points.append((lat, lon))
    logging.debug("Resampled from %d to %d points.", len(points), num_samples)
    return new_points

def compute_bounding_box(points: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box (min_lat, max_lat, min_lon, max_lon) for a list of points.
    """
    lats = [p['lat'] for p in points]
    lons = [p['lon'] for p in points]
    return (min(lats), max(lats), min(lons), max(lons))

def expand_bounding_box(bbox: Tuple[float, float, float, float], margin_m: float = 30) -> Tuple[float, float, float, float]:
    """
    Expand a bounding box by a given margin (in meters).

    Args:
      bbox: (min_lat, max_lat, min_lon, max_lon)
      margin_m: Margin in meters (default: 30 m)
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    margin_deg_lat = margin_m / 111000
    avg_lat = (min_lat + max_lat) / 2
    margin_deg_lon = margin_m / (111000 * math.cos(math.radians(avg_lat)))
    return (min_lat - margin_deg_lat, max_lat + margin_deg_lat,
            min_lon - margin_deg_lon, max_lon + margin_deg_lon)

def point_in_bbox(point: Dict[str, Any], bbox: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point (with 'lat' and 'lon') is inside the given bounding box.
    """
    lat, lon = point['lat'], point['lon']
    min_lat, max_lat, min_lon, max_lon = bbox
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

def refine_boundaries_with_shape(recorded_points: List[Dict[str, Any]],
                                 ref_points: List[Dict[str, Any]],
                                 current_start: int,
                                 current_end: int,
                                 search_window: int = 5,
                                 boundary_fraction: float = 0.1
                                ) -> Tuple[int, int]:
    """
    Refine candidate boundaries using a local shape-based mini-DTW over a portion of the segment.

    Args:
      recorded_points: List of points from the recorded track.
      ref_points: List of points from the reference segment.
      current_start: Initial candidate start index.
      current_end: Initial candidate end index.
      search_window: Number of indices to search around boundaries (default: 5).
      boundary_fraction: Fraction of the reference segment to use for matching (default: 0.1).

    Returns:
      (refined_start, refined_end)
    """
    import math
    def to_latlon(pts: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        return [(p['lat'], p['lon']) for p in pts]

    ref_len = len(ref_points)
    portion_size = max(2, int(ref_len * boundary_fraction))
    ref_start_shape = to_latlon(ref_points[:portion_size])
    ref_end_shape = to_latlon(ref_points[-portion_size:])

    # Refine start boundary.
    best_start = current_start
    best_start_dtw = math.inf
    start_low = max(0, current_start - search_window)
    start_high = min(len(recorded_points) - portion_size, current_start + search_window)
    for s in range(start_low, start_high + 1):
        candidate_slice = recorded_points[s:s+portion_size]
        if len(candidate_slice) < 2:
            continue
        cand_coords = to_latlon(candidate_slice)
        dtw_dist, _ = fastdtw(cand_coords, ref_start_shape, dist=haversine_distance)
        avg_dist = dtw_dist / len(cand_coords)
        if avg_dist < best_start_dtw:
            best_start_dtw = avg_dist
            best_start = s

    # Refine end boundary.
    best_end = current_end
    best_end_dtw = math.inf
    end_low = max(best_start + portion_size, current_end - search_window)
    end_high = min(len(recorded_points), current_end + search_window)
    for e in range(end_low, end_high + 1):
        candidate_slice = recorded_points[e-portion_size:e]
        if len(candidate_slice) < 2:
            continue
        cand_coords = to_latlon(candidate_slice)
        dtw_dist, _ = fastdtw(cand_coords, ref_end_shape, dist=haversine_distance)
        avg_dist = dtw_dist / len(cand_coords)
        if avg_dist < best_end_dtw:
            best_end_dtw = avg_dist
            best_end = e
    return best_start, best_end

def adjust_boundary_to_margin(recorded_points: List[Dict[str, Any]],
                              current_index: int,
                              ref_coord: Tuple[float, float],
                              margin_m: float,
                              window: int = 20) -> int:
    """
    Adjust the boundary index to the closest point within a search window.

    Args:
      recorded_points: List of points from the recorded track.
      current_index: Current boundary index.
      ref_coord: Reference coordinate (lat, lon).
      margin_m: The meter limit (from --bbox-margin).
      window: How many indices around current_index to search (default: 20).

    Returns:
      The adjusted index (which may be the original if no closer point is found).
    """
    best_index = current_index
    best_dist = haversine_distance((recorded_points[current_index]['lat'], recorded_points[current_index]['lon']), ref_coord)
    start_search = max(0, current_index - window)
    end_search = min(len(recorded_points) - 1, current_index + window)
    for i in range(start_search, end_search + 1):
        d = haversine_distance((recorded_points[i]['lat'], recorded_points[i]['lon']), ref_coord)
        if d < best_dist:
            best_dist = d
            best_index = i
    if best_dist > margin_m:
        logging.warning("Boundary adjustment: best distance %.2f m exceeds margin %.2f m.", best_dist, margin_m)
    return best_index

def find_all_segment_matches(
    recorded_points: List[Dict[str, Any]],
    ref_points: List[Dict[str, Any]],
    candidate_margin: float,
    dtw_threshold: float,
    resample_count: int,
    min_gap: int = 1,
    bbox_margin_m: float = 30
) -> List[Tuple[int, int, float]]:
    """
    Find all matching segments in a recorded GPX track that match a given reference segment.

    Uses cumulative distances to select candidate windows within an allowed variation
    (candidate_margin) and restricts candidates to those within an expanded bounding box.
    For each candidate meeting the DTW threshold, local shape-based refinement is applied.

    Args:
      recorded_points: Recorded track points (with 'lat', 'lon', 'time').
      ref_points: Reference segment points.
      candidate_margin: Allowed fractional error in total distance (e.g. 0.2 for ±20%).
      dtw_threshold: Maximum average DTW distance (meters per resampled point) for a match.
      resample_count: Number of points for resampling segments.
      min_gap: Minimum number of points to skip after a match.
      bbox_margin_m: Margin (m) to expand the reference bounding box.

    Returns a list of tuples (start_index, end_index, dtw_avg).
    """
    assert recorded_points, "Recorded track must contain points."
    assert ref_points, "Reference segment must contain points."
    assert candidate_margin > 0, "Candidate margin must be positive."
    assert dtw_threshold > 0, "DTW threshold must be positive."
    assert resample_count >= 2, "Resample count must be at least 2."

    ref_total_distance = compute_total_distance(ref_points)
    ref_resampled = resample_points(ref_points, resample_count)

    ref_bbox = compute_bounding_box(ref_points)
    expanded_bbox = expand_bounding_box(ref_bbox, margin_m=bbox_margin_m)

    rec_cum_dists: List[float] = [0.0]
    for i in range(1, len(recorded_points)):
        d = haversine_distance(
            (recorded_points[i-1]['lat'], recorded_points[i-1]['lon']),
            (recorded_points[i]['lat'], recorded_points[i]['lon'])
        )
        rec_cum_dists.append(rec_cum_dists[-1] + d)

    matches: List[Tuple[int, int, float]] = []
    rec_length = len(recorded_points)
    start = 0

    while start < rec_length - 1:
        if not point_in_bbox(recorded_points[start], expanded_bbox):
            start += 1
            continue

        lower_target = rec_cum_dists[start] + ref_total_distance * (1 - candidate_margin)
        upper_target = rec_cum_dists[start] + ref_total_distance * (1 + candidate_margin)
        lower_end = bisect.bisect_left(rec_cum_dists, lower_target, lo=start+1, hi=rec_length)
        upper_end = bisect.bisect_right(rec_cum_dists, upper_target, lo=start+1, hi=rec_length)
        logging.debug("Start index %d: candidate endpoints in range [%d, %d)", start, lower_end, upper_end)
        candidate_range_length = upper_end - lower_end
        dynamic_stride = max(1, candidate_range_length // 10)

        best_dtw = float('inf')
        best_candidate = None
        candidate_found = False

        for end in range(lower_end, upper_end, dynamic_stride):
            if not (point_in_bbox(recorded_points[start], expanded_bbox) and 
                    point_in_bbox(recorded_points[end], expanded_bbox)):
                continue

            candidate_segment = recorded_points[start:end+1]
            if len(candidate_segment) < 2:
                continue

            candidate_resampled = resample_points(candidate_segment, resample_count)
            dtw_distance, _ = fastdtw(candidate_resampled, ref_resampled, dist=haversine_distance)
            dtw_avg = dtw_distance / resample_count
            logging.debug("Candidate from %d to %d: dtw_avg=%.2f", start, end+1, dtw_avg)

            if dtw_avg < best_dtw:
                best_dtw = dtw_avg
                best_candidate = end

            if dtw_avg > best_dtw * 1.01:
                logging.debug("Early termination at candidate %d due to stalled improvement.", end)
                break

            if dtw_avg < dtw_threshold:
                refined_start, refined_end = refine_boundaries_with_shape(recorded_points, ref_points, start, end+1)
                matches.append((refined_start, refined_end, dtw_avg))
                logging.info("Match found from %d to %d with dtw_avg=%.2f", refined_start, refined_end, dtw_avg)
                start = end + min_gap
                candidate_found = True
                break

        if not candidate_found:
            start += 1

    return matches

def measure_segment_time(recorded_points: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Optional[float]:
    """
    Compute the elapsed time (in seconds) between the first and last point of a segment.

    Preconditions: 0 <= start_idx < end_idx <= len(recorded_points).
    """
    assert 0 <= start_idx < end_idx <= len(recorded_points), "Invalid indices."
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
    Export matched segments as separate GPX tracks in a file.
    Each track is named using the recorded file name, reference file name, and a match number.
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

def output_results(
    results: List[Dict[str, Any]],
    output_mode: str,
    output_file: Optional[str]
) -> None:
    """
    Output measurement results in one of three formats: stdout, csv, or xlsx.
    
    New columns added:
      - "Ref Start": Coordinates (lat, lon) for the reference segment start.
      - "Ref End": Coordinates for the reference segment end.
      - "Start Diff (m)": Distance from detected start to reference start.
      - "End Diff (m)": Distance from detected end to reference end.
    """
    header = ["Segment", "Start Idx", "End Idx", "DTW Avg (m)", "Time (s)", "Time (H:M:S)",
              "Ref Start", "Ref End", "Start Diff (m)", "End Diff (m)"]
    if output_mode == "stdout":
        print("{:<25} {:>10} {:>10} {:>15} {:>12} {:>15} {:>25} {:>25} {:>18} {:>15}".format(*header))
        print("-" * 170)
        for res in results:
            print("{:<25} {:>10} {:>10} {:>15.2f} {:>12.2f} {:>15} {:>25} {:>25} {:>18.2f} {:>15.2f}".format(
                res["segment"],
                res["start_index"],
                res["end_index"],
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
            raise ValueError("Output file must be specified for CSV output.")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for res in results:
                writer.writerow([
                    res["segment"],
                    res["start_index"],
                    res["end_index"],
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
            raise ValueError("Output file must be specified for XLSX output.")
        if Workbook is None:
            raise ImportError("openpyxl library is required for XLSX output.")
        wb = Workbook()
        ws = wb.active
        ws.title = "Segment Times"
        ws.append(header)
        for res in results:
            ws.append([
                res["segment"],
                res["start_index"],
                res["end_index"],
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
    Main routine: parses args, loads the recorded track and reference segments,
    finds matching segments, refines boundaries and measures elapsed time, checks boundary differences
    against the --bbox-margin, and outputs the results. Optionally exports matched segments as GPX.
    """
    parser = argparse.ArgumentParser(
        description="Measure elapsed time on reference segments within a recorded GPX track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Basic I/O.
    parser.add_argument("-r", "--recorded", required=True,
                        help="Path to the GPX file containing the recorded track.")
    parser.add_argument("-f", "--reference-folder", required=True,
                        help="Path to the folder containing reference GPX segment files.")
    parser.add_argument("-o", "--output-mode", choices=["stdout", "csv", "xlsx"], default="stdout",
                        help="Output format: pretty printed to STDOUT, CSV, or XLSX.")
    parser.add_argument("-O", "--output-file", default=None,
                        help="Output file path (required for CSV and XLSX outputs).")
    # Matching parameters.
    parser.add_argument("--candidate-margin", type=float, default=0.2,
                        help="Allowed variation (fraction) in candidate segment distance relative to reference.")
    parser.add_argument("--dtw-threshold", type=float, default=50,
                        help="Maximum allowed average DTW distance (m per resampled point) for a match.")
    parser.add_argument("--resample-count", type=int, default=50,
                        help="Number of points for resampling segments.")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="Minimum number of recorded points to skip after a detected match.")
    parser.add_argument("--bbox-margin", type=float, default=30,
                        help="Margin (in meters) to expand the reference bounding box and define boundary tolerance.")
    parser.add_argument("--refine-window", type=int, default=5,
                        help="Window size (in points) for refining match boundaries using shape-based alignment.")
    # Export option for matched segments as GPX.
    parser.add_argument("--export-gpx", action="store_true",
                        help="Export matched segments as individual tracks in a GPX file.")
    parser.add_argument("--export-gpx-file", default="matched_segments.gpx",
                        help="Output GPX file for exporting matched segments.")
    # Logging verbosity.
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging (INFO level).")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging (DEBUG level).")

    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.output_mode in ["csv", "xlsx"] and not args.output_file:
        parser.error("Output file must be provided for CSV or XLSX outputs.")

    logging.info("Loading recorded track: %s", args.recorded)
    recorded_points = load_gpx_points(args.recorded)
    if not recorded_points:
        logging.error("No points found in recorded GPX file: %s", args.recorded)
        return

    logging.info("Loading reference segments from folder: %s", args.reference_folder)
    ref_segments = load_reference_segments(args.reference_folder)
    if not ref_segments:
        logging.error("No valid reference GPX files found in: %s", args.reference_folder)
        return

    results: List[Dict[str, Any]] = []
    # Process each reference segment.
    for seg_filename, ref_points in ref_segments.items():
        logging.info("Processing reference segment: %s", seg_filename)
        if not ref_points:
            logging.warning("Reference segment %s has no points; skipping.", seg_filename)
            continue

        # Print reference segment start and end coordinates.
        ref_start_coords = (ref_points[0]['lat'], ref_points[0]['lon'])
        ref_end_coords = (ref_points[-1]['lat'], ref_points[-1]['lon'])
        logging.info("Reference '%s' start: (%.6f, %.6f), end: (%.6f, %.6f)",
                     seg_filename, ref_start_coords[0], ref_start_coords[1],
                     ref_end_coords[0], ref_end_coords[1])
        
        matches = find_all_segment_matches(
            recorded_points,
            ref_points,
            candidate_margin=args.candidate_margin,
            dtw_threshold=args.dtw_threshold,
            resample_count=args.resample_count,
            min_gap=args.min_gap,
            bbox_margin_m=args.bbox_margin
        )
        if not matches:
            logging.info("No matching segments found for reference '%s'.", seg_filename)
            continue

        for (start_idx, end_idx, dtw_avg) in matches:
            refined_start, refined_end = refine_boundaries_with_shape(recorded_points, ref_points,
                                                                       start_idx, end_idx,
                                                                       search_window=args.refine_window)
            # Adjust boundaries if they are not within the bbox margin.
            new_refined_start = adjust_boundary_to_margin(recorded_points, refined_start, ref_start_coords, args.bbox_margin)
            new_refined_end = adjust_boundary_to_margin(recorded_points, refined_end - 1, ref_end_coords, args.bbox_margin) + 1
            refined_start, refined_end = new_refined_start, new_refined_end

            time_seconds = measure_segment_time(recorded_points, refined_start, refined_end)
            time_str = str(datetime.timedelta(seconds=int(time_seconds))) if time_seconds is not None else "N/A"

            detected_start_coords = (recorded_points[refined_start]['lat'], recorded_points[refined_start]['lon'])
            detected_end_coords   = (recorded_points[refined_end - 1]['lat'], recorded_points[refined_end - 1]['lon'])
            start_diff = haversine_distance(detected_start_coords, ref_start_coords)
            end_diff = haversine_distance(detected_end_coords, ref_end_coords)

            logging.info("Detected start for '%s' is %.2f m from ref start (limit: %.2f m).",
                         seg_filename, start_diff, args.bbox_margin)
            logging.info("Detected end for '%s' is %.2f m from ref end (limit: %.2f m).",
                         seg_filename, end_diff, args.bbox_margin)

            result = {
                "segment": seg_filename,
                "start_index": refined_start,
                "end_index": refined_end,
                "dtw_avg": dtw_avg,
                "time_seconds": time_seconds,
                "time_str": time_str,
                "ref_start": f"({ref_start_coords[0]:.6f}, {ref_start_coords[1]:.6f})",
                "ref_end": f"({ref_end_coords[0]:.6f}, {ref_end_coords[1]:.6f})",
                "start_diff": start_diff,
                "end_diff": end_diff
            }
            results.append(result)
            logging.info("Segment %s: indices (%d, %d), dtw_avg=%.2f, time=%s, start_diff=%.2f, end_diff=%.2f",
                         seg_filename, refined_start, refined_end, dtw_avg, time_str, start_diff, end_diff)

            if args.export_gpx:
                export_filename = f"{os.path.splitext(args.export_gpx_file)[0]}_{seg_filename}_match{results.index(result)+1}.gpx"
                export_matches_to_gpx([(refined_start, refined_end, dtw_avg)],
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

