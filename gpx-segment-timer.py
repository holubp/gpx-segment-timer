#!/usr/bin/env python3
"""
GPX Segment Time Measurement Script

This script compares a recorded GPX track against a set of reference segments (GPX files)
to determine the elapsed time on each segment. It accounts for varying sampling frequencies,
nonuniform point counts, and segments that can appear multiple times (e.g. laps). The matching
process uses a distance-based candidate extraction and DTW (Dynamic Time Warping) on resampled
points for robust shape comparison.

Output can be printed to STDOUT as a pretty-printed table, or written to CSV/XLSX files.

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

# Try to import openpyxl for XLSX output
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
        - p1 and p2 are tuples with two floats (latitude, longitude)
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

    Each dictionary represents a point and has the keys:
      - 'lat': latitude (float)
      - 'lon': longitude (float)
      - 'time': timestamp (datetime) or None

    Preconditions:
        - filepath exists and is a valid GPX file.
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
                    'time': point.time  # May be None if not provided
                })
    logging.debug("Loaded %d points from %s.", len(points), filepath)
    return points

def load_reference_segments(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all GPX files from the given folder as reference segments.

    Returns:
        A dictionary mapping filename to a list of point dictionaries.

    Preconditions:
        - folder exists.
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
    Compute the total distance of a track (in meters) by summing the distance between successive points.

    Preconditions:
        - points contains at least 1 point.
    """
    assert len(points) >= 1, "At least one point is required to compute distance."
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
      points: List of point dictionaries (each with keys 'lat' and 'lon').
      num_samples: Target number of samples.
    
    Returns:
      A list of (lat, lon) tuples resampled uniformly along the track.
    
    Preconditions:
      - num_samples >= 2 if len(points) >= 2.
    """
    if not points:
        return []
    if len(points) < 2:
        return [(points[0]['lat'], points[0]['lon'])]
    
    # Compute cumulative distances.
    cum_dists = [0.0]
    for i in range(1, len(points)):
        d = haversine_distance(
            (points[i-1]['lat'], points[i-1]['lon']),
            (points[i]['lat'], points[i]['lon'])
        )
        cum_dists.append(cum_dists[-1] + d)
    total_dist = cum_dists[-1]
    # Create equally spaced target distances.
    target_dists = [i * total_dist / (num_samples - 1) for i in range(num_samples)]
    new_points: List[Tuple[float, float]] = []
    j = 0
    for t in target_dists:
        while j < len(cum_dists) - 1 and cum_dists[j+1] < t:
            j += 1
        if j >= len(points)-1:
            new_points.append((points[-1]['lat'], points[-1]['lon']))
        else:
            segment_length = cum_dists[j+1] - cum_dists[j]
            if segment_length == 0:
                new_points.append((points[j]['lat'], points[j]['lon']))
            else:
                ratio = (t - cum_dists[j]) / segment_length
                lat = points[j]['lat'] + ratio * (points[j+1]['lat'] - points[j]['lat'])
                lon = points[j]['lon'] + ratio * (points[j+1]['lon'] - points[j]['lon'])
                new_points.append((lat, lon))
    logging.debug("Resampled from %d to %d points.", len(points), num_samples)
    return new_points

def find_all_segment_matches(
    recorded_points: List[Dict[str, Any]],
    ref_points: List[Dict[str, Any]],
    candidate_margin: float,
    dtw_threshold: float,
    resample_count: int,
    min_gap: int = 1
) -> List[Tuple[int, int, float]]:
    """
    Find all matching segments in a recorded GPX track that match a given reference segment.
    
    This function uses the cumulative distance of the recorded track and the total distance of the
    reference segment to determine a candidate window (with an allowed error margin). Within that
    candidate endpoint range, a dynamic stride is computed (set to roughly 1/10th of the candidate range size)
    to skip candidate evaluations. For each candidate window, the segment is resampled to a fixed
    number of points (resample_count) and compared with the resampled reference segment via DTW,
    using the haversine distance as a metric. An early-termination check is used: if the DTW average (dtw_avg)
    ceases to improve by more than 1%, the candidate loop for that start index is terminated.
    
    Args:
      recorded_points: List of dictionaries for the recorded track, each with keys 'lat', 'lon', and 'time'.
      ref_points: List of dictionaries for the reference segment with the same keys.
      candidate_margin: Allowed fractional error in candidate total distance relative to the reference
                        segment’s total distance (e.g., 0.2 for ±20%).
      dtw_threshold: Maximum allowed average DTW distance (meters per resampled point) for a valid match.
      resample_count: Number of points to which both candidate and reference segments are resampled.
      min_gap: Minimum number of recorded points to skip after a detected match to avoid overlaps.
    
    Returns:
      A list of tuples (start_index, end_index, dtw_avg) for each detected matching segment.
    
    Preconditions:
      - recorded_points and ref_points are non-empty.
      - candidate_margin > 0, dtw_threshold > 0.
      - resample_count >= 2.
    
    Note:
      This function assumes that supporting functions haversine_distance, compute_total_distance, and 
      resample_points are already defined.
    """
    assert recorded_points, "Recorded track must contain points."
    assert ref_points, "Reference segment must contain points."
    assert candidate_margin > 0, "Candidate margin must be positive."
    assert dtw_threshold > 0, "DTW threshold must be positive."
    assert resample_count >= 2, "Resample count must be at least 2."
    
    # Precompute the total distance of the reference segment and its resampled version.
    ref_total_distance = compute_total_distance(ref_points)
    ref_resampled = resample_points(ref_points, resample_count)
    
    # Precompute cumulative distances for the recorded track.
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
        # Determine target cumulative distance bounds for a candidate segment.
        lower_target = rec_cum_dists[start] + ref_total_distance * (1 - candidate_margin)
        upper_target = rec_cum_dists[start] + ref_total_distance * (1 + candidate_margin)
        lower_end = bisect.bisect_left(rec_cum_dists, lower_target, lo=start+1, hi=rec_length)
        upper_end = bisect.bisect_right(rec_cum_dists, upper_target, lo=start+1, hi=rec_length)
        
        logging.debug("Start index %d: candidate endpoints in range [%d, %d)", start, lower_end, upper_end)
        candidate_range_length = upper_end - lower_end
        # Compute dynamic stride: skip approximately 1/10th of the candidate points.
        dynamic_stride = max(1, candidate_range_length // 10)
        
        best_dtw = float('inf')
        best_candidate = None
        candidate_found = False
        
        # Iterate over candidate endpoints using the dynamic stride.
        for end in range(lower_end, upper_end, dynamic_stride):
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
            
            # Early termination: if dtw_avg starts to worsen by more than 1% compared to the best so far, break out.
            if dtw_avg > best_dtw * 1.01:
                logging.debug("Early termination at candidate %d due to stalled improvement.", end)
                break
            
            if dtw_avg < dtw_threshold:
                matches.append((start, end + 1, dtw_avg))
                logging.info("Match found from %d to %d with dtw_avg=%.2f", start, end+1, dtw_avg)
                # Advance start to avoid overlapping matches.
                start = end + min_gap
                candidate_found = True
                break  # Proceed to the next start index.
        
        if not candidate_found:
            start += 1
    
    return matches


def measure_segment_time(recorded_points: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Optional[float]:
    """
    Compute the time difference (in seconds) between the first and last point of a segment.

    Returns:
      - The time difference in seconds, or None if either endpoint is missing a timestamp.

    Preconditions:
      - 0 <= start_idx < end_idx <= len(recorded_points)
    """
    assert 0 <= start_idx < end_idx <= len(recorded_points), "Invalid indices for measured segment."
    start_time = recorded_points[start_idx]['time']
    end_time = recorded_points[end_idx - 1]['time']
    if start_time and end_time:
        seconds = (end_time - start_time).total_seconds()
        logging.debug("Measured time for segment (%d, %d): %.2f seconds", start_idx, end_idx, seconds)
        return seconds
    logging.warning("Missing timestamp at one or both endpoints for indices %d and %d.", start_idx, end_idx)
    return None

def output_results(
    results: List[Dict[str, Any]],
    output_mode: str,
    output_file: Optional[str]
) -> None:
    """
    Output the measurement results in one of three formats:
      - 'stdout': pretty-printed table on STDOUT.
      - 'csv': CSV file.
      - 'xlsx': Excel XLSX file.
    
    Args:
      results: List of result dictionaries with keys: 'segment', 'start_index', 'end_index',
               'dtw_avg', 'time_seconds', and 'time_str'.
      output_mode: One of "stdout", "csv", "xlsx".
      output_file: Path to the output file (required for 'csv' and 'xlsx').
    
    Preconditions:
      - output_mode is one of the permitted options.
      - For csv/xlsx, output_file is not None.
    """
    if output_mode == "stdout":
        # Determine column widths and print header.
        header = ["Segment", "Start Idx", "End Idx", "DTW Avg (m)", "Time (s)", "Time (H:M:S)"]
        print("{:<25} {:>10} {:>10} {:>15} {:>12} {:>15}".format(*header))
        print("-" * 90)
        for res in results:
            print("{:<25} {:>10} {:>10} {:>15.2f} {:>12.2f} {:>15}".format(
                res["segment"],
                res["start_index"],
                res["end_index"],
                res["dtw_avg"],
                res["time_seconds"] if res["time_seconds"] is not None else -1,
                res["time_str"] if res["time_str"] is not None else "N/A"
            ))
    elif output_mode == "csv":
        if output_file is None:
            raise ValueError("Output file must be specified for CSV output.")
        with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Segment", "Start Index", "End Index", "DTW Avg (m)", "Time (s)", "Time (H:M:S)"])
            for res in results:
                writer.writerow([
                    res["segment"],
                    res["start_index"],
                    res["end_index"],
                    f"{res['dtw_avg']:.2f}",
                    f"{res['time_seconds']:.2f}" if res["time_seconds"] is not None else "",
                    res["time_str"] if res["time_str"] is not None else ""
                ])
        logging.info("Results written to CSV file: %s", output_file)
    elif output_mode == "xlsx":
        if output_file is None:
            raise ValueError("Output file must be specified for XLSX output.")
        if Workbook is None:
            raise ImportError("openpyxl library is required for XLSX output. Please install it via pip.")
        wb = Workbook()
        ws = wb.active
        ws.title = "Segment Times"
        ws.append(["Segment", "Start Index", "End Index", "DTW Avg (m)", "Time (s)", "Time (H:M:S)"])
        for res in results:
            ws.append([
                res["segment"],
                res["start_index"],
                res["end_index"],
                float(f"{res['dtw_avg']:.2f}"),
                float(f"{res['time_seconds']:.2f}") if res["time_seconds"] is not None else None,
                res["time_str"] if res["time_str"] is not None else ""
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
    Main routine for the GPX segment measurement script.
    Parses command-line arguments, loads the recorded track and reference segments,
    finds all matching segments, computes their elapsed time, and outputs the results.
    """
    parser = argparse.ArgumentParser(
        description="Measure elapsed time on reference segments within a recorded GPX track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Basic I/O
    parser.add_argument("-r", "--recorded", required=True,
                        help="Path to the GPX file containing the recorded track.")
    parser.add_argument("-f", "--reference-folder", required=True,
                        help="Path to the folder containing reference GPX segment files.")
    parser.add_argument("-o", "--output-mode", choices=["stdout", "csv", "xlsx"], default="stdout",
                        help="Output format: pretty printed to STDOUT, CSV file, or XLSX file.")
    parser.add_argument("-O", "--output-file", default=None,
                        help="Output file path (required for CSV and XLSX output modes).")
    # Matching parameters
    parser.add_argument("--candidate-margin", type=float, default=0.2,
                        help="Allowed variation (fraction) in candidate segment distance relative to the reference distance.")
    parser.add_argument("--dtw-threshold", type=float, default=50,
                        help="Maximum allowed average DTW distance (in meters per resampled point) for a valid match.")
    parser.add_argument("--resample-count", type=int, default=50,
                        help="Number of points for resampling candidate and reference segments.")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="Minimum number of recorded points to skip after a detected match to avoid duplicates.")
    # Logging verbosity
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging (info level).")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging (more detailed).")

    args = parser.parse_args()

    # Set up logging.
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Sanity check for output mode.
    if args.output_mode in ["csv", "xlsx"] and not args.output_file:
        parser.error("Output file path must be provided for CSV or XLSX output mode.")

    logging.info("Loading recorded track: %s", args.recorded)
    recorded_points = load_gpx_points(args.recorded)
    if not recorded_points:
        logging.error("No points found in the recorded GPX file: %s", args.recorded)
        return

    logging.info("Loading reference segments from folder: %s", args.reference_folder)
    ref_segments = load_reference_segments(args.reference_folder)
    if not ref_segments:
        logging.error("No valid reference GPX files found in folder: %s", args.reference_folder)
        return

    results: List[Dict[str, Any]] = []
    # Process each reference segment (each file)
    for seg_filename, ref_points in ref_segments.items():
        logging.info("Processing reference segment: %s", seg_filename)
        if not ref_points:
            logging.warning("Reference segment %s contains no points; skipping.", seg_filename)
            continue
        matches = find_all_segment_matches(
            recorded_points,
            ref_points,
            candidate_margin=args.candidate_margin,
            dtw_threshold=args.dtw_threshold,
            resample_count=args.resample_count,
            min_gap=args.min_gap
        )
        if not matches:
            logging.info("No matching segments found for reference '%s'.", seg_filename)
            continue
        for (start_idx, end_idx, dtw_avg) in matches:
            time_seconds = measure_segment_time(recorded_points, start_idx, end_idx)
            time_str = str(datetime.timedelta(seconds=int(time_seconds))) if time_seconds is not None else "N/A"
            result = {
                "segment": seg_filename,
                "start_index": start_idx,
                "end_index": end_idx,
                "dtw_avg": dtw_avg,
                "time_seconds": time_seconds,
                "time_str": time_str
            }
            results.append(result)
            logging.info("Segment %s: indices (%d, %d), dtw_avg=%.2f, time=%s",
                         seg_filename, start_idx, end_idx, dtw_avg, time_str)

    if results:
        output_results(results, args.output_mode, args.output_file)
    else:
        logging.info("No matching segments were detected in the recorded track.")

if __name__ == "__main__":
    main()

