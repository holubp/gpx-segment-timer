#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

import gpxpy
import gpxpy.gpx


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def meters_to_lat(m):
    return m / 111111.0


def meters_to_lon(m, lat):
    return m / (111111.0 * math.cos(math.radians(lat)))


def apply_noise(points, sigma_m, rng):
    out = []
    for lat, lon in points:
        dlat = meters_to_lat(rng.gauss(0.0, sigma_m))
        dlon = meters_to_lon(rng.gauss(0.0, sigma_m), lat)
        out.append((lat + dlat, lon + dlon))
    return out


def apply_translation(points, east_m, north_m):
    out = []
    for lat, lon in points:
        dlat = meters_to_lat(north_m)
        dlon = meters_to_lon(east_m, lat)
        out.append((lat + dlat, lon + dlon))
    return out


def insert_linger(points, idx, count):
    linger = points[idx:idx + 5]
    return points[:idx] + linger * count + points[idx:]


def insert_detour(points, idx, detour_m):
    if idx <= 0 or idx >= len(points) - 1:
        return points
    lat, lon = points[idx]
    dlat = meters_to_lat(detour_m)
    dlon = meters_to_lon(detour_m, lat)
    detour = [(lat + dlat, lon), (lat + dlat, lon + dlon), (lat, lon + dlon)]
    return points[:idx] + detour + points[idx:]


def shift_endpoints(points, shift_m):
    if len(points) < 6:
        return points
    lat0, lon0 = points[0]
    dlat = meters_to_lat(shift_m)
    dlon = meters_to_lon(shift_m, lat0)
    head = [(lat + dlat, lon - dlon) for lat, lon in points[:3]]
    tail = [(lat - dlat, lon + dlon) for lat, lon in points[-3:]]
    return head + points[3:-3] + tail


def add_lead_in_out(points, count=5, spacing_m=2.0):
    if len(points) < 2:
        return points
    lat0, lon0 = points[0]
    lat1, lon1 = points[1]
    base_dist = max(haversine_m(lat0, lon0, lat1, lon1), 1e-6)
    scale = spacing_m / base_dist
    step_lat = (lat1 - lat0) * scale
    step_lon = (lon1 - lon0) * scale
    lead = [(lat0 - step_lat * i, lon0 - step_lon * i) for i in range(count, 0, -1)]

    lat_last, lon_last = points[-1]
    lat_prev, lon_prev = points[-2]
    base_dist_tail = max(haversine_m(lat_prev, lon_prev, lat_last, lon_last), 1e-6)
    scale_tail = spacing_m / base_dist_tail
    step_lat_tail = (lat_last - lat_prev) * scale_tail
    step_lon_tail = (lon_last - lon_prev) * scale_tail
    tail = [(lat_last + step_lat_tail * i, lon_last + step_lon_tail * i) for i in range(1, count + 1)]

    return lead + points + tail


def load_segment_points(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.latitude, pt.longitude))
    return points


def write_recording(path: Path, points, start_time, interval_s):
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    current = start_time
    for lat, lon in points:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, time=current))
        current += timedelta(seconds=interval_s)
    path.write_text(gpx.to_xml())


def build_case(case_dir: Path, expected_dir: Path, segments: Path, pattern: str, rng: random.Random, mode: str, interval_s: float = 0.2):
    refs = sorted(segments.glob(pattern))
    points = []
    expected = []
    idx = 0
    for ref in refs:
        ref_pts = load_segment_points(ref)
        if mode == "match_noise":
            seg_pts = apply_noise(ref_pts, 2.0, rng)
            seg_pts = add_lead_in_out(seg_pts)
            start = idx
            points.extend(seg_pts)
            idx += len(seg_pts)
            expected.append({"segment": ref.name, "start_idx": start, "end_idx": idx - 1})
        elif mode == "match_linger":
            seg_pts = apply_noise(ref_pts, 1.0, rng)
            seg_pts = insert_linger(seg_pts, len(seg_pts) // 2, 3)
            seg_pts = add_lead_in_out(seg_pts)
            start = idx
            points.extend(seg_pts)
            idx += len(seg_pts)
            expected.append({"segment": ref.name, "start_idx": start, "end_idx": idx - 1})
        elif mode == "match_detour":
            seg_pts = apply_noise(ref_pts, 1.0, rng)
            seg_pts = insert_detour(seg_pts, len(seg_pts) // 3, 6.0)
            seg_pts = add_lead_in_out(seg_pts)
            start = idx
            points.extend(seg_pts)
            idx += len(seg_pts)
            expected.append({"segment": ref.name, "start_idx": start, "end_idx": idx - 1})
        elif mode == "match_endpoint_shift":
            seg_pts = apply_noise(ref_pts, 1.0, rng)
            seg_pts = shift_endpoints(seg_pts, 3.0)
            seg_pts = add_lead_in_out(seg_pts)
            start = idx
            points.extend(seg_pts)
            idx += len(seg_pts)
            expected.append({"segment": ref.name, "start_idx": start, "end_idx": idx - 1})
        elif mode == "nonmatch_short":
            seg_pts = ref_pts[: max(5, len(ref_pts) // 2)]
            seg_pts = add_lead_in_out(seg_pts)
            points.extend(seg_pts)
            idx += len(seg_pts)
        elif mode == "nonmatch_shift":
            seg_pts = apply_translation(ref_pts, 80.0, 80.0)
            seg_pts = add_lead_in_out(seg_pts)
            points.extend(seg_pts)
            idx += len(seg_pts)
        else:
            continue

        # Insert a small gap between segments
        if points:
            last = points[-1]
            points.append((last[0] + meters_to_lat(3.0), last[1]))
            idx += 1

    case_dir.mkdir(parents=True, exist_ok=True)
    recorded_path = case_dir / "recorded.gpx"
    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    write_recording(recorded_path, points, start_time, interval_s)
    expected_dir.mkdir(parents=True, exist_ok=True)
    expected_path = expected_dir / "expected.json"
    expected_path.write_text(json.dumps({"matches": expected, "mode": mode}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="segments")
    parser.add_argument("--segment-glob", default="*.gpx", help="Glob for reference segment selection")
    parser.add_argument("--output", default="tests/data/synthetic")
    parser.add_argument("--expected", default="tests/expected/synthetic")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--modes",
        default="match_noise,match_linger,match_detour,nonmatch_shift",
        help="Comma-separated list of synthetic modes to generate",
    )
    parser.add_argument(
        "--output-layout",
        choices=["combined", "per-segment"],
        default="combined",
        help="Write one recorded.gpx per mode (combined) or per segment (per-segment)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    segments = Path(args.segments)
    output = Path(args.output)
    expected_root = Path(args.expected)
    output.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.output_layout == "combined":
        for mode in modes:
            build_case(output / mode, expected_root / mode, segments, args.segment_glob, rng, mode)
        return

    refs = sorted(segments.glob(args.segment_glob))
    for mode in modes:
        for ref in refs:
            ref_name = ref.stem
            build_case(
                output / mode / ref_name,
                expected_root / mode / ref_name,
                segments,
                ref.name,
                rng,
                mode,
            )


if __name__ == "__main__":
    main()
