# GPX Segment Timer

The **GPX Segment Timer** is a Python tool for measuring elapsed time on specific segments within a recorded GPX track by comparing them against reference segment files.  
It is designed to handle varying sampling rates, nonuniform point counts, and repeated laps robustly.

---

## Overview

The script performs **multi-stage matching and refinement**:

1. **Coarse Candidate Selection**  
   Based on cumulative distance and bounding boxes around reference segment endpoints.  
   Candidates are generated for each start–end combination within the `--candidate-margin` distance window.

2. **DTW-Based Refinement**  
   Each candidate is uniformly resampled (`--resample-count`) and compared to the reference using **Dynamic Time Warping (DTW)**.  
   Candidates with average DTW distance below `--dtw-threshold` are accepted.

3. **Iterative Grid Search**  
   Candidate start/end indices are refined using a distance-based grid search within `--iterative-window-start` / `--iterative-window-end`,  
   with a penalty (`--penalty-weight`) for endpoint distance deviation.

4. **Endpoint Anchoring (Local Refinement)**  
   A local sliding-window optimization (`--endpoint-window-start`, `--endpoint-window-end`) anchors start and end subsegments  
   to match the reference endpoints more precisely.

5. **Optional Single-Passage Check**  
   Ensures the detected segment only touches the start and end buffers once — useful for tracks with self-intersections.

Segments with endpoint deviations larger than `--bbox-margin` meters are rejected by default, unless `--no-rejection` is used.

---

## Features

- Multi-stage matching with **DTW + endpoint refinement**
- **Resampling-independent** comparison of track shapes
- Adjustable distance, DTW, and endpoint tolerances
- **Candidate dump and GPX export** for debugging
- Output in **stdout**, **CSV**, or **XLSX**

---

## Installation

Requires Python 3 and the following libraries:

    pip install gpxpy fastdtw openpyxl

---

## Usage

Run from the command line:

    ./gpx-segment-timer.py -r <recorded_track.gpx> -f <reference_folder> [OPTIONS]

---

### Input / Output Options

| Option | Description |
|--------|--------------|
| `-r, --recorded` | Path to the recorded GPX file. |
| `-f, --reference-folder` | Folder containing reference segment GPX files. |
| `-o, --output-mode` | Output format: `stdout` (default), `csv`, or `xlsx`. |
| `-O, --output-file` | Output file path (required for CSV/XLSX). |
| `--export-gpx` | Export matched segments as GPX tracks. |
| `--export-gpx-file` | Base name for exported GPX files (default: `matched_segments.gpx`). |
| `--dump-candidates-gpx` | Dump all candidate segments (after bbox filtering) into GPX files using pattern placeholders `{ref}`, `{run}`, `{rs}`, `{re}`, `{n}`. Useful for debugging. |

---

### Matching and Refinement Parameters

| Option | Description |
|--------|--------------|
| `--candidate-margin` | Allowed relative distance variation for candidate search (default: `0.2`). |
| `--dtw-threshold` | Maximum allowed average DTW distance in meters per resampled point (default: `50`). |
| `--resample-count` | Number of points to use for resampling segments (default: `50`). |
| `--min-gap` | Minimum number of recorded points to skip after a match (default: `1`). |
| `--bbox-margin` | Allowed endpoint deviation (in meters) from reference (default: `30`). |
| `--bbox-margin-overall` | Overall bounding box margin (default: same as `--bbox-margin`). |
| `--iterative-window-start` | Search window for start refinement (default: `20`). |
| `--iterative-window-end` | Search window for end refinement (default: `20`). |
| `--penalty-weight` | Weight for Euclidean endpoint penalty during refinement (default: `2.0`). |
| `--anchor-beta1`, `--anchor-beta2` | Weights for DTW cost on start/end subsegments (default: `1.0`). |
| `--endpoint-window-start`, `--endpoint-window-end` | Local window (in points) for endpoint sliding refinement (default: `1000`). |
| `--no-refinement` | Disable all refinement steps (use DTW indices directly). |
| `--allow-length-mismatch` | Allow larger deviation of detected vs. reference length. |
| `--no-rejection` | Do not reject segments beyond `--bbox-margin` (only log warning). |

---

### Advanced Options

| Option | Description |
|--------|--------------|
| `--single-passage` | Enforce single entry into start buffer and single exit from end buffer. |
| `--passage-radius` | Radius (m) for buffers used by single-passage check (default: `30`). |
| `--passage-edge-frac` | Fraction of segment length near edges where passage must occur (default: `0.10`). |
| `-v, --verbose` | Enable detailed logging (INFO). |
| `-d, --debug` | Enable debug-level logs (DEBUG). |

---

### Example Commands

Basic run:

    ./gpx-segment-timer.py -v -f segments -r example.gpx --export-gpx

Resulting in:

| Segment                            | Start Idx | End Idx | Ref Dist (m) | Detected Dist (m) | DTW Avg (m) | Time (s) | Time (H:M:S) | Ref Start              | Ref End                | Start Diff (m) | End Diff (m) |
| :--------------------------------- | --------: | ------: | -----------: | ----------------: | ----------: | -------: | :----------- | :--------------------- | :--------------------- | -------------: | -----------: |
| automotodrom-okruh-offroad-adv.gpx |      1037 |    2721 |      4881.91 |           5384.69 |        9.87 |  1683.00 | 0:28:03      | (49.204458, 16.458708) | (49.204733, 16.458846) |          20.08 |        17.87 |
| automotodrom-okruh-offroad-adv.gpx |      6392 |    7539 |      4881.91 |           4995.58 |        4.90 |  1146.00 | 0:19:06      | (49.204458, 16.458708) | (49.204733, 16.458846) |          22.66 |        16.03 |
| automotodrom-okruh-offroad-adv.gpx |      8330 |    9431 |      4881.91 |           4960.26 |        4.19 |  1100.00 | 0:18:20      | (49.204458, 16.458708) | (49.204733, 16.458846) |          17.95 |         6.97 |
| automotodrom-okruh-offroad-les.gpx |      2191 |    2594 |      1546.04 |           1520.37 |        1.81 |   402.00 | 0:06:42      | (49.204479, 16.459172) | (49.204745, 16.461521) |          27.07 |        24.10 |
| automotodrom-okruh-offroad-les.gpx |      7021 |    7430 |      1546.04 |           1529.81 |        1.77 |   408.00 | 0:06:48      | (49.204479, 16.459172) | (49.204745, 16.461521) |          22.08 |        22.25 |
| automotodrom-okruh-offroad-les.gpx |      8991 |    9334 |      1546.04 |           1519.41 |        2.04 |   342.00 | 0:05:42      | (49.204479, 16.459172) | (49.204745, 16.461521) |          13.03 |        25.69 |
| automotodrom-okruh-offroad.gpx     |      1037 |    2721 |      4940.61 |           5384.69 |        9.53 |  1683.00 | 0:28:03      | (49.204458, 16.458708) | (49.204733, 16.458846) |          20.08 |        17.87 |
| automotodrom-okruh-offroad.gpx     |      6392 |    7539 |      4940.61 |           4995.58 |        4.01 |  1146.00 | 0:19:06      | (49.204458, 16.458708) | (49.204733, 16.458846) |          22.66 |        16.03 |
| automotodrom-okruh-offroad.gpx     |      8330 |    9431 |      4940.61 |           4960.26 |        4.18 |  1100.00 | 0:18:20      | (49.204458, 16.458708) | (49.204733, 16.458846) |          17.95 |         6.97 |

| Column                | Meaning                                                                                                                                                           |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Segment**           | Name of the reference GPX file (segment) that was matched against the recorded track. Each match corresponds to one detected traversal of that reference segment. |
| **Start Idx**         | Index (zero-based) of the first GPX point in the recorded track where the detected segment starts.                                                                |
| **End Idx**           | Index of the last GPX point of the detected segment (exclusive). The segment consists of all points between `Start Idx` and `End Idx – 1`.                        |
| **Ref Dist (m)**      | Total length (in meters) of the reference segment computed from its GPX file.                                                                                     |
| **Detected Dist (m)** | Distance (in meters) of the detected segment in the recorded track, measured between the matched start and end indices.                                           |
| **DTW Avg (m)**       | Average Dynamic Time Warping distance per resampled point between the reference and detected segment; measures geometric similarity (lower = better match).       |
| **Time (s)**          | Elapsed time in seconds between the timestamps of the first and last GPX points of the detected segment.                                                          |
| **Time (H:M:S)**      | Same elapsed time, formatted as hours : minutes : seconds for readability.                                                                                        |
| **Ref Start**         | Latitude/longitude coordinates of the start point of the reference segment.                                                                                       |
| **Ref End**           | Latitude/longitude coordinates of the end point of the reference segment.                                                                                         |
| **Start Diff (m)**    | Distance (in meters) between the reference segment’s start and the detected segment’s start in the recorded track. Indicates positional deviation of the start.   |
| **End Diff (m)**      | Distance (in meters) between the reference segment’s end and the detected segment’s end in the recorded track. Indicates positional deviation of the end.         |


Tighter tolerance with finer curly and possibly self-intersecting segments - with debugging and candidate dump:

    ./gpx-segment-timer.py -v -f segments -r example.gpx \
        --export-gpx \
        --candidate-margin 0.005 \
        --bbox-margin 5 \
        --bbox-margin-overall 100 \
        --resample-count 200 \
        --dump-candidates-gpx "candidates-0.2_{ref}run{run}{rs}-{re}_n{n}.gpx"

./gpx-segment-timer.py -v -f segments -r 20250904_1852.gpx  --export-gpx --candidate-margin 0.005 --bbox-margin 5 --bbox-margin-overall 100 --resample-count 200 --endpoint-window-start 20 --endpoint-window-end 20 --dump-candidates-gpx "candidates-0.005_{ref}run{run}{rs}-{re}_n{n}.gpx"

---

## Tuning & Debugging Tips

- **Adjust Global Matching:**  
  Tune `--candidate-margin` and `--dtw-threshold` to balance sensitivity and false matches.

- **Boundary Refinement:**  
  Use `--iterative-window-*` and `--penalty-weight` to improve alignment precision.

- **Endpoint Anchoring:**  
  Modify `--anchor-beta1` / `--anchor-beta2` for finer boundary corrections.

- **Candidate Inspection:**  
  Use `--dump-candidates-gpx` to export all potential matches for visual verification in a GPX viewer.

- **Boundary Rejection:**  
  Disable endpoint strictness with `--no-rejection` if you want to keep all detections.

---

## License

GNU GPL v3

## Author

Petr Holub

