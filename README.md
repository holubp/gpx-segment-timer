# GPX Segment Timer

GPX Segment Timer is a Python CLI that measures elapsed time on reference GPX segments inside a recorded GPX track. It is built for uneven sampling rates, repeated laps, self-intersections, overlapping segments, and noisy GPS.

---

## How It Works

The matcher uses a multi-stage pipeline that is translation-tolerant but spatially bounded:

1) **Coarse candidate selection**
   - Finds recorded points near the reference start/end using endpoint and overall bounding boxes.
   - Generates candidate windows using cumulative distance with a `--candidate-margin` tolerance.
   - Optional envelope/polylinedistance prefilters prune candidates that drift beyond GPS error bounds.

2) **Shape matching (DTW)**
   - Compares candidate windows to the reference using DTW on translation-invariant shape sequences.
   - Shape modes: step vectors, heading, or centered coordinates.

3) **Boundary refinement**
   - DTW warping path anchoring, iterative grid search, and endpoint refinement.

4) **Start/finish line crossings**
   - Start/finish lines are perpendicular to the first/last segment vectors.
   - Lines are finite segments (default total length 8m) and can be tuned.
   - Crossing points are interpolated between bracket points.

5) **Crossing disambiguation by shape**
   - When lines intersect multiple times (tight kinks near the start/finish), local and full-segment shape matching disambiguate the correct crossings.

6) **Output + GPX export**
   - Emits results to stdout/CSV/XLSX.
   - Optional GPX export includes matched window, reference segment, start/finish lines, and crossing points (with interpolation annotations).

---

## Installation

Requires Python 3 and:

```
pip install gpxpy fastdtw openpyxl
```

---

## Usage

```
./gpx-segment-timer.py -r <recorded_track.gpx> -f <reference_folder> [OPTIONS]
```

---

## Command Line Options

### Input / Output

| Option | Description |
|--------|-------------|
| `-r, --recorded` | Path to recorded GPX file. |
| `-f, --reference-folder` | Folder with reference segment GPX files. |
| `-o, --output-mode` | `stdout` (default), `csv`, or `xlsx`. |
| `-O, --output-file` | Output file path for CSV/XLSX. |
| `--export-gpx` | Export matched segments as GPX. |
| `--export-gpx-file` | Base name for exported GPX files. |
| `--dump-candidates-gpx` | Dump bbox-filtered candidates with placeholders `{ref}`, `{run}`, `{rs}`, `{re}`, `{n}`. |

### Matching / Refinement

| Option | Description |
|--------|-------------|
| `--candidate-margin` | Relative distance tolerance for candidates (default `0.2`). |
| `--start-end-margin-m` | Start/end bbox margin for candidate selection (meters); negative uses `--gps-error-m`. |
| `--envelope-max-m` | Max distance from reference polyline for envelope prefilter; negative uses `--gps-error-m`. |
| `--envelope-allow-off` | Allowed off-envelope samples per meters: `<points> <meters>` (default `2 100`). |
| `--envelope-sample-max` | Max number of samples per candidate for envelope prefilter; `0` uses all points. |
| `--prefilter-xtrack-p95-m` | Enable x-track p95 prefilter (meters); negative disables. |
| `--prefilter-xtrack-samples` | Sample count for x-track p95 prefilter. |
| `--allow-length-mismatch` | Allow candidates outside length window. |
| `--dtw-threshold` | Max avg DTW cost (default `50`). |
| `--dtw-penalty` | DTW penalty: `linear`, `quadratic`, `huber`. |
| `--dtw-penalty-scale-m` | Scale for quadratic DTW penalty (meters). |
| `--dtw-penalty-huber-k` | Huber k parameter for DTW penalty (meters). |
| `--shape-mode` | `step_vectors`, `heading`, `centered`, or `auto`. |
| `--gps-error-m` | GPS error estimate in meters (default `12`). |
| `--target-spacing-m` | Target meters between resampled points (default `8`). |
| `--resample-max` | Max resample count when using target spacing (default `400`). |
| `--resample-count` | Fixed resample count when target spacing is not used (default `200`). |
| `--min-gap` | Minimum points to skip after a match (default `1`). |
| `--bbox-margin` | Endpoint deviation tolerance in meters (default `30`). |
| `--bbox-margin-overall` | Overall bbox margin in meters (default `100`). |
| `--refine-window` | Legacy window (kept for compatibility). |
| `--iterative-window-start` | Start refinement window (default `20`). |
| `--iterative-window-end` | End refinement window (default `20`). |
| `--penalty-weight` | Endpoint distance penalty during refinement (default `2.0`). |
| `--anchor-beta1` | Start subsegment weight (default `1.0`). |
| `--anchor-beta2` | End subsegment weight (default `1.0`). |
| `--endpoint-window-start` | Start endpoint sliding window in points (default `1000`). |
| `--endpoint-window-end` | End endpoint sliding window in points (default `1000`). |
| `--endpoint-spatial-weight` | Spatial weight in endpoint refinement (default `0.25`). |
| `--no-refinement` | Disable refinement steps. |
| `--no-rejection` | Keep matches even if endpoint diffs exceed `--bbox-margin`. |
| `--skip-endpoint-checks` | Skip endpoint rejection checks. |

### Start/Finish Crossing Logic

| Option | Description |
|--------|-------------|
| `--line-length-m` | Start/finish line total length in meters (default `8.0`). |
| `--crossing-endpoint-weight` | Endpoint proximity weight when selecting crossings. |
| `--crossing-shape-weight` | Shape weight when selecting crossings. |
| `--crossing-shape-window-frac` | Local shape window fraction of resample count (default `0.2`). |
| `--crossing-shape-window-min` | Minimum window size for local crossing shape matching (default `3`). |
| `--crossing-length-weight` | Length weight for crossing selection (negative = auto). |
| `--crossing-window-max` | Max crossing search expansion window (default `200`). |

### Optional Single-Passage Check

| Option | Description |
|--------|-------------|
| `--single-passage` | Enforce single pass through start/end buffers. |
| `--passage-radius` | Buffer radius in meters (default `30`). |
| `--passage-edge-frac` | Fraction of segment length for passage checks (default `0.10`). |

### Logging

| Option | Description |
|--------|-------------|
| `-v, --verbose` | INFO logs. |
| `-d, --debug` | DEBUG logs. |

---

## Examples

### Basic matching

```
./gpx-segment-timer.py -v -f segments -r example.gpx
```

### Export GPX with start/finish lines and interpolated crossings

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --export-gpx --export-gpx-file matched_segments.gpx
```

### Tight line width for kink-heavy finishes

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --line-length-m 6
```

### Wider line width for noisy GPS

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --gps-error-m 12 --line-length-m 12
```

### Shape-sensitive crossing disambiguation

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --crossing-shape-weight 2.0 --crossing-shape-window-frac 0.3
```

### Higher shape fidelity for long/complex segments

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --target-spacing-m 5 --resample-max 600
```

### Candidate dump for debugging

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --dump-candidates-gpx "candidates_{ref}_run{run}_{rs}-{re}_n{n}.gpx"
```

### Single-passage validation for non-repeating segments

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --single-passage --passage-radius 25 --passage-edge-frac 0.1
```

---

## Parameter Tuning Guidance

### Start/Finish Lines
- **`--line-length-m`** controls how wide the finite line segment is.
- Default 8m matches typical GPS +-4m accuracy.
- Use smaller values for kink-heavy segments that intersect the line multiple times.
- Use larger values for noisy recordings where line crossings are offset.

### Shape Matching
- **`--shape-mode`**: `step_vectors` is robust for general use, `heading` can help for curvature emphasis, `centered` is useful for centroid-stable tracks.
- **`--target-spacing-m`** and **`--resample-max`** control resampling fidelity; reduce spacing for complex geometry.
- **`--dtw-threshold`** governs strictness; lower values are stricter.
- **`--dtw-penalty`** can be set to `quadratic` or `huber` to penalize large deviations more aggressively.

### Crossing Disambiguation
- **`--crossing-shape-weight`** increases how strongly shape decides which crossing to choose.
- **`--crossing-shape-window-frac`** controls local shape context near the line; increase if a tight kink is near the finish.

### Candidate Selection
- **`--candidate-margin`** expands or tightens candidate distance windows.
- **`--bbox-margin`** controls endpoint tolerance; tighter values reject more noise but can miss shifted tracks.
 - **`--start-end-margin-m`** tightens start/end candidate bboxes independent of endpoint rejection.
 - **`--envelope-max-m`** and **`--envelope-allow-off-per-100m`** prune candidates that drift off the reference polyline.
 - **`--prefilter-xtrack-p95-m`** adds an optional percentile-based cross-track gate.

### Refinement
- **`--endpoint-window-start/end`** and **`--endpoint-spatial-weight`** can tighten endpoint placement.
- **`--iterative-window-start/end`** and **`--penalty-weight`** adjust the iterative search around boundaries.

### Debugging
- Use **`--dump-candidates-gpx`** and **`--export-gpx`** to visually inspect candidate windows, line crossings, and interpolation points.

---

## License

GNU GPL v3

## Author

Petr Holub
