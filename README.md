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

## Matching Presets (recommended first)

Presets set defaults for strict envelope + DTW sliding window parameters. Any explicit flag overrides the preset value. Default preset is `standard`.

Preset summary:

| Parameter | `tight` | `standard` | `loose` | `loosest` |
|-----------|---------|------------|---------|-----------|
| `strict-envelope-window-m` | 30.0 | 30.0 | 30.0 | 30.0 |
| `strict-envelope-off-pct` | 0.2 | 0.2 | 0.2 | 0.2 |
| `envelope-max-m` | 2.0 | 5.0 | 10.0 | 50.0 |
| `dtw-window-m` | 30.0 | 30.0 | 30.0 | 30.0 |
| `dtw-window-max-avg` | 1.5 | 5.0 | 15.0 | 25.0 |
| `gps-error-m` | 12.0 | 12.0 | 12.0 | 24.0 |

Preset intent:
- **`tight`**: packed tracks with similar shapes (e.g., Jinacovice).
- **`standard`**: general-purpose default.
- **`loose`**: noisy/forested tracks; allows more drift.
- **`loosest`**: extreme GPS drift; consider enabling x-track gates (`--prefilter-xtrack-p95-m`, `--prefilter-xtrack-max-m`, `--final-xtrack-p95-m`, `--final-xtrack-max-m`).
- **`none`**: skip preset defaults; use explicit flags or legacy defaults.

Examples:

```
./gpx-segment-timer.py -v -f segments -r example.gpx --matching-preset tight
./gpx-segment-timer.py -v -f segments -r example.gpx --matching-preset loose
```

---

## Advanced Options

### Input / Output

| Option | Description |
|--------|-------------|
| `-r, --recorded` | Path to recorded GPX file (required). |
| `-f, --reference-folder` | Folder with reference segment GPX files (required). |
| `-o, --output-mode` | Output format: `stdout` (default), `csv`, or `xlsx`. |
| `-O, --output-file` | Output file path for CSV/XLSX (default: unset). |
| `--export-gpx` | Export matched segments as GPX (default: off). |
| `--export-gpx-file` | Base name for exported GPX files (default: `matched_segments.gpx`). |
| `--export-unmatched-gpx` | Export unmatched recorded segments as a single GPX with multiple tracks (default: off). |
| `--export-unmatched-gpx-file` | Output GPX file for unmatched segments (default: `unmatched_segments.gpx`). |
| `--export-unmatched-min-m` | Minimum length (meters) for unmatched segment export; negative disables (default `-1.0`, off). |
| `--export-unmatched-max-m` | Maximum length (meters) for unmatched segment export; negative disables (default `-1.0`, off). |
| `--dump-candidates-gpx` | Dump bbox-filtered candidates with placeholders `{ref}`, `{run}`, `{rs}`, `{re}`, `{n}` (default: off). |
| `--group-by-segment` | Group output by segment name (default: off; default output sorted by start index). |

### Matching / Refinement

#### Methods Summary (defaults + tradeoffs)

| Method | Purpose | Pros | Cons | Default |
|--------|---------|------|------|---------|
| Overall bbox | Fast spatial prune of impossible refs. | Very cheap. | Can miss if bbox margin too small. | On (derived from `--bbox-margin`). |
| Envelope prefilter | Enforce max distance to reference polyline. | Strong drift guard. | Can drop true matches when GPS offset is large. | On if `--envelope-max-m > 0` (preset). |
| Strict envelope window | Sliding window drift guard. | Catches local detours. | Sensitive to window size. | On in presets (`--strict-envelope-window-m`). |
| X-track prefilter (p95/max) | Quick percentile/max gate on candidates. | Removes obvious off-track matches. | Sampling can miss brief detours. | Off unless `--prefilter-xtrack-*-m` set. |
| DTW threshold | Global shape similarity gate. | Ensures overall shape match. | Can hide local deviations on long segments. | On (`--dtw-threshold` default 50). |
| DTW sliding window | Max local DTW guard. | Detects localized detours. | Needs tuning for noisy tracks. | On in presets (`--dtw-window-m`, `--dtw-window-max-avg`). |
| Endpoint refinement | Tightens boundaries near start/finish. | Improves timing precision. | Too wide can snap to wrong crossings. | On (window defaults 10m). |
| Endpoint deviation checks | Rejects matches with start/end too far. | Avoids bad anchors. | Can drop matches with large GPS shift. | On unless `--skip-endpoint-checks`. |
| Length floor (`min-length-ratio`) | Rejects matches that are too short. | Filters partial laps. | Can drop valid linger/loop cases if too strict. | On by default (0.8). |
| Final x-track gates | Last-chance drift guard. | Robust final cleanup. | Can drop matches under heavy GPS drift. | Off unless `--final-xtrack-*-m` set. |
| Single-passage check | Rejects re-entries for non-repeating refs. | Useful for non-looping segments. | Not valid for repeating/self-intersecting refs. | Off unless `--single-passage`. |

#### Candidate Selection + Spatial Prefilters

| Option | Description |
|--------|-------------|
| `--matching-preset` | Preset defaults for strict envelope + DTW window (`standard`, `tight`, `loose`, `loosest`, `none`; default `standard`). |
| `--candidate-margin` | Relative distance tolerance for candidates (default `0.2`). |
| `--candidate-endpoint-margin-m` | Start/end bbox margin for candidate selection (meters); negative uses `--gps-error-m` (default `-1.0`). |
| `--bbox-margin` | Endpoint deviation tolerance in meters (default `30`); overall bbox margin is `--bbox-margin * 3.33`. |
| `--envelope-max-m` | Max distance from reference polyline for envelope prefilter; negative uses `--gps-error-m` (default: preset; standard `5.0`, on when `> 0`). |
| `--envelope-allow-off` | Allowed off-envelope samples per meters: `<points> <meters>` (default `2 100`). |
| `--envelope-sample-max` | Max number of samples per candidate for envelope prefilter; `0` uses all points (default `0`, off only when `--envelope-max-m <= 0`). |
| `--strict-envelope-window-m` | Sliding strict envelope window length in meters; negative disables (default: preset; standard `30.0`, on when `> 0`). |
| `--strict-envelope-off-pct` | Allowed off-envelope percentage within each strict window; `0` disables (default: preset; standard `0.2`, on when `> 0`). |
| `--prefilter-xtrack-p95-m` | Enable x-track p95 prefilter (meters); negative disables (default `-1.0`, off). |
| `--prefilter-xtrack-max-m` | Enable x-track max prefilter (meters); negative disables (default `-1.0`, off). |
| `--prefilter-xtrack-samples` | Sample count for x-track prefilter (default `80`; applies to p95/max). |

#### DTW + Shape Matching

| Option | Description |
|--------|-------------|
| `--dtw-threshold` | Max avg DTW cost (default `50`). |
| `--dtw-window-m` | Sliding DTW window length in meters; negative disables (default: preset; standard `30.0`, on when `> 0`). |
| `--dtw-window-max-avg` | Reject candidates whose max avg DTW within the window exceeds this; negative disables (default: preset; standard `5.0`, on when `> 0`). |
| `--dtw-penalty` | DTW penalty: `linear`, `quadratic`, `huber` (default `linear`). |
| `--dtw-penalty-scale-m` | Scale for quadratic DTW penalty (meters, default `10.0`). |
| `--dtw-penalty-huber-k` | Huber k parameter for DTW penalty (meters, default `5.0`). |
| `--shape-mode` | `step_vectors`, `heading`, `centered`, or `auto` (default `step_vectors`). |
| `--target-spacing-m` | Target meters between resampled points (default `8.0`; `<=0` disables). |
| `--resample-max` | Max resample count when using target spacing (default `400`). |
| `--resample-count` | Fixed resample count when target spacing is not used (default `200`, ignored when `--target-spacing-m > 0`). |

#### Refinement + Final Acceptance

| Option | Description |
|--------|-------------|
| `--allow-length-mismatch` | Allow candidates outside length window (default: off). |
| `--min-length-ratio` | Reject matches shorter than this fraction of reference length; `0` disables (default `0.8`, on). |
| `--endpoint-window-start` | Start endpoint sliding window in meters (default `10.0`, converted to points using cumulative distance). |
| `--endpoint-window-end` | End endpoint sliding window in meters (default `10.0`, converted to points using cumulative distance). |
| `--endpoint-spatial-weight` | Spatial weight in endpoint refinement (default `0.25`). |
| `--iterative-window-start` | Start refinement window (default `20`). |
| `--iterative-window-end` | End refinement window (default `20`). |
| `--penalty-weight` | Endpoint distance penalty during refinement (default `2.0`). |
| `--anchor-beta1` | Start subsegment weight (default `1.0`). |
| `--anchor-beta2` | End subsegment weight (default `1.0`). |
| `--min-gap` | Minimum points to skip after a match (default `1`). |
| `--final-xtrack-p95-m` | Reject final matches if x-track p95 exceeds this (meters); negative disables (default `-1.0`, off). |
| `--final-xtrack-max-m` | Reject final matches if x-track max exceeds this (meters); negative disables (default `-1.0`, off). |
| `--final-xtrack-samples` | Sample count for final x-track stats; `0` uses all points (default `0`). |
| `--gps-error-m` | GPS error estimate in meters (default `12.0`; preset `loosest` sets `24.0` unless overridden). |
| `--no-refinement` | Disable refinement steps (default: off). |
| `--skip-endpoint-checks` | Keep matches even if endpoint diffs exceed `--bbox-margin` (default: off). |

### Start/Finish Crossing Logic

| Option | Description |
|--------|-------------|
| `--line-length-m` | Start/finish line total length in meters (default `8.0`). |
| `--crossing-endpoint-weight` | Endpoint proximity weight when selecting crossings (default `1.0`). |
| `--crossing-shape-weight` | Shape weight when selecting crossings (default `1.0`). |
| `--crossing-shape-window-frac` | Local shape window fraction of resample count (default `0.2`). |
| `--crossing-shape-window-min` | Minimum window size for local crossing shape matching (default `3`). |
| `--crossing-length-weight` | Length weight for crossing selection (negative = auto, default `-1.0`). |
| `--crossing-window-max` | Max crossing search expansion window (default `200`). |
| `--crossing-edge-window-s` | Start/end crossing search window in seconds (default `1.0`, uses median sampling rate). |
| `--crossing-expand-mode` | Crossing search expansion mode when no crossings are found (`fixed` or `ratio`, default `ratio`). |
| `--crossing-expand-k` | Scale factor for ratio-based crossing expansion (default `1.0`). |

### Optional Single-Passage Check

| Option | Description |
|--------|-------------|
| `--single-passage` | Enforce single pass through start/end buffers (default: off). |
| `--passage-radius` | Buffer radius in meters (default `30`). |
| `--passage-edge-frac` | Fraction of segment length for passage checks (default `0.10`). |

### Logging

| Option | Description |
|--------|-------------|
| `-v, --verbose` | INFO logs (default: off). |
| `-d, --debug` | DEBUG logs (default: off). |

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

## Testing

### Generate synthetic data

Default layout aggregates all segments per mode:

```
python3 tests/tools/generate_synthetic_data.py \
  --segments segments \
  --segment-glob "*.gpx" \
  --output tests/data/synthetic \
  --expected tests/expected/synthetic
```

Per-segment layout writes one recorded track per reference segment (Option B):

```
python3 tests/tools/generate_synthetic_data.py \
  --segments segments \
  --segment-glob "*.gpx" \
  --output tests/data/synthetic \
  --expected tests/expected/synthetic \
  --output-layout per-segment
```

## Parameter Tuning Guidance

### Start/Finish Lines
- **`--line-length-m`** controls how wide the finite line segment is.
- Default 8m matches typical GPS +-4m accuracy.
- Use smaller values for kink-heavy segments that intersect the line multiple times.

### Strict Envelope + DTW Window Presets
See the **Matching Presets** section for defaults and guidance.

### Strict Envelope
- **`--strict-envelope-window-m`** controls the sliding window length used to enforce local envelope adherence.
- **`--strict-envelope-off-pct`** caps how many points in each window can be outside the envelope.
- Combine with **`--envelope-max-m`** to set the envelope width (meters from the reference polyline).
Defaults are enabled via presets; set negative values or `0` to disable.

### DTW Sliding Window
- **`--dtw-window-m`** and **`--dtw-window-max-avg`** guard against long segments that hide local deviations.
- If you see long candidates with local detours that still pass average DTW, tighten `--dtw-window-max-avg`.
- Use larger values for noisy recordings where line crossings are offset.
Defaults are enabled via presets; set negative values to disable.

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
- **`--candidate-endpoint-margin-m`** tightens start/end candidate bboxes independent of endpoint rejection.
- **`--envelope-max-m`** and **`--envelope-allow-off`** prune candidates that drift off the reference polyline.
- **`--prefilter-xtrack-p95-m`** and **`--prefilter-xtrack-max-m`** add optional percentile/max cross-track gates.

### Refinement
- **`--endpoint-window-start/end`** and **`--endpoint-spatial-weight`** can tighten endpoint placement.
- **`--iterative-window-start/end`** and **`--penalty-weight`** adjust the iterative search around boundaries.
- **`--final-xtrack-p95-m`**, **`--final-xtrack-max-m`**, and **`--final-xtrack-samples`** add optional final x-track gates.

### Debugging
- Use **`--dump-candidates-gpx`** and **`--export-gpx`** to visually inspect candidate windows, line crossings, and interpolation points.
- Use **`tools/debug_match_metrics.py`** to compute matching metrics for an exported match GPX against its embedded reference, and optionally compare against other reference segments.

```
python3 tools/debug_match_metrics.py tuning-needed/example_match.gpx \
  --ref-segment segments/Jinacovice-vnejsi-dlouhy.gpx \
  --shape-mode step_vectors --target-spacing-m 8 --line-length-m 8
```

---

## License

GNU GPL v3

## Author

Petr Holub
