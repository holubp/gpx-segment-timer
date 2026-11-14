# GPX Segment Timer

GPX Segment Timer is a Python CLI that measures elapsed time on reference GPX segments inside a recorded GPX track. It is built for uneven sampling rates, repeated laps, self-intersections, overlapping segments, and noisy GPS.

---

## How It Works

The matcher uses a multi-stage pipeline that is translation-tolerant but spatially bounded:

1) **Reference mode selection (auto per GPX)**
   - `shape` mode when no start/finish markers are present.
   - `point`, `line`, or `hybrid` mode when markers are present.
   - Marker labels are read from `name`, then `desc/description`, then `cmt/comment`.

2) **Coarse candidate selection (shape mode)**
   - Finds recorded points near the reference start/end using endpoint and overall bounding boxes.
   - Generates candidate windows using cumulative distance with a `--candidate-margin` tolerance.
   - Optional envelope/polylinedistance prefilters prune candidates that drift beyond GPS error bounds.

3) **Shape matching (DTW, shape mode)**
   - Compares candidate windows to the reference using DTW on translation-invariant shape sequences.
   - Shape modes: step vectors, heading, or centered coordinates.

4) **Boundary refinement (shape mode)**
   - DTW warping path anchoring, iterative grid search, and endpoint refinement.

5) **Start/finish crossings**
   - Start/finish lines are perpendicular to the first/last segment vectors.
   - Lines are finite segments (default total length 8m) and can be tuned.
   - Crossing points are interpolated between bracket points.
   - In marker modes:
     - Point mode starts on leaving start circle and ends on entering finish circle.
     - Line mode starts/ends on explicit start/finish line crossings.
     - Hybrid mode enforces marker point + line constraints.

6) **Crossing disambiguation by shape (shape mode)**
   - When lines intersect multiple times (tight kinks near the start/finish), local and full-segment shape matching disambiguate the correct crossings.

7) **Output + GPX export**
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

Presets set defaults for strict envelope, DTW sliding windows, crossing-search limits, and iterative refinement radii. Any explicit flag overrides the preset value. Default preset is `normal`. The old name `standard` remains available as an equivalent legacy alias.

Preset summary:

| Parameter | `tight` | `normal` | `normal-widetrack` | `loose` | `loosest` |
|-----------|---------|------------|-------------------|---------|-----------|
| `strict-envelope-window-m` | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 |
| `strict-envelope-off-pct` | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| `envelope-max-m` | 2.0 | 5.0 | 10.0 | 10.0 | 50.0 |
| `dtw-window-m` | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 |
| `dtw-window-max-avg` | 1.5 | 5.0 | 5.0 | 15.0 | 25.0 |
| `gps-error-m` | 12.0 | 12.0 | 12.0 | 12.0 | 24.0 |
| `crossing-window-max` (meters) | 20.0 | 20.0 | 20.0 | 100.0 | 200.0 |
| `iterative-window-start/end` (meters) | 50.0 | 50.0 | 50.0 | 200.0 | 200.0 |

Preset intent:
- **`tight`**: packed tracks with similar shapes (e.g., Jinacovice).
- **`normal`**: general-purpose default (`standard` is a legacy alias).
- **`normal-widetrack`**: normal matching with a wider corridor (10 m on each side), useful for motocross tracks. All other settings are identical to `normal`.
- **`loose`**: noisy/forested tracks; allows more drift.
- **`loosest`**: extreme GPS drift; consider enabling x-track gates (`--prefilter-xtrack-p95-m`, `--prefilter-xtrack-max-m`, `--final-xtrack-p95-m`, `--final-xtrack-max-m`).
- **`none`**: skip preset defaults; use explicit flags or legacy defaults. The crossing-search cap remains 20 m and iterative windows remain 50 m unless overridden.

Examples:

```
./gpx-segment-timer.py -v -f segments -r example.gpx --matching-preset tight
./gpx-segment-timer.py -v -f segments -r example.gpx --matching-preset loose
./gpx-segment-timer.py -v -f segments -r example.gpx --matching-preset normal-widetrack
```

Profile settings are defined in `apply_matching_preset()` in `gpx-segment-timer.py`.
`normal-widetrack` is equivalent to `--matching-preset normal --envelope-max-m 10`.

---

## Reference Marker Modes

Marker modes are selected automatically per reference GPX file and bypass DTW shape discovery for that file.

- Point mode:
  - Requires marker points `start` and `finish`.
  - Uses virtual circles around markers (default 10m radius).
  - Match starts when recorded track leaves start circle.
  - Match ends when recorded track enters finish circle.
- Line mode:
  - Requires `start` and `finish` lines (2 points each) as named tracks/routes.
  - Match starts/ends on crossing these explicit lines.
- Hybrid mode:
  - Enabled when both point and line markers are present.
  - Applies point constraints and line constraints together.
- Trap points:
  - Any marker point named `trap*` creates additional consecutive subsegments:
    `start->trap1`, `trap1->trap2`, ..., `trapN->finish`.
  - Each complete trap visit uses the midpoint of its interpolated circle entry and exit, with the timestamp halfway between their times. Adjacent subsegments share this event; there is no timing gap while inside the trap circle.
  - This works with many, one, or no recorded points inside the circle. Tangent contact also counts for traps: entry and exit coincide. A visit missing its entry or exit does not produce a trap midpoint.

For a curved passage, the trap coordinate is the entry/exit **chord midpoint**, which can lie off the recorded path. Its timing is the average of the two boundary times, not the time of the sample nearest the center. Distances follow the original recorded path at that time. If timestamps are unavailable, the mapping uses the visit's half-distance position and does not invent a timestamp. If either timed end of a trap subsegment is unavailable, its duration stays unavailable rather than falling back to surrounding raw sample times. Start and finish circles still use exit and entry respectively; their tangencies do not count as crossings.

Trap GPX exports show entry, midpoint and exit in the crossing track, plus separate entry/exit tracks retaining the original bracketing samples. The table's trap indices bracket its midpoint time where timestamps are usable, or its half-distance mapping otherwise. Trap deviation is measured from the midpoint to the trap center; ordinary start/finish deviation remains relative to the circle boundary.

Marker labels can be defined in GPX `name`, `desc/description`, or `cmt/comment`.
This supports marker-only references like `segments/semmering.gpx` with waypoint descriptions.

Circle crossings are interpolated on the edges between original recorded points. A passage can therefore be detected even when both surrounding samples lie outside the circle. For ordinary start/finish circles, tangent contact alone is not an entry; re-entry is checked in event order, including multiple events within one sample interval. Timing remains **start-circle exit to finish-circle entry**. It does not use the point closest to the circle center. The exported recording retains original samples; sparse passages may include bracketing points outside the timed interval. Point-mode detected distance is measured between the interpolated boundaries. Its minimum-length check accounts for the start/finish circle radii omitted by exit-to-entry timing, rather than depending on which samples bracket those events.

---

## Advanced Options

### Input / Output

| Option | Description |
|--------|-------------|
| `-r, --recorded` | Path to recorded GPX file (required). |
| `-f, --reference-folder` | Folder with reference segment GPX files (required). |
| `-o, --output-mode` | Output format: `stdout` (default), `csv`, or `xlsx`; stdout prints a compact table unless `-v`/`-d` is enabled. |
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
| Marker reference mode | Marker-driven matching for point/line/hybrid refs. | No DTW needed for markerized refs; good for start/finish-only definitions. | Depends on correct marker labels and geometry. | Auto on when a reference contains start/finish markers. |
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
| `--matching-preset` | Preset defaults for strict envelope, DTW window and crossing cap (`normal`, `normal-widetrack`, `tight`, `loose`, `loosest`, `none`; default `normal`). `standard` remains an equivalent legacy alias for `normal`. |
| `--candidate-margin` | Relative distance tolerance for candidates (default `0.2`). |
| `--candidate-endpoint-margin-m` | Start/end bbox margin for candidate selection (meters); negative uses `--gps-error-m` (default `-1.0`). |
| `--bbox-margin` | Endpoint deviation tolerance in meters (default `30`); overall bbox margin is `--bbox-margin * 3.33`. |
| `--envelope-max-m` | Max distance from reference polyline for envelope prefilter; negative uses `--gps-error-m` (default: preset; normal `5.0`, on when `> 0`). |
| `--envelope-allow-off` | Allowed off-envelope samples per meters: `<points> <meters>` (default `2 100`). |
| `--envelope-sample-max` | Max number of samples per candidate for envelope prefilter; `0` uses all points (default `0`, off only when `--envelope-max-m <= 0`). |
| `--strict-envelope-window-m` | Sliding strict envelope window length in meters; negative disables (default: preset; normal `30.0`, on when `> 0`). |
| `--strict-envelope-off-pct` | Allowed off-envelope percentage within each strict window; `0` disables (default: preset; normal `0.2`, on when `> 0`). |
| `--prefilter-xtrack-p95-m` | Enable x-track p95 prefilter (meters); negative disables (default `-1.0`, off). |
| `--prefilter-xtrack-max-m` | Enable x-track max prefilter (meters); negative disables (default `-1.0`, off). |
| `--prefilter-xtrack-samples` | Sample count for x-track prefilter (default `80`; applies to p95/max). |

#### Reference Marker Modes

| Option | Description |
|--------|-------------|
| `--marker-circle-m` | Default virtual circle radius (meters) for point mode markers (default `10.0`). |
| `--marker-start-circle-m` | Start circle radius (meters); negative uses `--marker-circle-m` (default `-1.0`, fallback on). |
| `--marker-finish-circle-m` | Finish circle radius (meters); negative uses `--marker-circle-m` (default `-1.0`, fallback on). |
| `--marker-trap-circle-m` | Trap circle radius (meters); negative uses `--marker-circle-m` (default `-1.0`, fallback on). |

Markerized references are auto-detected and can use:
- Waypoints or point markers named `start`, `finish`, `trap*`.
- Named tracks/routes `start` and `finish` with exactly 2 points each (line mode).

#### DTW + Shape Matching

| Option | Description |
|--------|-------------|
| `--dtw-threshold` | Max avg DTW cost (default `50`). |
| `--dtw-window-m` | Sliding DTW window length in meters; negative disables (default: preset; normal `30.0`, on when `> 0`). |
| `--dtw-window-max-avg` | Reject candidates whose max avg DTW within the window exceeds this; negative disables (default: preset; normal `5.0`, on when `> 0`). |
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
| `--endpoint-window-start` | Start endpoint search radius in meters (default `10.0`, on); each side is bounded independently. `0` restricts it to the same along-track position. |
| `--endpoint-window-end` | Finish endpoint search radius in meters (default `10.0`, on); independent bounds retain all original samples within the radius. |
| `--endpoint-spatial-weight` | Spatial weight in endpoint refinement (default `0.25`). |
| `--iterative-window-start` | Start refinement search radius in meters (on; default `50.0`, or `200.0` for loose/loosest); `0` restricts it to the same along-track position. |
| `--iterative-window-end` | Finish refinement search radius in meters (on; default `50.0`, or `200.0` for loose/loosest); `0` restricts it to the same along-track position. |
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

Original coordinates, timestamps and indices are retained throughout matching. Endpoint and crossing shape comparisons use separate distance-based copies with cached mappings to the original samples. Local endpoint context spans 10% of reference distance; crossing context uses `--crossing-shape-window-frac`. Increasing the recording frequency does not shorten these physical contexts. This does not change the whole-segment DTW scoring or remove GPS jitter from measured lengths.

**Migration:** `--iterative-window-start/end` now accept meters, including decimal values, instead of raw point counts. A previous value of `20` now means 20 m. Do not convert old values using Hz alone: riding speed also matters. The defaults were checked against the validated automotodrom and Jinacovice recordings.

### Start/Finish Crossing Logic

| Option | Description |
|--------|-------------|
| `--line-length-m` | Start/finish line total length in meters (default `8.0`). |
| `--crossing-endpoint-weight` | Endpoint proximity weight when selecting crossings (default `1.0`). |
| `--crossing-shape-weight` | Shape weight when selecting crossings (default `1.0`). |
| `--crossing-shape-window-frac` | Local shape context as a fraction of reference distance (default `0.2`, on; valid range `(0, 1]`); also sets the comparison sample budget. |
| `--crossing-shape-window-min` | Minimum number of comparison samples for crossing shape matching (default `3`); not a count of original recorded points. |
| `--crossing-length-weight` | Length weight for crossing selection (negative = auto, default `-1.0`). |
| `--crossing-window-max` | Maximum along-track search distance on either side of each start/finish boundary, in meters (on by default: `20.0` for tight/normal/normal-widetrack/none, `100.0` for loose, `200.0` for loosest). Accepts decimals; `0` allows only a crossing exactly at the boundary. |
| `--crossing-edge-window-s` | Initial crossing search radius in seconds (default `1.0`, on), measured using actual timestamps and interpolated crossing times; `<=0` disables the time limit while retaining the meter cap. |
| `--crossing-expand-mode` | Crossing search expansion mode when no crossings are found (`fixed` or `ratio`, default `ratio`). |
| `--crossing-expand-k` | Scale factor for ratio-based crossing expansion (default `1.0`). |

`--crossing-window-max` now uses **meters, not recorded points**. Existing commands
must be reviewed: `--crossing-window-max 20` now means 20 m each side, not 20
samples. There is no exact conversion from the previous 200-point default using
sampling frequency alone: riding speed and uneven sampling also affect distance.
The new profile limits are independent of recording frequency. Loose uses 100 m
to retain validated finish crossings on the public automotodrom example; 20 m
was too restrictive there. These are calibrated distance limits, not exact
conversions of the old point count.

The cap measures distance travelled along the recording, not straight-line distance
from the start/finish coordinates. It bounds initial searches, expanded searches,
and fallback crossing-pair searches. A sample just beyond the limit may bracket an
interpolated crossing, but the crossing itself must be inside the limit. Increasing
`--endpoint-window-start/end` does not override this cap. The cap applies to shape
matching; explicit marker modes search their named boundaries directly.

The initial seconds window and the maximum distance cap are independent. Initial searches check actual interpolated times rather than converting seconds through a median recording rate. Missing timestamps or reversed bracket times fall back to the meter cap for the affected crossing (diagnostic details at debug level). If the initial search finds no crossing, `fixed` expansion uses the corresponding endpoint radius in meters; `ratio` expansion uses reference length and detected/reference length ratio. Both remain bounded by `--crossing-window-max`.

If refinement loses a start or finish crossing, the matcher can restore the original DTW candidate when it adds a valid crossing without losing any already observed end, within the same configured caps and endpoint margins. The recovered interval must remain forward in recording order; an unobserved end retains its original sample boundary. It does not widen the search limits to recover a missing crossing.

The debug metrics tool uses the same time/distance searches and physical shape context. Its verbose metrics identify seconds and meters explicitly; legacy point-count window values from old logs are not used as physical limits.

The debug tool uses the same meter cap and reports `crossing_window_max_m`.
When importing older logs, it ignores their point-based crossing cap and warns;
set the new meter limit explicitly when reproducing an older analysis.

### Optional Single-Passage Check

| Option | Description |
|--------|-------------|
| `--single-passage` | Enforce single pass through start/end buffers (default: off). |
| `--passage-radius` | Buffer radius in meters (default `30`). |
| `--passage-edge-frac` | Fraction of segment length for passage checks (default `0.10`). |

### Logging

| Option | Description |
|--------|-------------|
| `-v, --verbose` | INFO logs and detailed stdout table (default: off). |
| `-d, --debug` | DEBUG logs and detailed stdout table (default: off). |

---

## Examples

### Basic matching (shape mode references in `segments/`)

Included shape-mode references include:
- `segments/automotodrom-okruh-offroad-les.gpx`
- `segments/automotodrom-okruh-offroad-adv.gpx`
- `segments/automotodrom-okruh-offroad.gpx`
- `segments/Jinacovice-vnejsi-dlouhy.gpx`
- `segments/Jinacovice-vnejsi-kratky.gpx`
- `segments/Jinacovice-vnejsi-stredni.gpx`
- `segments/Jinacovice-vnitrni-kratky.gpx`
- `segments/Jinacovice-vnitrni-stredni.gpx`
- `segments/Jinacovice-vnitrni-kratky-2x.gpx`

```
./gpx-segment-timer.py -v -f segments -r example.gpx
```

### Export GPX with start/finish lines and interpolated crossings

```
./gpx-segment-timer.py -v -f segments -r example.gpx \
  --export-gpx --export-gpx-file matched_segments.gpx
```

### Marker point mode (`segments/semmering.gpx`, including `desc`/`cmt` labels)

`segments/semmering.gpx` is a marker-only reference (no shape track points). It is matched in point mode.

```
./gpx-segment-timer.py -v -f segments -r 21558926845.gpx \
  --marker-circle-m 10
```

### Marker point mode with per-role circle radii

```
./gpx-segment-timer.py -v -f segments -r 21558926845.gpx \
  --marker-start-circle-m 8 --marker-finish-circle-m 12 --marker-trap-circle-m 10
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
- `--crossing-window-max` limits how far along the recording boundary searches may travel (normal default 20 m each side; loose 100 m; loosest 200 m). It is independent of the line's physical width and the spatial envelope.

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
