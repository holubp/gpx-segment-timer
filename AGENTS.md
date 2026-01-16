# Repository Guidelines

## Project Structure & Module Organization
- `gpx-segment-timer.py` is the primary CLI script.
- `segments/` holds reference GPX segment files used for matching.
- `example.gpx` and `20250904_1852.gpx` are sample recorded tracks.
  - `20250904_1852.gpx` or similarly named files should not be committed to the repo, even if locally present.
- Additional variants like `gpx-segment-timer_*` are experimental snapshots.

## Build, Test, and Development Commands
- `python3 gpx-segment-timer.py -r example.gpx -f segments` runs a basic match against bundled data.
- `python3 gpx-segment-timer.py -r <track.gpx> -f segments --export-gpx` exports matched segments for inspection.
- `python3 tools/debug_match_metrics.py <match.gpx> --ref-segment segments/<segment.gpx>` inspects metrics for an exported match GPX.
- `python3 tests/tools/generate_synthetic_data.py` regenerates synthetic GPX fixtures and expected results.
- `python3 tests/run_tests.py` runs the regression test manifest.
- `pip install gpxpy fastdtw openpyxl` installs required dependencies for GPX parsing, DTW, and XLSX output.

## Coding Style & Naming Conventions
- Follow Python 3, 4-space indentation, and `snake_case` for functions and variables.
- Keep CLI flags consistent with existing long/short option patterns in `gpx-segment-timer.py`.
- Prefer standard library modules first; external deps should be documented in `README.md` if added.

## Design
- Provide a Python CLI that measures elapsed times for reference GPX segments against a recorded track.
- Functionality requirements:
  - Must work with different sampling frequencies across recorded GPX (1Hz, 5Hz, 10Hz, 20Hz, dynamic frequency) and with a different number of points than the reference segment.
  - Must work in presence of:
    - Self-intersecting reference segments.
    - Segments with multiple repetitions of the same track parts (e.g., 3x circuit as reference should only match when the recording passes it 3x).
    - Segments with very close start and end points (e.g., circuit races).
    - Overlapping segments (one reference segment can fully or partially overlap another; both must match if present).
  - Must work on recorded tracks where:
    - The same track points may be reused in different segments (matching one does not preclude matching another).
    - One reference segment can match multiple times in the same recording.
  - Matching must be based on:
    - Start line anchored at the first point, perpendicular to the vector between the first two reference points.
    - Finish line anchored at the last point, perpendicular to the vector between the last two reference points.
    - Start/finish lines are finite segments; default total line length is 8m (tunable via CLI) to reflect typical GPS accuracy.
    - Shape of the reference segment (including repeated/self-intersecting cases), evaluated in a translation-invariant representation but bounded by spatial proximity limits.
    - Length of the reference segment.
    - Start/finish crossing detection with interpolation between the two recorded points that bracket the line; output must identify interpolated points and bracketing indices.
    - Start/finish matching that remains correct when the rider lingers around the start/finish area before/after the segment; use shape + line crossings, not local random walk.
    - Robustness to localized chaos (e.g., a rider falls or stops mid-segment, producing a dense point cloud); boundaries should still be determined by shape + line crossings with interpolation.
    - Start/finish crossing disambiguation using local shape matching near the crossings so that lines intersecting multiple times pick the shape-consistent crossing.
    - Endpoint deviation checks by default; `--skip-endpoint-checks` allows keeping matches even when endpoint diffs exceed the margin.
  - Consider GPS variations/inaccuracies:
    - Support normal GPS recordings as primary input; RTK GPX should also work (possibly with gaps).
    - Translation tolerance: shape matching should be translation-invariant while still enforcing spatial proximity limits so identical shapes in different locations do not match.
    - Typical GPS errors are +-4m; extreme errors can be up to +-12m.
  - Must be computationally efficient for large recordings and many segments:
    - Use coarse pruning before fine-grained matching.
    - Recognize that one recorded segment can match multiple reference segments.
- Emphasize robustness to uneven sampling, repeated laps, and self-intersections.
- Favor tunable thresholds (DTW, candidate margins, endpoint windows) to adapt to different track qualities.
  - Self-tuning is preferred and manual tuning should be only used when the automatic tuning fails; if manual tuning is needed, provide guidance based on the recorded GPX and reference segments.
- Support inspectability via optional GPX exports and candidate dumps.
- Keep outputs usable for quick review (stdout) and structured export (CSV/XLSX).

## Architecture
- Python CLI (`gpx-segment-timer.py`) that loads recorded/segment GPX data.
  - Can remain a single file or be decomposed later for maintainability.
- Pipeline: coarse candidate selection (distance + bounding boxes or other heuristics), DTW-based refinement, iterative boundary search, and endpoint anchoring with interpolation based on segment shape and length.
- Coarse gating uses multiple spatial filters: start/end candidate bboxes, overall bbox (derived from `--bbox-margin * 3.33`), rhomboid overlap, envelope distance, and optional x-track p95 filters.
- Matching is translation-tolerant but bounded: shape matching is translation-invariant while enforcing spatial proximity limits (GPS error bounds).
- Start/finish crossings are detected within the shape-matched window to avoid false matches when lingering around start/finish.
- Optional single-passage validation to reject re-entries on reference segments without repetitions/self-intersections (off by default; only on user request and only when references satisfy requirements).
- Output stage formats results to stdout/CSV/XLSX and optionally exports matched/candidate GPX files.
  - Statistics include:
    - Timing precision to 0.001s.
    - Length precision to 0.1m.
    - Start/end indices and coordinates; if interpolation was used, include bracketing indices, interpolated coordinates, and indication that interpolation was used.
    - Relevant deviation metrics (shape/length).
- GPX exports include:
    - Recorded track slice (matched window).
    - Reference segment.
    - Start and finish lines as separate tracks (two points each), matching the configured line length.
    - Start/finish crossing points, marking interpolated points and the bracketing points.
- Configuration is driven by CLI flags; defaults favor general-purpose matching on common GPX tracks.
  - Candidate endpoint margin is set by `--candidate-endpoint-margin-m` (negative uses `--gps-error-m`).

## Testing Guidelines
- Use `tests/run_tests.py` to validate synthetic fixtures and baselines defined in `tests/test_manifest.json`.
- Synthetic fixtures live in `tests/data/synthetic/` and must be generated from reference segments while covering real-world variability in recorded GPX (sampling jitter, noise, drift, short gaps).
- Synthetic reference segments must include both simple and difficult scenarios (repeated laps, self-intersections), plus combinations of those.
- Synthetic recorded tracks + synthetic references must cover cases where they should match (collocated), should not match despite colocation, and are not collocated at all.
- Each synthetic test must include ground-truth expectations (segment occurrences + expected durations with tolerances).
- Keep the synthetic dataset count reasonable so the full test run remains practical after updates.
- The bundled `example.gpx` + `segments/` baseline is a real-world example of a recorded track (example.gpx) and real segments
- Real-world validation uses private GPX files stored outside GitHub (or under `tests/data/real_world/`) with expected results in `tests/expected/`.
- Prepare a public real-world examples folder in GitHub (like `example.gpx` + `segments/`) based on publicly accessible, legal tracks.
- When adjusting matching logic, update expected results and verify DTW thresholds (or other thresholds if other methods are introduced) and endpoint deviations remain reasonable.


## Data Privacy
- Treat GPX files as sensitive location data; avoid committing personal tracks to the repository.
- If sharing outputs, scrub filenames or coordinates as needed and prefer small, anonymized samples.

## Release & Versioning
- Major versions: redesigns or fundamental updates.
- Minor versions: small or singular new features, optionally bundled with bugfixes.
- Patch versions: bugfix-only releases.

## Commit & Pull Request Guidelines
- Commit history uses short, descriptive, lowercase messages (e.g., "reworked docs", "more robust segment detection").
- PRs should include a concise summary, the exact command(s) used for verification, and sample output if behavior changes.
- If new flags or outputs are added, update `README.md` with usage and examples.
- New versions must be tested on reference data and verified that it does not compromise performance 

## Configuration & Data Tips
- GPX inputs can contain multiple tracks/segments; the script flattens them into point lists.
- Use `--dump-candidates-gpx` to debug candidate selection and `--export-gpx` to inspect matches visually.
