# Repository Guidelines

## Project Structure & Module Organization
- `gpx-segment-timer.py` is the primary CLI script.
- `segments/` holds reference GPX segment files used for matching.
- `example.gpx` and `20250904_1852.gpx` are sample recorded tracks.
-- `20250904_1852.gpx` or similarly named files should not be committed to the repo, even if locally present
- Additional variants like `gpx-segment-timer_*` are experimental snapshots.

## Build, Test, and Development Commands
- `python3 gpx-segment-timer.py -r example.gpx -f segments` runs a basic match against bundled data.
- `python3 gpx-segment-timer.py -r <track.gpx> -f segments --export-gpx` exports matched segments for inspection.
- `pip install gpxpy fastdtw openpyxl` installs required dependencies for GPX parsing, DTW, and XLSX output.

## Coding Style & Naming Conventions
- Follow Python 3, 4-space indentation, and `snake_case` for functions and variables.
- Keep CLI flags consistent with existing long/short option patterns in `gpx-segment-timer.py`.
- Prefer standard library modules first; external deps should be documented in `README.md` if added.

## Testing Guidelines
- There is no automated test suite currently.
- Use `example.gpx` with `segments/` to validate changes manually and compare output stability.
- When adjusting matching logic, verify DTW thresholds and endpoint deviations remain reasonable with the sample data.

## Design
- Provide a Python CLI that measures elapsed times for reference GPX segments against a recorded track.
- Functionality requirements:
-- must work with different sampling frequencies across different recorded GPX (e.g., 1Hz, 5Hz, 10Hz, 20Hz, also dynamic frequency) and this being different from the number of points in the reference segment
-- must work in presence of:
--- self-intersecting reference segments
--- segments which contain multiple repetition of the same parts of the track/route (e.g., if the same circuit is used 3x as a reference segment - only when the recorded track passes it also 3x, it should match and not otherwise)
--- segments which have very close start and end points (e.g., measuring a circuit race)
--- segments which are overlapping (e.g., one reference segment may be fully or partially overlapping with another reference segment - and the measured GPX must recognize both if they match)
-- must work on recorded tracks 
--- the same parts of which might be used in different segments (i.e., the fact that the given series of track points in the recorded track matches some reference segment or a part of it does not preclude the same track points to match also another reference segment or its part)
-- matching of the reference segments must be done based on
--- starting and final points of the reference segments should be defined as a line perpendicular to the first/last vector connecting first/last two points of the reference segment
--- the shape of the reference segment (note it has to work in presence of repeated or self-intersecting reference segments)
--- length of the reference segment
--- if the recorder track does not contain a point precisely on the start or finish line, the optimum point has to be interpolated between the two closes matching points and output needs to clarify where that interpolated point is
--- due to the nature of GPX recording, the shape of the reference segment needs to be taken somewhat as translation invariant, i.e., the recorded track might contain inaccuracies that might make it somewhat shifted
-- must be computationally efficient for big track recordings and many segments
- Emphasize robustness to uneven sampling, repeated laps, and self-intersections.
- Favor tunable thresholds (DTW, candidate margins, endpoint windows) to adapt to different track qualities.
-- self-tuning is preferred and manual tuning should be only used when the automatic tuning fails (and if manual tuning is needed, the script should provide guidance based on the provided recorded GPX and reference GPX segments)
- Support inspectability via optional GPX exports and candidate dumps.
- Keep outputs usable for quick review (stdout) and structured export (CSV/XLSX).

## Architecture
- Python CLI (`gpx-segment-timer.py`) that loads recorded/segment GPX data into in-memory point lists.
-- can be a single file or later decomposed in different modules for better maintainability of the code
- Pipeline: coarse candidate selection by distance + bounding boxes, DTW-based refinement, iterative boundary search, and endpoint anchoring.
-- can change to achieve best computational performance and also matching capability
- Optional single-passage validation to reject re-entries on self-intersecting tracks.
-- but these are not default and should be only done when requested by the user (and in such a case the reference GPX segments should also be checked that they are not violating these requirements)
- Output stage formats results to stdout/CSV/XLSX and optionally exports matched/candidate GPX files.
- Configuration is driven by CLI flags; defaults favor general-purpose matching on common GPX tracks.

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

## Configuration & Data Tips
- GPX inputs can contain multiple tracks/segments; the script flattens them into point lists.
- Use `--dump-candidates-gpx` to debug candidate selection and `--export-gpx` to inspect matches visually.
