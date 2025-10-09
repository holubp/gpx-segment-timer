# GPX Segment Timer

The **GPX Segment Timer** is a Python tool for measuring elapsed time on specific segments within a recorded GPX track by comparing them against reference segments. It is designed to be robust against varying sampling frequencies, nonuniform point counts, and repeated segments (e.g., laps in a race).

It uses a multi-stage matching process that combines:

1. **Coarse Candidate Selection**  
   The tool first selects candidate windows based on cumulative distances from the recorded track that are within a specified variation (`--candidate-margin`) of the reference segment’s total distance.

2. **DTW-Based Refinement**  
   The candidate segments are uniformly resampled (using `--resample-count` points) and compared with the resampled reference segment using Dynamic Time Warping (DTW). A candidate with a low average DTW cost (below `--dtw-threshold`) is selected.

3. **Iterative Grid Search**  
   The candidate’s boundaries are adjusted in a coarse grid search independently for the start and end (controlled by `--iterative-window-start` and `--iterative-window-end`) so that the overall candidate segment length comes closer to the reference segment’s length. A penalty term (weighted by `--penalty-weight`) is added based on how far the candidate endpoints are from the reference endpoints.

4. **Endpoint Anchoring (Local Refinement)**  
   Finally, a local sliding-window search (controlled by `--endpoint-window-start` and `--endpoint-window-end`) is performed on each endpoint. A short subsegment of length L (default is 10% of the resample count, but at least 3 points) is extracted from the candidate and compared against the corresponding subsegment from the reference. This “anchors” the candidate’s start and end as close as possible to the reference endpoints. The costs for the start and end subsegments are weighted by `--anchor-beta1` and `--anchor-beta2`.

By default, if the detected segment’s start or end deviates by more than `--bbox-margin` meters from the reference, the segment is **rejected**. However, you can disable this rejection (and only log a warning) by using the flag `--no-boundary-rejection`.

---

## Features

- **Robust Matching**  
  Uses a combination of global DTW and local endpoint refinement to align the candidate segment’s shape with the reference.

- **Configurable Tolerance**  
  The allowed deviation at the endpoints (`--bbox-margin`) can be enforced by default or disabled with `--no-boundary-rejection`.

- **Flexible Output Options**  
  Output may be printed to standard output (as a formatted table), CSV, or XLSX. Matched segments can also be exported as separate GPX tracks for further inspection.

---

## Installation

Make sure you have Python 3 installed. Install the required Python packages using pip:

`pip install gpxpy fastdtw openpyxl`

---

## Usage

Run the script from the command line as follows:

`./gpx-segment-timer.py -r <recorded_track.gpx> -f <reference_folder> [OPTIONS]`

### Key Command-Line Options

- **Input / Output Options**  
  - `-r, --recorded`  
    Path to the GPX file containing the recorded track.  
  - `-f, --reference-folder`  
    Path to the folder containing reference GPX segment files.  
  - `-o, --output-mode`  
    Output format: `stdout` (default), `csv`, or `xlsx`.  
  - `-O, --output-file`  
    Output file path (required for CSV and XLSX).

- **Matching Parameters**  
  - `--candidate-margin`  
    Allowed variation (fraction) in candidate segment distance relative to the reference (default: 0.2).  
  - `--dtw-threshold`  
    Maximum allowed average DTW distance (in meters per resampled point) for a match (default: 50).  
  - `--resample-count`  
    Number of points for resampling (default: 50).  
  - `--min-gap`  
    Minimum number of recorded points to skip after a match (default: 1).  
  - `--bbox-margin`  
    Tolerance (in meters) for the detected endpoints compared to the reference (default: 30).

- **Boundary Refinement Parameters**  
  - `--iterative-window-start` and `--iterative-window-end`  
    Grid search window (in points) for adjusting the start and end boundaries, respectively (default: 50 each).  
  - `--penalty-weight`  
    Weight for the Euclidean endpoint penalty in grid refinement (default: 2.0).  
  - `--anchor-beta1` and `--anchor-beta2`  
    Weights for the DTW cost on the start and end subsegments (default: 1.0 each).  
  - `--endpoint-window-start` and `--endpoint-window-end`  
    Local sliding window (in points) for refining the boundaries (default: 1000 each).

- **Boundary Rejection**  
  By default, if the candidate’s endpoints deviate by more than `--bbox-margin`, the segment is rejected. Use `--no-boundary-rejection` to allow segments even if endpoints deviate beyond the margin (with a warning).

- **Logging & Export**  
  - `-v, --verbose`  
    Enable verbose logging (INFO level).  
  - `-d, --debug`  
    Enable debug logging (DEBUG level).  
  - `--export-gpx`  
    Export matched segments as individual GPX tracks.  
  - `--export-gpx-file`  
    Base filename for the exported GPX file(s) (default: `matched_segments.gpx`).

---

### Example Commands

Example how to run it on test data:

```./gpx-segment-timer.py -v -f segments -r example.gpx  --export-gpx```

```./gpx-segment-timer.py -v -f segments -r example.gpx --export-gpx --candidate-margin 0.05 --bbox-margin 20 --bbox-margin-overall 100 --resample-count 100```


## Tuning Tips

- **Adjust Global Matching**  
  Modify `--candidate-margin` and `--dtw-threshold` if the coarse candidate selection misses good matches or returns too many false positives.

- **Endpoint Refinement**  
  Increase `--iterative-window-start / --iterative-window-end` if the coarse grid search does not allow enough boundary movement.  
  Adjust `--penalty-weight` if the candidate endpoints consistently drift away from the reference endpoints.  
  Tweak `--anchor-beta1 / --anchor-beta2` if the shapes near the boundaries need more precise alignment.  
  Adjust `--endpoint-window-start / --endpoint-window-end` to control the local (final) sliding.

- **Boundary Rejection**  
  If you want to keep all segments (even if boundaries deviate), use `--no-boundary-rejection`.

---

## License

GNU GPL v3

## Author

Petr Holub
