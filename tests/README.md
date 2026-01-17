# Test Framework

This test framework replaces the legacy suite. It uses a manifest-driven runner and supports
public, private, and synthetic datasets.

## Layout

- `tests/manifest.json` defines test cases, inputs, CLI args, and tolerances.
- `tests/run_tests.py` runs one or more cases, compares outputs, and reports deltas.
- `tests/data/real_world/public/` holds public recorded tracks.
- `tests/data/real_world/private/` holds private recorded tracks (ignored by git).
- `tests/expected/real_world/public/` stores expected outputs for public datasets.
- `tests/expected/real_world/private/` stores expected outputs for private datasets (ignored by git).
- `tests/data/synthetic/` and `tests/expected/synthetic/` are used for generated fixtures.
- `tests/tools/generate_synthetic_data.py` creates synthetic tracks and expected results.

## Running Tests

- Run all available cases:
  `python3 tests/run_tests.py`

- Run only public or private cases:
  `python3 tests/run_tests.py --suite public`
  `python3 tests/run_tests.py --suite private`

- Run specific case IDs:
  `python3 tests/run_tests.py --case real_public_automotodrom`

## Updating Expected Outputs

Use `--update-expected` to refresh expected outputs from the current binary:

```
python3 tests/run_tests.py --suite public --update-expected
```

This overwrites `stdout.txt`, `trace.txt`, match GPX files, and unmatched GPX files for the case.

## Synthetic Data

The generator builds synthetic tracks with matching and non-matching variations.

```
python3 tests/tools/generate_synthetic_data.py \
  --segments segments \
  --segment-glob "automotodrom-okruh-offroad-les.gpx" \
  --output tests/data/synthetic \
  --expected tests/expected/synthetic
```

Useful options:

- `--segment-glob` selects which reference segments are synthesized.
- `--modes` controls which variation modes are generated (default: match_noise, match_linger,
  match_detour, nonmatch_shift).

## Result Interpretation

- The runner compares match tables, GPX outputs, and verbose traces.
- Exact matches pass; small deviations within tolerances are reported as "soft" mismatches.
- Large deviations are reported as hard failures.
- Use `--trace-diff` to print a normalized unified diff when trace files differ.
