import argparse
import ast
import bisect
import contextlib
import io
import logging
import math
import sys
from pathlib import Path
import types
import typing
import unittest


def load_geometry():
    source = ast.parse((Path(__file__).resolve().parents[1] / 'gpx-segment-timer.py').read_text())
    names = {'haversine_distance', '_point_to_xy', 'find_line_crossings',
             'find_line_crossings_in_distance_window', 'compute_line_normal', 'apply_matching_preset'}
    namespace = dict(vars(typing), math=math, bisect=bisect, logging=logging)
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'geometry', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


class CrossingDistanceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = load_geometry()
        self.assertTrue(hasattr(self.geometry, 'find_line_crossings_in_distance_window'),
                        'crossing searches need a shared distance-bounded implementation')

    def crossings(self, eastings, center, cap, line=12.0):
        points = [dict(lat=0.0, lon=east / 111320.0, time=None) for east in eastings]
        cumulative = [0.0]
        for left, right in zip(eastings, eastings[1:]):
            cumulative.append(cumulative[-1] + abs(right - left))
        return self.geometry.find_line_crossings_in_distance_window(
            points, cumulative, (0.0, line / 111320.0), (1.0, 0.0),
            center, cap, 0.0, 4.0)

    def test_same_distance_limit_at_1_10_20_hz(self):
        for rate in (1, 10, 20):
            with self.subTest(rate=rate):
                eastings = [offset * 10.0 / rate for offset in range(5 * rate + 1)]
                crossings = self.crossings(eastings, 0, 20.0)
                self.assertTrue(crossings)
                self.assertAlmostEqual(crossings[0]['lon'] * 111320, 12.0)
                self.assertEqual(self.crossings(eastings, 0, 10.0), [])

    def test_gap_bracketing_point_may_be_outside_cap(self):
        self.assertTrue(self.crossings([0.0, 50.0], 0, 20.0))
        self.assertEqual(self.crossings([0.0, 50.0], 0, 10.0), [])

    def test_uneven_sampling_caps_both_sides(self):
        self.assertTrue(self.crossings([0.0, 2.0, 25.0, 60.0], 2, 14.0))
        self.assertEqual(self.crossings([0.0, 2.0, 25.0, 60.0], 2, 12.0), [])

    def test_repeated_lap_crossing_outside_cap_is_excluded(self):
        self.assertEqual(len(self.crossings([0., 20., 50., 0., 20.], 0, 20.0)), 1)

    def test_invalid_distance_limit_is_rejected(self):
        for cap in (-1.0, float('nan'), float('inf')):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.crossings([0., 20.], 0, cap)

    def test_zero_limit_and_recording_edges(self):
        self.assertTrue(self.crossings([0., 20.], 0, 0.0, line=0.0))
        self.assertTrue(self.crossings([0., 20.], 1, 0.0, line=20.0))
        self.assertEqual(self.crossings([0., 20.], 0, 0.0), [])

    def test_profiles_use_meter_default_and_preserve_override(self):
        for profile in ('none', 'tight', 'normal', 'standard', 'normal-widetrack', 'loose', 'loosest'):
            for override in (None, 7.5):
                with self.subTest(profile=profile, override=override):
                    args = argparse.Namespace(matching_preset=profile, crossing_window_max=override,
                                              strict_envelope_window_m=None, strict_envelope_off_pct=None,
                                              envelope_max_m=None, dtw_window_m=None,
                                              dtw_window_max_avg=None, gps_error_m=12.0)
                    self.geometry.apply_matching_preset(args)
                    defaults = {'loose': 100.0, 'loosest': 200.0}
                    self.assertEqual(args.crossing_window_max, defaults.get(profile, 20.0) if override is None else override)

    def test_normal_widetrack_only_changes_normal_envelope(self):
        for envelope_override in (None, 7.5):
            with self.subTest(envelope_override=envelope_override):
                defaults = dict(crossing_window_max=None, strict_envelope_window_m=None,
                                strict_envelope_off_pct=None, envelope_max_m=envelope_override,
                                dtw_window_m=None, dtw_window_max_avg=None, gps_error_m=12.0)
                normal = argparse.Namespace(matching_preset='normal', **defaults)
                wide = argparse.Namespace(matching_preset='normal-widetrack', **defaults)
                self.geometry.apply_matching_preset(normal)
                self.geometry.apply_matching_preset(wide)
                self.assertEqual(wide.envelope_max_m, 10.0 if envelope_override is None else envelope_override)
                normal_values = vars(normal).copy()
                wide_values = vars(wide).copy()
                for key in ('matching_preset', 'envelope_max_m'):
                    normal_values.pop(key)
                    wide_values.pop(key)
                self.assertEqual(wide_values, normal_values)

    def test_standard_is_a_legacy_alias_for_normal(self):
        for explicit in (None, 7.5):
            with self.subTest(explicit=explicit):
                values = dict(crossing_window_max=explicit, strict_envelope_window_m=explicit,
                              strict_envelope_off_pct=explicit, envelope_max_m=explicit,
                              dtw_window_m=explicit, dtw_window_max_avg=explicit, gps_error_m=12.0)
                normal = argparse.Namespace(matching_preset='normal', **values)
                legacy = argparse.Namespace(matching_preset='standard', **values)
                self.geometry.apply_matching_preset(normal)
                self.geometry.apply_matching_preset(legacy)
                self.assertEqual(normal.envelope_max_m, 5.0 if explicit is None else explicit)
                self.assertEqual({key: value for key, value in vars(normal).items() if key != 'matching_preset'},
                                 {key: value for key, value in vars(legacy).items() if key != 'matching_preset'})

    def test_cli_defaults_to_normal_and_accepts_legacy_name(self):
        path = Path(__file__).resolve().parents[1] / 'gpx-segment-timer.py'
        tree = ast.parse(path.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        option = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                      and any(isinstance(arg, ast.Constant) and arg.value == '--matching-preset' for arg in node.args))
        parser = argparse.ArgumentParser()
        namespace = {'parser': parser}
        option_module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=option)], type_ignores=[]))
        exec(compile(option_module, str(path), 'exec'), namespace)
        self.assertEqual(parser.parse_args([]).matching_preset, 'normal')
        for profile in ('normal', 'standard', 'normal-widetrack'):
            self.assertEqual(parser.parse_args(['--matching-preset', profile]).matching_preset, profile)

    def test_debug_fallback_respects_distance_cap(self):
        path = Path(__file__).resolve().parents[1] / 'tools/debug_match_metrics.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_compute_crossings')
        namespace = dict(vars(typing))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        points = [dict(lat=0.0, lon=east / 111320.0, time=None) for east in (0., 50., 100.)]
        reference = [dict(lat=0.0, lon=east / 111320.0) for east in (12., 88.)]
        result = namespace['_compute_crossings'](self.geometry, points, reference, 1, 8.0, 10.0)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        result = namespace['_compute_crossings'](self.geometry, points, reference, 1, 8.0, 20.0)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])

    def test_debug_does_not_reinterpret_old_point_caps_as_meters(self):
        path = Path(__file__).resolve().parents[1] / 'tools/debug_match_metrics.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_apply_log_config')
        namespace = dict(vars(typing), argparse=argparse, sys=sys)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        args = argparse.Namespace(crossing_window_max=20.0)
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            namespace['_apply_log_config'](args, {'crossing_window_max': '200'})
        self.assertEqual(args.crossing_window_max, 20.0)
        self.assertIn('legacy point-based', warning.getvalue())
        namespace['_apply_log_config'](args, {'crossing_window_max_m': '100.5'})
        self.assertEqual(args.crossing_window_max, 100.5)


if __name__ == '__main__':
    unittest.main()
