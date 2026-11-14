"""Marker regressions use analytic paths, independent of GPX/DTW dependencies."""
import ast
import datetime
import math
from pathlib import Path
import types
import typing
import unittest


def geometry():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'gpx-segment-timer.py').read_text())
    names = {'haversine_distance', '_prefix_counts', '_count_true', '_find_point_pair_matches'}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and (n.name in names or n.name.startswith('_circle_'))]
    namespace = dict(vars(typing), math=math)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'marker_geometry', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


class MarkerGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mod = geometry()
        self.origin = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        self.scale = 6371000.0 * math.pi / 180.0

    def points(self, coordinates):
        return [dict(lat=y / self.scale, lon=x / self.scale,
                     time=self.origin + datetime.timedelta(seconds=t))
                for x, y, t in coordinates]

    def matches(self, coordinates):
        return self.mod._find_point_pair_matches(
            self.points(coordinates), (0., 0.), (0., 100. / self.scale), 10., 10.)

    def test_outside_samples_can_cross_both_circles_in_one_edge(self):
        matches = self.matches([(-20, 0, 0), (120, 0, 14)])
        self.assertEqual(len(matches), 1)
        start, end, sc, ec, _, _ = matches[0]
        self.assertEqual((start, end), (0, 2))
        self.assertAlmostEqual((sc['time'] - self.origin).total_seconds(), 3., places=5)
        self.assertAlmostEqual((ec['time'] - self.origin).total_seconds(), 11., places=5)
        self.assertEqual((sc['idx0'], sc['idx1'], ec['idx0'], ec['idx1']), (0, 1, 0, 1))

    def test_off_center_chord_uses_circle_boundary_not_linear_radial_distance(self):
        matches = self.matches([(-20, 6, 0), (0, 6, 2), (20, 6, 4),
                                (80, 6, 10), (100, 6, 12), (120, 6, 14)])
        self.assertEqual(len(matches), 1)
        # sqrt(10**2 - 6**2) = 8: start exits at x=8, finish enters at x=92.
        self.assertAlmostEqual((matches[0][2]['time'] - self.origin).total_seconds(), 2.8, places=5)
        self.assertAlmostEqual((matches[0][3]['time'] - self.origin).total_seconds(), 11.2, places=5)

    def test_boundary_tangency_does_not_create_a_visit(self):
        self.assertEqual(self.matches([(-20, 10, 0), (120, 10, 14)]), [])

    def test_reentry_inside_an_edge_invalidates_earlier_start(self):
        matches = self.matches([(0, 0, 0), (20, 0, 2), (-20, 0, 6), (120, 0, 20)])
        self.assertEqual(len(matches), 1)
        # Later exits at t=5 and t=9 replace the first exit at t=1.
        self.assertAlmostEqual((matches[0][2]['time'] - self.origin).total_seconds(), 9., places=5)
        self.assertAlmostEqual((matches[0][3]['time'] - self.origin).total_seconds(), 17., places=5)

    def test_uniform_frequencies_keep_interpolated_times(self):
        for hz in (1, 10, 20):
            with self.subTest(hz=hz):
                matches = self.matches([(-20 + 10*i/hz, 0, i/hz) for i in range(14*hz+1)])
                self.assertEqual(len(matches), 1)
                self.assertAlmostEqual((matches[0][2]['time']-self.origin).total_seconds(), 3., places=5)
                self.assertAlmostEqual((matches[0][3]['time']-self.origin).total_seconds(), 11., places=5)

    def test_stationary_points_and_incomplete_visits(self):
        matches = self.matches([(0, 0, 0), (0, 0, 5), (20, 0, 7), (100, 0, 15)])
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual((matches[0][2]['time']-self.origin).total_seconds(), 6., places=5)
        self.assertEqual(self.matches([(0, 0, 0), (0, 0, 5)]), [])

    def test_recording_can_start_on_start_boundary_and_end_on_finish_boundary(self):
        matches = self.matches([(10, 0, 0), (90, 0, 8)])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][2]['time'], self.origin)
        self.assertEqual(matches[0][3]['time'], self.origin + datetime.timedelta(seconds=8))

    def test_multiple_visits_are_separate_matches(self):
        matches = self.matches([(-20, 0, 0), (120, 0, 14), (-20, 0, 28), (120, 0, 42)])
        self.assertEqual(len(matches), 2)
        self.assertAlmostEqual((matches[1][2]['time']-self.origin).total_seconds(), 31., places=5)

    def test_missing_times_preserve_geometric_crossings(self):
        points = self.points([(-20, 0, 0), (120, 0, 14)])
        points[0]['time'] = None
        matches = self.mod._find_point_pair_matches(points, (0.,0.), (0.,100/self.scale), 10.,10.)
        self.assertEqual(len(matches), 1)
        self.assertIsNone(matches[0][2]['time'])


if __name__ == '__main__':
    unittest.main()
