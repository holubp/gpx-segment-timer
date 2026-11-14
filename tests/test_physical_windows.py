"""Analytic regressions for original-sample time and distance windows."""
import ast
import bisect
import datetime
import logging
import math
from pathlib import Path
import types
import typing
import unittest


def geometry():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'gpx-segment-timer.py').read_text())
    names = {'haversine_distance', '_point_to_xy', 'find_line_crossings',
             'find_line_crossings_in_distance_window', 'find_line_crossings_in_time_window',
             'distance_window_bounds', 'refine_endpoint_boundary', 'resample_points',
             'build_shape_sequence', '_points_to_xy', '_vector_distance', 'refine_boundaries_iteratively',
             'sample_distance_interval', 'make_distance_sampler', 'select_crossing_window'}
    namespace = dict(vars(typing), math=math, bisect=bisect, logging=logging)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name in names], type_ignores=[]), 'physical_windows', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


class PhysicalWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod = geometry()

    def test_independent_distance_bounds_do_not_mirror_dense_side(self):
        self.assertTrue(hasattr(self.mod, 'distance_window_bounds'))
        self.assertEqual(self.mod.distance_window_bounds([0.,100.,101.,102.,110.,120.], 1, 10.), (1,5))
        self.assertEqual(self.mod.distance_window_bounds([0.,100.,101.], 1, 0.), (1,2))

    def test_time_window_checks_interpolated_time_across_gap(self):
        self.assertTrue(hasattr(self.mod, 'find_line_crossings_in_time_window'))
        origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        points = [dict(lat=0.,lon=x/111320.,time=origin+datetime.timedelta(seconds=t))
                  for x,t in [(0,0),(50,5)]]
        def crossings(line):
            return self.mod.find_line_crossings_in_time_window(
                points,[0.,50.],(0.,line/111320.),(1.,0.),0,100.,0.,1.,4.)
        self.assertEqual(len(crossings(8)),1)
        self.assertEqual(crossings(12),[])

    def test_time_window_missing_timestamps_falls_back_to_meter_cap(self):
        self.assertTrue(hasattr(self.mod, 'find_line_crossings_in_time_window'))
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in (0,50)]
        result = self.mod.find_line_crossings_in_time_window(
            points,[0.,50.],(0.,12/111320.),(1.,0.),0,20.,0.,1.,4.)
        self.assertEqual(len(result),1)

    def test_time_window_is_consistent_at_uniform_and_uneven_rates(self):
        origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        for times in ([i/hz for i in range(3*hz+1)] for hz in (1,10,20)):
            points = [dict(lat=0.,lon=10*t/111320.,time=origin+datetime.timedelta(seconds=t)) for t in times]
            for line,expected in ((8.,1),(12.,0)):
                result = self.mod.find_line_crossings_in_time_window(
                    points,[10*t for t in times],(0.,line/111320.),(1.,0.),0,100.,0.,1.,4.)
                self.assertEqual(len(result),expected)
        times = (0.,.1,1.2,3.)
        points = [dict(lat=0.,lon=10*t/111320.,time=origin+datetime.timedelta(seconds=t)) for t in times]
        for line,expected in ((10.,1),(12.,0)):
            result = self.mod.find_line_crossings_in_time_window(
                points,[10*t for t in times],(0.,line/111320.),(1.,0.),1,100.,0.,1.,4.)
            self.assertEqual(len(result),expected)

    def test_reversed_time_bracket_uses_distance_limit(self):
        origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        points = [dict(lat=0.,lon=x/111320.,time=origin-datetime.timedelta(seconds=t)) for x,t in ((0,0),(50,5))]
        result = self.mod.find_line_crossings_in_time_window(
            points,[0.,50.],(0.,12/111320.),(1.,0.),0,20.,0.,1.,4.)
        self.assertEqual(len(result),1)

    def test_invalid_physical_bounds_are_rejected(self):
        for radius in (-1.,float('nan'),float('inf')):
            with self.subTest(radius=radius),self.assertRaises(ValueError):
                self.mod.distance_window_bounds([0.,10.],0,radius)
        with self.assertRaises(ValueError):
            self.mod.sample_distance_interval([dict(lat=0.,lon=0.)],[0.],0.,1.,3)

    def test_endpoint_search_cannot_escape_distance_limit_on_sparse_side(self):
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in (0,100,101,102,103,104,105,106,107,108,109,110)]
        ref = [(0.,x/111320.) for x in (0,1,2)]
        chosen = self.mod.refine_endpoint_boundary(points,1,ref,3,10,True,
                                                   'step_vectors',0.,(0.,0.),1000.)
        self.assertEqual(chosen,1)

    def test_iterative_search_uses_meters_for_each_boundary(self):
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in range(0,101,10)]
        result = self.mod.refine_boundaries_iteratively(
            points,4,8,(0.,0.),(0.,80/111320.),20.,list(range(0,111,10)),10,10,10.)
        self.assertEqual(result[0],3)
        self.assertIn(result[1],(7,8,9))

    def test_iterative_window_can_reach_terminal_sample_without_overrun(self):
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in range(12)]
        result = self.mod.refine_boundaries_iteratively(
            points,3,10,(0.,3/111320.),(0.,11/111320.),8.,list(range(12)),50.,50.,2.)
        self.assertLess(result[0], result[1])
        self.assertLessEqual(result[1],len(points))

    def test_distance_samples_keep_original_bracket_mapping(self):
        self.assertTrue(hasattr(self.mod,'sample_distance_interval'))
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in (0,100,101,110)]
        samples, positions = self.mod.sample_distance_interval(points,[0.,100.,101.,110.],10.,105.,3)
        self.assertEqual(len(points),4)
        self.assertAlmostEqual(samples[0][1]*111320.,10.)
        self.assertAlmostEqual(samples[-1][1]*111320.,105.)
        for actual,expected in zip(positions,(.1,.575,2+4/9)):
            self.assertAlmostEqual(actual,expected)

    def test_comparison_cache_reuses_interval_and_does_not_mutate_recording(self):
        self.assertTrue(hasattr(self.mod,'make_distance_sampler'))
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in (0,10,20)]
        before = [dict(p) for p in points]
        sample = self.mod.make_distance_sampler(points,[0.,10.,20.],2)
        first = sample(0.,10.,3)
        self.assertIs(first,sample(0.,10.,3))
        sample(10.,20.,3)
        sample(0.,20.,5)
        self.assertEqual(points,before)
        self.assertEqual(first,sample(0.,10.,3))

    def test_endpoint_shape_compares_same_physical_extent_at_different_rates(self):
        reference = [(0.,0.),(0.,50/111320.),(50/111320.,50/111320.)]
        for hz in (1,10,20):
            with self.subTest(hz=hz):
                coordinates = [(min(50.,5*i/hz),max(0.,5*i/hz-50.)) for i in range(20*hz+1)]
                points = [dict(lat=y/111320.,lon=x/111320.,time=None) for x,y in coordinates]
                chosen = self.mod.refine_endpoint_boundary(points,0,reference,3,100.,True,
                                                           'step_vectors',0.,(0.,0.),0.)
                self.assertEqual(chosen,0)

    def test_looping_reference_context_keeps_extent_after_chord_collapse(self):
        scale = 6371000. * math.pi / 180.
        points = [dict(lat=0.,lon=x/scale,time=None) for x in (0,20,15,20,15,20,15,40)]
        # A 20m out/back reference can have three colocated comparison points.
        # Its physical extent must still be 20m, not the zero-length chords.
        result = self.mod.refine_endpoint_boundary(
            points,0,[(0.,0.)]*3,3,40.,True,'step_vectors',0.,(0.,0.),0.,
            [0.,20.,25.,30.,35.,40.,45.,70.],context_length_m=20.)
        self.assertEqual(result,1)

    def test_refinement_cannot_discard_a_complete_original_crossing_pair(self):
        self.assertTrue(hasattr(self.mod,'select_crossing_window'))
        xs = [0.,10.,20.,30.,40.,50.,60.,70.,75.,80.,90.,100.,110.]
        points = [dict(lat=0.,lon=x/111320.,time=None) for x in xs]
        def evaluate(start,end):
            starts = self.mod.find_line_crossings_in_distance_window(
                points,xs,(0.,0.),(1.,0.),start,20.,0.,4.)
            finishes = self.mod.find_line_crossings_in_distance_window(
                points,xs,(0.,100/111320.),(1.,0.),end-1,20.,0.,4.)
            return None,None,0.,0. if finishes else 100-xs[end-1],starts[0] if starts else None,finishes[-1] if finishes else None
        window, metrics, restored = self.mod.select_crossing_window((0,9),(0,12),evaluate,30.)
        self.assertEqual(window,(0,12))
        self.assertTrue(restored)
        self.assertIsNotNone(metrics[5])
        # An original candidate with no additional crossing cannot replace refinement.
        self.assertFalse(self.mod.select_crossing_window((0,9),(0,8),evaluate,30.)[2])
        reverse_points = list(reversed(points))
        reverse_cumulative = [0.]
        for a,b in zip(reversed(xs),list(reversed(xs))[1:]):
            reverse_cumulative.append(reverse_cumulative[-1]+abs(b-a))
        def reverse_evaluate(start,end):
            starts = self.mod.find_line_crossings_in_distance_window(
                reverse_points,reverse_cumulative,(0.,0.),(1.,0.),start,200.,0.,4.,start,end)
            finishes = self.mod.find_line_crossings_in_distance_window(
                reverse_points,reverse_cumulative,(0.,100/111320.),(1.,0.),end-1,200.,0.,4.,start,end)
            return None,None,0.,0.,starts[0] if starts else None,finishes[-1] if finishes else None
        self.assertFalse(self.mod.select_crossing_window((0,2),(0,len(points)),reverse_evaluate,30.)[2])

    def test_recovery_retains_valid_finish_when_start_line_is_missed(self):
        xs = [0.,10.,20.,30.,40.,50.,60.,70.,75.,80.,90.,100.,110.]
        points = [dict(lat=(5.-x*.05)/111320.,lon=x/111320.,time=None) for x in xs]
        def evaluate(start,end):
            starts = self.mod.find_line_crossings_in_distance_window(
                points,xs,(0.,0.),(1.,0.),start,20.,0.,4.)
            ends = self.mod.find_line_crossings_in_distance_window(
                points,xs,(0.,100/111320.),(1.,0.),end-1,20.,0.,4.)
            return None,None,5.,0. if ends else 25.,starts[0] if starts else None,ends[-1] if ends else None
        selected, metrics, restored = self.mod.select_crossing_window((0,9),(0,12),evaluate,30.)
        self.assertEqual(selected,(0,12))
        self.assertTrue(restored)
        self.assertIsNone(metrics[4])
        self.assertIsNotNone(metrics[5])


if __name__ == '__main__':
    unittest.main()
