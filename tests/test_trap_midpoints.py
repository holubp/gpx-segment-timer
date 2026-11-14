"""Analytic trap midpoint behavior, independent of GPX/DTW imports."""
import ast
import bisect
import datetime
import math
from pathlib import Path
import types
import typing
import unittest


def geometry():
    tree = ast.parse((Path(__file__).resolve().parents[1]/'gpx-segment-timer.py').read_text())
    names = {'haversine_distance','crossing_distance_along_track','_find_trap_pair_matches',
             '_find_point_pair_matches','_prefix_counts','_count_true'}
    namespace = dict(vars(typing),math=math,bisect=bisect)
    funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef)
             and (n.name in names or n.name.startswith('_circle_'))]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'trap_geometry','exec'),namespace)
    return types.SimpleNamespace(**namespace)


class TrapMidpointTests(unittest.TestCase):
    def setUp(self):
        self.mod = geometry()
        self.scale = 6371000.*math.pi/180.
        self.origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)

    def points(self, coordinates):
        return [dict(lat=y/self.scale,lon=x/self.scale,
                     time=self.origin+datetime.timedelta(seconds=t)) for x,y,t in coordinates]

    def pairs(self, coords, start_trap=False, finish_trap=True):
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        return self.mod._find_trap_pair_matches(
            self.points(coords),(0.,0.),(0.,50/self.scale),10.,10.,start_trap,finish_trap)

    def test_dense_one_inside_and_no_inside_samples_share_trap_midpoint(self):
        cases = [[(x,0,(x+20)/10) for x in range(-20,81)],
                 [(-20,0,0),(20,0,4),(50,0,7),(80,0,10)],
                 [(-20,0,0),(80,0,10)]]
        for coords in cases:
            with self.subTest(samples=len(coords)):
                matches = self.pairs(coords)
                self.assertEqual(len(matches),1)
                start, trap = matches[0][2:4]
                self.assertAlmostEqual((start['time']-self.origin).total_seconds(),3.,places=5)
                self.assertAlmostEqual((trap['time']-self.origin).total_seconds(),7.,places=5)
                self.assertAlmostEqual(trap['lon']*self.scale,50.,places=5)
                self.assertAlmostEqual(trap['entry']['lon']*self.scale,40.,places=5)
                self.assertAlmostEqual(trap['exit']['lon']*self.scale,60.,places=5)

    def test_midpoint_uses_boundary_coordinates_and_times_on_bent_visit(self):
        # Entry (40,0) at t=6; exit (50,10) at t=14.
        # Chord midpoint (45,5), time10; no assumption about interior samples.
        matches = self.pairs([(-20,0,0),(20,0,4),(40,0,6),(50,0,9),(50,10,14),(50,20,16)])
        self.assertEqual(len(matches),1)
        event = matches[0][3]
        self.assertAlmostEqual(event['lon']*self.scale,45.,places=5)
        self.assertAlmostEqual(event['lat']*self.scale,5.,places=5)
        self.assertEqual(event['time'],self.origin+datetime.timedelta(seconds=10))
        self.assertAlmostEqual(event['position'],3.2,places=5)

    def test_tangent_contact_between_outside_points_is_a_trap_event(self):
        # Start exits at x8 (radius10,y6). Trap is centered at (50,16): tangent y6.
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        matches = self.mod._find_trap_pair_matches(
            self.points([(-20,6,0),(80,6,10)]),(0.,0.),(16/self.scale,50/self.scale),10.,10.,False,True)
        self.assertEqual(len(matches),1)
        trap=matches[0][3]
        self.assertAlmostEqual(trap['lon']*self.scale,50.,places=5)
        self.assertAlmostEqual(trap['lat']*self.scale,6.,places=5)
        self.assertEqual(trap['entry']['time'],trap['exit']['time'])
        self.assertAlmostEqual((trap['time']-self.origin).total_seconds(),7.,places=5)

    def test_incomplete_trap_visit_does_not_fabricate_exit(self):
        self.assertEqual(self.pairs([(-20,0,0),(20,0,4),(50,0,7)]),[])
        # Merely reaching the entry boundary at EOF is not a tangent visit.
        self.assertEqual(self.pairs([(-20,0,0),(40,0,6)]),[])
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        self.assertEqual(self.mod._find_trap_pair_matches(
            self.points([(10,0,0),(80,0,7)]),(0.,0.),(0.,50/self.scale),10.,10.,True,False),[])

    def test_adjacent_pairs_share_trap_event_instead_of_entry_exit_gap(self):
        coords=[(-20,0,0),(120,0,14)]
        points=self.points(coords)
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        left=self.mod._find_trap_pair_matches(points,(0.,0.),(0.,50/self.scale),10.,10.,False,True)
        right=self.mod._find_trap_pair_matches(points,(0.,50/self.scale),(0.,100/self.scale),10.,10.,True,False)
        self.assertEqual(len(left),1)
        self.assertEqual(len(right),1)
        self.assertEqual(left[0][3]['time'],right[0][2]['time'])
        self.assertAlmostEqual((right[0][3]['time']-self.origin).total_seconds(),11.,places=5)

    def test_missing_timestamps_use_distance_mapping_without_fabricated_time(self):
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        points=self.points([(-20,0,0),(20,0,4),(50,0,7),(80,0,10)])
        for p in points: p['time']=None
        matches=self.mod._find_trap_pair_matches(points,(0.,0.),(0.,50/self.scale),10.,10.,False,True)
        self.assertEqual(len(matches),1)
        trap=matches[0][3]
        self.assertIsNone(trap['time'])
        self.assertAlmostEqual(trap['along_track_m'],70.,places=5)

    def test_repeated_visits_and_reentry_choose_correct_start(self):
        matches=self.pairs([(0,0,0),(20,0,2),(0,0,4),(20,0,6),(80,0,12),
                            (-20,0,22),(80,0,32)])
        self.assertEqual(len(matches),2)
        self.assertEqual(matches[0][2]['time'],self.origin+datetime.timedelta(seconds=5))
        self.assertEqual(matches[0][3]['time'],self.origin+datetime.timedelta(seconds=9))
        self.assertEqual(matches[1][3]['time'],self.origin+datetime.timedelta(seconds=29))

    def test_overlapping_trap_circles_pair_ordered_midpoints(self):
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        points=self.points([(-20,0,0),(80,0,10)])
        matches=self.mod._find_trap_pair_matches(points,(0.,20/self.scale),(0.,30/self.scale),10.,10.,True,True)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0][2]['time'],self.origin+datetime.timedelta(seconds=4))
        self.assertEqual(matches[0][3]['time'],self.origin+datetime.timedelta(seconds=5))

    def test_tangent_vertex_is_one_visit_and_incomplete_initial_trap_is_skipped(self):
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        points=self.points([(-20,6,0),(50,6,7),(80,6,10)])
        matches=self.mod._find_trap_pair_matches(points,(0.,0.),(16/self.scale,50/self.scale),10.,10.,False,True)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0][3]['time'],self.origin+datetime.timedelta(seconds=7))
        # Start trap has no observed entry because recording begins at its center.
        self.assertEqual(self.mod._find_trap_pair_matches(
            self.points([(0,0,0),(80,0,8)]),(0.,0.),(0.,50/self.scale),10.,10.,True,False),[])

    def test_stationary_ordinary_tangency_is_not_a_start(self):
        points=self.points([(-20,10,0),(0,10,2),(0,10,4),(80,10,12),(120,10,16)])
        self.assertTrue(hasattr(self.mod,'_find_trap_pair_matches'))
        self.assertEqual(self.mod._find_trap_pair_matches(
            points,(0.,0.),(10/self.scale,50/self.scale),10.,10.,False,True),[])

    def test_midpoint_timing_is_consistent_at_one_ten_and_twenty_hz(self):
        for hz in (1,10,20):
            with self.subTest(hz=hz):
                matches=self.pairs([(-20+10*i/hz,0,i/hz) for i in range(14*hz+1)])
                self.assertEqual(len(matches),1)
                self.assertEqual(matches[0][3]['time'],self.origin+datetime.timedelta(seconds=7))

    def test_complete_boundary_to_boundary_trap_at_nonzero_latitude(self):
        lat=49.
        factor=self.scale*math.cos(math.radians(lat))
        points=[dict(lat=lat,lon=x/factor,time=self.origin+datetime.timedelta(seconds=t))
                for x,t in ((40,0),(60,2),(120,8))]
        matches=self.mod._find_trap_pair_matches(points,(lat,50/factor),(lat,100/factor),10.,10.,True,False)
        self.assertEqual(len(matches),1)
        self.assertAlmostEqual((matches[0][2]['time']-self.origin).total_seconds(),1.,places=5)
        self.assertAlmostEqual((matches[0][3]['time']-self.origin).total_seconds(),5.,places=5)


if __name__ == '__main__':
    unittest.main()
