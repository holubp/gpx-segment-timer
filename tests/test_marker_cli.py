"""End-to-end marker timing/export regressions (requires CLI dependencies)."""
import re
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


class MarkerCliTests(unittest.TestCase):
    def test_shape_elapsed_time_is_consistent_across_recording_rates(self):
        import datetime
        scale = 6371000. * 3.141592653589793 / 180.
        origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        def gpx(xs):
            content = ''.join(f'<trkpt lat="0" lon="{x/scale}"><time>{(origin+datetime.timedelta(seconds=(x+30)/10)).isoformat()}</time></trkpt>' for x in xs)
            return f'<gpx version="1.1" creator="test"><trk><trkseg>{content}</trkseg></trk></gpx>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root/'refs'
            refs.mkdir()
            (refs/'shape.gpx').write_text(gpx(range(0,101,10)))
            for hz in (1,10,20):
                with self.subTest(hz=hz):
                    (root/'recorded.gpx').write_text(gpx([10*i/hz for i in range(-3*hz,13*hz+1)]))
                    proc = subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(root/'recorded.gpx'),'-f',str(refs)],
                                          capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
                    self.assertEqual(proc.returncode,0,proc.stderr)
                    rows = [re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if line.startswith('match1')]
                    self.assertEqual(len(rows),1,proc.stderr)
                    self.assertEqual(rows[0][-2],'10.00')

    def test_shape_search_may_reach_recording_end_without_crashing(self):
        import datetime
        scale = 6371000. * 3.141592653589793 / 180.
        origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        def gpx(xs):
            points = ''.join(f'<trkpt lat="0" lon="{x/scale}"><time>{(origin+datetime.timedelta(seconds=x+30)).isoformat()}</time></trkpt>' for x in xs)
            return f'<gpx version="1.1" creator="test"><trk><trkseg>{points}</trkseg></trk></gpx>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root/'refs'
            refs.mkdir()
            (refs/'shape.gpx').write_text(gpx(range(101)))
            (root/'recorded.gpx').write_text(gpx(range(-30,132)))
            proc = subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(root/'recorded.gpx'),'-f',str(refs)],
                                  capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows = [re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if line.startswith('match1')]
            self.assertEqual(len(rows),1,proc.stderr)
            self.assertEqual(rows[0][-2],'100.00')

    def test_circle_length_gate_does_not_depend_on_bracket_placement(self):
        scale = 6371000. * 3.141592653589793 / 180.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root/'refs'
            refs.mkdir()
            (refs/'circle.gpx').write_text(
                f'<gpx version="1.1" creator="test"><wpt lat="0" lon="0"><name>start</name></wpt>'
                f'<wpt lat="0" lon="{100/scale}"><name>finish</name></wpt></gpx>')
            for xs in (list(range(-20,121)),[0,89,100,120]):
                with self.subTest(samples=len(xs)):
                    import datetime
                    origin = datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
                    recorded = root/'recorded.gpx'
                    content = ''.join(f'<trkpt lat="0" lon="{x/scale}"><time>{(origin+datetime.timedelta(seconds=x+20)).isoformat()}</time></trkpt>' for x in xs)
                    recorded.write_text(f'<gpx version="1.1" creator="test"><trk><trkseg>{content}</trkseg></trk></gpx>')
                    proc = subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(recorded),'-f',str(refs)],
                                          capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
                    self.assertEqual(proc.returncode,0,proc.stderr)
                    rows = [re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if line.startswith('match1')]
                    self.assertEqual(len(rows),1,proc.stderr)
                    self.assertEqual(rows[0][-2],'80.00')
                    self.assertEqual(rows[0][3],'80.00')

    def test_single_edge_passage_at_recording_end_exports_and_prints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root / 'refs'
            refs.mkdir()
            scale = 6371000. * 3.141592653589793 / 180.
            (refs/'circle.gpx').write_text(
                f'<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
                f'<wpt lat="0" lon="0"><name>start</name></wpt>'
                f'<wpt lat="0" lon="{100/scale}"><name>finish</name></wpt></gpx>')
            recorded = root/'recorded.gpx'
            recorded.write_text(
                f'<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
                f'<trkpt lat="0" lon="{-20/scale}"><time>2025-01-01T00:00:00Z</time></trkpt>'
                f'<trkpt lat="0" lon="{120/scale}"><time>2025-01-01T00:00:14Z</time></trkpt>'
                f'</trkseg></trk></gpx>')
            proc = subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(recorded),
                                   '-f',str(refs),'--export-gpx','--export-gpx-file',str(root/'matched.gpx')],
                                  capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows = [re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if line.startswith('match1')]
            self.assertEqual(len(rows),1,proc.stdout)
            self.assertEqual(rows[0][-2],'8.00')
            exports = list(root.glob('*_match1.gpx'))
            self.assertEqual(len(exports),1)
            ns = {'g':'http://www.topografix.com/GPX/1/1'}
            tracks = ET.parse(exports[0]).getroot().findall('g:trk',ns)
            self.assertEqual(len(tracks[0].findall('g:trkseg/g:trkpt',ns)),2)
            crossing_tracks = {t.find('g:name',ns).text:t for t in tracks}
            for name,second in (('start crossing',3),('finish crossing',11)):
                times = crossing_tracks[name].findall('g:trkseg/g:trkpt/g:time',ns)
                self.assertEqual(len(times),3)
                self.assertEqual(datetime.datetime.fromisoformat(times[1].text.replace('Z','+00:00')),
                                 datetime.datetime(2025,1,1,0,0,second,tzinfo=datetime.timezone.utc))


if __name__ == '__main__':
    unittest.main()
