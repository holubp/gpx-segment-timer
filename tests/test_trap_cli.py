"""Trap midpoint timing through the real CLI and GPX exporter."""
import datetime
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


class TrapCliTests(unittest.TestCase):
    def test_missing_midpoint_time_is_not_replaced_by_raw_sample_duration(self):
        scale=6371000.*math.pi/180.
        origin=datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); refs=root/'refs'; refs.mkdir()
            waypoints=''.join(f'<wpt lat="0" lon="{x/scale}"><name>{name}</name></wpt>'
                              for name,x in (('start',0),('trap1',50),('finish',100)))
            (refs/'markers.gpx').write_text(f'<gpx version="1.1" creator="test">{waypoints}</gpx>')
            content=''
            for x,t in ((-20,0),(20,4),(50,None),(80,10),(120,14)):
                time_xml='' if t is None else '<time>'+(origin+datetime.timedelta(seconds=t)).isoformat()+'</time>'
                content+=f'<trkpt lat="0" lon="{x/scale}">{time_xml}</trkpt>'
            (root/'recorded.gpx').write_text(f'<gpx version="1.1" creator="test"><trk><trkseg>{content}</trkseg></trk></gpx>')
            proc=subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(root/'recorded.gpx'),'-f',str(refs)],
                                capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1],timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows=[re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if re.match(r'^match\d+\s',line)]
            parts=[r for r in rows if ':' in r[1]]
            self.assertEqual(len(parts),2)
            self.assertTrue(all(r[-1]=='N/A' and r[-2]=='-1.00' for r in parts),proc.stdout)

    def test_multiple_traps_and_tangent_are_not_suppressed_by_shared_brackets(self):
        scale=6371000.*math.pi/180.
        origin=datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        cases=[([('trap1',30,0),('trap2',60,0)],0,['2.00','3.00','3.00'],'8.00'),
               ([('trap1',50,16)],6,['4.20','4.20'],'8.40')]
        for traps,y,expected,total in cases:
            with self.subTest(traps=len(traps),y=y),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); refs=root/'refs'; refs.mkdir()
                markers=[('start',0,0),*traps,('finish',100,0)]
                waypoints=''.join(f'<wpt lat="{north/scale}" lon="{east/scale}"><name>{name}</name></wpt>' for name,east,north in markers)
                (refs/'markers.gpx').write_text(f'<gpx version="1.1" creator="test">{waypoints}</gpx>')
                points=''.join(f'<trkpt lat="{y/scale}" lon="{x/scale}"><time>{(origin+datetime.timedelta(seconds=t)).isoformat()}</time></trkpt>' for x,t in ((-20,0),(120,14)))
                (root/'recorded.gpx').write_text(f'<gpx version="1.1" creator="test"><trk><trkseg>{points}</trkseg></trk></gpx>')
                proc=subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-r',str(root/'recorded.gpx'),'-f',str(refs)],
                                    capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1],timeout=30)
                self.assertEqual(proc.returncode,0,proc.stderr)
                rows=[re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if re.match(r'^match\d+\s',line)]
                full=[r for r in rows if ':' not in r[1]]
                parts=[r for r in rows if ':' in r[1]]
                self.assertEqual(len(full),1,proc.stdout)
                self.assertEqual(full[0][-2],total)
                self.assertEqual([r[-2] for r in parts],expected)

    def test_trap_subsegments_share_midpoint_and_export_entry_exit(self):
        scale=6371000.*math.pi/180.
        origin=datetime.datetime(2025,1,1,tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); refs=root/'refs'; refs.mkdir()
            waypoints=''.join(f'<wpt lat="0" lon="{x/scale}"><name>{name}</name></wpt>'
                              for name,x in (('start',0),('trap1',50),('finish',100)))
            (refs/'markers.gpx').write_text(f'<gpx version="1.1" creator="test">{waypoints}</gpx>')
            points=''.join(f'<trkpt lat="0" lon="{x/scale}"><time>{(origin+datetime.timedelta(seconds=t)).isoformat()}</time></trkpt>'
                           for x,t in ((-20,0),(120,14)))
            (root/'recorded.gpx').write_text(f'<gpx version="1.1" creator="test"><trk><trkseg>{points}</trkseg></trk></gpx>')
            proc=subprocess.run([sys.executable,'-B','gpx-segment-timer.py','-v','-r',str(root/'recorded.gpx'),
                                 '-f',str(refs),'--export-gpx','--export-gpx-file',str(root/'matches.gpx')],
                                capture_output=True,text=True,cwd=Path(__file__).resolve().parents[1],timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr)
            rows=[re.split(r'\s{2,}',line.strip()) for line in proc.stdout.splitlines() if re.match(r'^match\d+\s',line)]
            by_segment={r[1]:r for r in rows}
            self.assertEqual(set(by_segment),{'markers.gpx','markers.gpx:start->trap1','markers.gpx:trap1->finish'})
            self.assertEqual(by_segment['markers.gpx'][13],'8.00')
            self.assertEqual(by_segment['markers.gpx:start->trap1'][13],'4.00')
            self.assertEqual(by_segment['markers.gpx:trap1->finish'][13],'4.00')
            ns={'g':'http://www.topografix.com/GPX/1/1'}
            trap_exports=[]
            for path in root.glob('*_match*.gpx'):
                tracks={t.find('g:name',ns).text:t for t in ET.parse(path).getroot().findall('g:trk',ns)}
                if 'finish crossing trap entry' in tracks:
                    trap_exports.append(tracks)
                    for role,seconds in (('entry',6),('exit',8)):
                        track=tracks[f'finish crossing trap {role}']
                        times=track.findall('g:trkseg/g:trkpt/g:time',ns)
                        self.assertEqual(datetime.datetime.fromisoformat(times[1].text.replace('Z','+00:00')),
                                         origin+datetime.timedelta(seconds=seconds))
                    mids=tracks['finish crossing'].findall('g:trkseg/g:trkpt',ns)
                    self.assertEqual(len(mids),3)
                    self.assertAlmostEqual(float(mids[1].get('lon'))*scale,50.,places=5)
            self.assertEqual(len(trap_exports),1)


if __name__ == '__main__':
    unittest.main()
