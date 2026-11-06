"""Numerical/protocol checks; fixtures are not research observations."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_holoassist_pilot import gaze_summary, write_csv
from download_holoassist import recording


class HoloAssistPilotTests(unittest.TestCase):
    def test_shuffle_preserves_static_and_quality(self):
        t=np.arange(100)/30
        a=np.column_stack([np.sin(t),np.cos(t)])
        valid=np.ones(100,bool)
        q,s,d=gaze_summary(t,a,valid,0,3.3)
        qq,ss,dd=gaze_summary(t,a,valid,0,3.3,True,13)
        np.testing.assert_allclose(q,qq)
        np.testing.assert_allclose(s,ss)
        self.assertGreater(np.linalg.norm(d-dd),1)

    def test_future_samples_do_not_change_summary(self):
        t=np.arange(100)/30
        a=np.column_stack([t,t*.1])
        valid=np.ones(100,bool)
        original=gaze_summary(t,a,valid,0,1)
        a[t>1]=1000
        modified=gaze_summary(t,a,valid,0,1)
        for x,y in zip(original,modified):
            np.testing.assert_allclose(x,y)

    def test_missing_gaze_is_finite(self):
        values=gaze_summary(np.arange(10)/30,np.zeros((10,2)),np.zeros(10,bool),0,1)
        self.assertTrue(all(np.isfinite(v).all() for v in values))
        self.assertTrue(all((v==0).all() for v in values[1:]))

    def test_mixed_metrics_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'metrics.csv'
            write_csv(p,[{'task':'verb','f1':.2},{'task':'mistake','f1':.3,'ap':.1}])
            self.assertIn('ap',p.read_text().splitlines()[0])

    def test_recording_paths(self):
        self.assertEqual(recording('./R142-31Aug-RAM/Export_py/Video.mp4'),'R142-31Aug-RAM')
        self.assertEqual(recording('z095-july-11-22-gopro/Export_py/Eyes/Eyes_sync.txt'),'z095-july-11-22-gopro')


if __name__=='__main__':
    unittest.main()
