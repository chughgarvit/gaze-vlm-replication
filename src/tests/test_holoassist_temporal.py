"""Research-input and loss-mask checks; synthetic fixtures are not efficacy data."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_holoassist_temporal import JointEncoder,loss_function,motion_summary,sequence_data


class TemporalStudyTests(unittest.TestCase):
    def test_equal_capacity_and_shapes(self):
        counts=[]
        for _ in range(5):
            model=JointEncoder(99);counts.append(sum(p.numel() for p in model.parameters()))
            a,e=model(torch.zeros(2,4,99))
            self.assertEqual(tuple(a.shape),(2,8));self.assertEqual(tuple(e.shape),(2,2))
        self.assertEqual(len(set(counts)),1)

    def test_unknown_correctness_is_not_a_negative_target(self):
        a=torch.randn(3,8,requires_grad=True);e=torch.randn(3,2,requires_grad=True)
        loss=loss_function((a,e),torch.tensor([0,1,2]),torch.full((3,),-1),torch.ones(8),torch.ones(2))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertIsNone(e.grad)

    def test_motion_does_not_bridge_invalid_pose(self):
        poses=np.tile(np.eye(4),(3,1,1));poses[1]=np.nan;poses[2,0,3]=100
        np.testing.assert_equal(motion_summary(np.array([0,.03,.06]),poses,0,.1),np.zeros(4))

    def test_unobserved_final_gaze_cannot_change_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'features').mkdir()
            np.savez(output/'features/fixture.npz',frames=np.ones((1,4,512)),event_ids=np.array(['1']))
            t=np.arange(31)/30;angles=np.column_stack([np.sin(t),np.cos(t)])
            valid=np.ones(31,bool);poses=np.tile(np.eye(4),(31,1,1))
            row={'id':'fixture:1','recording':'fixture','start':'0','end':'1'}
            with patch('run_holoassist_temporal.pilot.load_recording',return_value=(output,t,angles,valid,t,poses,1)):
                first=sequence_data([row],output)
                (output/'temporal_inputs.npz').unlink()
                angles[t>.85]=1000
                second=sequence_data([row],output)
            for key in first:
                np.testing.assert_allclose(first[key],second[key])


if __name__=='__main__':unittest.main()
