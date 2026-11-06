"""Checks for the exploratory budget/context diagnostics."""
import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_holoassist_hypotheses import confusion_f1,feature_parts


class HypothesisTests(unittest.TestCase):
    def test_fixed_vocabulary_f1_matches_reference(self):
        y=np.array([0,0,1,2,2,4,7]);pred=np.array([0,1,1,3,2,4,0])
        self.assertAlmostEqual(confusion_f1(y,pred),f1_score(y,pred,labels=list(range(8)),average='macro',zero_division=0))

    def test_oracle_does_not_change_gaze_feature_selection(self):
        for mode in ['control','static','temporal','shuffled']:
            self.assertEqual(feature_parts(mode),feature_parts('oracle_'+mode))

    def test_controls_are_shared_and_temporal_ablation_is_explicit(self):
        self.assertEqual(feature_parts('control'),['duration','quality','motion'])
        self.assertIn('static',feature_parts('shuffled'))
        self.assertNotIn('dynamic',feature_parts('shuffled'))
        self.assertIn('shuffled',feature_parts('shuffled'))
        self.assertEqual(set(feature_parts('temporal'))-set(feature_parts('static')),{'dynamic'})


if __name__=='__main__':
    unittest.main()
