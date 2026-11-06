import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gaze_behavior.data import (build_sequence, class_matched_support, gaze_features,
                                load_manifest, personalization_split, subject_split)
from gaze_behavior.model import BehaviorModel, adapt_head, normalization, paired_interval


class BehaviorTests(unittest.TestCase):
    def arrays(self):
        return {"visual": np.ones((3, 4), np.float32), "frame_time": np.array([0, 0.5, 1.0]),
                "gaze_time": np.array([0, 0.25, 0.5, 0.75, 1.0]),
                "gaze_xy": np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.1], [0.5, 0.5], [0.6, 0.6]]),
                "gaze_valid": np.ones(5, bool)}

    def test_features_do_not_observe_future_gaze(self):
        a = self.arrays()
        first = gaze_features(a["frame_time"], a["gaze_time"], a["gaze_xy"], a["gaze_valid"])
        a["gaze_xy"][-1] = [1, 1]
        second = gaze_features(a["frame_time"], a["gaze_time"], a["gaze_xy"], a["gaze_valid"])
        np.testing.assert_equal(first[:2], second[:2])
        self.assertFalse(np.array_equal(first[-1], second[-1]))

    def test_missing_samples_are_masked_not_center_imputed(self):
        a = self.arrays()
        a["gaze_valid"][:] = False
        a["gaze_xy"][:] = 0.5
        result = gaze_features(a["frame_time"], a["gaze_time"], a["gaze_xy"], a["gaze_valid"])
        self.assertEqual(result[:, :4].sum(), 0)

    def test_pixel_return_and_gap_are_not_object_fixations(self):
        a = self.arrays()
        result = gaze_features(a["frame_time"], a["gaze_time"], a["gaze_xy"], a["gaze_valid"], max_age=0.3)
        self.assertGreater(result[1, 7], 0)
        a["gaze_valid"][1] = False
        result = gaze_features(a["frame_time"], a["gaze_time"], a["gaze_xy"], a["gaze_valid"], max_age=0.3)
        self.assertEqual(result[1, 6], 0)

    def test_visual_baseline_receives_same_time_information(self):
        a = self.arrays()
        visual = build_sequence(a, "visual", "sample", 0)
        temporal = build_sequence(a, "temporal_gaze", "sample", 0)
        np.testing.assert_equal(visual[:, [8, 15]], temporal[:, [8, 15]])
        np.testing.assert_equal(visual[:, 4:8], np.zeros((3, 4)))

    def test_shuffle_is_reproducible_and_preserves_input(self):
        a = self.arrays()
        original = a["gaze_xy"].copy()
        first = build_sequence(a, "shuffled_gaze", "sample", 0)
        second = build_sequence(a, "shuffled_gaze", "sample", 0)
        np.testing.assert_equal(first, second)
        np.testing.assert_equal(a["gaze_xy"], original)

    def test_subject_split_and_later_session_query(self):
        examples = [dict(id=f"{s}-{o}-{c}", subject_id=s, session_id=f"{s}-{o}", session_order=o,
                         start_s=float(c), cutoff_s=float(c + 1.5))
                    for s in ("a", "b", "c") for o in range(2) for c in range(3)]
        train, dev, test = subject_split(examples, "c", "b")
        self.assertTrue(all(examples[i]["subject_id"] == "a" for i in train))
        support, query = personalization_split(examples, test)
        self.assertEqual(len(support), 2)  # overlapping middle window excluded
        self.assertTrue(all(examples[i]["session_order"] == 0 for i in support))
        self.assertTrue(all(examples[i]["session_order"] == 1 for i in query))
        self.assertTrue(set(support).isdisjoint(query))
        with self.assertRaises(ValueError):
            subject_split(examples, "c", "c")

    def test_normalization_uses_training_only(self):
        seq = [np.zeros((2, 3), np.float32), np.full((2, 3), 1000, np.float32)]
        mean, std = normalization(seq, [0])
        np.testing.assert_equal(mean, [0, 0, 0])
        self.assertTrue((std < 0.001).all())

    def test_adaptation_changes_only_copied_head(self):
        torch.manual_seed(0)
        model = BehaviorModel(4, 4, 2)
        old = copy.deepcopy(model.state_dict())
        head = adapt_head(model.head, torch.ones((2, 4)), torch.tensor([1, 1]), steps=3)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, old[key]))
        self.assertFalse(torch.equal(head.weight, model.head.weight))

    def test_bias_only_control_preserves_weights(self):
        torch.manual_seed(0)
        model = BehaviorModel(4, 4, 2)
        head = adapt_head(model.head, torch.ones((2, 4)), torch.tensor([1, 1]), steps=3, bias_only=True)
        self.assertTrue(torch.equal(head.weight, model.head.weight))
        self.assertFalse(torch.equal(head.bias, model.head.bias))

    def test_wrong_user_calibration_matches_label_distribution(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        selected = class_matched_support(labels, [0, 1], [4, 5])
        np.testing.assert_equal(labels[selected], labels[[0, 1]])
        self.assertEqual(class_matched_support(labels, [0, 0], [4, 5]), [])

    def test_bootstrap_averages_seeds_before_wearers(self):
        rows = [dict(subject=s, mode=m, macro_f1_present=v + seed * 0.01)
                for s in ("a", "b", "c") for m, v in (("gaze", 0.7), ("visual", 0.5)) for seed in range(3)]
        result = paired_interval(rows, "gaze", "visual")
        self.assertEqual(result["subjects"], 3)
        self.assertAlmostEqual(result["delta"], 0.2)

    def test_manifest_rejects_future_data_and_unverified_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.arrays()
            np.savez(root / "features.npz", **a)
            manifest = {"synthetic": True, "task": "recognition", "classes": ["a", "b"],
                        "audit": {k: True for k in ("alignment_verified", "subject_identity_verified",
                                                   "session_order_verified", "labels_independent_of_gaze")},
                        "sessions": [{"subject_id": s, "session_id": s, "order": 0} for s in ("a", "b", "c")],
                        "samples": [{"id": s, "subject_id": s, "session_id": s, "start_s": 0,
                                     "cutoff_s": 1, "label": 0, "features": "features.npz"} for s in ("a", "b", "c")]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            load_manifest(path, True)
            with self.assertRaises(ValueError):
                load_manifest(path, False)
            manifest["samples"][0]["cutoff_s"] = 0.75
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path, True)
            manifest["samples"][0]["cutoff_s"] = 1
            manifest["audit"]["session_order_verified"] = False
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path, True)

    def test_anticipation_requires_positive_observed_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "features.npz", **self.arrays())
            manifest = {"synthetic": True, "task": "anticipation", "anticipation_horizon_s": 1,
                        "classes": ["a", "b"],
                        "audit": {k: True for k in ("alignment_verified", "subject_identity_verified",
                                                   "session_order_verified", "labels_independent_of_gaze")},
                        "sessions": [{"subject_id": s, "session_id": s, "order": 0} for s in ("a", "b", "c")],
                        "samples": [{"id": s, "subject_id": s, "session_id": s, "start_s": 0,
                                     "cutoff_s": 1, "target_start_s": 1.5, "label": 0,
                                     "features": "features.npz"} for s in ("a", "b", "c")]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path, True)


if __name__ == "__main__":
    unittest.main()
