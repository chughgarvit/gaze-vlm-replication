import unittest
import numpy as np
import torch

from gaze_transfer.data import frame_times, gaze_variant, heatmaps, select_frames, window_targets
from gaze_transfer.knowledge import SymbolicState, validate_concepts
from gaze_transfer.model import GazeAttention
from gaze_transfer.personalization import GazeCalibration


class TransferTests(unittest.TestCase):
    def test_future_events_are_labels_only_and_horizon_bounded(self):
        record = {'events': [
            {'id': 1, 'label': 'Fine grained action', 'start': 1., 'end': 4.,
             'attributes': {'Verb': 'grab', 'Noun': 'cup', 'Action Correctness': 'Correct Action'}},
            {'id': 2, 'label': 'Fine grained action', 'start': 6., 'end': 7.,
             'attributes': {'Verb': 'place', 'Noun': 'cup', 'Action Correctness': 'Wrong Action'}},
            {'id': 3, 'label': 'Conversation', 'start': 8., 'end': 9.,
             'attributes': {'Conversation Purpose': 'correct wrong action'}}]}
        result = window_targets(record, 0., 5., 2.)
        self.assertEqual(result['activity']['verb'], 'grab')
        self.assertEqual(result['anticipation']['verb'], 'place')
        self.assertEqual(result['mistake'], 'correct')
        self.assertFalse(result['intervention'])
        self.assertTrue(window_targets(record, 0., 5., 5.)['intervention'])

    def test_unknown_is_not_a_correct_action(self):
        self.assertEqual(window_targets({'events': []}, 0., 5.)['mistake'], 'unknown')

    def test_strict_past_frames(self):
        times = frame_times(10., 15.)
        self.assertEqual(len(times), 4)
        self.assertTrue(all(10 <= t < 15 for t in times))

    def test_shuffle_preserves_time_mask_and_positions(self):
        gaze = np.random.default_rng(0).random((64, 5)).astype('float32')
        gaze[:, 3] = np.arange(64) % 2
        result = gaze_variant(gaze, 'shuffled', 1)
        np.testing.assert_array_equal(result[:, [0, 3, 4]], gaze[:, [0, 3, 4]])
        self.assertEqual(sorted(map(tuple, result[result[:, 3] > 0, 1:3])),
                         sorted(map(tuple, gaze[gaze[:, 3] > 0, 1:3])))

    def test_heatmap_does_not_observe_future_points(self):
        gaze = np.array([[.1, .2, .3, 1, 0], [.9, .8, .9, 1, 0]], np.float32)
        first, mask = heatmaps(gaze, [.5], 4, 0, 5)
        gaze[1, 1:3] = 0
        second, _ = heatmaps(gaze, [.5], 4, 0, 5)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(mask[0])

    def test_budget_and_context_frame(self):
        for budget in [1, 2, 4]:
            selected = select_frames(np.zeros((64, 5)), 4, budget)
            self.assertEqual(len(selected), budget)
            self.assertIn(0, selected)

    def test_missing_gaze_is_neutral_and_capacity_matches(self):
        torch.manual_seed(4)
        layer = GazeAttention(32)
        x = torch.randn(4, 16, 32)
        layer.mode = 'label_only'; control = layer(x)
        layer.gaze = torch.randn(64, 5); layer.gaze[:, 3] = 0
        layer.mode = 'temporal'; missing = layer(x)
        torch.testing.assert_close(control, missing)
        self.assertEqual(layer.attention.shape, (4, 16))
        torch.testing.assert_close(layer.attention.sum(-1), torch.ones(4))

    def test_masked_kl_has_no_invalid_supervision(self):
        layer = GazeAttention(32); layer(torch.randn(4, 16, 32))
        loss = layer.kl(torch.ones(4, 16)/16, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0.)

    def test_teacher_rejects_unobserved_evidence(self):
        c = {'caption': 'cup', 'entities': [{'id': 'e1', 'name': 'cup', 'frame': 4,
              'box': [0, 0, 1, 1], 'confidence': .8}]}
        with self.assertRaises(ValueError):
            validate_concepts(c, 4)

    def test_symbolic_unknown_and_chronology(self):
        c = validate_concepts({'caption': 'cup', 'entities': [{'id': 'e1', 'name': 'cup', 'frame': 0,
            'box': [0, 0, 1, 1], 'confidence': .8}], 'facts': [{'subject': 'e1', 'predicate': 'full',
            'value': 'unknown', 'frame': 0, 'confidence': .9}]}, 4)
        ledger = SymbolicState(); state = ledger.update(c, 5)
        self.assertEqual(state['predicates']['cup::full']['value'], 'unknown')
        with self.assertRaises(ValueError):
            ledger.update(c, 4)

    def test_personalization_updates_after_observation_and_is_bounded(self):
        calibration = GazeCalibration([.5, .5], [.2, .2], max_samples=4)
        gaze = np.array([[0., .3, .5, 1, 0], [1., .7, .5, 1, 0]], np.float32)
        initial = calibration.transform(gaze)
        np.testing.assert_allclose(initial[:, 1], [-1, 1], atol=1e-6)
        self.assertEqual(calibration.samples, [])
        for _ in range(8):
            calibration.observe(gaze)
        self.assertEqual(len(calibration.samples), 4)

    def test_symbolic_transition_requires_temporal_support(self):
        concepts = {'caption': 'cup', 'entities': [{'id': 'e1', 'name': 'cup', 'frame': 0,
            'box': [0, 0, 1, 1], 'confidence': .9}], 'facts': [
            {'subject': 'e1', 'predicate': 'held', 'value': 'false', 'frame': 0, 'confidence': .9},
            {'subject': 'e1', 'predicate': 'held', 'value': 'true', 'frame': 1, 'confidence': .9}]}
        no_edge = SymbolicState().update(validate_concepts(concepts, 4), 5)
        self.assertEqual(no_edge['predicates']['cup::held']['value'], 'unknown')
        concepts['relations'] = [{'before': 0, 'after': 1, 'predicate': 'visible_change', 'confidence': .9}]
        with_edge = SymbolicState().update(validate_concepts(concepts, 4), 5)
        self.assertEqual(with_edge['predicates']['cup::held']['value'], 'true')


if __name__ == '__main__':
    unittest.main()
