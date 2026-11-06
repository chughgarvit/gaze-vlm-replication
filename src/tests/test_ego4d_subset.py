import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("audit_ego4d_subset", Path(__file__).resolve().parents[1] / "scripts/audit_ego4d_subset.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Ego4dSubsetTests(unittest.TestCase):
    def metadata(self):
        return {"version": "SYNTHETIC_SCHEMA_FIXTURE", "videos": [
            {"video_uid": f"fixture-{person}-{recording}", "has_gaze": True,
             "fb_participant_id": person, "video_source": "fixture-source",
             "origin_video_id": f"recording{recording}", "duration_sec": 10}
            for person in range(3) for recording in range(2)]}

    def annotations(self, metadata):
        return {"fixture_action_annotations": {"clips": [{"video_uid": v["video_uid"], "label": "fixture"}
                                                         for v in metadata["videos"]]}}

    def test_joint_coverage_selection_stays_small_and_not_research_ready(self):
        m = self.metadata()
        report, rows = audit.inspect_metadata(m, self.annotations(m), max_videos=6)
        self.assertEqual(len(report["candidate_video_uids"]), 6)
        self.assertEqual(report["selected_candidate_wearers"], 3)
        self.assertFalse(report["research_ready"])
        self.assertFalse(report["chronology_verified"])

    def test_missing_gaze_identity_or_annotation_excludes_candidates(self):
        m = self.metadata()
        annotations = self.annotations(m)
        m["videos"][0]["has_gaze"] = False
        m["videos"][2]["fb_participant_id"] = None
        annotations["fixture_action_annotations"]["clips"].pop()
        report, _ = audit.inspect_metadata(m, annotations)
        self.assertEqual(report["candidate_video_uids"], [])

    def test_two_redacted_pieces_of_same_origin_are_not_two_recordings(self):
        m = self.metadata()
        for v in m["videos"]:
            v["origin_video_id"] = "one-capture"
        report, _ = audit.inspect_metadata(m, self.annotations(m))
        self.assertEqual(report["repeat_origin_candidate_wearers"], 0)
        self.assertEqual(report["candidate_video_uids"], [])

    def test_no_annotations_never_produces_video_selection(self):
        report, _ = audit.inspect_metadata(self.metadata(), {})
        self.assertEqual(report["candidate_video_uids"], [])

    def test_narration_root_keys_and_video_fields_are_both_supported(self):
        known = {"fixture-1", "fixture-2"}
        references = audit.annotation_video_refs({"fixture-1": {"narrations": []},
                                                 "clips": [{"video_uid": "fixture-2"},
                                                           {"video_uid": "unrelated"}]}, known)
        self.assertEqual(references, known)


if __name__ == "__main__":
    unittest.main()
