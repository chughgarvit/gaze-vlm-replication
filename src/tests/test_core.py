import unittest
from gaze_memory.core import BoundedMemory, Observation, corrupt_roles


def obs(uid, time, name, role="primary"):
    return Observation(uid, "v", int(time), time-1, time, 1, name, "description", role)


class MemoryTests(unittest.TestCase):
    def test_no_future_and_monotonic_input(self):
        memory = BoundedMemory(2, "fifo")
        memory.add(obs("a", 5, "cup"))
        with self.assertRaises(ValueError):
            memory.snapshot(4)
        with self.assertRaises(ValueError):
            memory.add(obs("b", 3, "bowl"))

    def test_all_storage_is_bounded(self):
        for policy in ["fifo", "reservoir", "category_recency", "gaze_priority", "gaze_context"]:
            memory = BoundedMemory(4, policy)
            for i in range(50):
                memory.add(obs(str(i), i+1, f"obj{i}", "peripheral" if i%3 else "primary"))
                self.assertLessEqual(len(memory.records), 4)
                self.assertLessEqual(len(memory.keys), 4)

    def test_updated_category_replaces_stale_record(self):
        memory = BoundedMemory(2, "gaze_priority")
        memory.add(obs("a", 2, "cup"))
        memory.add(obs("b", 3, "cup", "peripheral"))
        self.assertEqual([o.uid for o in memory.records], ["b"])

    def test_context_quota_and_backfill(self):
        memory = BoundedMemory(4, "gaze_context")
        for i in range(10):
            memory.add(obs(str(i), i+1, f"obj{i}", "peripheral" if i==0 else "primary"))
        self.assertEqual(len(memory.records), 4)
        self.assertTrue(any(o.uid == "0" for o in memory.records))

    def test_corruption_leaves_visual_evidence_unchanged(self):
        original = [obs("a", 2, "cup"), obs("b", 2, "chair", "peripheral")]
        corrupted = corrupt_roles(original, 1, 0)
        self.assertEqual({o.uid:o.text for o in original}, {o.uid:o.text for o in corrupted})
        self.assertEqual(sorted(o.role for o in original), sorted(o.role for o in corrupted))


if __name__ == "__main__":
    unittest.main()
