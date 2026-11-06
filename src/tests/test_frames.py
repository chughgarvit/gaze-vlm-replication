import unittest
from PIL import Image
from gaze_memory.frames import MODES, prepare_frame, select_past_frames, usable_gaze


class FrameTests(unittest.TestCase):
    def test_matched_canvas_and_edge_crop(self):
        image=Image.new("RGB",(1280,960),"white")
        for mode in MODES:
            for x,y in [(0,0),(1,1),(.5,.5)]:
                result=prepare_frame(image,{"x":x,"y":y,"confidence":1},mode)
                self.assertEqual(result.size,(448,448))

    def test_invalid_gaze_fallback(self):
        self.assertFalse(usable_gaze({"x":float("nan"),"y":.2,"confidence":1}))
        self.assertFalse(usable_gaze({"x":.5,"y":.5,"confidence":.1}))
        image=Image.new("RGB",(100,80),"white")
        self.assertEqual(prepare_frame(image,None,"full").tobytes(),
                         prepare_frame(image,None,"gaze_crop").tobytes())

    def test_cutoff(self):
        frames=[{"timestamp_s":t} for t in [0,1,2,3,4]]
        selected=select_past_frames(frames,0,2.5,3)
        self.assertTrue(all(f["timestamp_s"]<=2.5 for f in selected))
        self.assertNotIn(frames[3],selected)

    def test_single_frame_uses_last_observed_frame(self):
        selected=select_past_frames([{"timestamp_s":t} for t in [1,2,3]],.5,2.5,1)
        self.assertEqual(selected,[{"timestamp_s":2}])


if __name__=="__main__":
    unittest.main()
