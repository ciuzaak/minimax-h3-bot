import unittest

from minimax_h3.jobs import (
    enforce_no_background_music,
    frame_count,
    image_resolution,
)


class JobHelpersTest(unittest.TestCase):
    def test_frame_count_uses_h3_grid(self):
        for seconds in (0.2, 5, 10, 15):
            frames = frame_count(seconds)
            self.assertGreaterEqual(frames, 5)
            self.assertEqual(frames % 17, 5)

    def test_explicit_resolution_requires_multiples_of_32(self):
        self.assertEqual(image_resolution("unused", 576, 768), (576, 768))
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            image_resolution("unused", 575, 768)

    def test_background_music_is_forced_off_once(self):
        prompt = "Scene\nnon_diegetic_music: loud orchestra\nAudio: rain"
        result = enforce_no_background_music(prompt)
        self.assertNotIn("loud orchestra", result)
        self.assertEqual(result.count("non_diegetic_music:"), 1)
        self.assertTrue(result.endswith("non_diegetic_music: N/A"))


if __name__ == "__main__":
    unittest.main()

