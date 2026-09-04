import unittest

import numpy as np

from minicpmo_dflash.timeline import (
    _max_subtitle_units,
    _split_text_by_units,
    analyze_windows,
    build_subtitle_cues,
    display_units,
    format_srt_time,
    mix_timeline,
    write_srt,
)

SAMPLE_RATE = 24000


def _silence(samples: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


def _padded(pad_seconds: float, pcm_seconds: float) -> np.ndarray:
    pad = np.zeros(int(pad_seconds * SAMPLE_RATE), dtype=np.float32)
    pcm = np.full(int(pcm_seconds * SAMPLE_RATE), 0.5, dtype=np.float32)
    return np.concatenate([pad, pcm])


def _record(
    chunk_idx: int,
    *,
    is_listen: bool = False,
    text: str = "",
    end_of_turn: bool = False,
    codec_tokens: int = 0,
) -> dict:
    return {
        "chunk_idx": chunk_idx,
        "is_listen": is_listen,
        "text": text,
        "end_of_turn": end_of_turn,
        "tts_codec_tokens": codec_tokens,
    }


class AnalyzeWindowsTest(unittest.TestCase):
    def test_pad_and_stream_positions(self) -> None:
        timed = [
            (0, _silence()),
            (1, _padded(0.8, 0.2)),
            (2, np.full(SAMPLE_RATE, 0.5, dtype=np.float32)),
        ]
        windows = analyze_windows(timed)
        self.assertEqual([w.chunk_index for w in windows], [1, 2])
        self.assertAlmostEqual(windows[0].start, 2.8)
        self.assertEqual(windows[0].committed, int(0.2 * SAMPLE_RATE))
        self.assertEqual(windows[0].stream_pos, 0)
        self.assertAlmostEqual(windows[1].start, 3.0)
        self.assertEqual(windows[1].stream_pos, int(0.2 * SAMPLE_RATE))

    def test_silence_only_session(self) -> None:
        self.assertEqual(analyze_windows([(0, _silence()), (1, _silence())]), [])


class SubtitleCueTest(unittest.TestCase):
    def test_padded_turn_start_shifts_and_merges_cues(self) -> None:
        timed = [
            (0, _silence()),
            (1, _silence()),
            (2, _padded(0.8, 0.2)),
            (3, np.full(SAMPLE_RATE, 0.5, dtype=np.float32)),
        ]
        results_log = [
            _record(0, is_listen=True),
            _record(1, is_listen=True),
            _record(2, text="你好", codec_tokens=5),
            _record(3, text="世界", end_of_turn=True, codec_tokens=25),
        ]
        cues = build_subtitle_cues(results_log, timed, video_duration=10.0)
        # Speech only becomes audible at 3.8s; the two texts merge because
        # their spans are contiguous.
        self.assertEqual(len(cues), 1)
        start, end, text = cues[0]
        self.assertAlmostEqual(start, 3.8, places=3)
        self.assertAlmostEqual(end, 5.0, places=3)
        self.assertEqual(text, "你好世界")

    def test_codec_tokens_cover_lookahead_deferred_audio(self) -> None:
        # 15 codec tokens = 0.6s of speech, but the window only commits
        # 0.2s; the remaining 0.4s plays at the start of the next window.
        timed = [
            (0, _silence()),
            (1, _padded(0.8, 0.2)),
            (2, np.full(SAMPLE_RATE, 0.5, dtype=np.float32)),
        ]
        results_log = [
            _record(0, is_listen=True),
            _record(1, text="三个折线图", codec_tokens=15),
            _record(2, text="", end_of_turn=True, codec_tokens=25),
        ]
        cues = build_subtitle_cues(results_log, timed, video_duration=10.0)
        self.assertEqual(len(cues), 1)
        start, end, _ = cues[0]
        self.assertAlmostEqual(start, 2.8, places=3)
        self.assertAlmostEqual(end, 3.4, places=3)

    def test_without_codec_stats_falls_back_to_committed_audio(self) -> None:
        timed = [(0, _silence()), (1, _padded(0.8, 0.2))]
        results_log = [_record(0, is_listen=True), _record(1, text="你好", end_of_turn=True)]
        cues = build_subtitle_cues(results_log, timed, video_duration=10.0)
        self.assertEqual(len(cues), 1)
        start, end, text = cues[0]
        self.assertAlmostEqual(start, 2.8, places=3)
        self.assertAlmostEqual(end, 3.0, places=3)
        self.assertEqual(text, "你好")

    def test_turn_boundary_resyncs_stream_position(self) -> None:
        # The first turn flushes more audio than its codec tokens account
        # for; the second turn must still map to its own window.
        timed = [
            (0, _silence()),
            (1, _padded(0.0, 0.72)),  # turn 1 flush: 0.72s committed
            (2, _silence()),
            (3, _silence()),
            (4, _padded(0.8, 0.2)),  # turn 2 starts late in its window
        ]
        results_log = [
            _record(0, is_listen=True),
            _record(1, text="第一轮", end_of_turn=True, codec_tokens=11),
            _record(2, is_listen=True),
            _record(3, is_listen=True),
            _record(4, text="第二轮", end_of_turn=True, codec_tokens=7),
        ]
        cues = build_subtitle_cues(results_log, timed, video_duration=10.0)
        self.assertEqual(len(cues), 2)
        # Turn 1 spans its whole flushed audio; turn 2 starts when its
        # window becomes audible and cannot spill into other turns.
        self.assertAlmostEqual(cues[0][0], 2.0, places=3)
        self.assertAlmostEqual(cues[0][1], 2.72, places=3)
        self.assertAlmostEqual(cues[1][0], 5.8, places=3)
        self.assertAlmostEqual(cues[1][1], 6.0, places=3)

    def test_clamps_and_drops_cues_beyond_video(self) -> None:
        timed = [(0, _padded(0.8, 0.2))]
        results_log = [_record(0, text="你好", end_of_turn=True, codec_tokens=5)]
        clamped = build_subtitle_cues(results_log, timed, video_duration=1.95)
        self.assertEqual(len(clamped), 1)
        start, end, _ = clamped[0]
        self.assertAlmostEqual(start, 1.8, places=3)
        self.assertAlmostEqual(end, 1.95, places=3)
        dropped = build_subtitle_cues(results_log, timed, video_duration=1.5)
        self.assertEqual(dropped, [])

    def test_special_tokens_are_stripped(self) -> None:
        timed = [(0, _padded(0.0, 1.0))]
        results_log = [_record(0, text="你<|chunk_eos|>好", end_of_turn=True, codec_tokens=25)]
        cues = build_subtitle_cues(results_log, timed, video_duration=10.0)
        self.assertEqual(cues[0][2], "你好")


class MixTimelineTest(unittest.TestCase):
    def test_chunks_land_on_current_time_grid(self) -> None:
        timed = [(0, _silence()), (1, _padded(0.8, 0.2))]
        mixed = mix_timeline(timed, number_of_chunks=2)
        # Window for chunk 1 occupies second 2, speech at its 0.8s offset.
        nonzero = np.flatnonzero(mixed != 0.0)
        self.assertAlmostEqual(nonzero[0] / SAMPLE_RATE, 2.8, places=3)
        self.assertAlmostEqual(nonzero[-1] / SAMPLE_RATE, 3.0 - 1 / SAMPLE_RATE, places=3)

    def test_length_covers_last_response_window(self) -> None:
        timed = [(0, np.full(SAMPLE_RATE, 0.5, dtype=np.float32))]
        mixed = mix_timeline(timed, number_of_chunks=1)
        self.assertEqual(len(mixed), 2 * SAMPLE_RATE)


class SubtitleWidthTest(unittest.TestCase):
    def test_display_units(self) -> None:
        self.assertEqual(display_units("你好"), 2.0)
        self.assertEqual(display_units("ab"), 1.0)
        self.assertEqual(display_units("你a"), 1.5)

    def test_split_prefers_punctuation_boundaries(self) -> None:
        text = "一二三四五，六七八九十。十一十二十三"
        pieces = _split_text_by_units(text, 8.0)
        for piece in pieces:
            self.assertLessEqual(display_units(piece), 8.0)
        self.assertEqual("".join(pieces), text)
        # The first break lands after the comma, not mid-word.
        self.assertEqual(pieces[0], "一二三四五，")

    def test_split_without_break_chars_falls_back_to_chars(self) -> None:
        pieces = _split_text_by_units("一二三四五六七八九", 4.0)
        self.assertEqual("".join(pieces), "一二三四五六七八九")
        for piece in pieces:
            self.assertLessEqual(display_units(piece), 4.0)

    def test_max_units_from_video_width(self) -> None:
        # 960px with 8% side margins (~77px) at 38px per full-width unit,
        # two lines.
        side = max(20, round(960 * 0.08))
        self.assertAlmostEqual(
            _max_subtitle_units(960), (960 - 2 * side) / 38 * 2, places=3
        )
        self.assertEqual(_max_subtitle_units(0), 8.0)

    def test_long_text_splits_into_proportional_cues(self) -> None:
        timed = [(0, _padded(0.0, 1.0))]
        results_log = [
            _record(0, text="一二三四五六七八九，十一十二", end_of_turn=True, codec_tokens=25)
        ]
        cues = build_subtitle_cues(
            results_log, timed, video_duration=10.0, max_display_units=8.0
        )
        self.assertEqual(len(cues), 2)
        first, second = cues
        # No punctuation before the overflow point, so it splits per char.
        self.assertEqual(first[2], "一二三四五六七八")
        self.assertAlmostEqual(first[0], 1.0, places=3)
        self.assertAlmostEqual(first[1], 1.0 + 8 / 14, places=3)
        self.assertAlmostEqual(second[0], first[1], places=3)
        self.assertAlmostEqual(second[1], 2.0, places=3)

    def test_merge_respects_display_width(self) -> None:
        timed = [
            (0, _padded(0.0, 1.0)),
            (1, np.full(SAMPLE_RATE, 0.5, dtype=np.float32)),
            (2, np.full(SAMPLE_RATE, 0.5, dtype=np.float32)),
        ]
        results_log = [
            _record(0, text="一二三四", codec_tokens=25),
            _record(1, text="五六七八", codec_tokens=25),
            _record(2, text="九十十一", end_of_turn=True, codec_tokens=25),
        ]
        cues = build_subtitle_cues(
            results_log, timed, video_duration=10.0, max_display_units=9.0
        )
        # 4+4 units merge, adding the third would reach 12 > 9.
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0][2], "一二三四五六七八")
        self.assertEqual(cues[1][2], "九十十一")


class SrtFormatTest(unittest.TestCase):
    def test_formatting(self) -> None:
        self.assertEqual(format_srt_time(3.8), "00:00:03,800")
        self.assertEqual(format_srt_time(61.256), "00:01:01,256")

    def test_write_srt_pre_wraps_long_lines(self) -> None:
        import tempfile
        from pathlib import Path

        cues = [(1.0, 5.0, "一二三四五，六七八九十。十一十二")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.srt"
            write_srt(cues, path, per_line_units=8.0)
            body = path.read_text(encoding="utf-8")
        # The wrapped cue keeps its characters and every line fits the budget.
        wrapped = [l for l in body.splitlines() if l and not l[0].isdigit() and "-->" not in l]
        self.assertGreater(len(wrapped), 1)
        self.assertEqual("".join(wrapped), "一二三四五，六七八九十。十一十二")
        for line in wrapped:
            self.assertLessEqual(display_units(line), 8.0)


if __name__ == "__main__":
    unittest.main()
