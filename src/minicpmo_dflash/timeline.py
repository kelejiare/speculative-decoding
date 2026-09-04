"""Timeline assembly for recorded duplex sessions.

The duplex loop returns one waveform per one-second input chunk, but the
model's token2wav stage holds back a lookahead buffer and left-pads short
chunks with silence, so audible speech inside window ``chunk_idx + 1``
often starts late - by up to a second at turn starts. Subtitles keyed to
the raw chunk index therefore lead the speech. The helpers here measure
the committed (non zero-padded) audio of each window and map every speak
chunk's text onto the time span where its codec tokens are actually
audible in the mixed track.
"""

from __future__ import annotations

import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
TOKENS_PER_SECOND = 25
SPECIAL_SUBTITLE_TOKENS = (
    "<|tts_pad|>",
    "<|turn_eos|>",
    "<|chunk_eos|>",
    "<|listen|>",
    "<|speak|>",
)
# Merge consecutive chunk texts while the combined cue stays readable.
MAX_CUE_SECONDS = 6.0
CUE_GAP_SECONDS = 0.05
MIN_CUE_SECONDS = 0.05
# Keep cues within two rendered subtitle lines of the video width.
# Calibrated empirically at 960px: one full-width glyph plus its opaque-box
# outline renders ~37-38px at FontSize=28.
SUBTITLE_CHAR_PX = 38
SUBTITLE_SIDE_MARGIN = 20
MAX_SUBTITLE_LINES = 2
_BREAK_CHARS = '，。！？；：、,.!?;: …'


def to_numpy(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def display_units(text: str) -> float:
    """Approximate rendered width in full-width character units."""

    return sum(_char_units(ch) for ch in text)


def _char_units(ch: str) -> float:
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5


def _split_text_by_units(text: str, max_units: float) -> list[str]:
    """Split text into pieces that each fit ``max_units`` display units.

    Breaks prefer the latest punctuation inside the overflowing piece so
    segments end on natural boundaries.
    """

    if display_units(text) <= max_units:
        return [text]
    segments: list[str] = []
    current = ""
    current_units = 0.0
    for ch in text:
        width = _char_units(ch)
        if current and current_units + width > max_units:
            cut = max(current.rfind(ch_break) for ch_break in _BREAK_CHARS) + 1
            if cut > len(current) // 2:
                segments.append(current[:cut].strip())
                current = current[cut:].lstrip() + ch
            else:
                segments.append(current.strip())
                current = ch
            current_units = display_units(current)
        else:
            current += ch
            current_units += width
    if current.strip():
        segments.append(current.strip())
    return segments


@dataclass
class Window:
    chunk_index: int
    start: float  # wall-clock start of the committed audio
    committed: int  # committed sample count
    stream_pos: int  # committed samples before this window


def analyze_windows(
    timed_output_audio: list[tuple[int, object]], sample_rate: int = SAMPLE_RATE
) -> list[Window]:
    """Describe the audible audio contributed by each one-second window.

    token2wav pads short chunks with exact zeros on the left, so the first
    non-zero sample marks where the committed PCM begins.
    """

    windows: list[Window] = []
    stream_pos = 0
    for chunk_index, waveform in timed_output_audio:
        values = to_numpy(waveform)
        nonzero = np.flatnonzero(values != 0.0)
        if nonzero.size == 0:
            continue
        pad = int(nonzero[0])
        committed = len(values) - pad
        if committed <= 0:
            continue
        start = (chunk_index + 1) + pad / sample_rate
        windows.append(Window(chunk_index, start, committed, stream_pos))
        stream_pos += committed
    return windows


def _stream_time(turn_windows: list[Window], pos: int, sample_rate: int) -> float:
    """Wall-clock time of stream sample offset ``pos`` within one turn.

    Windows of different turns are contiguous in stream coordinates but
    far apart on the wall clock, so mapping is bounded by the turn's own
    audio and clamps at both ends.
    """

    if not turn_windows:
        return 0.0
    first = turn_windows[0]
    if pos <= first.stream_pos:
        return first.start
    last = turn_windows[-1]
    if pos >= last.stream_pos + last.committed:
        return last.start + last.committed / sample_rate
    for window in turn_windows:
        if pos < window.stream_pos + window.committed:
            return window.start + (pos - window.stream_pos) / sample_rate
    return last.start + last.committed / sample_rate


def _split_turns(results_log: list[dict]) -> list[list[dict]]:
    turns: list[list[dict]] = []
    current: list[dict] = []
    for record in results_log:
        if record.get("is_listen", True):
            if current:
                turns.append(current)
                current = []
            continue
        current.append(record)
        if record.get("end_of_turn", False):
            turns.append(current)
            current = []
    if current:
        turns.append(current)
    return turns


def build_subtitle_cues(
    results_log: list[dict],
    timed_output_audio: list[tuple[int, object]],
    video_duration: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    max_display_units: float | None = None,
) -> list[tuple[float, float, str]]:
    """Map each speak chunk's text onto its audible time span.

    Text generated at chunk N is spoken by the codec tokens generated in
    the same chunk. Their position in the audible stream is tracked in
    committed samples; ``tts_codec_tokens`` refines the span because the
    lookahead buffer defers part of a chunk's audio to the next window.
    ``max_display_units`` bounds each cue's rendered width in full-width
    character units; overflowing texts split into consecutive cues whose
    time spans divide proportionally.
    """

    windows = analyze_windows(timed_output_audio, sample_rate)
    committed = {window.chunk_index: window.committed for window in windows}
    spans: list[list] = []
    for turn in _split_turns(results_log):
        first_chunk = turn[0]["chunk_idx"]
        last_chunk = turn[-1]["chunk_idx"]
        turn_windows = [
            window for window in windows if first_chunk <= window.chunk_index <= last_chunk
        ]
        if not turn_windows:
            continue
        turn_pos = turn_windows[0].stream_pos
        for record in turn:
            chunk_idx = record["chunk_idx"]
            length = committed.get(chunk_idx, 0)
            codec_tokens = int(record.get("tts_codec_tokens") or 0)
            if codec_tokens > 0:
                length = max(length, codec_tokens * sample_rate // TOKENS_PER_SECOND)
            text = record.get("text", "") or ""
            for token in SPECIAL_SUBTITLE_TOKENS:
                text = text.replace(token, "")
            text = text.strip()
            if text:
                start = _stream_time(turn_windows, turn_pos, sample_rate)
                end = _stream_time(turn_windows, turn_pos + length, sample_rate)
                pieces = (
                    _split_text_by_units(text, max_display_units)
                    if max_display_units is not None
                    else [text]
                )
                total_units = display_units(text)
                prefix_units = 0.0
                for piece in pieces:
                    piece_start = start + (end - start) * prefix_units / total_units
                    prefix_units += display_units(piece)
                    piece_end = start + (end - start) * prefix_units / total_units
                    spans.append([piece_start, piece_end, piece])
            turn_pos += length

    merged: list[list] = []
    for start, end, text in spans:
        fits_width = (
            max_display_units is None
            or not merged
            or display_units(merged[-1][2] + text) <= max_display_units
        )
        if (
            merged
            and start - merged[-1][1] <= CUE_GAP_SECONDS
            and end - merged[-1][0] <= MAX_CUE_SECONDS
            and fits_width
        ):
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] += text
        else:
            merged.append([start, end, text])

    cues: list[tuple[float, float, str]] = []
    for start, end, text in merged:
        if video_duration is not None:
            if start >= video_duration:
                continue
            end = min(end, video_duration)
        if end - start < MIN_CUE_SECONDS:
            continue
        cues.append((start, end, text))
    return cues


def format_srt_time(seconds: float) -> str:
    total_millis = round(seconds * 1000)
    hours, rem = divmod(total_millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(
    cues: list[tuple[float, float, str]],
    path: Path | str,
    per_line_units: float | None = None,
) -> None:
    """Write cues as SRT, optionally pre-wrapping text to a line budget.

    libass does not honor subtitle margins when auto-wrapping, so explicit
    line breaks are the only reliable way to keep cues off the edges.
    """

    lines: list[str] = []
    for index, (start, end, text) in enumerate(cues, start=1):
        if per_line_units is not None and display_units(text) > per_line_units:
            text = "\n".join(_split_text_by_units(text, per_line_units))
        lines.append(f"{index}")
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def mix_timeline(
    timed_output_audio: list[tuple[int, object]],
    number_of_chunks: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix per-chunk waveforms onto the shared ``(chunk_idx + 1)``-second grid.

    The grid matches the model's ``current_time`` clock (chunk N is the
    response to input audio heard during second N+1) and the video AI
    track, so the standalone WAV and the MP4 stay in sync.
    """

    waveforms = [(index, to_numpy(waveform)) for index, waveform in timed_output_audio]
    total = (number_of_chunks + 1) * sample_rate
    for chunk_index, values in waveforms:
        total = max(total, (chunk_index + 1) * sample_rate + len(values))
    output = np.zeros(total, dtype=np.float32)
    for chunk_index, values in waveforms:
        start = (chunk_index + 1) * sample_rate
        end = min(total, start + len(values))
        output[start:end] += values[: end - start]
    peak = float(np.abs(output).max()) if output.size else 0.0
    if peak > 1.0:
        output /= peak
    return output


def _subtitle_side_margin(video_width: int) -> int:
    """Left/right margin that keeps long lines off the video edges."""

    return max(SUBTITLE_SIDE_MARGIN, round(video_width * 0.08))


def _max_subtitle_units(video_width: int) -> float:
    side = _subtitle_side_margin(video_width)
    per_line = max(4.0, (video_width - 2 * side) / SUBTITLE_CHAR_PX)
    return per_line * MAX_SUBTITLE_LINES


def _video_width(video_path: str) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception as error:
        print(f"ffprobe width failed ({error}); using 960 default")
        return 960


def _video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as error:
        print(f"ffprobe duration failed ({error}); using 60s default")
        return 60.0


def _stream_has_audio(video_path: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "audio"
    except Exception as error:
        print(f"ffprobe audio stream check failed ({error}); assuming no audio")
        return False


def render_duplex_video(
    video_path: str,
    output_video_path: str,
    results_log: list[dict],
    timed_output_audio: list[tuple[int, object]],
    ai_audio_path: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Mux the duplex demo video: original audio + AI track + synced SRT."""

    video_duration = _video_duration(video_path)
    video_width = _video_width(video_path)
    side_margin = _subtitle_side_margin(video_width)
    max_display_units = _max_subtitle_units(video_width)
    cues = build_subtitle_cues(
        results_log, timed_output_audio, video_duration, sample_rate, max_display_units
    )

    has_ai_audio = False
    if ai_audio_path and Path(ai_audio_path).is_file():
        track, _ = sf.read(ai_audio_path, dtype="float32")
        has_ai_audio = bool(track.size and float(np.abs(track).max()) > 0.0)
    has_original_audio = _stream_has_audio(video_path)

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path = output_path.with_name(f".{output_path.stem}.srt")
    try:
        has_subtitles = bool(cues)
        if has_subtitles:
            write_srt(cues, srt_path, max_display_units / MAX_SUBTITLE_LINES)
            escaped = (
                str(srt_path).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
            )
            subtitle_filter = (
                f"subtitles='{escaped}':"
                f"force_style='FontSize=28,"
                f"PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,"
                f"BorderStyle=3,"
                f"Outline=2,"
                f"Shadow=1,"
                f"MarginV=30,"
                f"MarginL={side_margin},"
                f"MarginR={side_margin},"
                f"Alignment=2'"
            )

        cmd = ["ffmpeg", "-y", "-i", video_path]
        if has_ai_audio:
            cmd.extend(["-i", ai_audio_path])
            if has_original_audio:
                audio_chain = "[0:a][1:a]amix=inputs=2:duration=longest[aout]"
                if has_subtitles:
                    filter_complex = f"[0:v]{subtitle_filter}[vout];{audio_chain}"
                    cmd.extend(
                        ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]"]
                    )
                else:
                    cmd.extend(
                        ["-filter_complex", audio_chain, "-map", "0:v", "-map", "[aout]"]
                    )
            elif has_subtitles:
                cmd.extend(
                    [
                        "-filter_complex",
                        f"[0:v]{subtitle_filter}[vout]",
                        "-map",
                        "[vout]",
                        "-map",
                        "1:a",
                    ]
                )
            else:
                cmd.extend(["-map", "0:v", "-map", "1:a"])
        else:
            if has_subtitles:
                cmd.extend(["-vf", subtitle_filter])
            if has_original_audio:
                cmd.extend(["-c:a", "copy"])

        cmd.extend(
            ["-c:v", "libx264", "-c:a", "aac", "-preset", "medium", "-crf", "23", str(output_path)]
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if srt_path.is_file():
            srt_path.unlink()
    return str(output_path)
