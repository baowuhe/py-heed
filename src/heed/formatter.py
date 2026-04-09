"""Output formatting for transcription results."""

from typing import List, Iterator
import math


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_txt(seconds: float) -> str:
    """Convert seconds to TXT timestamp format: HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_srt(segments: Iterator, output_path: str) -> int:
    """Write SRT format subtitles from segments.

    Args:
        segments: Iterator of segment objects with start, end, text attributes
        output_path: Path to write the SRT file

    Returns:
        Number of segments written
    """
    segment_list = list(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segment_list, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")
    return len(segment_list)


def write_txt(segments: Iterator, output_path: str) -> int:
    """Write TXT format with timestamps from segments.

    Format: [start_time - end_time] <transcribed text>

    Args:
        segments: Iterator of segment objects with start, end, text attributes
        output_path: Path to write the TXT file

    Returns:
        Number of segments written
    """
    segment_list = list(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segment_list:
            start = format_timestamp_txt(segment.start)
            end = format_timestamp_txt(segment.end)
            f.write(f"[{start} - {end}] {segment.text.strip()}\n")
    return len(segment_list)
