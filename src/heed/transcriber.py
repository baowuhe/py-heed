"""Core transcription logic using faster-whisper."""

import os
import subprocess
import tempfile
from typing import List, Optional, Callable

import ffmpeg
from faster_whisper import WhisperModel


class FFmpegError(Exception):
    """Raised when ffmpeg fails."""

    pass


def extract_audio(video_path: str, audio_path: Optional[str] = None) -> str:
    """Extract audio from video file to WAV format.

    Args:
        video_path: Path to input video/audio file
        audio_path: Optional output path. If None, creates a temp file.

    Returns:
        Path to the extracted audio file

    Raises:
        FFmpegError: If ffmpeg fails to extract audio
    """
    if audio_path is None:
        audio_path = tempfile.mktemp(suffix=".wav")

    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream,
        audio_path,
        acodec="pcm_s16le",
        ar=16000,
        ac=1,
        format="wav",
    )
    try:
        ffmpeg.run(
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as e:
        raise FFmpegError(
            f"Failed to extract audio: {e.stderr.decode() if e.stderr else str(e)}"
        )
    return audio_path


def transcribe(
    audio_path: str,
    model_path: str,
    device: str = "auto",
    progress_callback: Optional[Callable[[float, float], None]] = None,
    **kwargs,
) -> tuple[List, float]:
    """Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to audio file
        model_path: Path to faster-whisper model directory
        device: Device to use: "auto", "cuda", "cpu". Default: "auto"
        progress_callback: Optional callback(elapsed, total) for progress updates
        **kwargs: Additional arguments passed to model.transcribe()
            (e.g., language, beam_size, vad_filter)

    Returns:
        Tuple of (list of segment objects, total_duration)
    """
    if device == "auto":
        device = "cuda" if is_cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    try:
        segments_generator, info = model.transcribe(audio_path, **kwargs)

        # Collect segments and report progress
        segments = []
        for segment in segments_generator:
            segments.append(segment)
            if progress_callback and info.duration > 0:
                progress_callback(segment.end, info.duration)

        return segments, info.duration

    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower():
            if device == "cuda":
                # Fall back to CPU if CUDA fails
                model.close()
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    model_path, device=device, compute_type=compute_type
                )
                segments_generator, info = model.transcribe(audio_path, **kwargs)

                # Collect segments and report progress
                segments = []
                for segment in segments_generator:
                    segments.append(segment)
                    if progress_callback and info.duration > 0:
                        progress_callback(segment.end, info.duration)

                return segments, info.duration
            else:
                raise
        else:
            raise
    finally:
        pass


def is_cuda_available() -> bool:
    """Check if CUDA (GPU) is available."""
    # Check via torch if available
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        pass
    # Check via ctranslate2 (used by faster-whisper)
    try:
        import ctranslate2

        # ctranslate2 has a method to get device map
        return "cuda" in str(ctranslate2.get_available_devices())
    except (ImportError, AttributeError):
        pass
    # Fallback: check for nvidia-smi
    import shutil

    if shutil.which("nvidia-smi"):
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    return False


def get_audio_info(audio_path: str) -> dict:
    """Get audio file information.

    Args:
        audio_path: Path to audio file

    Returns:
        Dict with duration and other info
    """
    try:
        probe = ffmpeg.probe(audio_path)
    except ffmpeg.Error as e:
        raise FFmpegError(
            f"Failed to probe audio: {e.stderr.decode() if e.stderr else str(e)}"
        )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    if audio_stream:
        return {
            "duration": float(audio_stream.get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 16000)),
            "channels": int(audio_stream.get("channels", 1)),
        }
    return {"duration": 0, "sample_rate": 16000, "channels": 1}


def convert_video_to_audio(
    video_path: str,
    audio_path: str,
    device: str = "gpu",
    quality: str = "best",
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> float:
    """Convert video file to audio.

    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
        device: Device to use: "gpu" or "cpu" (default: "gpu")
        quality: Audio quality: "best", "high", "medium", "low" (default: "best")
        progress_callback: Optional callback(elapsed, total) for progress updates

    Returns:
        Duration of the audio in seconds
    """
    import re

    # Audio codec mapping for different formats
    codec_map = {
        "mp3": "libmp3lame",
        "aac": "aac",
        "m4a": "aac",
        "flac": "flac",
        "ogg": "libvorbis",
        "wav": "pcm_s16le",
        "wma": "wmav2",
    }

    # Quality presets (bitrate for mp3, general for others)
    # For mp3: best=320k, high=192k, medium=128k, low=96k
    # For aac/m4a: best=192k, high=128k, medium=96k, low=64k
    quality_bitrates = {
        "mp3": {"best": "320k", "high": "192k", "medium": "128k", "low": "96k"},
        "aac": {"best": "192k", "high": "128k", "medium": "96k", "low": "64k"},
        "m4a": {"best": "192k", "high": "128k", "medium": "96k", "low": "64k"},
        "ogg": {"best": "192k", "high": "128k", "medium": "96k", "low": "64k"},
        "wma": {"best": "192k", "high": "128k", "medium": "96k", "low": "64k"},
        "flac": None,  # Lossless, no bitrate needed
        "wav": None,  # Lossless, no bitrate needed
    }

    output_format = os.path.splitext(audio_path)[1].lstrip(".").lower()
    codec = codec_map.get(output_format, "libmp3lame")

    # Get duration and audio stream info for progress reporting
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise FFmpegError(
            f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}"
        )
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    duration = float(
        probe.get("format", {}).get(
            "duration", video_stream.get("duration", 0) if video_stream else 0
        )
    )

    # Determine bitrate: for "auto" use source bitrate, otherwise use preset
    if quality == "auto":
        source_bitrate = None
        source_sample_rate = None
        source_channels = None

        if audio_stream:
            # Try to get bitrate from audio stream
            source_bitrate = audio_stream.get("bit_rate")
            if not source_bitrate:
                # Try format level bitrate as fallback
                source_bitrate = probe.get("format", {}).get("bit_rate")
            source_sample_rate = audio_stream.get("sample_rate")
            source_channels = audio_stream.get("channels")

        # Use source bitrate if available, otherwise fall back to medium
        if source_bitrate:
            bitrate = str(source_bitrate)
        else:
            # Fall back to medium quality preset
            bitrate = quality_bitrates.get(output_format, {}).get("medium")
        # Keep sample rate and channels for auto mode
        sample_rate = source_sample_rate
        channels = source_channels
    else:
        bitrate = quality_bitrates.get(output_format, {}).get(quality)
        sample_rate = None
        channels = None

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", video_path, "-progress", "pipe:1"]

    # Add codec
    cmd.extend(["-acodec", codec])

    # Add bitrate if applicable (not for lossless formats)
    if bitrate:
        cmd.extend(["-b:a", bitrate])

    # Add sample rate if available
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])

    # Add channel count if available
    if channels:
        cmd.extend(["-ac", str(channels)])

    # Add output path
    cmd.append(audio_path)

    if progress_callback:
        # Use subprocess to read progress in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )

        try:
            # Parse progress from stdout line by line
            time_pattern = re.compile(r"out_time_ms=(\d+)")

            for line in process.stdout:
                line_str = line.decode("utf-8", errors="ignore")
                match = time_pattern.search(line_str)
                if match:
                    time_ms = int(match.group(1))
                    elapsed = time_ms / 1000000.0  # Convert to seconds
                    progress_callback(elapsed, duration)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

        if process.returncode != 0:
            stderr = process.stderr.read().decode("utf-8", errors="ignore")
            raise FFmpegError(f"ffmpeg failed: {stderr}")

        return duration
    else:
        stream = ffmpeg.input(video_path)
        output_options = {"acodec": codec}
        if bitrate:
            output_options["b:a"] = bitrate
        stream = ffmpeg.output(
            stream, audio_path, format=output_format, **output_options
        )
        try:
            ffmpeg.run(
                stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )
        except ffmpeg.Error as e:
            raise FFmpegError(
                f"ffmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
            )
        return duration


def slice_media(
    input_path: str,
    output_dir: str,
    num_parts: int,
    device: str = "auto",
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> list[str]:
    """Split a video or audio file into equal parts.

    Args:
        input_path: Path to input video or audio file
        output_dir: Directory to save output slices
        num_parts: Number of equal parts to split into (must be > 1)
        device: Device to use: "auto", "cuda", "cpu" (default: "auto")
        progress_callback: Optional callback(elapsed, total) for progress updates

    Returns:
        List of paths to the output slice files

    Raises:
        FFmpegError: If ffmpeg fails
        ValueError: If num_parts <= 1
    """
    if num_parts <= 1:
        raise ValueError("num_parts must be greater than 1")

    input_ext = os.path.splitext(input_path)[1].lower()
    input_basename = os.path.splitext(os.path.basename(input_path))[0]

    # Get duration
    try:
        probe = ffmpeg.probe(input_path)
    except ffmpeg.Error as e:
        raise FFmpegError(
            f"Failed to probe input: {e.stderr.decode() if e.stderr else str(e)}"
        )

    # Get duration from format or streams
    duration = None
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    if video_stream:
        duration = float(video_stream.get("duration", 0))
    if duration is None or duration == 0:
        duration = float(probe.get("format", {}).get("duration", 0))
    if audio_stream and (duration is None or duration == 0):
        duration = float(audio_stream.get("duration", 0))
    if duration is None or duration == 0:
        raise FFmpegError("Could not determine duration of input file")

    part_duration = duration / num_parts
    width = len(str(num_parts))
    output_paths = []

    for i in range(num_parts):
        start_time = i * part_duration
        output_path = os.path.join(
            output_dir, f"{str(i + 1).zfill(width)}_{input_basename}{input_ext}"
        )
        output_paths.append(output_path)

        stream = ffmpeg.input(input_path, ss=start_time)
        stream = ffmpeg.output(
            stream,
            output_path,
            t=part_duration,
            c="copy",  # Copy codecs, no re-encoding
        )
        try:
            ffmpeg.run(
                stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )
        except ffmpeg.Error as e:
            raise FFmpegError(
                f"Failed to slice part {i + 1}: {e.stderr.decode() if e.stderr else str(e)}"
            )

        if progress_callback:
            progress_callback((i + 1) * part_duration, duration)

    return output_paths
