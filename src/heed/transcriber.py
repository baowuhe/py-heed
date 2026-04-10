"""Core transcription logic using faster-whisper."""

import os
import tempfile
from typing import List, Optional, Callable

import ffmpeg
from faster_whisper import WhisperModel


def extract_audio(video_path: str, audio_path: Optional[str] = None) -> str:
    """Extract audio from video file to WAV format.

    Args:
        video_path: Path to input video/audio file
        audio_path: Optional output path. If None, creates a temp file.

    Returns:
        Path to the extracted audio file
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
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    return audio_path


def transcribe(
    audio_path: str,
    model_path: str,
    device: str = "auto",
    progress_callback: Optional[Callable[[float, float], None]] = None,
    **kwargs
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
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(model_path, device=device, compute_type=compute_type)
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
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    return False


def get_audio_info(audio_path: str) -> dict:
    """Get audio file information.

    Args:
        audio_path: Path to audio file

    Returns:
        Dict with duration and other info
    """
    probe = ffmpeg.probe(audio_path)
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
        "wav": None,   # Lossless, no bitrate needed
    }

    output_format = os.path.splitext(audio_path)[1].lstrip(".").lower()
    codec = codec_map.get(output_format, "libmp3lame")
    bitrate = quality_bitrates.get(output_format, {}).get(quality)

    # Get duration first for progress reporting
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    duration = float(probe.get("format", {}).get("duration", video_stream.get("duration", 0) if video_stream else 0))

    stream = ffmpeg.input(video_path)

    # Build output options
    output_options = {
        "acodec": codec,
    }

    # Add bitrate for formats that support it
    if bitrate:
        output_options["b:a"] = bitrate

    stream = ffmpeg.output(stream, audio_path, format=output_format, **output_options)

    # Run with progress if callback provided
    if progress_callback:
        # Add -progress pipe:1 to get progress output
        stream = stream.global_args("-progress", "pipe:1")

        # Run and capture output
        stdout, stderr = ffmpeg.run(stream, overwrite_output=True, pipe=True, capture_stderr=True)

        # Parse progress from stdout
        # ffmpeg outputs progress in format: out_time_ms=<timestamp>
        time_pattern = re.compile(r'out_time_ms=(\d+)')

        for line in stdout.decode('utf-8', errors='ignore').split('\n'):
            match = time_pattern.search(line)
            if match:
                time_ms = int(match.group(1))
                elapsed = time_ms / 1000000.0  # Convert to seconds
                progress_callback(elapsed, duration)

        return duration
    else:
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return duration
