"""Command-line interface for py-heed."""

import argparse
import json
import os
import sys
import tempfile
import time

from .transcriber import extract_audio, transcribe, is_cuda_available
from .formatter import write_srt, write_txt


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="py-heed",
        description="Transcribe audio/video to text using faster-whisper",
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to audio or video file",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to faster-whisper model directory",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["srt", "txt", "all"],
        default="srt",
        help="Output format: srt, txt, or all (default: srt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory or file path (default: current directory)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use: auto, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output progress in JSON format",
    )
    return parser.parse_args(argv)


def determine_output_paths(
    input_path: str,
    output_arg: str,
    output_format: str,
) -> dict:
    """Determine output file paths based on input and output argument.

    Args:
        input_path: Path to input file
        output_arg: Output argument from CLI
        output_format: Format selection (srt/txt/both)

    Returns:
        Dict with 'srt_path' and/or 'txt_path' keys, values are paths
    """
    input_basename = os.path.splitext(os.path.basename(input_path))[0]

    if output_arg is None:
        output_dir = os.getcwd()
    elif os.path.isdir(output_arg):
        output_dir = output_arg
    else:
        # output_arg is a file path
        if output_format == "all":
            # User must specify a directory when using 'all'
            output_dir = os.path.dirname(output_arg) or os.getcwd()
        else:
            ext = ".srt" if output_format == "srt" else ".txt"
            if os.path.splitext(output_arg)[1].lower() == ext:
                return {f"{output_format}_path": output_arg}
            # User gave a path without extension, treat as directory
            output_dir = output_arg

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    srt_path = os.path.join(output_dir, f"{input_basename}.srt")
    txt_path = os.path.join(output_dir, f"{input_basename}.txt")

    if output_format in ("srt", "all"):
        paths["srt_path"] = srt_path
    if output_format in ("txt", "all"):
        paths["txt_path"] = txt_path

    return paths


def main(argv=None):
    """Main entry point for py-heed CLI."""
    args = parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine output paths
    output_paths = determine_output_paths(
        args.input, args.output, args.format
    )

    # Extract audio
    if not args.json:
        print(f"Extracting audio from {args.input}...")
    audio_path = None
    try:
        audio_path = extract_audio(args.input)

        # Resolve actual device
        if args.device == "auto":
            actual_device = "cuda" if is_cuda_available() else "cpu"
        else:
            actual_device = args.device

        # Progress tracking variables
        start_time = time.time()
        progress_data = {
            "elapsed": 0,
            "total": 0,
            "percent": 0,
        }

        def progress_callback(elapsed: float, total: float):
            progress_data["elapsed"] = elapsed
            progress_data["total"] = total
            if total > 0:
                progress_data["percent"] = min(100, int(elapsed / total * 100))
            else:
                progress_data["percent"] = 0

            if not args.json:
                print(f"\r{format_timestamp(elapsed)} / {format_timestamp(total)} ({progress_data['percent']}%)", end="", flush=True)

        # Transcribe
        if not args.json:
            print(f"Transcribing with model: {args.model} (device: {actual_device})...")

        segments, total_duration = transcribe(
            audio_path,
            args.model,
            device=actual_device,
            progress_callback=progress_callback,
        )

        if not args.json:
            print()  # New line after progress

        # Write outputs
        results = {}
        if "srt_path" in output_paths:
            if not args.json:
                print(f"Writing SRT: {output_paths['srt_path']}")
            count = write_srt(segments, output_paths["srt_path"])
            results["srt"] = {"path": output_paths["srt_path"], "segments": count}

        if "txt_path" in output_paths:
            if not args.json:
                print(f"Writing TXT: {output_paths['txt_path']}")
            count = write_txt(segments, output_paths["txt_path"])
            results["txt"] = {"path": output_paths["txt_path"], "segments": count}

        if args.json:
            elapsed_time = time.time() - start_time
            print(json.dumps({
                "type": "complete",
                "results": results,
                "total_segments": len(segments),
                "elapsed_time": f"{elapsed_time:.1f}s",
            }))
        else:
            print("Done!")

        return 0

    finally:
        # Clean up temp audio file
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    sys.exit(main())
