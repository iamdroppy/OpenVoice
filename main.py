from __future__ import annotations

import argparse
import enum
import os
import warnings
from os import times_result
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import openai
import torch
from dotenv import load_dotenv
from loguru import logger

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SegmentationMethod(enum.Enum):
    VAD = "vad"
    WHISPER = "whisper"


TEMP_DIR = Path(".tmp")
DEFAULT_PROMPT = "Salespitch"
DEFAULT_ENCODE_MESSAGE = "@MyShell"


def parse_arguments() -> argparse.Namespace:
    """Create and parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_converter", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--reference", type=str, default=None, help="Path to the reference audio file.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-tts", help="OpenAI Model for TTS.")
    parser.add_argument("--voice", type=str, default="Sage", help="The voice to use for TTS.")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Set the logging level.")

    tts_group = parser.add_mutually_exclusive_group(required=False)
    tts_group.add_argument("--text", type=str, default=None, help="Text to be synthesized.")
    tts_group.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to a text file containing the text to be synthesized.",
    )

    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument(
        "--text-prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for the TTS model.",
    )
    prompt_group.add_argument(
        "--text-file-prompt",
        type=str,
        default=None,
        help="Path to a text file containing the prompt for the TTS model.",
    )

    segmentation_group = parser.add_mutually_exclusive_group(required=False)
    segmentation_group.add_argument(
        "--vad",
        action="store_const",
        const=SegmentationMethod.VAD,
        dest="segmentation_method",
        help="Use VAD for speaker embedding extraction.",
    )
    segmentation_group.add_argument(
        "--whisper",
        action="store_const",
        const=SegmentationMethod.WHISPER,
        dest="segmentation_method",
        help="Use Whisper for speaker embedding extraction.",
    )

    parser.add_argument(
        "--whisper-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use for segmentation.",
    )
    parser.add_argument(
        "--whisper-ctype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Whisper model compute type to use for segmentation.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio user interface instead of running the CLI pipeline.",
    )

    parser.set_defaults(segmentation_method=SegmentationMethod.VAD)
    return parser.parse_args()

def configure_logger(log_level: str) -> None:
    """Configure terminal and file log outputs."""

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, level=log_level)
    logger.add("app.log", rotation="1 MB", level=log_level)


def load_text_from_file(path: Path) -> str:
    """Read and return trimmed text from a file."""

    return path.read_text(encoding="utf-8").strip()


def resolve_text_input(config: argparse.Namespace) -> str:
    """Resolve the text input either from CLI or file."""

    if config.text:
        return config.text.strip()

    if config.text_file:
        return load_text_from_file(Path(config.text_file))

    raise ValueError("No text provided for synthesis. Use --text or --text-file to provide input text.")

def resolve_prompt_input(config: argparse.Namespace) -> str:
    """Resolve the prompt input either from CLI or file."""

    if config.text_prompt:
        return config.text_prompt.strip()

    if config.text_file_prompt:
        return load_text_from_file(Path(config.text_file_prompt))

    raise ValueError("No prompt provided for TTS. Use --text-prompt or --text-file-prompt to provide a prompt.")


def resolve_segmentation_kwargs(
    segmentation_method: SegmentationMethod,
    whisper_size: Optional[str],
    whisper_ctype: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return Whisper-related kwargs if the method requires them."""

    if segmentation_method == SegmentationMethod.WHISPER:
        return whisper_size or "medium", whisper_ctype or "float16"

    return None, None

def ensure_directories(output_dir: Path) -> None:
    """Ensure that the temporary and output directories exist."""

    TEMP_DIR.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)


def create_tone_color_converter(ckpt_root: Path, device: str) -> ToneColorConverter:
    """Instantiate and load the tone color converter."""

    config_path = ckpt_root / "config.json"
    checkpoint_path = ckpt_root / "checkpoint.pth"

    converter = ToneColorConverter(str(config_path), device=device)
    converter.load_ckpt(str(checkpoint_path))
    return converter


def get_audio(
    client: openai.Client,
    text: str,
    *,
    model: str = "gpt-4o-mini-tts",
    voice: str = "sage",
    instructions: Optional[str] = None,
) -> Path:
    """Generate an audio file using the OpenAI TTS client."""

    logger.info(f"Generating audio with model: {model}, voice: {voice}")

    temp_audio_path = TEMP_DIR / f"{uuid4().hex[:8]}.mp3"
    resolved_path = temp_audio_path.resolve()
    logger.debug(f"Temporary OpenAI audio path: {resolved_path}")

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(str(resolved_path))

    logger.debug(f"Generated audio. Flushed to: {resolved_path}")
    return resolved_path


def build_openai_client() -> openai.Client:
    """Create an OpenAI client, ensuring the API key is present."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in your .env file. Please configure your environment.")

    return openai.Client(api_key=api_key)

def convert_voice(
    converter: ToneColorConverter,
    reference: Path,
    source_path: Path,
    output_path: Path,
    segmentation_method: SegmentationMethod = SegmentationMethod.VAD,
    whisper_size: Optional[str] = None,
    whisper_ctype: Optional[str] = None,
    *,
    encode_message: str = DEFAULT_ENCODE_MESSAGE,
) -> None:
    """Convert the source audio to match the target speaker."""
    vad = segmentation_method == SegmentationMethod.VAD
    if vad:
        logger.info("Using VAD for speaker embedding extraction")
    else:
        logger.info("Using Whisper for speaker embedding extraction")
        logger.debug(
            "Whisper params",
            size=whisper_size or "medium",
            compute_type=whisper_ctype or "float16",
        )

    whisper_kwargs = {}
    if not vad:
        whisper_kwargs["whisper_size"] = whisper_size or "medium"
        whisper_kwargs["whisper_ctype"] = whisper_ctype or "float16"

    source_se, audio_name = se_extractor.get_se(
        str(source_path),
        converter,
        vad=vad,
        **whisper_kwargs,
    )
    logger.debug(f"[SOURCE] Extracted embedding: {audio_name}")

    target_se, target_audio_name = se_extractor.get_se(
        str(reference),
        converter,
        vad=vad,
        **whisper_kwargs,
    )
    logger.debug(f"[TARGET] Extracted embedding: {target_audio_name}")

    logger.info(f"Converting voice and flushing to: {output_path.resolve()}")
    converter.convert(
        audio_src_path=str(source_path),
        src_se=source_se,
        tgt_se=target_se,
        output_path=str(output_path),
        message=encode_message,
    )
    logger.success(f"Voice conversion completed: {output_path.resolve()}")

def calculate_elapsed_time(start: times_result, end: times_result, join:str = " ") -> str:
    """
    Calculate and format the elapsed time between two times_result objects.
    Args:
        start (times_result): The start time.
        end (times_result): The end time.
        join (str): The string to join the time components. Default is a single space.
    Returns:
        str: Formatted elapsed time string (e.g. 3m 20s 150ms) up to hours.
    """
    elapsed_time = end.elapsed - start.elapsed
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    seconds = int(seconds)
    total_time_str = []
    if hours > 0:
        total_time_str.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0:
        total_time_str.append(f"{int(minutes)}m")
    if seconds > 0 or minutes > 0 or hours > 0:
        total_time_str.append(f"{seconds}s")
    if milliseconds > 0 or seconds > 0 or minutes > 0 or hours > 0:
        total_time_str.append(f"{int(milliseconds)}ms")
    return join.join(total_time_str)

def main() -> None:
    """Run the voice conversion pipeline."""

    start_time = os.times()
    load_dotenv()
    config = parse_arguments()
    configure_logger(config.log_level)

    logger.trace("Configuration parsed")

    output_dir = Path(config.output_dir).resolve()
    ensure_directories(output_dir)

    if config.ui:
        launch_gradio(config)
        return

    if not config.reference:
        raise ValueError("Please provide a --reference audio file when running in CLI mode.")

    reference = Path(config.reference).resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference audio file not found: {reference}")
    logger.success(f"Using reference audio file: {reference}")

    converter = create_tone_color_converter(Path(config.ckpt_converter), config.device)
    client = build_openai_client()

    text = resolve_text_input(config)
    prompt = resolve_prompt_input(config)

    whisper_size, whisper_ctype = resolve_segmentation_kwargs(
        config.segmentation_method,
        config.whisper_size,
        config.whisper_ctype,
    )

    generated_audio, output_path = run_pipeline(
        converter,
        client,
        reference=reference,
        text=text,
        prompt=prompt,
        output_dir=output_dir,
        model=config.model,
        voice=config.voice,
        segmentation_method=config.segmentation_method,
        whisper_size=whisper_size,
        whisper_ctype=whisper_ctype,
    )

    logger.success(f"Audio generated: {generated_audio}")
    logger.success(f"Converted audio saved: {output_path}")
    end_time = os.times()
    total_time_str = calculate_elapsed_time(start_time, end_time)
    logger.success(f"Voice conversion pipeline completed successfully in {total_time_str}")


def run_pipeline(
    converter: ToneColorConverter,
    client: openai.Client,
    *,
    reference: Path,
    text: str,
    prompt: Optional[str],
    output_dir: Path,
    model: str,
    voice: str,
    segmentation_method: SegmentationMethod,
    whisper_size: Optional[str],
    whisper_ctype: Optional[str],
) -> Tuple[Path, Path]:
    """Generate speech via OpenAI and convert it to the reference speaker."""

    generated_audio = get_audio(
        client,
        text,
        model=model,
        voice=voice,
        instructions=prompt,
    )

    output_path = output_dir / f"{generated_audio.stem}.wav"
    convert_voice(
        converter,
        reference,
        generated_audio,
        output_path,
        segmentation_method=segmentation_method,
        whisper_size=whisper_size,
        whisper_ctype=whisper_ctype,
    )

    return generated_audio, output_path


def launch_gradio(config: argparse.Namespace) -> None:
    """Spin up a Gradio Blocks interface for interactive usage."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Gradio is required to launch the UI. Install it with `pip install gradio`."
        ) from exc

    load_dotenv()
    output_dir = Path(config.output_dir).resolve()
    ensure_directories(output_dir)

    converter = create_tone_color_converter(Path(config.ckpt_converter), config.device)
    client = build_openai_client()

    segmentation_labels = {
        "Voice Activity Detection (VAD)": SegmentationMethod.VAD,
        "Whisper (speech-aware)": SegmentationMethod.WHISPER,
    }

    def inference(
        text: str,
        prompt: str,
        reference_audio: str,
        voice: str,
        model: str,
        segmentation_label: str,
        whisper_size: str,
        whisper_ctype: str,
    ):
        if not reference_audio:
            raise gr.Error("Please provide a reference audio sample.")

        text_value = (text or "").strip()
        if not text_value:
            raise gr.Error("Please enter text to synthesize.")
        
        voice = voice.lower()
        model = model.lower()
        
        prompt_value = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT

        segmentation_method = segmentation_labels[segmentation_label]
        whisper_size_value, whisper_ctype_value = resolve_segmentation_kwargs(
            segmentation_method,
            whisper_size,
            whisper_ctype,
        )

        tmp_path, output_path = run_pipeline(
            converter,
            client,
            reference=Path(reference_audio),
            text=text_value,
            prompt=prompt_value,
            output_dir=output_dir,
            model=model,
            voice=voice,
            segmentation_method=segmentation_method,
            whisper_size=whisper_size_value,
            whisper_ctype=whisper_ctype_value,
        )

        return str(tmp_path), str(output_path)

    initial_segmentation_label = "Voice Activity Detection (VAD)"

    with gr.Blocks(title="OpenVoice Tone Conversion") as demo:
        gr.Markdown("## OpenVoice Tone Conversion\nGenerate speech with OpenAI TTS and match a reference speaker.")
    
        with gr.Row():
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter the content you want to synthesize",
                lines=4,
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                placeholder="Describe how the speech should be delivered",
                lines=4,
            )

        reference_input = gr.Audio(
            label="Reference Audio",
            type="filepath",
            sources=["upload"]
        )

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                label="Voice",
                choices=[
                    "Alloy", "Ash", "Ballad", "Cedar", "Coral", "Echo",
                    "Fable", "Marin", "Nova", "Onyx", "Sage", "Shimmer", "Verse"
                ],
                value=config.voice if config.voice else "Sage",
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[
                    "gpt-4o-mini-tts",
                    "tts-1",
                    "tts-1-hd",
                ],
                value=config.model if config.model else "gpt-4o-mini-tts",
            )

        with gr.Row():
            segmentation_dropdown = gr.Dropdown(
                label="Segmentation Method",
                choices=list(segmentation_labels.keys()),
                value=initial_segmentation_label,
            )
            whisper_size_dropdown = gr.Dropdown(
                label="Whisper size",
                choices=["tiny", "base", "small", "medium", "large"],
                value=config.whisper_size,
            )
            whisper_ctype_dropdown = gr.Dropdown(
                label="Whisper compute type",
                choices=["float16", "float32"],
                value=config.whisper_ctype,
            )

        generate_button = gr.Button("Generate & Clone Audio", primary=True)
        with gr.Row():
            openai_audio = gr.Audio(label="OpenAI Audio", type="filepath")
            output_audio = gr.Audio(label="Converted Audio", type="filepath")

        generate_button.click(
            fn=inference,
            inputs=[
                text_input,
                prompt_input,
                reference_input,
                voice_dropdown,
                model_dropdown,
                segmentation_dropdown,
                whisper_size_dropdown,
                whisper_ctype_dropdown,
            ],
            outputs=[openai_audio, output_audio],
        )

    demo.launch()

if __name__ == "__main__":
    main()