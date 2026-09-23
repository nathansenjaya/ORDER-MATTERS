import json
import random
import sys
from pathlib import Path

import pandas as pd
import torchaudio
from huggingface_hub import snapshot_download

COSYVOICE_ROOT = Path(__file__).parent / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from speaker_identities import load_audio_speaker


MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_DIR = Path(__file__).parent / "pretrained_models" / "Fun-CosyVoice3-0.5B"


def main():
    script_path = sys.argv[1]
    reference_audio_path = sys.argv[2]
    output_path = sys.argv[3]
    item = sys.argv[4]
    instruct_path = sys.argv[5]
    speaker_file = sys.argv[6] if len(sys.argv) > 6 else None
    speaker_index = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    transcript_index = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] else None
    transcript_text = sys.argv[9] if len(sys.argv) > 9 else None
    reference_audio_override = sys.argv[10] if len(sys.argv) > 10 else None

    if item != "clone":
        raise ValueError("CosyVoice supports only clone mode")
    if speaker_file is None:
        raise ValueError("A LibriTTS-R speaker file is required for CosyVoice cloning")

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))

    model = AutoModel(model_dir=str(MODEL_DIR))
    text = transcript_text or random_transcript(script_path, transcript_index)
    instruct_text = random_instruction(instruct_path)
    _, _, _, reference_audio = load_audio_speaker(speaker_file, speaker_index)
    if reference_audio_override:
        reference_audio = reference_audio_override
    print(f"Selected CosyVoice instruction: {instruct_text}")

    result = next(model.inference_instruct2(
        text,
        instruct_text,
        reference_audio,
        stream=False,
    ))
    torchaudio.save(output_path, result["tts_speech"], model.sample_rate)


def random_transcript(script_path, transcript_index=None):
    df = pd.read_csv(
        script_path,
        sep="\t",
        encoding="utf-8",
        on_bad_lines="error",
    )
    english = df[df["lang"] == "en"]
    if transcript_index is None:
        return english.sample(n=1).iloc[0]["sentence"]
    if transcript_index < 0 or transcript_index >= len(english):
        raise IndexError(f"Transcript index out of range: {transcript_index}")
    return english.iloc[transcript_index]["sentence"]


def random_instruction(instruct_path):
    with Path(instruct_path).open("r", encoding="utf-8") as file:
        instructions = json.load(file)

    traits = {}
    for category, values in instructions["speaking_traits"].items():
        if not values:
            raise ValueError(f"No values configured for instruction category: {category}")
        traits[category] = random.choice(values)

    instruction = instructions["prompt_template"]
    for category, value in traits.items():
        instruction = instruction.replace(f"[{category}]", value)
    return instruction


if __name__ == "__main__":
    main()
