import sys
import json
import random
import pandas as pd
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from speaker_identities import load_audio_speaker, load_identity_prompt


def main():
    script_path = sys.argv[1]
    speaker_prompt_path = sys.argv[2]
    output_path = sys.argv[3]
    reference_audio_path = sys.argv[4]
    item = sys.argv[5]
    identity_file = sys.argv[6] if len(sys.argv) > 6 else None
    identity_index = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    speaker_file = sys.argv[8] if len(sys.argv) > 8 else None
    speaker_index = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    transcript_index = int(sys.argv[10]) if len(sys.argv) > 10 and sys.argv[10] else None
    transcript_text = sys.argv[11] if len(sys.argv) > 11 else None
    reference_audio_override = sys.argv[12] if len(sys.argv) > 12 else None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if item == "generate":
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda:0" else torch.float32,
            attn_implementation="sdpa",
        )

        if identity_file is None:
            raise ValueError("An identity file is required for Qwen voice design")
        identity_number, instruct, split = load_identity_prompt(identity_file, identity_index)
        print(f"Selected identity {identity_number} ({split}) from {identity_file}")
        
        text = transcript_text or random_transcript(script_path, transcript_index)

        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language="English",
            instruct= instruct,
        )

        sf.write(output_path, wavs[0], sample_rate)
    
    elif item == "clone":
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda:0" else torch.float32,
            attn_implementation="sdpa",
        )

        text = transcript_text or random_transcript(script_path, transcript_index)
        if speaker_file is None:
            raise ValueError("A LibriTTS-R speaker file is required for Qwen voice cloning")
        speaker_number, speaker_id, split, reference_audio = load_audio_speaker(
            speaker_file, speaker_index
        )
        if reference_audio_override:
            reference_audio = reference_audio_override
        print(f"Selected LibriTTS-R speaker {speaker_number} ({speaker_id}, {split})")

        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=reference_audio,
            x_vector_only_mode = True,
        )

        sf.write(output_path, wavs[0], sample_rate)

def random_audio_file(dataset_path):
    root = Path(dataset_path)

    audio_files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"}
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {root}")

    selected = random.choice(audio_files)
    print(f"Selected reference audio: {selected}")
    return str(selected)

def random_transcript(script_path, transcript_index=None):

    '''
    Function to select a random english transcript from the utterance file in the specified path.
    Returns a randomly selected english transcript from the utterance file.
    '''
    df = pd.read_csv(script_path,
        sep="\t",
        encoding="utf-8",
        on_bad_lines="error",
    )

    english = df[df["lang"] == "en"]
    if transcript_index is None:
        random_row = english.sample(n=1).iloc[0]
    else:
        if transcript_index < 0 or transcript_index >= len(english):
            raise IndexError(f"Transcript index out of range: {transcript_index}")
        random_row = english.iloc[transcript_index]
    return random_row['sentence']


if __name__ == "__main__":
    result = main()
    print(f"WORKER_RESULT:{result}")