import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from speaker_identities import load_audio_speaker


def main():
    script_path = sys.argv[1]
    reference_audio_path = sys.argv[2]
    output_path = sys.argv[3]
    item = sys.argv[4]
    speaker_file = sys.argv[5] if len(sys.argv) > 5 else None
    speaker_index = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    transcript_index = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None
    transcript_text = sys.argv[8] if len(sys.argv) > 8 else None
    reference_audio_override = sys.argv[9] if len(sys.argv) > 9 else None

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")

    model = ChatterboxTTS.from_pretrained(device="cuda:0")

    if item != "clone":
        raise ValueError("Chatterbox supports only clone mode")

    if item == "clone":
        text = transcript_text or random_transcript(script_path, transcript_index)
        if speaker_file is None:
            raise ValueError("A LibriTTS-R speaker file is required for Chatterbox cloning")
        speaker_number, speaker_id, split, reference_audio = load_audio_speaker(
            speaker_file, speaker_index
        )
        if reference_audio_override:
            reference_audio = reference_audio_override
        print(f"Selected LibriTTS-R speaker {speaker_number} ({speaker_id}, {split})")

        wav = model.generate(text, audio_prompt_path=reference_audio)
        ta.save(output_path, wav, sample_rate=24000)

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