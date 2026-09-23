import json
import random
from pathlib import Path


def create_identity_pool(prompt_path, identity_count, seed, split_proportions=None):
    prompt_path = Path(prompt_path)
    with prompt_path.open("r", encoding="utf-8") as file:
        descriptions = json.load(file)

    if identity_count < 1:
        raise ValueError("identity_count must be at least 1")

    categories = {}
    category_number = 0
    for group_name in ("speaker_profile", "speaking_traits"):
        for category, values in descriptions[group_name].items():
            if not values:
                raise ValueError(f"No values configured for speaker category: {category}")

            shuffled_values = list(values)
            random.Random(seed + category_number).shuffle(shuffled_values)
            categories[category] = [
                shuffled_values[index % len(shuffled_values)]
                for index in range(identity_count)
            ]
            category_number += 1

    split_counts = _split_counts(identity_count, split_proportions)
    split_labels = (
        ["train"] * split_counts["train"]
        + ["validation"] * split_counts["validation"]
        + ["test"] * split_counts["test"]
    )

    identities = []
    identity_order = list(range(identity_count))
    random.Random(seed).shuffle(identity_order)
    for identity_position, identity_number in enumerate(identity_order):
        traits = {
            category: values[identity_number]
            for category, values in categories.items()
        }
        prompt = descriptions["prompt_template"]
        for category, value in traits.items():
            prompt = prompt.replace(f"[{category}]", value)

        identities.append({
            "identity_number": identity_number + 1,
            "split": split_labels[identity_position],
            "traits": traits,
            "prompt": prompt,
        })

    return identities


def _split_counts(identity_count, split_proportions=None):
    proportions = split_proportions or {
        "train": 0.7,
        "validation": 0.1,
        "test": 0.2,
    }
    counts = {name: int(identity_count * proportion) for name, proportion in proportions.items()}
    remaining = identity_count - sum(counts.values())

    fractional_parts = sorted(
        proportions,
        key=lambda name: identity_count * proportions[name] - counts[name],
        reverse=True,
    )
    for name in fractional_parts[:remaining]:
        counts[name] += 1

    return counts


def write_identity_pool(prompt_path, output_path, identity_count, seed, split_proportions=None):
    identities = create_identity_pool(
        prompt_path, identity_count, seed, split_proportions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({
            "prompt_source": str(prompt_path),
            "identity_count": identity_count,
            "seed": seed,
            "identities": identities,
        }, file, indent=2)
    print(f"Speaker identity pool written to {output_path}")
    return output_path


def write_audio_speaker_pool(dataset_path, output_path, seed, split_proportions=None):
    dataset_path = Path(dataset_path)
    audio_extensions = {".wav", ".flac", ".mp3"}
    speakers = {}
    for audio_path in dataset_path.rglob("*"):
        if not audio_path.is_file() or audio_path.suffix.lower() not in audio_extensions:
            continue

        relative_parts = audio_path.relative_to(dataset_path).parts
        speaker_id = relative_parts[0] if len(relative_parts) > 1 else audio_path.parent.name
        speakers.setdefault(speaker_id, []).append(str(audio_path))

    if not speakers:
        raise FileNotFoundError(f"No audio files found under {dataset_path}")

    speaker_ids = sorted(speakers)
    random.Random(seed).shuffle(speaker_ids)
    split_counts = _split_counts(len(speaker_ids), split_proportions)
    split_labels = (
        ["train"] * split_counts["train"]
        + ["validation"] * split_counts["validation"]
        + ["test"] * split_counts["test"]
    )
    pool = [
        {
            "speaker_number": index + 1,
            "speaker_id": speaker_id,
            "split": split_labels[index],
            "audio_files": sorted(speakers[speaker_id]),
        }
        for index, speaker_id in enumerate(speaker_ids)
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({
            "dataset_path": str(dataset_path),
            "speaker_count": len(pool),
            "seed": seed,
            "speakers": pool,
        }, file, indent=2)
    print(f"LibriTTS-R speaker pool written to {output_path}")
    return output_path


def write_identity_manifest(identity_file, speaker_file, output_path):
    with Path(identity_file).open("r", encoding="utf-8") as file:
        generated_identities = json.load(file)
    with Path(speaker_file).open("r", encoding="utf-8") as file:
        libritts_speakers = json.load(file)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({
            "generated_prompt_identities": generated_identities,
            "libritts_r_speakers": libritts_speakers,
        }, file, indent=2)
    print(f"Combined identity manifest written to {output_path}")
    return output_path


def load_identity_prompt(identity_file, identity_index):
    selected = get_identity(identity_file, identity_index)
    return selected["identity_number"], selected["prompt"], selected["split"]


def get_identity(identity_file, identity_index):
    with Path(identity_file).open("r", encoding="utf-8") as file:
        identity_pool = json.load(file)

    identities = identity_pool.get("identities", [])
    if not identities:
        raise ValueError(f"No identities found in {identity_file}")
    return identities[identity_index % len(identities)]


def load_audio_speaker(identity_file, identity_index):
    selected = get_audio_speaker(identity_file, identity_index)
    audio_path = random.choice(selected["audio_files"])
    return selected["speaker_number"], selected["speaker_id"], selected["split"], audio_path


def get_audio_speaker(identity_file, identity_index):
    with Path(identity_file).open("r", encoding="utf-8") as file:
        speaker_pool = json.load(file)

    speakers = speaker_pool.get("speakers", [])
    if not speakers:
        raise ValueError(f"No speakers found in {identity_file}")

    return speakers[identity_index % len(speakers)]
