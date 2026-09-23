import argparse
import json
import os
import random
import subprocess
import sys
import torch
from collections import Counter
from pathlib import Path

from data_storing import MetaDataStorage
from evaluate_generated_audio import detector_variants
from evaluate_generated_audio import load_detector, score_file
from speaker_identities import (
    write_identity_manifest,
    write_audio_speaker_pool,
    write_identity_pool,
)

def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _worker_python(mode, config):
    configured = config.get("python") or os.environ.get(
        f"IEMOCAP_{mode.upper()}_PYTHON"
    )
    return configured or sys.executable


def _make_split_schedule(data_number, seed, configured_proportions=None):
    '''
    Create a randomize schedule to split the identity from the cloning source and generated identity
    based on the configured proportions for train, validation, and test splits. The schedule is generated using a random seed to ensure reproducibility.
    The function returns a list of split names corresponding to the number of identities to be generated.

    Data_number: The total number of identities to be generated.
    Seed: The random seed for reproducibility.
    Configured_proportions: A dictionary specifying the proportions for train, validation, and test
    '''

    proportions = configured_proportions or {
        "train": 0.7,
        "validation": 0.1,
        "test": 0.2,
    }
    if set(proportions) != {"train", "validation", "test"}:
        raise ValueError(
            "generation_split_proportions must define train, validation, and test"
        )
    if any(proportion < 0 for proportion in proportions.values()):
        raise ValueError("generation split proportions cannot be negative")
    if abs(sum(proportions.values()) - 1.0) > 1e-8:
        raise ValueError("generation split proportions must sum to 1")
    counts = {
        split: int(data_number * proportion)
        for split, proportion in proportions.items()
    }
    remaining = data_number - sum(counts.values())
    fractional = sorted(
        proportions,
        key=lambda split: data_number * proportions[split] - counts[split],
        reverse=True,
    )
    for split in fractional[:remaining]:
        counts[split] += 1

    schedule = [split for split, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def _balanced_pool_indices(pool, split, count, seed, number_key):
    '''
    Select a balanced set of indices from a pool of items based on a specific split and count.
    '''

    candidates = [
        index for index, item in enumerate(pool)
        if item.get("split") == split
    ]

    # If the count is greater than the number of candidates, raise an error
    if count and not candidates:
        raise ValueError("No {} identities available for {} split".format(
            number_key, split))
    random.Random(seed).shuffle(candidates)
    selected = [candidates[index % len(candidates)] for index in range(count)]
    random.Random(seed + 1).shuffle(selected)
    return selected


def _make_generation_schedule(data_number, generated_identities, source_speakers, config_specs, split_proportions,
                                method_proportions, seed):
    '''
    Create a randomized schedule for generating audio data based on the specified number of identities, source speakers, and configuration specifications.
    The schedule is generated using a random seed to ensure reproducibility. The function returns a list of dictionaries, each containing the configuration path, generation type, dataset split, and identity/speaker indices
    for each generated audio item.
    '''

    if set(method_proportions) != {"generated", "clone"}:
        raise ValueError(
            "generation_method_proportions must define generated and clone"
        )
    if any(value < 0 for value in method_proportions.values()):
        raise ValueError("generation method proportions cannot be negative")
    if abs(sum(method_proportions.values()) - 1.0) > 1e-8:
        raise ValueError("generation method proportions must sum to 1")

    method_counts = {
        method: int(data_number * proportion)
        for method, proportion in method_proportions.items()
    }

    remaining = data_number - sum(method_counts.values())
    fractional = sorted(
        method_proportions,
        key=lambda method: data_number * method_proportions[method]
        - method_counts[method],
        reverse=True,
    )
    for method in fractional[:remaining]:
        method_counts[method] += 1

    generated_configs = [
        spec for spec in config_specs if "generate" in spec["generation_types"]
    ]
    clone_configs = [
        spec for spec in config_specs if "clone" in spec["generation_types"]
    ]
    if method_counts["generated"] and not generated_configs:
        raise ValueError("No configuration supports generated voice design")
    if method_counts["clone"] and not clone_configs:
        raise ValueError("No configuration supports voice cloning")

    jobs = []
    for method_index, (method, method_count) in enumerate(method_counts.items()):
        method_splits = _make_split_schedule(
            method_count, seed + method_index, split_proportions)
        pool = generated_identities if method == "generated" else source_speakers
        number_key = "identity_number" if method == "generated" else "speaker_number"
        candidates_by_split = {}
        for split in ("train", "validation", "test"):
            split_count = method_splits.count(split)
            candidates_by_split[split] = _balanced_pool_indices(
                pool, split, split_count, seed + method_index * 10 + len(split), number_key)

        config_pool = generated_configs if method == "generated" else clone_configs
        if method == "generated":
            config_order = [spec for index in range(method_count)
                            for spec in [config_pool[index % len(config_pool)]]]
            random.Random(seed + method_index + 100).shuffle(config_order)
        else:
            # Allocate each speaker's jobs across models as evenly as possible,
            # then shuffle each speaker's model assignments independently.
            config_order = [None] * method_count
            config_random = random.Random(seed + method_index + 100)
            model_cursor = 0
            for split in ("train", "validation", "test"):
                split_indices = [
                    index for index, split_name in enumerate(method_splits)
                    if split_name == split
                ]
                speaker_sequence = candidates_by_split[split]
                speaker_counts = Counter(speaker_sequence)
                speaker_order = list(speaker_counts)
                config_random.shuffle(speaker_order)
                assignments = {}
                for speaker_index in speaker_order:
                    speaker_configs = []
                    for _ in range(speaker_counts[speaker_index]):
                        speaker_configs.append(
                            config_pool[model_cursor % len(config_pool)]
                        )
                        model_cursor += 1
                    config_random.shuffle(speaker_configs)
                    assignments[speaker_index] = speaker_configs

                for index, speaker_index in zip(split_indices, speaker_sequence):
                    config_order[index] = assignments[speaker_index].pop()
        split_positions = {split: 0 for split in candidates_by_split}
        for index, split in enumerate(method_splits):
            pool_index = candidates_by_split[split][split_positions[split]]
            split_positions[split] += 1
            jobs.append({
                "config_path": config_order[index]["path"],
                "generation_type": "generate" if method == "generated" else "clone",
                "dataset_split": split,
                "identity_index": pool_index if method == "generated" else 0,
                "speaker_index": pool_index if method == "clone" else 0,
            })

    random.Random(seed + 200).shuffle(jobs)
    return jobs


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    from apply_attack import AudioAttackClass
    from quality_check import QualityCheckClass

    detector_scorers = {}
    detector_device = None

    if args.evaluate_model:
        # Set the device for the detector scorers based on GPU availability
        detector_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the specified detector models and their variants
        for requested_model_name in args.evaluate_model:
            model_names = (
                ("sonar-full",)
                if requested_model_name == "sonar"
                else (requested_model_name,)
            )
            for model_name in model_names:
                for variant_name, checkpoint_path in detector_variants(model_name):
                    scorer_name = f"{model_name}_{variant_name}"
                    print(f"Loading {scorer_name} detector on {detector_device} ...")
                    detector_scorers[scorer_name] = load_detector(
                        model_name, detector_device, checkpoint_path
                    )

    with open(args.config, "r") as file:
        pipeline = json.load(file)
    
    """
    Extract parameters from the pipeline configuration file
    1. The number of data to be generated form this pipeline
    2. Maximum number of attacks to apply to each generated audio
    3. The number of speaker identities to generate from Voice Design Models
    4. The seed for randomized speaker identities generation, which ensures reproducibility
    5. The folder containing the configuration files for the models to be used in this pipeline
    6. The path to the output folder for the generated audio
    7. The path to the log file for the generation process
    8. The path to the speaker identity file, which will be generated if it does not exist
    9. The paths to the cafe and street audio datasets, which will be used for applying realistic audio attacks
    10. The paths to the model configuration files for QwenTTS and Chatterbox, which will be randomly selected
    11. The paths to the model configuration files for the detector models, which will be used for quality checking
    """

    data_number = pipeline.get("data_number", 0)
    max_attack_number = pipeline.get("maximum_attack", 0)
    remove_if_all_baselines_fake = pipeline.get(
        "remove_if_all_baselines_fake", False)
    speaker_identity_number = max(1, pipeline.get("generated_speaker_number", 1))
    speaker_identity_seed = pipeline.get("speaker_identity_seed", 1)
    config_base_dir = Path(args.config).resolve().parent
    config_folder = _resolve_path(
        pipeline["config_folder_path"], config_base_dir)
    
    # Load all configuration files in the specified folder for random selection during data generation
    config_files = [
        path for path in config_folder.glob("*.config")
        if path.name != Path(args.config).name
    ]

    if not config_files:
        raise FileNotFoundError(f"No model configs found in {config_folder}")
    
    # Create output folder and generation log file if they don't exist
    audio_output_folder = _resolve_path(
        pipeline["audio_output_folder"], config_base_dir)
    generation_log_path = _resolve_path(
        pipeline["generation_log_path"], config_base_dir)
    identity_file = _resolve_path(pipeline["speaker_identity_file"], config_base_dir)
    libritts_speaker_file = _resolve_path(
        pipeline.get(
            "libritts_speaker_file",
            identity_file.with_name("libritts_speakers.json"),
        ),
        config_base_dir,
    )
    identity_manifest_file = _resolve_path(
        pipeline.get(
            "identity_manifest_file",
            identity_file.with_name("identity_manifest.json"),
        ),
        config_base_dir,
    )

    split_proportions = pipeline.get("generation_split_proportions")

    instruct_path = _resolve_path(pipeline["instruct_path"], config_base_dir)
    description_path = _resolve_path(pipeline["description_path"], config_base_dir)

    # Generate the speaker identity pool using the specified description file, number of identities, and seed for reproducibility
    write_identity_pool(
        description_path,
        identity_file,
        speaker_identity_number,
        speaker_identity_seed,
        split_proportions,
    )

    reference_config_paths = [
        path for path in config_files
        if json.loads(path.read_text(encoding="utf-8")).get("mode") in {"qwen", "chatterbox"}
    ]
    
    if reference_config_paths:
        with reference_config_paths[0].open("r", encoding="utf-8") as file:
            reference_config = json.load(file)
        write_audio_speaker_pool(
            reference_config["reference_audio_path"], libritts_speaker_file,
            speaker_identity_seed, split_proportions,
        )
        write_identity_manifest(
            identity_file,
            libritts_speaker_file,
            identity_manifest_file,
        )

    with identity_file.open("r", encoding="utf-8") as file:
        generated_identities = json.load(file)["identities"]
    with libritts_speaker_file.open("r", encoding="utf-8") as file:
        source_speakers = json.load(file)["speakers"]
    config_specs = []
    for config_path in config_files:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        mode = config["mode"]
        generation_types = config.get("generation_type")
        if generation_types is None:
            generation_types = ["generate", "clone"] if mode == "qwen" else ["clone"]
        else:
            generation_types = [generation_types]
        config_specs.append({
            "path": config_path,
            "generation_types": generation_types,
        })

    
    # Start the randomized generation schedule for generated identites for Voice Design Models
    # and cloning from the LibriTTS-R speaker pool for the three models (QwenTTS, Chatterbox, and CosyVoice)

    jobs = _make_generation_schedule(
        data_number,
        generated_identities,
        source_speakers,
        config_specs,
        split_proportions,
        pipeline.get("generation_method_proportions", {
            "generated": 0.5,
            "clone": 0.5,
        }),
        speaker_identity_seed,
    )

    # Initialize the path for the WHAM background audio dataset
    cafe_audio_path = _resolve_path(pipeline["cafe_audio_path"], config_base_dir)

    """
    Initialize the attack, quality check, and data storing classes
    For applying random audio attacks
    For checking the quality of the attacked audio
    For storing metadata about the generated audio and attacks
    """

    attack = AudioAttackClass(cafe_audio_path)
    quality_check = QualityCheckClass()
    data_storing = MetaDataStorage(audio_output_folder, generation_log_path)

    # Loop through the number of data to be generated based on the configuration, randomly selecting a model config for each iteration
    for i, job in enumerate(jobs):
        print(f"Data Generation {i + 1}/{data_number} on GPU {args.gpu} ...")
        selected_config_path = job["config_path"]
        dataset_split = job["dataset_split"]

        with selected_config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        mode = config["mode"]

        # Define the output path for the generated audio file based on the mode and iteration index
        output_path = audio_output_folder / f"{mode}_{i + 1:05d}.wav"
        current_audio_path = output_path

        # Use a configured generation type when provided; otherwise choose randomly.
        item = job["generation_type"]
        identity_metadata = {
            "selection": item,
            "dataset_split": dataset_split,
        }

        # Generating subprocess command for inferencing with QwenTTS, while passing the item parameter for task assignment
        if mode == "qwen":
            if item == "generate":
                identity_index = job["identity_index"]
                selected_identity = generated_identities[identity_index]
                identity_metadata.update({
                    "type": "generated_prompt",
                    "pool_file": str(identity_file),
                    "identity_number": selected_identity["identity_number"],
                    "split": selected_identity["split"],
                    "traits": selected_identity["traits"],
                })
                speaker_index = 0
            else:
                identity_index = 0
                speaker_index = job["speaker_index"]
                selected_speaker = source_speakers[speaker_index]
            if item == "clone":
                identity_metadata.update({
                    "type": "libritts_r_speaker",
                    "pool_file": str(libritts_speaker_file),
                    "speaker_number": selected_speaker["speaker_number"],
                    "speaker_id": selected_speaker["speaker_id"],
                    "split": selected_speaker["split"],
                })

            command = [
                _worker_python("qwen", config),
                str(Path(__file__).parent / "qwen_worker.py"),
                str(_resolve_path(config["script_path"], config_base_dir)),
                str(description_path),
                str(output_path),
                str(_resolve_path(config["reference_audio_path"], config_base_dir)),
                str(item),
                str(identity_file),
                str(identity_index),
                str(libritts_speaker_file),
                str(speaker_index),
            ]

        # Generating subprocess command for inferencing with Chatterbox, while passing the item parameter for task assignment
        elif mode == "chatterbox":
            speaker_index = job["speaker_index"]
            selected_speaker = source_speakers[speaker_index]
            if item == "clone":
                identity_metadata.update({
                    "type": "libritts_r_speaker",
                    "pool_file": str(libritts_speaker_file),
                    "speaker_number": selected_speaker["speaker_number"],
                    "speaker_id": selected_speaker["speaker_id"],
                    "split": selected_speaker["split"],
                })
            else:
                speaker_index = 0

            command = [
                _worker_python("chatterbox", config),
                str(Path(__file__).parent / "chatterbox_worker.py"),
                str(_resolve_path(config["script_path"], config_base_dir)),
                str(_resolve_path(config["reference_audio_path"], config_base_dir)),
                str(output_path),
                str(item),
                str(libritts_speaker_file),
                str(speaker_index),
            ]

        elif mode == "cosyvoice":
            if item != "clone":
                raise ValueError("CosyVoice supports only clone mode")

            speaker_index = job["speaker_index"]
            selected_speaker = source_speakers[speaker_index]
            if item == "clone":
                identity_metadata.update({
                    "type": "libritts_r_speaker",
                    "pool_file": str(libritts_speaker_file),
                    "speaker_number": selected_speaker["speaker_number"],
                    "speaker_id": selected_speaker["speaker_id"],
                    "split": selected_speaker["split"],
                })

            command = [
                _worker_python("cosyvoice", config),
                str(Path(__file__).parent / "cosyvoice_worker.py"),
                str(_resolve_path(config["script_path"], config_base_dir)),
                str(_resolve_path(config["reference_audio_path"], config_base_dir)),
                str(output_path),
                str(item),
                str(instruct_path),
                str(libritts_speaker_file),
                str(speaker_index),
            ]
        else:
            raise ValueError("mode must be either 'qwen', 'chatterbox', or 'cosyvoice'")

        # Since we are using subprocesses to run the TTS models, we enforce the use of the correct Python environment
        # and the correct GPU for the subprocesses by setting the environment variables accordingly
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        if mode == "chatterbox":
            worker_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        # Execute the subprocess command to generate audio using the selected model
        subprocess.run(command, check=True, env=worker_env)

        # Store the first metadata about the generated audio file, including its model and generation type
        original_scores = {
            model_name: score_file(output_path, scorer, detector_device)
            for model_name, scorer in detector_scorers.items()
        }

        # Store the first metadata about the generated audio file, including its model and generation type
        data_storing.store_metadata(output_path.name, mode, item, None, None, None, None, None, None, None, None, first=True, detector_scores=original_scores, identity=identity_metadata)

        # Preserve the generated WAV as level 0, then use a compressed copy as the
        # starting point for the randomized attack chain.
        compression_method = attack.choose_compression()
        print(f"Initial compression: {compression_method.__name__}")
        current_audio_path, _, _ = attack.apply_compression(
            str(output_path),
            str(output_path.parent),
            0,
            compression_method,
        )

        compressed_scores = {
            model_name: score_file(current_audio_path, scorer, detector_device)
            for model_name, scorer in detector_scorers.items()
        }
        data_storing.store_metadata(
            output_path.name,
            None,
            None,
            None,
            None,
            None,
            current_audio_path,
            "compression",
            compression_method.__name__,
            0,
            passed_condition=True,
            first=False,
            detector_scores=compressed_scores,
        )

        # Apply only randomized attacks at levels 1 through maximum_attack.
        for attack_index in range(max_attack_number):
            print(f"Applying attack {attack_index + 1} from maximum {max_attack_number} ...")

            attacked_path, attack_type, attack_parameter = attack.random_attack(
                str(current_audio_path),
                str(output_path.parent),
                attack_index + 1,
            )

            # Compute the quality metrics for the attacked audio, including STOI, PESQ, and UTMOS scores
            stoi_score = quality_check.compute_stoi(str(output_path), str(attacked_path))
            pesq_score = quality_check.compute_pesq(str(output_path), str(attacked_path))
            utmos_score = quality_check.compute_utmos(str(attacked_path))

            # Quality failures are recorded, then discarded from the audio dataset.
            bad_metrics = sum([utmos_score < 1.5, stoi_score < 0.35, pesq_score < 1.15])
            if bad_metrics >= 2:
                data_storing.store_metadata(
                    output_path.name, None, None, stoi_score, pesq_score,
                    utmos_score, attacked_path, attack_type, attack_parameter,
                    attack_index + 1, passed_condition=False, first=False,
                    rejection_reason="quality",
                )
                print("Removing rejected attack: {} (quality thresholds failed)".format(
                    attacked_path))
                Path(attacked_path).unlink(missing_ok=True)
                break

            attack_scores = {
                model_name: score_file(attacked_path, scorer, detector_device)
                for model_name, scorer in detector_scorers.items()
            }
            all_baselines_fake = bool(attack_scores) and all(
                score["prediction"] == "fake"
                for score in attack_scores.values()
            )

            baseline_rejected = (
                remove_if_all_baselines_fake and all_baselines_fake)

            # Consensus-fake files are logged for auditability before removal.
            data_storing.store_metadata(output_path.name, None, None, stoi_score, pesq_score, utmos_score, 
                    attacked_path, attack_type, attack_parameter, attack_index + 1,
                    passed_condition=not baseline_rejected, first=False,
                    detector_scores=attack_scores,
                    rejection_reason=("baseline_consensus" if baseline_rejected else None))

            if baseline_rejected:
                print("Removing rejected attack: {} (all baseline models predicted fake)".format(
                    attacked_path))
                Path(attacked_path).unlink(missing_ok=True)
                break

            current_audio_path = attacked_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU ID to use for inference",
    )

    parser.add_argument(
        "--config",
        default="pipeline.config",
        help="Path to the pipeline configuration file",
    )

    parser.add_argument(
        "--evaluate-model",
        action="append",
        choices=["rawnet", "aasist", "sonar", "hyperpotter", "slsforasvspoof"],
        help="Score original and accepted attack audio in generation.json; repeat for multiple models",
    )

    main(parser.parse_args())