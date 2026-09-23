import json
from pathlib import Path

class MetaDataStorage:
    def __init__(self, audio_output_folder, generation_log_path):
        """
        Check if the specified folders exist, and create them if they don't.

        Args:
            audio_output_folder (str): Path to the audio output folder.
            generation_log_path (str): Path to the generation log file.
        """
        self.audio_output_folder = Path(audio_output_folder)
        self.generation_log_path = Path(generation_log_path)

        if not self.audio_output_folder.exists():
            self.audio_output_folder.mkdir(parents=True, exist_ok=True)

        # Check and create generation log file's parent folder
        if not self.generation_log_path.parent.exists():
            self.generation_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.generation_log_path.exists():
            with self.generation_log_path.open("r", encoding="utf-8") as file:
                self.generation_log = json.load(file)
        else:
            self.generation_log = {}
    
    def store_metadata(self, audio_file_name, mode, item, stoi_score, pesq_score, utmos_score, attacked_audio_path, attack_type, attack_parameter, attack_index, passed_condition, first, detector_scores=None, identity=None, rejection_reason=None):
        if first:
            self.generation_log[audio_file_name] = {
                "mode": mode,
                "generation_type": item,
                "original_audio": audio_file_name,
                "levels": [],
                "detector_scores": detector_scores or {},
                "identity": identity or {}
            }
        else:
            level_metadata = {
                "level": attack_index,
                "attack_type": attack_type,
                "attack_parameter": attack_parameter,
                "attacked_audio": str(attacked_audio_path),
                "passed_condition": passed_condition
            }
            if stoi_score is not None:
                level_metadata["stoi"] = stoi_score
            if pesq_score is not None:
                level_metadata["pesq"] = pesq_score
            if utmos_score is not None:
                level_metadata["utmos"] = utmos_score
            if rejection_reason is not None:
                level_metadata["rejection_reason"] = rejection_reason
            self.generation_log[audio_file_name]["levels"].append(level_metadata)
            if detector_scores is not None:
                self.generation_log[audio_file_name]["levels"][-1]["detector_scores"] = detector_scores
        
        with self.generation_log_path.open("w", encoding="utf-8") as file:
            json.dump(self.generation_log, file, indent=4)
