import argparse
import csv
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# The bundled Fairseq version still references aliases removed in NumPy 2.x.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".opus", ".flac", ".aiff"}
ROOT = Path(__file__).resolve().parent


def load_waveform(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    return torch.from_numpy(audio.astype(np.float32))


def score_logits(logits):
    probs = F.softmax(logits, dim=1)[0]
    spoof = float(probs[0].item())
    bona_fide = float(probs[1].item())
    return spoof, bona_fide, "fake" if spoof >= 0.5 else "real"


def output_rows(files, scorer, device):
    with torch.no_grad():
        for path in files:
            waveform = load_waveform(str(path)).unsqueeze(0).to(device)
            logits = scorer(waveform)
            spoof, bona_fide, prediction = score_logits(logits)
            yield {
                "file": str(path),
                "name": path.name,
                "spoof_probability": f"{spoof:.6f}",
                "bonafide_probability": f"{bona_fide:.6f}",
                "prediction": prediction,
            }


def write_scores(files, scorer, output_path, device):
    rows = list(output_rows(files, scorer, device))
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)
    fake_rate = sum(row["prediction"] == "fake" for row in rows) / max(1, len(rows))
    print(f"Scored {len(rows)} files; fake detection rate: {fake_rate:.4f}")
    print(f"Saved: {output_path}")


def load_aasist(device, checkpoint_path=None):
    sys.path.insert(0, str(ROOT / "aasist"))
    config = json.loads((ROOT / "aasist/config/AASIST_in_the_wild.conf").read_text())
    module = importlib.import_module("models.AASIST")
    model = module.Model(config["model_config"]).to(device)
    checkpoint_path = checkpoint_path or ROOT / "aasist/models/weights/AASIST.pth"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return lambda x: model(x)[1]


def load_rawnet(device, checkpoint_path=None):
    import yaml
    module = _load_module_from_path(
        "rawnet2_detector_model", ROOT / "2021/DF/Baseline-RawNet2/model.py"
    )
    config = yaml.safe_load((ROOT / "2021/DF/Baseline-RawNet2/model_config_RawNet.yaml").read_text())
    model = module.RawNet(config["model"], str(device)).to(device)
    checkpoint_path = checkpoint_path or ROOT / "2021/pre_trained_DF_RawNet2.pth"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_sonar_full(device, checkpoint_path=None):
    sys.path.insert(0, str(ROOT / "SONAR"))
    from sonar.inference import SONARDetector
    detector = SONARDetector(
        ckpt=str(checkpoint_path or ROOT / "SONAR/checkpoints/sonar_full_xlsr_aasist_eer6.pth"),
        device=str(device),
        xlsr_ckpt=str(ROOT / "SONAR/checkpoints/xlsr2_300m.pt"),
    )

    def scorer(waveform):
        result = detector.score_audio(waveform[0].cpu().numpy())
        return torch.tensor([[result.spoof_prob, result.bonafide_prob]], device=device)

    return scorer

def load_sonar_finetune(device, checkpoint_path=None):
    sys.path.insert(0, str(ROOT / "SONAR"))
    from sonar.inference import SONARDetector
    detector = SONARDetector(
        ckpt=str(checkpoint_path or ROOT / "SONAR/checkpoints/sonar_finetune_xlsr_mamba_eer5p5.pth"),
        device=str(device),
        xlsr_ckpt=str(ROOT / "SONAR/checkpoints/xlsr2_300m.pt"),
    )

    def scorer(waveform):
        result = detector.score_audio(waveform[0].cpu().numpy())
        return torch.tensor([[result.spoof_prob, result.bonafide_prob]], device=device)

    return scorer


def load_hyperpotter(device, checkpoint_path=None):
    hyperpotter_dir = ROOT / "HyperPotter"
    xlsr_checkpoint = ROOT / "SONAR/checkpoints/xlsr2_300m.pt"
    if not xlsr_checkpoint.is_file():
        raise FileNotFoundError(
            f"HyperPotter requires the XLS-R checkpoint at {xlsr_checkpoint}."
        )
    os.environ["HYPERPOTTER_XLSR_CKPT"] = str(xlsr_checkpoint)
    if str(hyperpotter_dir) not in sys.path:
        sys.path.insert(0, str(hyperpotter_dir))
    module = _load_module_from_path(
        "hyperpotter_detector_model", hyperpotter_dir / "model.py"
    )
    model = module.Model(argparse.Namespace(), device).to(device)
    checkpoint_path = checkpoint_path or ROOT / "HyperPotter/HyperPotter.pth"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if any(key.startswith("module.") for key in state):
        state = {
            key[len("module."):]: value
            for key, value in state.items()
        }
    model.load_state_dict(state)
    model.eval()
    return model


def load_slsforasvspoof(device, checkpoint_path=None):
    sls_dir = ROOT / "SLSforASVspoof-2021-DF"
    xlsr_checkpoint = ROOT / "SONAR/checkpoints/xlsr2_300m.pt"
    if not xlsr_checkpoint.is_file():
        raise FileNotFoundError(
            f"SLS requires the XLS-R checkpoint at {xlsr_checkpoint}. "
            "Download it before using --evaluate-model slsforasvspoof."
        )
    sls_dir.mkdir(parents=True, exist_ok=True)
    if str(sls_dir) not in sys.path:
        sys.path.insert(0, str(sls_dir))
    fairseq_dir = sls_dir / "fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"
    if str(fairseq_dir) not in sys.path:
        sys.path.insert(0, str(fairseq_dir))
    os.environ["SLS_XLSR_CKPT"] = str(xlsr_checkpoint)
    module = _load_module_from_path("sls_detector_model", sls_dir / "model.py")
    model = module.Model(argparse.Namespace(), device).to(device)
    checkpoint_path = checkpoint_path or sls_dir / "MMpaper_model.pth"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {
            key[len("module."):]: value
            for key, value in state.items()
        }
    model.load_state_dict(state)
    model.eval()

    def scorer(waveform):
        audio = waveform[0].detach().cpu().numpy()
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(sls_dir)) as temp_handle:
            feature_path = Path(temp_handle.name)
        sf.write(feature_path, audio, SAMPLE_RATE)
        try:
            import librosa
            audio_features = librosa.load(str(feature_path), sr=16000)[0]
            if audio_features.size >= 64600:
                audio_features = audio_features[:64600]
            else:
                repeats = 64600 // max(1, audio_features.size) + 1
                audio_features = np.tile(audio_features, repeats)[:64600]
            logits = model(torch.from_numpy(audio_features).float().unsqueeze(0).to(device))
            return logits
        finally:
            feature_path.unlink(missing_ok=True)

    return scorer


def _load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load detector module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_detector(model_name, device, checkpoint_path=None):
    loaders = {
        "rawnet": load_rawnet,
        "aasist": load_aasist,
        "sonar-full": load_sonar_full,
        "sonar-finetune": load_sonar_finetune,
        "hyperpotter": load_hyperpotter,
        "slsforasvspoof": load_slsforasvspoof,
    }
    return loaders[model_name](device, checkpoint_path)


def detector_variants(model_name):
    checkpoints = {
        "rawnet": [
            ("pretrained", ROOT / "2021/pre_trained_DF_RawNet2.pth"),
        ],
        "aasist": [
            ("pretrained", ROOT / "aasist/models/weights/AASIST.pth"),
        ],
        "sonar-full": [
            ("pretrained", ROOT / "SONAR/checkpoints/sonar_full_xlsr_aasist_eer6.pth"),
        ],
        "hyperpotter": [
            ("pretrained", ROOT / "HyperPotter/HyperPotter.pth"),
        ],
        "slsforasvspoof": [
            ("pretrained", ROOT / "SLSforASVspoof-2021-DF/MMpaper_model.pth"),
        ],
    }
    return [variant for variant in checkpoints[model_name] if variant[1].is_file()]


def score_file(path, scorer, device):
    waveform = load_waveform(str(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = scorer(waveform)
    spoof, bona_fide, prediction = score_logits(logits)
    return {
        "spoof_probability": spoof,
        "bonafide_probability": bona_fide,
        "prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--model", choices=["rawnet", "aasist", "sonar-full", "sonar-finetune", "hyperpotter", "slsforasvspoof"], required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = sorted(path for path in args.audio_dir.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No audio files found in {args.audio_dir}")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = load_detector(args.model, device)
    output_path = args.output_dir / f"{args.model}_scores.csv"
    write_scores(files, scorer, output_path, device)


if __name__ == "__main__":
    main()
