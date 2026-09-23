# IEMOCAP Main Generation Pipeline

This repository contains the main audio-generation pipeline in `main.py`.
Evaluation, trajectory, and sweep scripts are not required for the minimal run.
The pipeline needs a CUDA-capable machine for practical TTS inference, `ffmpeg`,
one supported TTS worker, a transcript TSV, reference audio, and WHAM noise.

## Minimal setup

```bash
sudo apt-get install ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-main.txt
python -m pip install -r requirements-qwen.txt
```

Install the Qwen TTS package in the same environment for the example config.
The worker interpreter can instead be set per model config with `python`, or by
setting `IEMOCAP_QWEN_PYTHON`, `IEMOCAP_CHATTERBOX_PYTHON`, or
`IEMOCAP_COSYVOICE_PYTHON`.

CosyVoice uses a separate dependency environment. Install its source checkout
and upstream requirements, then add the packages imported by this worker:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
python -m venv .venv-cosyvoice
source .venv-cosyvoice/bin/activate
python -m pip install -r CosyVoice/requirements.txt
python -m pip install -r requirements-cosyvoice.txt
```

Set `"python": "/absolute/path/to/.venv-cosyvoice/bin/python"` in a CosyVoice
model config. The worker downloads `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
into `pretrained_models/` on its first run. CosyVoice supports clone mode only
in this pipeline.

Chatterbox can also use a separate environment because its PyTorch and audio
dependencies may conflict with the other workers:

```bash
python -m venv .venv-chatterbox
source .venv-chatterbox/bin/activate
python -m pip install -r requirements-main.txt
python -m pip install -r requirements-chatterbox.txt
```

Set `"python": "/absolute/path/to/.venv-chatterbox/bin/python"` in a
Chatterbox model config. Chatterbox supports clone mode only in this pipeline.

## Inputs

Create these local paths, which are ignored by Git:

```text
data/
  transcripts.tsv       # must contain a `lang` column with `en` rows and a `sentence` column
  reference_audio/      # one or more WAV/FLAC/MP3 files, grouped by speaker directory when possible
  wham_noise/           # WAV/FLAC/MP3 files used by the attack stage
```

The `data/wham_noise` directory must contain audio even when
`maximum_attack` is zero because the attack component is initialized by the
main pipeline.

## Run

```bash
source .venv/bin/activate
python main.py --config config/main.example.json --gpu 0
```

The example generates one clone sample and writes audio and metadata under
`outputs/`. Increase `data_number`, `maximum_attack`, or add model configs in
`config/models/` for a real run. Use absolute paths in a production config when
inputs live outside the repository.

## GitHub

Do not commit model weights, datasets, generated audio, or generated metadata.
The included `.gitignore` keeps those artifacts local. A minimal first push is:

```bash
git init
git add main.py qwen_worker.py chatterbox_worker.py cosyvoice_worker.py \
  apply_attack.py quality_check.py data_storing.py speaker_identities.py \
  description.json instruct.json requirements-main.txt requirements-qwen.txt \
  requirements-cosyvoice.txt \
  requirements-chatterbox.txt \
  config .gitignore README.md
git commit -m "Add main generation pipeline"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```