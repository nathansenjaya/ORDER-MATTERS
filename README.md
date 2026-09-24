# Iterative Transformation Main Generation Pipeline

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

## Repository map

This project is organized around a single generation pipeline and a few
supporting assets. The main execution flow is driven by [main.py](main.py),
which reads a JSON config, creates speaker identities, dispatches jobs to the
TTS workers, applies randomized audio attacks, records metadata, and scores the
resulting audio with detector models.

### Core pipeline files

- [main.py](main.py): main orchestration script for scheduling data generation,
  dispatching TTS workers, applying attacks, and saving metadata.
- [apply_attack.py](apply_attack.py): randomized audio attack pipeline used to
  create adversarial or degraded variants of generated speech.
- [quality_check.py](quality_check.py): STOI, PESQ, and UTMOS quality checks used
  to reject low-quality attacked outputs.
- [data_storing.py](data_storing.py): writes generation metadata and attack logs to
  JSON files.
- [speaker_identities.py](speaker_identities.py): builds prompt identities and
  LibriTTS-R speaker pools used for generated and cloned voice jobs.
- [evaluate_generated_audio.py](evaluate_generated_audio.py): loads detector models
  and scores generated or attacked audio files.

### Model workers

- [qwen_worker.py](qwen_worker.py): Qwen TTS voice design and clone implementation.
- [chatterbox_worker.py](chatterbox_worker.py): Chatterbox clone-only pipeline.
- [cosyvoice_worker.py](cosyvoice_worker.py): CosyVoice clone-only pipeline using a
  separate model checkout and model download step.

### Config and prompt files

- [pipeline.config](pipeline.config): production config used by the local pipeline.
- [config/main.example.json](config/main.example.json): portable example config for
  running from a public repository checkout.
- [config/models/qwen.example.config](config/models/qwen.example.config): minimal
  example model configuration for a Qwen job.
- [description.json](description.json): generated-speaker description schema used to
  build synthetic voice identity prompts.
- [instruct.json](instruct.json): prompt instructions used by CosyVoice-style
  generation tasks.

### Generated data and logs

- [generation_previous.json](generation_previous.json):  Data log for attack progression and scoring across samples on the main 11,660 audio data.
- [generation_clean.json](generation_clean.json): Data log for attack progression and scoring across samples on order-permuted experiment for LibriTTS.
- [generation_trajectory.json](generation_trajectory.json): Data log for attack progression and scoring across samples on order-permuted experiment for LibriTTS-R, Qwen3TTS-1.7B-base, Chatterbox-V3, and FunCosyVoice3.
- [speaker_identity.json](speaker_identity.json): Generated speaker identity pool.
- [identity_manifest.json](identity_manifest.json): Combined manifest pairing generated identities and LibriTTS-R speaker references.
- [checkpoint/aasist/best.pth](checkpoint/aasist/best.pth): The fine-tuned AASIST weight on our pipeline's generated data.
  
### Top-level utilities and evaluation scripts

The repository also contains additional helper scripts for dataset expansion,
trajectory processing, and evaluation, such as trajectory builders, sweeps,
quality filtering, and attack-level analysis. Those are not required for the
minimal runnable generation pipeline, but they are useful for the full research
workflow behind the manuscript.

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
