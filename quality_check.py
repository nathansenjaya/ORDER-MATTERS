from contextlib import contextmanager
import tempfile

import ffmpeg
import soundfile as sf
import torch

from scipy.signal import resample_poly
import numpy as np

import utmos
from pystoi import stoi
from pesq import pesq

class QualityCheckClass:
    def __init__(self):
        original_torch_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_load
        try:
            self.utmos_model = utmos.Score()
        finally:
            torch.load = original_torch_load

    @staticmethod
    @contextmanager
    def _decoded_wav(path):
        with tempfile.TemporaryDirectory() as temporary_directory:
            decoded_path = f"{temporary_directory}/decoded.wav"
            (
                ffmpeg
                .input(str(path))
                .output(decoded_path, format="wav", acodec="pcm_s16le")
                .run(overwrite_output=True, quiet=True)
            )
            yield decoded_path

    @classmethod
    def _load_audio(cls, path):
        with cls._decoded_wav(path) as decoded_path:
            audio, rate = sf.read(decoded_path, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = np.nan_to_num(audio)
        return audio, rate

    def _resample(self, audio, original_rate, target_rate):
        if original_rate == target_rate:
            return audio

        gcd = np.gcd(original_rate, target_rate)

        up = target_rate // gcd
        down = original_rate // gcd

        return resample_poly(audio, up, down)

    def _load_aligned_audio(self, original_audio_path, attacked_audio_path):
        reference, rate_ref = self._load_audio(original_audio_path)
        degraded, rate_deg = self._load_audio(attacked_audio_path)

        reference = self._resample(reference, rate_ref, 16000)
        degraded = self._resample(degraded, rate_deg, 16000)

        length = min(len(reference), len(degraded))
        if length == 0:
            raise ValueError("Audio files must contain at least one sample")

        return reference[:length], degraded[:length]

    def compute_stoi(self, original_audio_path, attacked_audio_path):
        # Compute STOI
        reference, degraded = self._load_aligned_audio(
            original_audio_path, attacked_audio_path
        )
        stoi_score = stoi(reference, degraded, 16000, extended=False)
        return stoi_score

    def compute_pesq(self, original_audio_path, attacked_audio_path):
        # Compute PESQ
        reference, degraded = self._load_aligned_audio(
            original_audio_path, attacked_audio_path
        )
        pesq_score = pesq(16000, reference, degraded, 'wb')
        return pesq_score

    def compute_utmos(self, attacked_audio_path):
        # Compute UTMOS
        with self._decoded_wav(attacked_audio_path) as decoded_path:
            utmos_score = self.utmos_model.calculate_wav_file(decoded_path)
        return utmos_score