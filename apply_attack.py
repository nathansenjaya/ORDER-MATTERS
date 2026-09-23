import numpy as np
import subprocess
import ffmpeg
import random
import tempfile
from pathlib import Path
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile

class AudioAttackClass:
    def __init__(self, cafe_audio_path, parameter_profile="all"):

        """Initialize the AudioAttackClass with the WHAM background audio path."""
        self.cafe_audio_path = Path(cafe_audio_path)
        if parameter_profile not in {"all", "moderate"}:
            raise ValueError(f"Unknown attack parameter profile: {parameter_profile}")
        self.parameter_profile = parameter_profile
        self._cafe_audio_files = None
        self._fixed_parameters = {}

    def _snr_values(self):
        return [0, 10, 20, 30] if self.parameter_profile == "all" else [0, 10]

    def _bitrate_values(self):
        return (
            ["16k", "32k", "64k", "128k", "192k", "256k"]
            if self.parameter_profile == "all"
            else ["32k", "64k", "128k", "192k", "256k"]
        )

    def _reverb_values(self):
        return [0.1, 0.4, 0.7] if self.parameter_profile == "all" else [0.4, 0.7]

    def _sample_rate_values(self, opus=False):
        if self.parameter_profile == "all":
            return [8000, 12000, 16000, 24000, 48000] if opus else [8000, 16000, 22050, 48000]
        return [16000, 24000, 48000] if opus else [16000, 22050, 48000]

    def _fixed_or_random(self, key, values):
        fixed_parameters = getattr(self, "_fixed_parameters", {})
        return fixed_parameters.get(key, random.choice(values))

    @staticmethod
    def _prepare_paths(input_audio_path, output_folder):
        """
        Prepare the input and output paths.
        """
        return Path(input_audio_path), Path(output_folder)

    @staticmethod
    def _read_audio(input_audio_path, sample_rate=None, num_channels=None):

        """ Read an audio file using ffmpeg and return the audio data, sample rate, and number of channels. """
        
        with tempfile.TemporaryDirectory() as temporary_directory:
            decoded_path = Path(temporary_directory) / "decoded.wav"
            output_options = {"format": "wav", "acodec": "pcm_s16le"}
            if sample_rate is not None:
                output_options["ar"] = sample_rate
            if num_channels is not None:
                output_options["ac"] = num_channels
            (
                ffmpeg
                .input(str(input_audio_path))
                .output(str(decoded_path), **output_options)
                .run(overwrite_output=True, quiet=True)
            )

            with AudioFile(str(decoded_path), 'r') as audio_file:
                audio = audio_file.read(audio_file.frames)
                return audio, audio_file.samplerate, audio_file.num_channels

    @staticmethod
    def _audio_duration(input_audio_path):
        probe = ffmpeg.probe(str(input_audio_path))
        audio_stream = next(
            stream for stream in probe["streams"]
            if stream.get("codec_type") == "audio"
        )
        duration = audio_stream.get("duration") or probe["format"].get("duration")
        if duration is None:
            raise ValueError(f"No duration found for {input_audio_path}")
        return float(duration)

    @staticmethod
    def _effect_output_suffix(input_audio_path):
        """ Return the suffix of the effect output file. """
        return input_audio_path.suffix

    def _write_effected_audio(self, audio, sample_rate, num_channels, output_audio_path, bitrate=None):

        """ Write the effected audio to the specified output path using ffmpeg. """

        with tempfile.TemporaryDirectory() as temporary_directory:
            decoded_path = Path(temporary_directory) / "effected.wav"
            with AudioFile(
                str(decoded_path),
                'w',
                samplerate=sample_rate,
                num_channels=num_channels,
            ) as audio_file:
                audio_file.write(audio)

            suffix = output_audio_path.suffix.lower()
            codec_by_suffix = {
                ".m4a": "aac",
                ".mp3": "libmp3lame",
                ".opus": "libopus",
                ".flac": "flac",
                ".wav": "pcm_s16le",
                ".aiff": "pcm_s16le",
            }
            codec = codec_by_suffix.get(suffix)
            if codec is None:
                raise ValueError(f"Unsupported effect output format: {suffix}")

            output_options = {"acodec": codec, "ar": sample_rate}
            if suffix in {".m4a", ".mp3", ".opus"}:
                output_options["audio_bitrate"] = (
                    bitrate
                    or self._fixed_parameters.get("bitrate")
                    or self._fixed_parameters.get("output_bitrate")
                    or self._bitrate_values()[-1]
                )

            (
                ffmpeg
                .input(str(decoded_path))
                .output(str(output_audio_path), **output_options)
                .run(overwrite_output=True, quiet=True)
            )

    def random_attack(self, input_audio_path, output_folder, attack_index):

        """
        Randomly select and apply an attack method to the input audio.
        Call the function corresponding to the selected attack method,
        receive the output audio path, attack type, and attack parameter, and return them.

        Args:
            input_audio_path (str or Path): Path to the input audio file.
            output_folder (str or Path): Path to the folder where the attacked audio will be saved
            attack_index (int): Index of the attack (used for naming the output file).

        Returns:
            tuple: (output_audio_path, attack_type, attack_parameter)
        
        Attack methods include:
            - apply_reverb: Apply reverb effect to the audio.
            - apply_gaussian_noise: Add Gaussian noise to the audio.
            - apply_white_noise: Add white noise to the audio.
            - apply_cafe_background: Mix a cafe background recording into the audio.
            - apply_resampling: Resample the audio to a different sample rate.
            - compress_AAC: Compress the audio using AAC codec.
            - compress_MP3: Compress the audio using MP3 codec.
            - compress_Opus: Compress the audio using Opus codec.
        """

        attack_methods = [
            self.apply_reverb,
            self.apply_gaussian_noise,
            self.apply_white_noise,
            self.apply_cafe_background,
            self.apply_resampling,
            self.compress_AAC,
            self.compress_MP3,
            self.compress_Opus
        ]

        selected_attack = np.random.choice(attack_methods)
        return selected_attack(input_audio_path, output_folder, attack_index)

    def apply_named_attack(
        self, attack_name, input_audio_path, output_folder, attack_index,
        parameters=None,
    ):
        attack_methods = {
            "reverb": self.apply_reverb,
            "gaussian_noise": self.apply_gaussian_noise,
            "white_noise": self.apply_white_noise,
            "cafe_background": self.apply_cafe_background,
            "resampling": self.apply_resampling,
            "compression_aac": self.compress_AAC,
            "compression_mp3": self.compress_MP3,
            "compression_opus": self.compress_Opus,
        }
        try:
            selected_attack = attack_methods[attack_name]
        except KeyError as error:
            raise ValueError(f"Unknown named attack: {attack_name}") from error
        self._fixed_parameters = parameters or {}
        try:
            return selected_attack(input_audio_path, output_folder, attack_index)
        finally:
            self._fixed_parameters = {}

    def apply_cafe_background(self, input_audio_path, output_folder, attack_index):

        """
        Mix a randomly selected cafe recording into the input at a random SNR.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)
        audio, sample_rate, num_channels = self._read_audio(input_audio_path)
        target_frames = audio.shape[1]
        if self._cafe_audio_files is None:
            self._cafe_audio_files = sorted(self.cafe_audio_path.rglob("*.wav"))
        noise_files = self._cafe_audio_files

        if not noise_files:
            raise FileNotFoundError(f"No WHAM WAV files found in {self.cafe_audio_path}")

        # Probe files in random order so selection stays random without decoding
        # the entire WHAM dataset for every attack.
        random_noise_files = noise_files.copy()
        random.shuffle(random_noise_files)
        target_duration = target_frames / sample_rate
        cafe_path = next(
            (
                noise_path for noise_path in random_noise_files
                if self._audio_duration(noise_path) >= target_duration
            ),
            None,
        )
        if cafe_path is None:
            cafe_path = max(noise_files, key=self._audio_duration)

        # Use the best available noise recording even if it is shorter than the input.
        # The code below loops/crops the selected WHAM clip to match the target length
        # instead of hard-failing on a length mismatch.

        # Randomly select an SNR value.
        snr_db = self._fixed_or_random("snr_db", self._snr_values())
        cafe_audio, cafe_rate, cafe_channels = self._read_audio(
            cafe_path,
            sample_rate=sample_rate,
            num_channels=1,
        )

        # Adjust the number of channels in the cafe audio to match the input audio
        if cafe_channels != num_channels:
            if cafe_channels == 1:
                cafe_audio = np.repeat(cafe_audio, num_channels, axis=0)
            else:
                cafe_audio = cafe_audio[:num_channels]

        # Select a random section while preserving the source-length invariant.
        # If the WHAM clip is shorter than the source, loop it until it reaches the
        # required length instead of crashing.
        cafe_frames = cafe_audio.shape[1]
        if cafe_frames < target_frames:
            repeats = int(np.ceil(target_frames / cafe_frames)) + 1
            cafe_audio = np.tile(cafe_audio, (1, repeats))[:, :target_frames]
        else:
            start_frame = random.randint(0, cafe_frames - target_frames)
            cafe_audio = cafe_audio[:, start_frame:start_frame + target_frames]
        
        # Calculate the power of the input audio and the cafe audio, and scale the cafe audio to achieve the desired SNR.
        signal_power = np.mean(audio ** 2)
        cafe_power = np.mean(cafe_audio ** 2)

        # If the signal power or cafe power is too low, skip the mixing to avoid division by zero.
        if cafe_power <= 1e-12 or signal_power <= 1e-12:
            mixed_audio = audio
        else:
            target_cafe_power = signal_power / (10 ** (snr_db / 10))
            cafe_audio *= np.sqrt(target_cafe_power / cafe_power)
            mixed_audio = np.clip(audio + cafe_audio, -1.0, 1.0)

        # Keep the same format as the input
        output_audio_path = output_folder / (
            f"{input_audio_path.stem}_{attack_index}{self._effect_output_suffix(input_audio_path)}"
        )
        self._write_effected_audio(mixed_audio, sample_rate, num_channels, output_audio_path)
        print(
            f"Applied WHAM background {cafe_path.name} "
            f"with SNR {snr_db} dB"
        )
        return output_audio_path, "cafe_background", f"{snr_db}"

    def random_compression(self, input_audio_path, output_folder, attack_index):
        return self.apply_compression(input_audio_path, output_folder, attack_index)

    def choose_compression(self):

        """
        Randomly select a compression method from the available options.
        Returns the selected compression method.
        """

        attack_methods = [
            self.compress_AAC,
            self.compress_MP3,
            self.compress_Opus
        ]
        return np.random.choice(attack_methods)

    def apply_compression(self, input_audio_path, output_folder, attack_index, compression_method=None):
        
        """
        Apply a compression method to the input audio.
        If no compression method is provided, randomly select one.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """
        selected_attack = compression_method or self.choose_compression()
        return selected_attack(input_audio_path, output_folder, attack_index)

    def apply_reverb(self, input_audio_path, output_folder, attack_index):

        """
        Apply a reverb effect to the input audio using the Pedalboard library.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)
        
        # Randomly select a room size for the reverb effect
        room_size = self._fixed_or_random("room_size", self._reverb_values())

        audio, sample_rate, num_channels = self._read_audio(input_audio_path)

        # Create a pedalboard instance with a reverb effect
        board = Pedalboard([Reverb(room_size=room_size)])

        # Apply the effect
        effected_audio = board(audio, sample_rate)

        # Keep the same format as the input
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}{self._effect_output_suffix(input_audio_path)}"
        )

        self._write_effected_audio(
            effected_audio, sample_rate, num_channels, output_audio_path
        )

        print(f"Applied reverb with {room_size} strength")

        return output_audio_path, "reverb", f"{room_size}"
    
    def apply_gaussian_noise(self, input_audio_path, output_folder, attack_index):

        """
        Add Gaussian noise to the input audio at a randomly selected SNR.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Randomly select SNR for the Gaussian noise
        snr_db = self._fixed_or_random("snr_db", self._snr_values())

        audio, sample_rate, num_channels = self._read_audio(input_audio_path)

        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_amplitude = np.sqrt(noise_power)

        # Generate Gaussian noise
        gaussian_noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=audio.shape)

        # Add the Gaussian noise to the original audio
        noisy_audio = audio + gaussian_noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # Keep the same format as the input
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}{self._effect_output_suffix(input_audio_path)}"
        )

        self._write_effected_audio(
            noisy_audio, sample_rate, num_channels, output_audio_path
        )

        print(f"Applied gaussian noise with SNR {snr_db} dB")

        return output_audio_path, "gaussian_noise", f"{snr_db}"
    
    def apply_white_noise(self, input_audio_path, output_folder, attack_index):

        """
        Add white noise to the input audio at a randomly selected SNR.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Randomly select SNR for the Gaussian noise
        snr_db = self._fixed_or_random("snr_db", self._snr_values())

        audio, sample_rate, num_channels = self._read_audio(input_audio_path)

        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_amplitude = np.sqrt(noise_power)

        white_noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=audio.shape)
        noisy_audio = audio + white_noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # Keep the same format as the input
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}{self._effect_output_suffix(input_audio_path)}"
        )

        self._write_effected_audio(
            noisy_audio, sample_rate, num_channels, output_audio_path
        )

        print(f"Applied white noise with SNR {snr_db} dB")

        return output_audio_path, "white_noise", f"{snr_db}"
    
    def compress_AAC(self, input_audio_path, output_folder, attack_index):
        
        """
        Compress the input audio using AAC codec at a randomly selected bitrate.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Use ffmpeg to compress the audio file with randomly selected bitrate
        target_bitrate = self._fixed_or_random("bitrate", self._bitrate_values())

        # Change the format as the codec to AAC
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}.m4a"
        )

        (
            ffmpeg
            .input(str(input_audio_path))
            .output(str(output_audio_path), acodec='aac', audio_bitrate=target_bitrate)
            .run(overwrite_output=True)
        )
        print(f"Compressed to AAC with bitrate {target_bitrate}")

        return output_audio_path, "compression", f"AAC_{target_bitrate}"
    
    def compress_MP3(self, input_audio_path, output_folder, attack_index):

        """
        Compress the input audio using MP3 codec at a randomly selected bitrate.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Use ffmpeg to compress the audio file with randomly selected bitrate
        target_bitrate = self._fixed_or_random("bitrate", self._bitrate_values())

        # Change the format as the codec to MP3
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}.mp3"
        )

        (
            ffmpeg
            .input(str(input_audio_path))
            .output(str(output_audio_path), acodec='libmp3lame', audio_bitrate=target_bitrate)
            .run(overwrite_output=True)
        )
        print(f"Compressed to MP3 with bitrate {target_bitrate}")

        return output_audio_path, "compression", f"MP3_{target_bitrate}"

    def compress_Opus(self, input_audio_path, output_folder, attack_index):

        """
        Compress the input audio using Opus codec at a randomly selected bitrate.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """

        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Use ffmpeg to compress the audio file with randomly selected bitrate
        target_bitrate = self._fixed_or_random("bitrate", self._bitrate_values())

        # Change the format as the codec to Opus
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}.opus"
        )

        (
            ffmpeg
            .input(str(input_audio_path))
            .output(str(output_audio_path), acodec='libopus', audio_bitrate=target_bitrate)
            .run(overwrite_output=True)
        )
        print(f"Compressed to Opus with bitrate {target_bitrate}")

        return output_audio_path, "compression", f"Opus_{target_bitrate}"
    
    def apply_car_background(self, input_audio_path, output_folder, attack_index):
        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)

        # Randomly select SNR for the Car Background noise
        snr_db = self._fixed_or_random("snr_db", self._snr_values())

        audio, sample_rate, num_channels = self._read_audio(input_audio_path)

        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_amplitude = np.sqrt(noise_power)
    
    def apply_resampling(self, input_audio_path, output_folder, attack_index):

        """
        Resample the input audio to a randomly selected sample rate.
        Returns the path to the output audio file, the attack type, and the attack parameter.
        """
        
        input_audio_path, output_folder = self._prepare_paths(input_audio_path, output_folder)
        if not input_audio_path.is_file() or input_audio_path.stat().st_size == 0:
            raise ValueError(f"Cannot resample missing or empty audio file: {input_audio_path}")

        # Keep the same format as the input
        output_audio_path = (
            output_folder
            / f"{input_audio_path.stem}_{attack_index}{self._effect_output_suffix(input_audio_path)}"
        )

        suffix = output_audio_path.suffix.lower()
        supported_sample_rates = self._sample_rate_values()
        if suffix == ".opus":
            supported_sample_rates = self._sample_rate_values(opus=True)
        new_sample_rate = self._fixed_or_random("sample_rate", supported_sample_rates)

        codec_by_suffix = {
            ".m4a": "aac",
            ".mp3": "libmp3lame",
            ".opus": "libopus",
            ".flac": "flac",
            ".wav": "pcm_s16le",
            ".aiff": "pcm_s16le",
        }
        codec = codec_by_suffix.get(suffix)
        if codec is None:
            raise ValueError(f"Unsupported resampling output format: {suffix}")

        output_options = {
            "ar": new_sample_rate,
            "acodec": codec,
        }
        if suffix in {".m4a", ".mp3", ".opus"}:
            output_options["audio_bitrate"] = random.choice(self._bitrate_values())

        try:
            (
                ffmpeg
                .input(str(input_audio_path))
                .output(str(output_audio_path), **output_options)
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as error:
            output_audio_path.unlink(missing_ok=True)
            stderr = error.stderr.decode(errors="replace") if error.stderr else ""
            raise RuntimeError(
                f"Resampling failed for {input_audio_path} -> {output_audio_path}:\n{stderr}"
            ) from error

        if not output_audio_path.is_file() or output_audio_path.stat().st_size == 0:
            output_audio_path.unlink(missing_ok=True)
            raise RuntimeError(f"Resampling produced an empty file: {output_audio_path}")

        print(f"Applied resampling to {new_sample_rate} Hz")

        return output_audio_path, "resampling", f"{new_sample_rate}"
        