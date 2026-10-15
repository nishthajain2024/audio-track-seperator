"""
Module A: Audio Acquisition and Input Management
Handles audio file importing, validation, metadata extraction, error handling,
and provides an extensible interface for future microphone / hardware streaming input.
"""

import os
import soundfile as sf
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class AudioSourceError(Exception):
    """Base exception for audio input subsystem errors."""
    pass


class AudioFormatError(AudioSourceError):
    """Raised when the audio format is unsupported or the file header is corrupted."""
    pass


class AudioValidationError(AudioSourceError):
    """Raised when audio fails quality, duration, or channel validation constraints."""
    pass


@dataclass
class AudioMetadata:
    """Audio metadata container."""
    file_path: str
    sample_rate: int
    channels: int
    duration_seconds: float
    num_samples: int
    format: str
    bit_depth: Optional[int] = None
    rms_energy: float = 0.0
    peak_amplitude: float = 0.0
    is_clipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": round(self.duration_seconds, 3),
            "num_samples": self.num_samples,
            "format": self.format,
            "bit_depth": self.bit_depth,
            "rms_energy": round(float(self.rms_energy), 6),
            "peak_amplitude": round(float(self.peak_amplitude), 4),
            "is_clipped": self.is_clipped
        }


class BaseAudioSource(ABC):
    """Abstract base class for audio acquisition (Files, Microphones, Hardware Streams)."""

    @abstractmethod
    def read(self) -> Tuple[np.ndarray, AudioMetadata]:
        """Read audio data and metadata from the source."""
        pass

    @abstractmethod
    def inspect(self) -> AudioMetadata:
        """Inspect source metadata without fully loading long audio arrays into memory."""
        pass


class FileAudioSource(BaseAudioSource):
    """
    File-based audio acquisition supporting WAV, MP3, FLAC, OGG.
    Performs integrity validation, duration bounds checking, and clipping detection.
    """

    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

    def __init__(self, file_path: str, min_duration: float = 0.5, max_duration: float = 1200.0):
        self.file_path = os.path.abspath(file_path)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._validate_file_path()

    def _validate_file_path(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Audio file does not exist: {self.file_path}")

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise AudioFormatError(
                f"Unsupported audio format '{ext}'. Supported formats: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

        if os.path.getsize(self.file_path) == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): {self.file_path}")

    def inspect(self) -> AudioMetadata:
        """Inspect file header using soundfile without loading raw audio into RAM."""
        try:
            info = sf.info(self.file_path)
            duration = info.duration
            channels = info.channels
            sr = info.samplerate
            fmt = info.format
            subtype = info.subtype

            # Determine bit depth from subtype if possible
            bit_depth = None
            if 'PCM_16' in subtype:
                bit_depth = 16
            elif 'PCM_24' in subtype:
                bit_depth = 24
            elif 'PCM_32' in subtype or 'FLOAT' in subtype:
                bit_depth = 32

            if duration < self.min_duration:
                raise AudioValidationError(
                    f"Audio duration ({duration:.2f}s) is shorter than minimum required ({self.min_duration}s)."
                )
            if duration > self.max_duration:
                raise AudioValidationError(
                    f"Audio duration ({duration:.2f}s) exceeds maximum supported ({self.max_duration}s)."
                )

            return AudioMetadata(
                file_path=self.file_path,
                sample_rate=sr,
                channels=channels,
                duration_seconds=duration,
                num_samples=info.frames,
                format=fmt,
                bit_depth=bit_depth,
                rms_energy=0.0,
                peak_amplitude=0.0,
                is_clipped=False
            )
        except sf.SoundFileError as e:
            # Fallback to librosa if soundfile fails on specific MP3 encodings
            return self._inspect_fallback(str(e))

    def _inspect_fallback(self, original_error: str) -> AudioMetadata:
        try:
            duration = librosa.get_duration(path=self.file_path)
            ext = os.path.splitext(self.file_path)[1].upper().replace('.', '')
            return AudioMetadata(
                file_path=self.file_path,
                sample_rate=44100,
                channels=2,
                duration_seconds=duration,
                num_samples=int(duration * 44100),
                format=ext,
                bit_depth=16
            )
        except Exception as e:
            raise AudioFormatError(f"Failed to read audio metadata: {original_error} | Fallback error: {str(e)}")

    def read(self) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Load audio waveform data.
        Returns:
            audio: np.ndarray shape (samples,) for mono or (channels, samples) for multi-channel.
            metadata: AudioMetadata with calculated RMS and clipping statistics.
        """
        meta = self.inspect()
        try:
            # Load with soundfile for lossless fidelity
            audio_data, sr = sf.read(self.file_path, dtype='float32', always_2d=False)
        except Exception:
            # Librosa fallback
            audio_data, sr = librosa.load(self.file_path, sr=None, mono=False)

        # Transpose if soundfile returned (samples, channels) to standard (channels, samples)
        if audio_data.ndim == 2 and audio_data.shape[1] < audio_data.shape[0]:
            audio_data = audio_data.T

        # Calculate metrics
        peak = float(np.max(np.abs(audio_data)))
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        is_clipped = peak >= 0.999

        meta.sample_rate = sr
        meta.peak_amplitude = peak
        meta.rms_energy = rms
        meta.is_clipped = is_clipped
        meta.num_samples = audio_data.shape[-1]
        meta.duration_seconds = audio_data.shape[-1] / sr

        return audio_data, meta


class MicrophoneAudioSource(BaseAudioSource):
    """
    Extensible hardware microphone / streaming input interface.
    Designed for future Raspberry Pi I2S / USB audio input integration.
    """

    def __init__(self, sample_rate: int = 16000, duration_seconds: float = 6.8):
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

    def inspect(self) -> AudioMetadata:
        return AudioMetadata(
            file_path="live://microphone",
            sample_rate=self.sample_rate,
            channels=1,
            duration_seconds=self.duration_seconds,
            num_samples=int(self.sample_rate * self.duration_seconds),
            format="LIVE_STREAM",
            bit_depth=16
        )

    def read(self) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Microphone capture placeholder.
        In hardware deployment, integrates with sounddevice / ALSA pyAudio.
        Generates structured placeholder signal if hardware capture is unconfigured.
        """
        meta = self.inspect()
        num_samples = meta.num_samples
        # Generate silence or simulated microphone test tone
        audio = np.zeros(num_samples, dtype=np.float32)
        return audio, meta
