"""
Unit tests for Module A: Audio Input and Acquisition Subsystem
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from core.audio_input import (
    FileAudioSource,
    MicrophoneAudioSource,
    AudioFormatError,
    AudioValidationError,
    AudioMetadata
)


class TestAudioInput:

    @pytest.fixture
    def synthetic_wav_file(self):
        """Creates a temporary 2-second stereo 44.1kHz WAV file."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz left, 880 Hz right
        ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo = np.vstack([ch1, ch2]).T

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name

        sf.write(temp_path, stereo, sr, subtype='PCM_16')
        yield temp_path, sr, duration
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_inspect_metadata(self, synthetic_wav_file):
        temp_path, expected_sr, expected_dur = synthetic_wav_file
        source = FileAudioSource(temp_path)
        meta = source.inspect()

        assert isinstance(meta, AudioMetadata)
        assert meta.sample_rate == expected_sr
        assert meta.channels == 2
        assert abs(meta.duration_seconds - expected_dur) < 0.05
        assert meta.format == 'WAV'

    def test_read_audio_data(self, synthetic_wav_file):
        temp_path, expected_sr, expected_dur = synthetic_wav_file
        source = FileAudioSource(temp_path)
        audio, meta = source.read()

        assert audio.ndim == 2
        assert audio.shape[0] == 2  # 2 channels
        assert meta.peak_amplitude > 0.4
        assert not meta.is_clipped

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
            f.write(b"not an audio file")

        try:
            with pytest.raises(AudioFormatError):
                FileAudioSource(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            FileAudioSource("non_existent_audio_sample_12345.wav")

    def test_short_duration_validation(self):
        sr = 44100
        short_audio = np.zeros((int(sr * 0.1), 1))  # 0.1s
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, short_audio, sr)

        try:
            source = FileAudioSource(temp_path, min_duration=0.5)
            with pytest.raises(AudioValidationError):
                source.inspect()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_microphone_source(self):
        mic = MicrophoneAudioSource(sample_rate=16000, duration_seconds=1.0)
        meta = mic.inspect()
        assert meta.sample_rate == 16000
        assert meta.duration_seconds == 1.0

        audio, meta = mic.read()
        assert len(audio) == 16000
