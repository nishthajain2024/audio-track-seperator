"""
Unit tests for Module B: Audio Preprocessing and Overlap-Add Buffering Subsystem
"""

import numpy as np
import pytest

from core.audio_preprocessing import AudioPreprocessor, PreprocessingConfig, PreprocessingState


class TestAudioPreprocessing:

    @pytest.fixture
    def test_audio_signal(self):
        """Creates a 10-second stereo 44.1kHz chirp/sine mixture."""
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig1 = 0.6 * np.sin(2 * np.pi * 220 * t)
        sig2 = 0.4 * np.sin(2 * np.pi * 440 * t)
        stereo = np.vstack([sig1, sig2])
        return stereo, sr, duration

    def test_resampling_and_downmix(self, test_audio_signal):
        stereo, orig_sr, orig_dur = test_audio_signal
        preprocessor = AudioPreprocessor(PreprocessingConfig(target_sample_rate=11025, overlap_ratio=0.0))

        segments, state = preprocessor.process(stereo, orig_sr)

        assert state.was_stereo is True
        assert state.original_sample_rate == orig_sr
        expected_samples = int(orig_dur * 11025)
        assert abs(state.resampled_num_samples - expected_samples) <= 10
        assert segments.ndim == 3
        assert segments.shape[1] == 75008  # Model dim
        assert segments.shape[2] == 1      # Channel dim

    def test_segment_dimensions_are_multiples_of_32(self, test_audio_signal):
        stereo, orig_sr, _ = test_audio_signal
        preprocessor = AudioPreprocessor(PreprocessingConfig(overlap_ratio=0.5))

        segments, state = preprocessor.process(stereo, orig_sr)

        assert state.segment_length % 32 == 0
        assert state.hop_length % 32 == 0
        assert segments.shape[1] == 75008

    def test_invertible_overlap_add_reconstruction(self, test_audio_signal):
        stereo, orig_sr, orig_dur = test_audio_signal
        preprocessor = AudioPreprocessor(PreprocessingConfig(target_sample_rate=11025, overlap_ratio=0.5))

        segments, state = preprocessor.process(stereo, orig_sr)

        # Invert identity pass-through
        reconstructed = preprocessor.reconstruct(segments, state, restore_sample_rate=False)

        assert len(reconstructed) == state.resampled_num_samples
        # Peak amplitude restored properly
        assert np.max(np.abs(reconstructed)) > 0.4
        assert np.max(np.abs(reconstructed)) <= 1.0

    def test_zero_overlap_embedded_mode(self, test_audio_signal):
        stereo, orig_sr, orig_dur = test_audio_signal
        preprocessor = AudioPreprocessor(PreprocessingConfig(target_sample_rate=11025, overlap_ratio=0.0))

        segments, state = preprocessor.process(stereo, orig_sr)
        assert state.hop_length == state.segment_length

        reconstructed = preprocessor.reconstruct(segments, state)
        assert len(reconstructed) == state.resampled_num_samples

    def test_short_audio_padding(self):
        """Audio shorter than 1 U-Net segment (6.8s) must be cleanly padded to 75008."""
        sr = 11025
        short_audio = np.ones(10000, dtype=np.float32) * 0.5
        preprocessor = AudioPreprocessor(PreprocessingConfig(target_sample_rate=11025, overlap_ratio=0.0))

        segments, state = preprocessor.process(short_audio, sr)

        assert segments.shape[0] == 1
        assert segments.shape[1] == 75008
        reconstructed = preprocessor.reconstruct(segments, state)
        assert len(reconstructed) == 10000
        assert abs(reconstructed[5000] - 0.5) < 0.05
