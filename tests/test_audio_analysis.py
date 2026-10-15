"""
Unit tests for Module D: Audio Analysis Subsystem
"""

import numpy as np
import pytest

from core.audio_analysis import AudioAnalyzer, AudioAnalysisReport, SpectralBandEnergy


class TestAudioAnalysis:

    @pytest.fixture
    def test_tones(self):
        sr = 11025
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Low frequency kick (50 Hz) + High frequency cymbal (4500 Hz)
        low_tone = 0.5 * np.sin(2 * np.pi * 50 * t)
        high_tone = 0.3 * np.sin(2 * np.pi * 4500 * t)
        return (low_tone + high_tone).astype(np.float32), sr

    def test_full_analysis(self, test_tones):
        audio, sr = test_tones
        analyzer = AudioAnalyzer(sample_rate=sr)
        report = analyzer.analyze(audio)

        assert isinstance(report, AudioAnalysisReport)
        assert report.sample_rate == sr
        assert abs(report.duration_seconds - 1.0) < 0.05
        assert report.peak_amplitude > 0.4
        assert not report.is_clipped
        assert report.clipping_sample_count == 0

    def test_clipping_detection(self):
        sr = 11025
        clipped_audio = np.ones(500, dtype=np.float32)
        analyzer = AudioAnalyzer(sample_rate=sr)
        report = analyzer.analyze(clipped_audio)

        assert report.is_clipped is True
        assert report.clipping_sample_count == 500

    def test_spectral_bands(self, test_tones):
        audio, sr = test_tones
        analyzer = AudioAnalyzer(sample_rate=sr)
        bands = analyzer.compute_spectral_bands(audio)

        assert isinstance(bands, SpectralBandEnergy)
        # 50 Hz should have high sub-bass energy
        assert bands.sub_bass_energy > 0.1
        # 4500 Hz should have high frequency energy
        assert bands.high_energy > 0.1

    def test_spectrogram_and_rms(self, test_tones):
        audio, sr = test_tones
        analyzer = AudioAnalyzer(sample_rate=sr)
        spec = analyzer.compute_spectrogram(audio)
        rms = analyzer.compute_rms_envelope(audio)

        assert spec.ndim == 2
        assert len(rms) > 0
