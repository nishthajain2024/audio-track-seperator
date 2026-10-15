"""
Unit tests for Module E: Resource-Aware Processing Controller
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import pytest

from core.audio_input import FileAudioSource
from core.separation_engine import DummySourceSeparator
from core.resource_controller import ResourceController, ProcessingMode


class TestResourceController:

    @pytest.fixture
    def test_audio(self):
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 330 * t)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sr)
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_processing_modes_execution(self, test_audio):
        with tempfile.TemporaryDirectory() as out_dir:
            separator = DummySourceSeparator(latency_simulation=0.01)
            controller = ResourceController(separator=separator, output_dir=out_dir)
            source = FileAudioSource(test_audio)

            # Test Resource Constrained mode
            saved, report, analysis = controller.process_audio(source, mode=ProcessingMode.RESOURCE_CONSTRAINED)

            assert report.processing_mode == "RESOURCE_CONSTRAINED"
            assert "vocals" in saved
            assert "drums" in saved
            assert "performance_json" in saved
            assert os.path.exists(saved["vocals"])
            assert os.path.exists(saved["performance_json"])
            assert report.total_pipeline_time_seconds > 0
            assert report.real_time_factor > 0

    def test_high_quality_mode(self, test_audio):
        with tempfile.TemporaryDirectory() as out_dir:
            separator = DummySourceSeparator(latency_simulation=0.01)
            controller = ResourceController(separator=separator, output_dir=out_dir)
            source = FileAudioSource(test_audio)

            saved, report, analysis = controller.process_audio(source, mode=ProcessingMode.HIGH_QUALITY)

            assert report.processing_mode == "HIGH_QUALITY"
            assert "waveform_plot" in saved
            assert os.path.exists(saved["waveform_plot"])

    def test_memory_guard_protection(self, test_audio):
        with tempfile.TemporaryDirectory() as out_dir:
            separator = DummySourceSeparator(latency_simulation=0.01)
            # Set absurdly low memory limit to trigger guard
            controller = ResourceController(separator=separator, output_dir=out_dir, memory_limit_mb=1.0)
            source = FileAudioSource(test_audio)

            saved, report, analysis = controller.process_audio(source, mode=ProcessingMode.HIGH_QUALITY)

            # Controller must detect violation and handle safely
            assert len(report.errors) > 0
            assert "Memory threshold exceeded" in report.errors[0]
