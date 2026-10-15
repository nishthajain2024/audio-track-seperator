"""
Unit tests for Module C: Source Separation Engine
"""

import numpy as np
import pytest

from core.separation_engine import DummySourceSeparator, BaseSourceSeparator


class TestSeparationEngine:

    def test_dummy_separator_interface(self):
        separator = DummySourceSeparator(segment_length=75008, latency_simulation=0.0)
        assert isinstance(separator, BaseSourceSeparator)
        assert separator.stem_names == ["vocals", "drums"]
        assert separator.supported_segment_length == 75008

        info = separator.get_model_info()
        assert info["model_type"] == "DummySourceSeparator"
        assert info["segment_length"] == 75008

    def test_dummy_separator_inference(self):
        separator = DummySourceSeparator(segment_length=75008, latency_simulation=0.0)
        test_batch = np.ones((2, 75008, 1), dtype=np.float32)

        stems, latencies = separator.separate_batch(test_batch)

        assert "vocals" in stems
        assert "drums" in stems
        assert stems["vocals"].shape == (2, 75008)
        assert stems["drums"].shape == (2, 75008)
        assert len(latencies) == 2
        assert latencies[0] >= 0.0

    def test_dummy_2d_input_tolerance(self):
        separator = DummySourceSeparator(segment_length=75008, latency_simulation=0.0)
        test_2d = np.ones((1, 75008), dtype=np.float32)

        stems, latencies = separator.separate_batch(test_2d)
        assert stems["vocals"].shape == (1, 75008)
