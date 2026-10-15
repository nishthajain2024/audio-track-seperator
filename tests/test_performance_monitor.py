"""
Unit tests for Module F: Performance Monitoring Subsystem
"""

import os
import tempfile
import json
import csv
import pytest

from core.performance_monitor import PerformanceMonitor, PerformanceReport, SegmentMetric


class TestPerformanceMonitor:

    @pytest.fixture
    def sample_report(self):
        metrics = [
            SegmentMetric(segment_index=0, duration_seconds=6.8, inference_time_seconds=1.2, rtf=0.176, memory_rss_mb=250.0),
            SegmentMetric(segment_index=1, duration_seconds=6.8, inference_time_seconds=1.1, rtf=0.161, memory_rss_mb=255.0)
        ]
        return PerformanceReport(
            run_timestamp="2026-10-16T00:00:00",
            audio_file="test.wav",
            processing_mode="BALANCED",
            audio_duration_seconds=13.6,
            total_pipeline_time_seconds=3.0,
            preprocessing_time_seconds=0.3,
            total_inference_time_seconds=2.3,
            postprocessing_time_seconds=0.4,
            real_time_factor=0.22,
            is_realtime_capable=True,
            num_segments=2,
            avg_segment_latency_seconds=1.15,
            initial_memory_mb=240.0,
            peak_memory_mb=255.0,
            memory_delta_mb=15.0,
            memory_measurement_method="psutil_rss",
            output_validity={"vocals": {"has_nans": False, "is_silent": False}},
            segment_metrics=metrics,
            errors=[]
        )

    def test_json_and_csv_export(self, sample_report):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(output_dir=temp_dir)
            json_path = monitor.save_report_json(sample_report, "test_run")
            csv_path = monitor.save_report_csv(sample_report, "test_run")

            assert os.path.exists(json_path)
            assert os.path.exists(csv_path)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert data["processing_mode"] == "BALANCED"
                assert len(data["segments"]) == 2

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 3  # header + 2 segments
                assert rows[0] == ["segment_index", "duration_sec", "inference_latency_sec", "segment_rtf", "memory_rss_mb"]

    def test_plot_generation(self, sample_report):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = PerformanceMonitor(output_dir=temp_dir)
            plot_path = monitor.plot_performance_diagnostics(sample_report, "test_run")
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 1000
