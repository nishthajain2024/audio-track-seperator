"""
Module F: Performance Monitoring and Diagnostics Subsystem
Records latency, memory consumption, Real-Time Factor (RTF), error logs,
exports CSV/JSON reports, and plots system performance diagnostics.
"""

import os
import json
import csv
import time
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class SegmentMetric:
    """Individual segment performance observation."""
    segment_index: int
    duration_seconds: float
    inference_time_seconds: float
    rtf: float
    memory_rss_mb: float
    is_measured: bool = True


@dataclass
class PerformanceReport:
    """Consolidated performance audit for an entire processing run."""
    run_timestamp: str
    audio_file: str
    processing_mode: str
    audio_duration_seconds: float
    total_pipeline_time_seconds: float
    preprocessing_time_seconds: float
    total_inference_time_seconds: float
    postprocessing_time_seconds: float
    real_time_factor: float
    is_realtime_capable: bool
    num_segments: int
    avg_segment_latency_seconds: float
    initial_memory_mb: float
    peak_memory_mb: float
    memory_delta_mb: float
    memory_measurement_method: str
    output_validity: Dict[str, Any]
    segment_metrics: List[SegmentMetric] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "audio_file": self.audio_file,
            "processing_mode": self.processing_mode,
            "audio_duration_seconds": round(self.audio_duration_seconds, 3),
            "total_pipeline_time_seconds": round(self.total_pipeline_time_seconds, 3),
            "preprocessing_time_seconds": round(self.preprocessing_time_seconds, 3),
            "total_inference_time_seconds": round(self.total_inference_time_seconds, 3),
            "postprocessing_time_seconds": round(self.postprocessing_time_seconds, 3),
            "real_time_factor": round(self.real_time_factor, 3),
            "is_realtime_capable": self.is_realtime_capable,
            "num_segments": self.num_segments,
            "avg_segment_latency_seconds": round(self.avg_segment_latency_seconds, 4),
            "initial_memory_mb": round(self.initial_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
            "memory_measurement_method": self.memory_measurement_method,
            "output_validity": self.output_validity,
            "errors": self.errors
        }


class PerformanceMonitor:
    """Tracks and logs runtime metrics for embedded and desktop platforms."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._initial_mem = self.get_current_memory_mb()
        self._peak_mem = self._initial_mem
        tracemalloc.start()

    def get_current_memory_mb(self) -> float:
        """Reads resident memory in megabytes."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        else:
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)

    def update_peak_memory(self):
        curr = self.get_current_memory_mb()
        if curr > self._peak_mem:
            self._peak_mem = curr

    def save_report_json(self, report: PerformanceReport, filename_prefix: str) -> str:
        """Saves machine-readable JSON performance audit."""
        json_path = os.path.join(self.output_dir, f"{filename_prefix}_performance_report.json")
        data = report.to_dict()
        data["segments"] = [
            {
                "index": s.segment_index,
                "duration_s": round(s.duration_seconds, 3),
                "inference_s": round(s.inference_time_seconds, 4),
                "rtf": round(s.rtf, 3),
                "rss_mb": round(s.memory_rss_mb, 2)
            }
            for s in report.segment_metrics
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return json_path

    def save_report_csv(self, report: PerformanceReport, filename_prefix: str) -> str:
        """Saves segment-by-segment CSV metrics for data analysis."""
        csv_path = os.path.join(self.output_dir, f"{filename_prefix}_segment_metrics.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["segment_index", "duration_sec", "inference_latency_sec", "segment_rtf", "memory_rss_mb"])
            for s in report.segment_metrics:
                writer.writerow([s.segment_index, s.duration_seconds, s.inference_time_seconds, s.rtf, s.memory_rss_mb])
        return csv_path

    def plot_performance_diagnostics(self, report: PerformanceReport, filename_prefix: str) -> str:
        """
        Renders performance diagnostic charts:
        1. Inference latency per segment vs real-time threshold.
        2. Memory profile across the processing lifecycle.
        """
        plot_path = os.path.join(self.output_dir, f"{filename_prefix}_performance_diagnostics.png")

        if not report.segment_metrics:
            return ""

        seg_indices = [s.segment_index + 1 for s in report.segment_metrics]
        latencies = [s.inference_time_seconds for s in report.segment_metrics]
        durations = [s.duration_seconds for s in report.segment_metrics]
        memories = [s.memory_rss_mb for s in report.segment_metrics]

        plt.figure(figsize=(12, 7))

        # Subplot 1: Latency per segment
        plt.subplot(2, 1, 1)
        bars = plt.bar(seg_indices, latencies, color='cornflowerblue', alpha=0.85, label='Inference Latency (s)')
        # Plot audio duration line (Real-Time limit)
        if len(durations) > 0:
            avg_dur = np.mean(durations)
            plt.axhline(avg_dur, color='red', linestyle='--', linewidth=1.5, label=f'Real-Time Bound ({avg_dur:.2f}s)')

        plt.title(f"Processing Latency Profile - {report.processing_mode} (Total RTF: {report.real_time_factor:.2f}x)", fontweight='bold')
        plt.ylabel("Execution Time (s)")
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle=':', alpha=0.6)

        # Subplot 2: Memory consumption profile
        plt.subplot(2, 1, 2)
        plt.plot(seg_indices, memories, marker='o', color='darkorange', linewidth=2, label='Process RSS Memory (MB)')
        plt.title(f"Memory Footprint Across Execution (Peak: {report.peak_memory_mb:.1f} MB, Method: {report.memory_measurement_method})", fontweight='bold')
        plt.xlabel("Audio Segment Number")
        plt.ylabel("Memory (MB)")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()

        return plot_path
