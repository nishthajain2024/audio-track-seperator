"""
Module E: Resource-Aware Processing Controller (Primary Original Contribution)
Coordinates audio acquisition, preprocessing, neural source separation,
acoustic analysis, performance monitoring, and output generation.
Adapts execution strategies according to resource constraints and processing modes.
"""

import os
import gc
import time
from datetime import datetime
from enum import Enum
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from core.audio_input import BaseAudioSource, FileAudioSource, AudioMetadata
from core.audio_preprocessing import AudioPreprocessor, PreprocessingConfig, PreprocessingState
from core.separation_engine import BaseSourceSeparator, UNetSourceSeparator, DummySourceSeparator
from core.audio_analysis import AudioAnalyzer, AudioAnalysisReport
from core.audio_output import AudioOutputManager
from core.performance_monitor import PerformanceMonitor, PerformanceReport, SegmentMetric, PSUTIL_AVAILABLE


class ProcessingMode(str, Enum):
    """Execution modes tailoring resource consumption vs audio quality."""
    HIGH_QUALITY = "HIGH_QUALITY"                      # 50% OLA, full HPSS refinement, rich analysis
    BALANCED = "BALANCED"                              # 25% OLA, fast refinement, standard analysis
    RESOURCE_CONSTRAINED = "RESOURCE_CONSTRAINED"      # 0% OLA (low RAM), no heavy STFT refinement, GC enabled


class ResourceController:
    """
    Central controller managing lifecycle, pipeline execution, and performance monitoring.
    Designed for deployment on both laptop development machines and low-power embedded Linux boards (e.g. Raspberry Pi).
    """

    def __init__(
        self,
        separator: Optional[BaseSourceSeparator] = None,
        output_dir: str = "./output",
        default_mode: ProcessingMode = ProcessingMode.BALANCED,
        memory_limit_mb: float = 1800.0
    ):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.separator = separator
        self.default_mode = default_mode
        self.memory_limit_mb = memory_limit_mb
        self.output_manager = AudioOutputManager(output_dir=self.output_dir)
        self.analyzer = AudioAnalyzer(sample_rate=11025)

    def _configure_pipeline_for_mode(self, mode: ProcessingMode) -> Tuple[PreprocessingConfig, bool, float]:
        """Maps processing mode to concrete buffering and signal refinement parameters."""
        if mode == ProcessingMode.HIGH_QUALITY:
            config = PreprocessingConfig(target_sample_rate=11025, segment_length=75008, overlap_ratio=0.5)
            use_refinement = True
            drum_gain = 1.3
        elif mode == ProcessingMode.BALANCED:
            config = PreprocessingConfig(target_sample_rate=11025, segment_length=75008, overlap_ratio=0.25)
            use_refinement = True
            drum_gain = 1.2
        else:  # RESOURCE_CONSTRAINED / EMBEDDED
            config = PreprocessingConfig(target_sample_rate=11025, segment_length=75008, overlap_ratio=0.0)
            use_refinement = False
            drum_gain = 1.0

        return config, use_refinement, drum_gain

    def check_resource_guard(self, monitor: PerformanceMonitor) -> Tuple[bool, str]:
        """Checks whether process memory exceeds safety ceiling."""
        current_mem = monitor.get_current_memory_mb()
        if current_mem > self.memory_limit_mb:
            return False, f"Memory threshold exceeded: {current_mem:.1f} MB > limit {self.memory_limit_mb:.1f} MB"
        return True, "Resource bounds OK"

    def process_audio(
        self,
        audio_source: BaseAudioSource,
        mode: Optional[ProcessingMode] = None,
        generate_plots: bool = True
    ) -> Tuple[Dict[str, str], PerformanceReport, Dict[str, AudioAnalysisReport]]:
        """
        Executes the full audio processing workflow:
        Acquisition -> Preprocessing -> Inference -> Refinement -> Analysis -> Output & Diagnostics.
        """
        mode = mode or self.default_mode
        t_pipeline_start = time.perf_counter()
        monitor = PerformanceMonitor(output_dir=self.output_dir)
        errors = []

        # Ensure separator is ready
        if self.separator is None:
            self.separator = UNetSourceSeparator()

        prep_config, use_refinement, drum_gain = self._configure_pipeline_for_mode(mode)
        preprocessor = AudioPreprocessor(config=prep_config)

        # 1. Audio Acquisition
        raw_audio, in_meta = audio_source.read()
        base_name = os.path.splitext(os.path.basename(in_meta.file_path))[0]
        if base_name == "":
            base_name = f"audio_{int(time.time())}"

        # 2. Preprocessing & Buffering
        t_prep_start = time.perf_counter()
        segments, prep_state = preprocessor.process(raw_audio, in_meta.sample_rate)
        t_prep_elapsed = time.perf_counter() - t_prep_start
        monitor.update_peak_memory()

        # 3. Source Separation Engine Inference
        t_inf_start = time.perf_counter()
        segment_metrics = []
        num_segments = segments.shape[0]

        # Check memory before heavy inference
        safe, msg = self.check_resource_guard(monitor)
        if not safe:
            errors.append(msg)
            # Downgrade to constrained settings automatically
            mode = ProcessingMode.RESOURCE_CONSTRAINED

        # Execute inference
        separated_raw_stems, latencies = self.separator.separate_batch(segments)
        t_inf_elapsed = time.perf_counter() - t_inf_start

        # Record segment-by-segment latency & memory
        for i, lat in enumerate(latencies):
            seg_dur = prep_state.segment_length / prep_config.target_sample_rate
            seg_rtf = lat / seg_dur if seg_dur > 0 else 0.0
            mem_now = monitor.get_current_memory_mb()
            monitor.update_peak_memory()

            segment_metrics.append(SegmentMetric(
                segment_index=i,
                duration_seconds=seg_dur,
                inference_time_seconds=lat,
                rtf=seg_rtf,
                memory_rss_mb=mem_now,
                is_measured=True
            ))

        # 4. Overlap-Add Reconstruction
        reconstructed_stems = {}
        for stem_name, stem_segments in separated_raw_stems.items():
            reconstructed = preprocessor.reconstruct(stem_segments, prep_state, restore_sample_rate=False)
            reconstructed_stems[stem_name] = reconstructed

        # 5. Output Processing & Optional Spectral Refinement
        t_post_start = time.perf_counter()
        mix_resampled = preprocessor.reconstruct(segments, prep_state, restore_sample_rate=False)

        if use_refinement and "vocals" in reconstructed_stems and "drums" in reconstructed_stems:
            clean_voc, clean_drm, ref_metrics = self.output_manager.refine_stems(
                mixture=mix_resampled,
                pred_vocals=reconstructed_stems["vocals"],
                pred_drums=reconstructed_stems["drums"],
                sample_rate=prep_config.target_sample_rate,
                drum_gain=drum_gain
            )
            reconstructed_stems["vocals"] = clean_voc
            reconstructed_stems["drums"] = clean_drm
        else:
            ref_metrics = {"refinement": "disabled_or_skipped"}

        # Save output WAV stems
        saved_paths = self.output_manager.save_stems(
            stems=reconstructed_stems,
            sample_rate=prep_config.target_sample_rate,
            base_filename=base_name
        )

        if generate_plots:
            waveform_plot_path = self.output_manager.generate_waveform_plot(
                stems=reconstructed_stems,
                sample_rate=prep_config.target_sample_rate,
                base_filename=base_name,
                mixture=mix_resampled
            )
            saved_paths["waveform_plot"] = waveform_plot_path

        t_post_elapsed = time.perf_counter() - t_post_start
        monitor.update_peak_memory()

        # 6. Audio Analysis on Stems
        analysis_reports = {}
        analysis_reports["input"] = self.analyzer.analyze(mix_resampled)
        for stem_name, audio_data in reconstructed_stems.items():
            analysis_reports[stem_name] = self.analyzer.analyze(audio_data)

        # 7. Output Audio Validity Checks
        validity = {}
        for stem_name, audio_data in reconstructed_stems.items():
            validity[stem_name] = {
                "has_nans": bool(np.isnan(audio_data).any()),
                "has_infs": bool(np.isinf(audio_data).any()),
                "max_peak": round(float(np.max(np.abs(audio_data))), 4),
                "is_silent": bool(np.max(np.abs(audio_data)) < 1e-5)
            }

        # 8. Final Performance Reporting
        t_pipeline_total = time.perf_counter() - t_pipeline_start
        audio_dur = in_meta.duration_seconds
        total_rtf = t_pipeline_total / audio_dur if audio_dur > 0 else 0.0

        if mode == ProcessingMode.RESOURCE_CONSTRAINED:
            # Aggressive cleanup on embedded targets
            gc.collect()

        perf_report = PerformanceReport(
            run_timestamp=datetime.now().isoformat(),
            audio_file=in_meta.file_path,
            processing_mode=mode.value,
            audio_duration_seconds=audio_dur,
            total_pipeline_time_seconds=t_pipeline_total,
            preprocessing_time_seconds=t_prep_elapsed,
            total_inference_time_seconds=t_inf_elapsed,
            postprocessing_time_seconds=t_post_elapsed,
            real_time_factor=total_rtf,
            is_realtime_capable=(total_rtf <= 1.0),
            num_segments=num_segments,
            avg_segment_latency_seconds=float(np.mean(latencies)) if latencies else 0.0,
            initial_memory_mb=monitor._initial_mem,
            peak_memory_mb=monitor._peak_mem,
            memory_delta_mb=monitor._peak_mem - monitor._initial_mem,
            memory_measurement_method="psutil_rss" if PSUTIL_AVAILABLE else "tracemalloc_heap",
            output_validity=validity,
            segment_metrics=segment_metrics,
            errors=errors
        )

        # Persist diagnostic artifacts
        json_path = monitor.save_report_json(perf_report, base_name)
        csv_path = monitor.save_report_csv(perf_report, base_name)
        saved_paths["performance_json"] = json_path
        saved_paths["segment_csv"] = csv_path

        if generate_plots and len(segment_metrics) > 0:
            diag_plot = monitor.plot_performance_diagnostics(perf_report, base_name)
            if diag_plot:
                saved_paths["diagnostics_plot"] = diag_plot

        return saved_paths, perf_report, analysis_reports
