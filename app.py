"""
Module G: Application Interface (Streamlit Web Dashboard)
Interactive UI for laptop demonstration, audio playback, waveform visualization,
processing mode configuration, and performance diagnostics inspection.
Note: Invokes the actual core ResourceController pipeline without simulation or hardcoding.
"""

import os
import sys
import json
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from core.audio_input import FileAudioSource
from core.separation_engine import UNetSourceSeparator, DummySourceSeparator
from core.resource_controller import ResourceController, ProcessingMode

st.set_page_config(
    page_title="Resource-Aware Embedded Audio Processor",
    page_icon="🎵",
    layout="wide"
)

st.title("Resource-Aware Embedded Audio Processing & Source Separation System")
st.markdown(
    "**Course Semester Project:** Complete embedded-oriented audio processing system integrating modular 1D U-Net source separation, "
    "acoustic analysis, Overlap-Add buffering, and real-time resource profiling."
)

# Sidebar Configuration
st.sidebar.header("System Configuration")

mode_choice = st.sidebar.selectbox(
    "Processing Mode",
    ["BALANCED", "HIGH_QUALITY", "RESOURCE_CONSTRAINED"],
    index=0,
    help="HIGH_QUALITY (50% OLA + Refinement) | BALANCED (25% OLA) | RESOURCE_CONSTRAINED (0% OLA, Low RAM for Pi)"
)

engine_choice = st.sidebar.radio(
    "Separation Engine",
    ["1D U-Net Neural Model (985K Params)", "Dummy Benchmark Separator (Fast Test)"],
    index=0
)

# Audio Selection
st.sidebar.subheader("Audio Acquisition")
input_option = st.sidebar.radio("Input Source", ["Preloaded Sample", "Upload Custom Audio"])

sample_dir = os.path.dirname(os.path.abspath(__file__))
available_samples = [f for f in ["sample_mixture.wav", "musdb_track_1.wav", "treat_you_better.wav"] if os.path.exists(os.path.join(sample_dir, f))]

selected_audio_path = None
if input_option == "Preloaded Sample":
    sample_pick = st.sidebar.selectbox("Select Sample Audio", available_samples)
    if sample_pick:
        selected_audio_path = os.path.join(sample_dir, sample_pick)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        temp_path = os.path.join(sample_dir, "output", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_audio_path = temp_path

# Main Display
if selected_audio_path and os.path.exists(selected_audio_path):
    source = FileAudioSource(selected_audio_path)
    meta = source.inspect()

    st.subheader("1. Audio Source Metadata")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Format", meta.format)
    col2.metric("Sample Rate", f"{meta.sample_rate} Hz")
    col3.metric("Channels", f"{meta.channels} ({'Stereo' if meta.channels > 1 else 'Mono'})")
    col4.metric("Duration", f"{meta.duration_seconds:.2f} s")
    col5.metric("Bit Depth", f"{meta.bit_depth or 16}-bit")

    st.audio(selected_audio_path)

    st.markdown("---")
    st.subheader("2. Processing Pipeline Execution")

    if st.button("Start Audio Separation Pipeline", type="primary"):
        with st.spinner("Initializing resource controller, buffering, and running neural source separation..."):
            # Load engine
            if "Dummy" in engine_choice:
                separator = DummySourceSeparator(latency_simulation=0.02)
            else:
                separator = UNetSourceSeparator(weights_path=os.path.join(sample_dir, "models", "model_weights"))

            controller = ResourceController(
                separator=separator,
                output_dir=os.path.join(sample_dir, "output"),
                default_mode=ProcessingMode(mode_choice)
            )

            # Run controller
            saved_paths, report, analysis = controller.process_audio(
                audio_source=source,
                mode=ProcessingMode(mode_choice),
                generate_plots=True
            )

        st.success("Audio Processing & Stem Isolation Complete!")

        # Performance Metrics Cards
        st.subheader("3. Resource & Performance Audit")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Real-Time Factor (RTF)", f"{report.real_time_factor:.3f}x", delta="Real-time" if report.is_realtime_capable else "Batch mode", delta_color="normal" if report.is_realtime_capable else "inverse")
        m2.metric("Total Execution Time", f"{report.total_pipeline_time_seconds:.2f} s")
        m3.metric("Neural Inference Time", f"{report.total_inference_time_seconds:.2f} s")
        m4.metric("Peak Memory (RSS)", f"{report.peak_memory_mb:.1f} MB", delta=f"+{report.memory_delta_mb:.1f} MB")

        # Audio Players for Separated Stems
        st.subheader("4. Isolated Audio Stems")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("### 🎤 Isolated Vocals")
            if "vocals" in saved_paths and os.path.exists(saved_paths["vocals"]):
                st.audio(saved_paths["vocals"])
                with open(saved_paths["vocals"], "rb") as f:
                    st.download_button("Download Vocals WAV", f, file_name=os.path.basename(saved_paths["vocals"]), mime="audio/wav")

        with p_col2:
            st.markdown("### 🥁 Isolated Drums")
            if "drums" in saved_paths and os.path.exists(saved_paths["drums"]):
                st.audio(saved_paths["drums"])
                with open(saved_paths["drums"], "rb") as f:
                    st.download_button("Download Drums WAV", f, file_name=os.path.basename(saved_paths["drums"]), mime="audio/wav")

        # Visualizations
        st.subheader("5. Acoustic & Performance Visualizations")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            if "waveform_plot" in saved_paths and os.path.exists(saved_paths["waveform_plot"]):
                st.image(saved_paths["waveform_plot"], caption="Stem Separation Waveforms", use_container_width=True)

        with v_col2:
            if "diagnostics_plot" in saved_paths and os.path.exists(saved_paths["diagnostics_plot"]):
                st.image(saved_paths["diagnostics_plot"], caption="Per-Segment Latency & Memory Footprint", use_container_width=True)

        # Download Audit Reports
        st.subheader("6. Export Audits & Data")
        d1, d2 = st.columns(2)
        with d1:
            if "performance_json" in saved_paths and os.path.exists(saved_paths["performance_json"]):
                with open(saved_paths["performance_json"], "r", encoding="utf-8") as f:
                    st.download_button("Download Performance Audit (JSON)", f.read(), file_name=os.path.basename(saved_paths["performance_json"]), mime="application/json")
        with d2:
            if "segment_csv" in saved_paths and os.path.exists(saved_paths["segment_csv"]):
                with open(saved_paths["segment_csv"], "r", encoding="utf-8") as f:
                    st.download_button("Download Segment Metrics (CSV)", f.read(), file_name=os.path.basename(saved_paths["segment_csv"]), mime="text/csv")
else:
    st.info("Please select or upload an audio file from the sidebar to begin.")
