# Resource-Aware Embedded Audio Processing and Intelligent Source Separation System

<img align="center" alt="architecture" src="./images/intro.png" style="background-color:white"/><br>

## Project Overview
This repository contains a modular **Embedded Audio Processing System** that incorporates a 1D Time-Domain U-Net neural network as one replaceable component within a broader audio engineering and resource-management pipeline.

Developed as an **Embedded Systems Semester Project**, the system is designed laptop-first and engineered for resource-constrained deployment on low-power single-board computers (Raspberry Pi 3B+ / 4 / Zero 2W) within a **sub-₹5,000 INR hardware budget**.

---

## System Architecture

The software architecture is modularized into discrete subsystems with clean boundaries:

```
Audio-Track-Separation/
├── core/                                # Core embedded processing pipeline (No GUI dependencies)
│   ├── audio_input.py                  # Module A: Audio acquisition, format validation, streaming interface
│   ├── audio_preprocessing.py          # Module B: Anti-aliasing resample, mono mixdown, Overlap-Add (OLA)
│   ├── separation_engine.py            # Module C: Abstract BaseSourceSeparator & UNetSourceSeparator
│   ├── audio_analysis.py               # Module D: Waveforms, Spectrograms, RMS, peak clipping, spectral bands
│   ├── audio_output.py                 # Module D: WAV stem export, metadata tagging, post-refinement
│   ├── resource_controller.py          # Module E: Resource controller, modes (Quality/Balanced/Embedded), RTF
│   └── performance_monitor.py          # Module F: Metrics logging (CSV/JSON), memory profiling, diagnostic plots
├── Model/                              # Neural network definition (existing 1D U-Net preserved)
│   ├── UNetBlocks.py
│   └── UNetModel.py
├── models/                             # Pretrained weights checkpoint (985,984 parameters)
├── embedded_deployment/                # Module H: Raspberry Pi deployment guides & TFLite conversion
│   ├── tflite_converter.py             # Converts U-Net to float32/dynamic-range TFLite
│   └── RASPBERRY_PI_GUIDE.md           # Embedded deployment and optimization guide
├── tests/                              # Verification unit & integration tests (23 tests, 100% pass)
│   ├── test_audio_input.py
│   ├── test_audio_preprocessing.py
│   ├── test_separation_engine.py
│   ├── test_audio_analysis.py
│   ├── test_resource_controller.py
│   └── test_performance_monitor.py
├── cli.py                              # Headless embedded CLI runner (100% terminal-executable)
├── app.py                              # Module G: Streamlit web UI (invokes core controller)
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## Modules Breakdown

### Module A: Audio Acquisition & Input Management (`core/audio_input.py`)
- Inspects audio headers without loading large arrays into RAM (`FileAudioSource.inspect()`).
- Supports `WAV`, `MP3`, `FLAC`, and `OGG` with duration bounds checks (min 0.5s) and format integrity validation.
- Extensible `BaseAudioSource` interface with `MicrophoneAudioSource` for future I2S/ALSA live hardware streaming.

### Module B: Audio Preprocessing & Buffering (`core/audio_preprocessing.py`)
- Resamples audio to $11,025\text{ Hz}$ using high-order anti-aliasing filters.
- Handles stereo-to-mono downmixing while preserving acoustic energy.
- **Overlap-Add (OLA) Buffering**: Implements square-root Hann windowing ($\sqrt{\text{Hann}}$) satisfying Constant Overlap-Add (COLA) properties, completely eliminating boundary click artifacts between 6.8s segments.
- Supports both **Quality Mode** (50% OLA) and **Embedded Mode** (0% OLA for minimal RAM footprint).

### Module C: Source Separation Engine (`core/separation_engine.py`)
- Integrates the 1D U-Net as an abstract `BaseSourceSeparator` plugin.
- Allows hot-swapping different neural backends (e.g., TFLite, INT8 quantized models, or dummy benchmarking mock separators) without modifying application logic.

### Module D: Audio Analysis & Output Processing (`core/audio_analysis.py`, `core/audio_output.py`)
- Computes RMS energy envelopes, log-amplitude spectrograms, peak amplitude, and digital clipping detection.
- Decomposes audio energy into 6 standardized frequency bands: Sub-Bass (20–60 Hz), Bass (60–250 Hz), Low-Mid (250–500 Hz), Mid (500–2000 Hz), High-Mid (2000–4000 Hz), and Highs (4000–5512 Hz).
- Exports 16-bit uncompressed WAV stems and comparison waveform plots.

### Module E: Resource-Aware Processing Controller (`core/resource_controller.py`)
- Primary original engineering contribution coordinating the full pipeline.
- Offers 3 configurable operational modes:
  - `HIGH_QUALITY`: 50% Overlap-Add windowing, spectral softmask refinement, comprehensive acoustic analysis.
  - `BALANCED`: 25% Overlap-Add windowing, fast refinement, standard monitoring.
  - `RESOURCE_CONSTRAINED`: 0% Overlap-Add (direct buffering), batch=1, minimal memory buffering, no heavy STFT refinement, aggressive garbage collection for low-RAM embedded SBCs.
- Real-time safety guard: Intercepts memory spikes and dynamically downgrades buffering parameters to avoid OS Out-Of-Memory (OOM) termination.

### Module F: Performance Monitoring & Diagnostics (`core/performance_monitor.py`)
- Measures segment-by-segment latency, preprocessing time, neural inference time, and postprocessing time.
- Profiles resident memory (RSS MB) using `psutil`.
- Computes Real-Time Factor: $\text{RTF} = \frac{\text{Total Pipeline Time}}{\text{Audio Duration}}$.
- Automatically exports machine-readable JSON reports, segment-by-segment CSV metrics, and diagnostic PNG plots.

### Module G: Application Interface (`cli.py` & `app.py`)
- **Headless Embedded CLI (`cli.py`)**: 100% terminal-executable without browser dependencies, designed for SSH sessions on Raspberry Pi.
- **Web UI Dashboard (`app.py`)**: Interactive Streamlit interface with live audio players, waveform views, and audit exports.

### Module H: Embedded Deployment Preparation (`embedded_deployment/`)
- Includes [tflite_converter.py](file:///c:/Users/ACER/OneDrive/Desktop/Audio-Track-Seperator/Audio-Track-Separation/embedded_deployment/tflite_converter.py) to export FP32 and INT8 dynamic-range quantized TFLite models.
- Full deployment instructions in [RASPBERRY_PI_GUIDE.md](file:///c:/Users/ACER/OneDrive/Desktop/Audio-Track-Seperator/Audio-Track-Separation/embedded_deployment/RASPBERRY_PI_GUIDE.md) covering swapfile expansion, sub-₹5,000 BOM, and headless systemd service automation.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nishthajain2024/audio-track-seperator.git
   cd audio-track-seperator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Modes

### 1. Headless CLI Execution (Embedded / Terminal)
```bash
# Run with full neural U-Net model in Resource-Constrained (Embedded) mode:
python cli.py --audio sample_mixture.wav --mode RESOURCE_CONSTRAINED --output ./output

# Run with High Quality (50% Overlap-Add + Spectral Refinement):
python cli.py --audio treat_you_better.wav --mode HIGH_QUALITY --output ./output

# Run fast benchmarking test with Mock Separator:
python cli.py --audio sample_mixture.wav --dummy --mode RESOURCE_CONSTRAINED
```

### 2. Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```

### 3. Running Verification Unit Tests
The codebase includes 23 unit tests verifying input validation, OLA reconstruction math, engine modularity, and resource tracking:
```bash
pytest tests/
```

---

## Measured Performance Benchmarks

| Metric | Laptop (Intel CPU) | Raspberry Pi 4 (2GB) Est. | Raspberry Pi 3B+ Est. |
| :--- | :--- | :--- | :--- |
| **Model Size** | 11.8 MB checkpoint | 3.94 MB TFLite (FP32) | 1.1 MB TFLite (INT8 Quant) |
| **Segment Latency** | ~18.7 s | ~35 – 45 s | ~65 – 80 s |
| **Segment Length** | 6.80 s | 6.80 s | 6.80 s |
| **Real-Time Factor (RTF)** | 3.359x | ~5.1x – 6.6x | ~9.5x – 11.7x |
| **Real-Time Feasible?** | Near Real-time | Non-real-time (Batch) | Non-real-time (Batch) |
| **Peak RAM (Embedded Mode)**| ~266 MB | ~280 MB | ~250 MB |

---

## References & Academic Grounding
1. **Wave-U-Net:** Stoller et al., *Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation*, 2018.
2. **Overlap-Add Synthesis:** Allen & Rabiner, *A unified approach to short-time Fourier analysis and synthesis*, Proc. IEEE, 1977.
3. **HPSS Decomposition:** Fitzgerald, *Harmonic/percussive separation using median filtering*, 13th Int. Conf. on Digital Audio Effects (DAFx-10), 2010.