"""
Module D: Audio Analysis Subsystem
Provides signal-processing tools to inspect waveforms, spectrograms,
RMS energy, peak amplitude, digital clipping, and frequency band energy.
"""

import numpy as np
import librosa
import librosa.display
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class SpectralBandEnergy:
    """Frequency band energy distribution."""
    sub_bass_energy: float      # 20 - 60 Hz (Kick sub, 808s)
    bass_energy: float          # 60 - 250 Hz (Bassline, kick body)
    low_mid_energy: float       # 250 - 500 Hz (Warmth, snare body)
    mid_energy: float           # 500 - 2000 Hz (Vocal body, instruments)
    high_mid_energy: float      # 2000 - 4000 Hz (Vocal presence, attack)
    high_energy: float          # 4000 - 5512 Hz (Cymbals, air, sibilance)

    def to_dict(self) -> Dict[str, float]:
        return {
            "sub_bass": round(self.sub_bass_energy, 4),
            "bass": round(self.bass_energy, 4),
            "low_mid": round(self.low_mid_energy, 4),
            "mid": round(self.mid_energy, 4),
            "high_mid": round(self.high_mid_energy, 4),
            "high": round(self.high_energy, 4)
        }


@dataclass
class AudioAnalysisReport:
    """Comprehensive analysis report for an audio signal."""
    duration_seconds: float
    sample_rate: int
    num_samples: int
    peak_amplitude: float
    rms_energy: float
    dynamic_range_db: float
    is_clipped: bool
    clipping_sample_count: int
    spectral_bands: SpectralBandEnergy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "peak_amplitude": round(float(self.peak_amplitude), 4),
            "rms_energy": round(float(self.rms_energy), 6),
            "dynamic_range_db": round(float(self.dynamic_range_db), 2),
            "is_clipped": self.is_clipped,
            "clipping_sample_count": self.clipping_sample_count,
            "spectral_bands": self.spectral_bands.to_dict()
        }


class AudioAnalyzer:
    """Signal analysis toolkit for inspection and quality auditing."""

    def __init__(self, sample_rate: int = 11025):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray) -> AudioAnalysisReport:
        """Runs full suite of acoustic analyses on audio."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        num_samples = len(audio)
        duration = num_samples / self.sample_rate if self.sample_rate > 0 else 0.0

        peak = float(np.max(np.abs(audio))) if num_samples > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if num_samples > 0 else 0.0

        # Clipping detection (> -0.05 dBFS)
        clipping_threshold = 0.994
        clipping_samples = int(np.sum(np.abs(audio) >= clipping_threshold))
        is_clipped = clipping_samples > 0

        # Dynamic range in dB
        min_nonzero = np.min(np.abs(audio)[np.abs(audio) > 1e-6]) if np.any(np.abs(audio) > 1e-6) else 1e-6
        dyn_range = 20 * np.log10((peak + 1e-7) / (min_nonzero + 1e-7))

        bands = self.compute_spectral_bands(audio)

        return AudioAnalysisReport(
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            num_samples=num_samples,
            peak_amplitude=peak,
            rms_energy=rms,
            dynamic_range_db=dyn_range,
            is_clipped=is_clipped,
            clipping_sample_count=clipping_samples,
            spectral_bands=bands
        )

    def compute_spectral_bands(self, audio: np.ndarray, n_fft: int = 1024) -> SpectralBandEnergy:
        """Decomposes signal energy into 6 standardized frequency bands using STFT."""
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        stft_mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)

        total_energy = np.sum(stft_mag ** 2) + 1e-9

        def band_energy(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                return 0.0
            return float(np.sum(stft_mag[mask, :] ** 2) / total_energy)

        return SpectralBandEnergy(
            sub_bass_energy=band_energy(20, 60),
            bass_energy=band_energy(60, 250),
            low_mid_energy=band_energy(250, 500),
            mid_energy=band_energy(500, 2000),
            high_mid_energy=band_energy(2000, 4000),
            high_energy=band_energy(4000, self.sample_rate / 2)
        )

    def compute_spectrogram(self, audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
        """Calculates log-amplitude spectrogram in decibels (dB)."""
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return db_spec

    def compute_rms_envelope(self, audio: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
        """Calculates RMS energy trajectory over time."""
        return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).squeeze()

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Standalone Acoustic Signal Analyzer")
    parser.add_argument("-a", "--audio", type=str, default="sample_mixture.wav", help="Audio file to analyze")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory for plots")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        # Check relative to parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(parent_dir, audio_path)
        if os.path.exists(alt):
            audio_path = alt
        else:
            print(f"[-] Error: File '{args.audio}' not found.")
            sys.exit(1)

    print("================================================================================")
    print("                      MODULE D: AUDIO SIGNAL ANALYSIS                           ")
    print("================================================================================")
    print(f"[*] Target Audio : {audio_path}")

    # Load audio
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    analyzer = AudioAnalyzer(sample_rate=sr)
    report = analyzer.analyze(data)

    print(f"[*] Sample Rate  : {report.sample_rate} Hz")
    print(f"[*] Duration     : {report.duration_seconds:.2f} seconds ({report.num_samples} samples)")
    print(f"[*] Peak Amp     : {report.peak_amplitude:.4f} ({'CLIPPED' if report.is_clipped else 'Clean - No Clipping'})")
    print(f"[*] RMS Energy   : {report.rms_energy:.6f}")
    print(f"[*] Dynamic Range: {report.dynamic_range_db:.2f} dB")
    print("--------------------------------------------------------------------------------")
    print("FREQUENCY BAND ENERGY DISTRIBUTION:")
    bands = report.spectral_bands.to_dict()
    for band_name, ratio in bands.items():
        bar_len = int(ratio * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {band_name:<10} [{bar}] {ratio * 100:5.1f}%")
    print("--------------------------------------------------------------------------------")

    # Generate spectrogram and waveform plots
    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_plot = os.path.join(args.output, f"{base_name}_analysis_plot.png")

    plt.figure(figsize=(12, 7))

    # Waveform
    plt.subplot(2, 1, 1)
    plt.title(f"Acoustic Waveform - {base_name}", fontweight='bold')
    t_axis = np.linspace(0, report.duration_seconds, len(data))
    plt.plot(t_axis, data, color='royalblue', linewidth=0.7)
    plt.axhline(0.99, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Clipping Threshold')
    plt.axhline(-0.99, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')

    # Spectrogram
    plt.subplot(2, 1, 2)
    plt.title("Log-Frequency Spectrogram (dB)", fontweight='bold')
    spec_db = analyzer.compute_spectrogram(data)
    img = librosa.display.specshow(spec_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=130)
    plt.close()

    print(f"[+] Saved Visual Analysis Plot: {out_plot}")
    print("================================================================================\n")

