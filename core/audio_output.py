"""
Module D: Audio Output Subsystem
Handles audio saving, format encoding, metadata tagging, visualization generation,
and configurable spectral refinement.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


class AudioOutputManager:
    """Manages audio file persistence, visualization, and optional stem refinement."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_stems(self, stems: Dict[str, np.ndarray], sample_rate: int, base_filename: str) -> Dict[str, str]:
        """
        Persists isolated audio tracks as uncompressed 16-bit PCM WAV files.
        Returns:
            saved_paths: Mapping from stem name to saved file path.
        """
        saved_paths = {}
        for stem_name, audio_data in stems.items():
            out_name = f"{base_filename}_{stem_name}.wav"
            out_path = os.path.join(self.output_dir, out_name)
            # Clip safely to avoid integer wrapping
            safe_audio = np.clip(audio_data, -0.999, 0.999)
            sf.write(out_path, safe_audio, sample_rate, subtype='PCM_16')
            saved_paths[stem_name] = out_path

        return saved_paths

    def refine_stems(
        self,
        mixture: np.ndarray,
        pred_vocals: np.ndarray,
        pred_drums: np.ndarray,
        sample_rate: int = 11025,
        drum_gain: float = 1.3,
        hpss_margin: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Configurable harmonic-percussive spectral masking refinement.
        Applies non-linear power soft-masking to suppress vocal bleed into drums
        and drum percussion bleed into vocals.
        """
        n_fft = 1024
        hop_length = 256

        # STFT computation
        stft_mix = librosa.stft(mixture, n_fft=n_fft, hop_length=hop_length)
        stft_voc = librosa.stft(pred_vocals, n_fft=n_fft, hop_length=hop_length)
        stft_drm = librosa.stft(pred_drums, n_fft=n_fft, hop_length=hop_length)

        # Harmonic-Percussive Decomposition
        stft_harmonic, stft_percussive = librosa.decompose.hpss(stft_mix, margin=(hpss_margin, hpss_margin))

        mag_voc = np.abs(stft_voc)
        mag_drm = np.abs(stft_drm)

        # Sub-bass gate (<110Hz is drums/bass)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        sub_bass_mask = (freqs < 110)[:, np.newaxis]
        mag_voc[sub_bass_mask[:, 0], :] *= 0.01
        mag_drm[sub_bass_mask[:, 0], :] *= drum_gain

        # Steep non-linear power ratio
        power = 2.5
        voc_p = mag_voc ** power
        drm_p = mag_drm ** power
        sum_p = voc_p + drm_p + 1e-9

        mask_voc = voc_p / sum_p
        mask_drm = drm_p / sum_p

        # Re-synthesize
        clean_stft_voc = stft_harmonic * mask_voc
        clean_stft_drm = stft_percussive * mask_drm

        clean_voc = librosa.istft(clean_stft_voc, hop_length=hop_length, length=len(mixture))
        clean_drm = librosa.istft(clean_stft_drm, hop_length=hop_length, length=len(mixture))

        # Dynamic range matching
        mix_peak = np.max(np.abs(mixture)) + 1e-7
        if np.max(np.abs(clean_voc)) > 0:
            clean_voc = (clean_voc / np.max(np.abs(clean_voc))) * (mix_peak * 0.90)
        if np.max(np.abs(clean_drm)) > 0:
            clean_drm = (clean_drm / np.max(np.abs(clean_drm))) * (mix_peak * 0.95)

        # Retention metric: Verify refinement does not destroy vocal power
        raw_voc_rms = float(np.sqrt(np.mean(pred_vocals ** 2))) + 1e-9
        refined_voc_rms = float(np.sqrt(np.mean(clean_voc ** 2)))
        retention_ratio = refined_voc_rms / raw_voc_rms

        metrics = {
            "hpss_margin": hpss_margin,
            "drum_gain": drum_gain,
            "vocal_energy_retention_ratio": round(retention_ratio, 3),
            "refinement_verified_safe": (0.6 <= retention_ratio <= 1.4)
        }

        return clean_voc, clean_drm, metrics

    def generate_waveform_plot(
        self,
        stems: Dict[str, np.ndarray],
        sample_rate: int,
        base_filename: str,
        mixture: Optional[np.ndarray] = None
    ) -> str:
        """Renders stem comparison waveform plots."""
        plot_path = os.path.join(self.output_dir, f"{base_filename}_separated_waveform.png")
        num_rows = len(stems) + (1 if mixture is not None else 0)

        plt.figure(figsize=(12, 2.5 * num_rows))
        row = 1

        if mixture is not None:
            plt.subplot(num_rows, 1, row)
            plt.title("Mixture (Input Audio)", fontsize=11, fontweight='bold')
            time_axis = np.linspace(0, len(mixture) / sample_rate, len(mixture))
            plt.plot(time_axis, mixture, color='gray', alpha=0.8, linewidth=0.7)
            plt.ylabel("Amplitude")
            row += 1

        colors = {'vocals': 'blue', 'drums': 'orange', 'other': 'purple'}
        for name, audio_data in stems.items():
            plt.subplot(num_rows, 1, row)
            plt.title(f"Isolated {name.capitalize()}", fontsize=11, fontweight='bold')
            time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
            plt.plot(time_axis, audio_data, color=colors.get(name, 'teal'), linewidth=0.7)
            plt.ylabel("Amplitude")
            row += 1

        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()

        return plot_path
