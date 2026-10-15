"""
Module B: Audio Preprocessing and Buffering Subsystem
Handles resampling, mono/stereo conversion, dynamic range normalization,
overlap-based segmentation, and Overlap-Add (OLA) reconstruction.
Designed to be completely testable independently of neural network inference.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class PreprocessingConfig:
    """Configuration tokens for audio preprocessing."""
    target_sample_rate: int = 11025
    segment_length: int = 75008      # Exactly matches 1D U-Net receptive field
    overlap_ratio: float = 0.5       # 0.0 = direct tiling (embedded), 0.5 = 50% OLA (quality)
    normalize_peak: bool = True
    target_peak: float = 0.95


@dataclass
class PreprocessingState:
    """State preserved during preprocessing to allow exact reconstruction."""
    original_sample_rate: int
    original_num_samples: int
    resampled_num_samples: int
    was_stereo: bool
    stereo_channel_weights: Optional[np.ndarray] = None
    peak_gain_factor: float = 1.0
    segment_length: int = 75008
    hop_length: int = 37504
    num_segments: int = 0
    padded_length: int = 0
    overlap_ratio: float = 0.5


class AudioPreprocessor:
    """
    Modular preprocessing pipeline executing:
    1. Channel downmixing (Stereo -> Mono) with energy preservation
    2. High-quality anti-aliasing resampling
    3. Peak normalization and clipping prevention
    4. Overlap-aware windowed segmentation
    5. Invertible Overlap-Add (OLA) reconstruction
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def process(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, PreprocessingState]:
        """
        Transforms raw input audio into a batch of segmented model inputs.
        Args:
            audio: Shape (samples,) for mono or (channels, samples) for stereo/multi-channel.
            sample_rate: Native sample rate of the input audio.
        Returns:
            segments: Array of shape (num_segments, segment_length, 1) ready for model inference.
            state: PreprocessingState containing parameters needed for reconstruction.
        """
        orig_samples = audio.shape[-1]
        was_stereo = audio.ndim > 1 and audio.shape[0] > 1

        # 1. Channel downmixing to Mono
        if was_stereo:
            # Average stereo channels into mono
            mono_audio = np.mean(audio, axis=0).astype(np.float32)
        elif audio.ndim > 1:
            mono_audio = audio.squeeze().astype(np.float32)
        else:
            mono_audio = audio.astype(np.float32)

        # 2. Resampling to model target sample rate
        if sample_rate != self.config.target_sample_rate:
            resampled = librosa.resample(
                mono_audio,
                orig_sr=sample_rate,
                target_sr=self.config.target_sample_rate,
                res_type='kaiser_best'
            )
        else:
            resampled = mono_audio

        resampled_samples = len(resampled)

        # 3. Dynamic range peak normalization
        peak_gain = 1.0
        if self.config.normalize_peak:
            peak = float(np.max(np.abs(resampled)))
            if peak > 1e-7:
                peak_gain = peak / self.config.target_peak
                normalized = resampled / peak_gain
            else:
                normalized = resampled
        else:
            normalized = resampled

        # 4. Windowed Segmentation (Overlap-Add support)
        seg_len = self.config.segment_length
        overlap = self.config.overlap_ratio

        if overlap > 0.0:
            hop_len = int(seg_len * (1.0 - overlap))
            # Ensure hop_len is a multiple of 32 for U-Net pool/upsample alignment
            hop_len = max(32, (hop_len // 32) * 32)
        else:
            hop_len = seg_len

        # Pad end so all audio is covered by segments
        if len(normalized) < seg_len:
            pad_amount = seg_len - len(normalized)
            padded = np.pad(normalized, (0, pad_amount), mode='constant')
        else:
            remainder = (len(normalized) - seg_len) % hop_len
            pad_amount = (hop_len - remainder) if remainder != 0 else 0
            padded = np.pad(normalized, (0, pad_amount), mode='constant')

        padded_len = len(padded)
        num_segments = (padded_len - seg_len) // hop_len + 1

        segments = np.zeros((num_segments, seg_len, 1), dtype=np.float32)

        # Apply analysis window if overlapping
        if overlap > 0.0:
            # Square-root Hann window for perfect COLA (Constant Overlap-Add) reconstruction
            window = np.sqrt(np.hanning(seg_len)).astype(np.float32)
        else:
            window = np.ones(seg_len, dtype=np.float32)

        for i in range(num_segments):
            start = i * hop_len
            end = start + seg_len
            chunk = padded[start:end] * window
            segments[i, :, 0] = chunk

        state = PreprocessingState(
            original_sample_rate=sample_rate,
            original_num_samples=orig_samples,
            resampled_num_samples=resampled_samples,
            was_stereo=was_stereo,
            peak_gain_factor=peak_gain,
            segment_length=seg_len,
            hop_length=hop_len,
            num_segments=num_segments,
            padded_length=padded_len,
            overlap_ratio=overlap
        )

        return segments, state

    def reconstruct(self, segments: np.ndarray, state: PreprocessingState, restore_sample_rate: bool = False) -> np.ndarray:
        """
        Invertible Overlap-Add reconstruction.
        Assembles separated segments back into continuous audio.
        Args:
            segments: Array of shape (num_segments, segment_length) or (num_segments, segment_length, 1)
            state: PreprocessingState preserved from process()
            restore_sample_rate: If True, resamples audio back to original source sample rate.
        Returns:
            reconstructed: Continuous audio array matching source duration.
        """
        if segments.ndim == 3:
            segments = segments.squeeze(-1)

        seg_len = state.segment_length
        hop_len = state.hop_length
        padded_len = state.padded_length

        reconstructed = np.zeros(padded_len, dtype=np.float32)
        norm_buffer = np.zeros(padded_len, dtype=np.float32)

        if state.overlap_ratio > 0.0:
            # Synthesis window matching analysis window
            synthesis_window = np.sqrt(np.hanning(seg_len)).astype(np.float32)
        else:
            synthesis_window = np.ones(seg_len, dtype=np.float32)

        for i in range(state.num_segments):
            start = i * hop_len
            end = start + seg_len
            chunk = segments[i] * synthesis_window
            reconstructed[start:end] += chunk
            norm_buffer[start:end] += (synthesis_window ** 2)

        # Normalize by window overlap sum to prevent amplitude distortion
        mask = norm_buffer > 1e-4
        reconstructed[mask] /= norm_buffer[mask]

        # Truncate padding to exact resampled length
        reconstructed = reconstructed[:state.resampled_num_samples]

        # Restore original dynamic range
        reconstructed = reconstructed * state.peak_gain_factor

        # Prevent digital clipping
        reconstructed = np.clip(reconstructed, -1.0, 1.0)

        # Optional: Resample back to original sample rate
        if restore_sample_rate and state.original_sample_rate != self.config.target_sample_rate:
            reconstructed = librosa.resample(
                reconstructed,
                orig_sr=self.config.target_sample_rate,
                target_sr=state.original_sample_rate,
                res_type='kaiser_best'
            )
            # Match exact original length
            reconstructed = librosa.util.fix_length(reconstructed, size=state.original_num_samples)

        return reconstructed
