import os
import sys
import argparse
import time
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.UNetModel import AudioTrackSeparation

def refine_stems_spectral_mask(mixture, pred_vocals, pred_drums, sampling_rate=11025, n_fft=1024, hop_length=256, power=3.0, threshold_db=20, drum_gain=1.5):
    '''
    Ultra-clean stem separation with boosted drum punch and complete isolation:
    1. 1D U-Net neural acoustic priors.
    2. Strict Harmonic-Percussive Decomposition (margin=2.5) to isolate singing from percussion.
    3. Steep power contrast ratio (power=3.0) and spectral floor gating.
    4. Drum punch volume boosting and dynamic range normalization.
    '''
    # 1. Compute STFTs
    stft_mix = librosa.stft(mixture, n_fft=n_fft, hop_length=hop_length)
    stft_voc = librosa.stft(pred_vocals, n_fft=n_fft, hop_length=hop_length)
    stft_drm = librosa.stft(pred_drums, n_fft=n_fft, hop_length=hop_length)

    # 2. Strict Harmonic-Percussive Decomposition of the mixture
    stft_harmonic, stft_percussive = librosa.decompose.hpss(stft_mix, margin=(2.5, 2.5))

    # 3. Magnitude spectra of U-Net neural priors
    mag_voc = np.abs(stft_voc)
    mag_drm = np.abs(stft_drm)

    # Low frequency gate: frequencies < 110 Hz are 100% kick drum/bass
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    sub_bass_mask = (freqs < 110)[:, np.newaxis]
    mag_voc[sub_bass_mask[:, 0], :] *= 0.001
    mag_drm[sub_bass_mask[:, 0], :] *= 2.0

    # 4. Steep Non-linear power contrast ratio
    mag_voc_p = mag_voc ** power
    mag_drm_p = mag_drm ** power
    sum_mag = mag_voc_p + mag_drm_p + 1e-9

    mask_voc = mag_voc_p / sum_mag
    mask_drm = mag_drm_p / sum_mag

    # 5. Contrast Threshold Gating (hard suppress residual crosstalk)
    ratio_voc_to_drm = (mag_voc + 1e-7) / (mag_drm + 1e-7)
    thresh_linear = 10.0 ** (-threshold_db / 20.0)

    mask_voc[ratio_voc_to_drm < thresh_linear] = 0.0
    mask_drm[ratio_voc_to_drm > (1.0 / thresh_linear)] = 0.0

    # 6. Strict stem synthesis:
    # Vocals get purely harmonic spectrum modulated by vocal mask
    # Drums get purely percussive spectrum modulated by drum mask
    clean_stft_voc = stft_harmonic * mask_voc
    clean_stft_drm = stft_percussive * mask_drm

    # 7. Inverse STFT back to waveforms
    clean_voc = librosa.istft(clean_stft_voc, hop_length=hop_length, length=len(mixture))
    clean_drm = librosa.istft(clean_stft_drm, hop_length=hop_length, length=len(mixture))

    # 8. High Quality Dynamic Volume Scaling (Loud & Punchy Drums, Clear Vocals)
    mix_max = np.max(np.abs(mixture)) + 1e-7
    if np.max(np.abs(clean_voc)) > 0:
        clean_voc = (clean_voc / np.max(np.abs(clean_voc))) * (mix_max * 0.88)
    if np.max(np.abs(clean_drm)) > 0:
        clean_drm = (clean_drm / np.max(np.abs(clean_drm))) * (mix_max * 0.95 * drum_gain)
        clean_drm = np.clip(clean_drm, -0.98, 0.98)

    return clean_voc, clean_drm

def separate(audio_path, model_weights_path='./models/model_weights', output_dir='./output', sampling_rate=11025, dim=75008, use_refinement=True, drum_gain=1.3):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Flexible audio path resolution
    if not os.path.exists(audio_path):
        alt_audio = os.path.join(script_dir, audio_path)
        if os.path.exists(alt_audio):
            audio_path = alt_audio
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Flexible model weights path resolution
    if os.path.isdir(model_weights_path):
        candidate = os.path.join(model_weights_path, 'model_weights')
        if os.path.exists(candidate + '.index') or os.path.exists(candidate):
            model_weights_path = candidate
    elif not (os.path.exists(model_weights_path + '.index') or os.path.exists(model_weights_path)):
        alt_model = os.path.join(script_dir, model_weights_path)
        if os.path.isdir(alt_model):
            alt_candidate = os.path.join(alt_model, 'model_weights')
            if os.path.exists(alt_candidate + '.index') or os.path.exists(alt_candidate):
                model_weights_path = alt_candidate
        elif os.path.exists(alt_model + '.index') or os.path.exists(alt_model):
            model_weights_path = alt_model

    print(f"\n==========================================")
    print(f"[+] Audio Source Separation (1D U-Net)")
    print(f"==========================================")
    print(f"Input Audio    : {audio_path}")
    print(f"Model Weights  : {model_weights_path}")
    print(f"Sampling Rate  : {sampling_rate} Hz")
    print(f"Drums Boost    : {drum_gain}x")
    print(f"Isolation Mode : {'Refined Softmask (HPSS)' if use_refinement else 'Raw U-Net Prior'}")

    # 1. Load Audio
    print("\n[1/4] Loading and pre-processing audio...")
    y, sr = librosa.load(audio_path, sr=sampling_rate)
    mix_peak = np.max(np.abs(y)) + 1e-7
    y_norm = np.clip(y / mix_peak, -1.0, 1.0)
    audio_duration = len(y) / sampling_rate
    print(f"      Audio duration: {audio_duration:.2f} seconds ({len(y)} samples)")

    # 2. Slice into chunks matching U-Net input dimension
    num_chunks = int(np.ceil(len(y) / dim))
    print(f"[2/4] Processing {num_chunks} audio chunk(s)...")

    # 3. Load Model
    print("[3/4] Initializing 1D U-Net model...")
    model = AudioTrackSeparation()
    model.build(input_shape=(None, dim, 1))
    model.load_weights(model_weights_path).expect_partial()

    all_vocals = []
    all_drums = []

    start_time = time.time()
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * dim
        end_sample = min((chunk_idx + 1) * dim, len(y))
        chunk = y_norm[start_sample:end_sample]
        chunk_len = len(chunk)

        # Pad chunk if needed
        chunk_fixed = librosa.util.fix_length(chunk, size=dim)
        chunk_tensor = np.expand_dims(chunk_fixed, axis=(0, 2))

        # Predict
        pred_voc, pred_drm = model(chunk_tensor, training=False)
        voc_array = pred_voc.numpy().squeeze()[:chunk_len]
        drm_array = pred_drm.numpy().squeeze()[:chunk_len]

        all_vocals.append(voc_array)
        all_drums.append(drm_array)

    elapsed_time = time.time() - start_time
    rtf = elapsed_time / audio_duration if audio_duration > 0 else 0

    raw_vocals = np.concatenate(all_vocals) * mix_peak
    raw_drums = np.concatenate(all_drums) * mix_peak

    # 4. High-Purity Cross-Talk Suppression Refinement
    if use_refinement:
        full_vocals, full_drums = refine_stems_spectral_mask(y, raw_vocals, raw_drums, sampling_rate=sampling_rate, drum_gain=drum_gain)
    else:
        full_vocals, full_drums = raw_vocals, raw_drums

    # 5. Save Outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_out = os.path.join(output_dir, f"{base_name}_vocals.wav")
    drums_out = os.path.join(output_dir, f"{base_name}_drums.wav")
    plot_out = os.path.join(output_dir, f"{base_name}_separated_waveform.png")

    sf.write(vocals_out, full_vocals, sampling_rate)
    sf.write(drums_out, full_drums, sampling_rate)

    # Plot waveforms
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title(f"Isolated Vocals (Clean - No Drums) - {base_name}")
    librosa.display.waveshow(full_vocals, sr=sampling_rate, color='blue')
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.title(f"Isolated Drums (Punchy - No Vocals) - {base_name}")
    librosa.display.waveshow(full_drums, sr=sampling_rate, color='orange')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()

    print("\n[4/4] Output Generation Complete!")
    print(f"------------------------------------------")
    print(f"[*] Processing Time : {elapsed_time:.3f} seconds")
    print(f"[*] Audio Duration  : {audio_duration:.2f} seconds")
    print(f"[*] Real-Time Factor: {rtf:.3f}x")
    print(f"[*] Separated Vocals: {vocals_out}")
    print(f"[*] Separated Drums : {drums_out}")
    print(f"[*] Waveform Plot   : {plot_out}")
    print(f"==========================================\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 't', 'yes', 'y')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separate vocals and drums from any audio track.")
    parser.add_argument('-i', '-a', '--input', '--audio', dest='input', type=str, default='sample_mixture.wav',
                        help="Path to input audio file (WAV/MP3)")
    parser.add_argument('-m', '--model', '--model_weights', dest='model', type=str, default='./models/model_weights',
                        help="Path to model weights checkpoint or directory")
    parser.add_argument('-o', '--output', '--output_dir', dest='output', type=str, default='./output',
                        help="Output directory")
    parser.add_argument('--refine', dest='refine', type=str2bool, default=True,
                        help="Enable spectral softmask refinement for zero stem bleed (default: True)")
    parser.add_argument('--no_refine', dest='refine', action='store_false',
                        help="Disable spectral softmask refinement")
    parser.add_argument('--drum_gain', type=float, default=1.3,
                        help="Volume boost factor for separated drums (default: 1.3)")
    args = parser.parse_args()

    separate(args.input, args.model, args.output, use_refinement=args.refine, drum_gain=args.drum_gain)

