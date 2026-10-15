import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import time

from Training import train
from Test import test_model
from config import get_parameters

def main(config):
    if config.train:
        '''
           Train the AudioTrackSeparation Model
        '''
        print("--- Starting Model Training ---")
        os.makedirs(config.model_save_path, exist_ok=True)
        model = train(config)
        # saving model weights
        save_file = os.path.join(config.model_save_path, 'model_weights')
        model.save_weights(save_file)
        print(f"--- Model weights successfully saved to {save_file} ---")

    else:
        '''
           Validate the AudioTrackSeparation Model on Test Audio
        '''
        if not config.test:
            raise ValueError("Please provide an audio file path for testing using --test <path/to/audio.wav>")

        test_file = config.test
        if not os.path.exists(test_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_test = os.path.join(script_dir, test_file)
            if os.path.exists(alt_test):
                test_file = alt_test

        print(f"--- Loading Audio: {test_file} ---")
        y, sr = librosa.load(test_file, duration=6.80345, sr=config.sampling_rate)
        y = librosa.util.fix_length(y, size=config.dim)
        input_audio = np.expand_dims(y, axis=1)
        input_audio = np.expand_dims(input_audio, axis=0)

        print(f"Input audio shape: {input_audio.shape} at {sr} Hz")
        start_time = time.time()
        pred = test_model(config, input_audio)
        elapsed_time = time.time() - start_time

        audio_duration = len(y) / config.sampling_rate
        rtf = elapsed_time / audio_duration
        print(f"Inference complete in {elapsed_time:.3f}s (Audio duration: {audio_duration:.2f}s, RTF: {rtf:.3f})")

        os.makedirs(config.output_dir, exist_ok=True)
        vocals_wav = os.path.join(config.output_dir, 'predicted_vocals.wav')
        drums_wav = os.path.join(config.output_dir, 'predicted_drums.wav')

        vocals_audio = pred[0][0].squeeze()
        drums_audio = pred[1][0].squeeze()

        sf.write(vocals_wav, vocals_audio, config.sampling_rate)
        sf.write(drums_wav, drums_audio, config.sampling_rate)
        print(f"Saved separated vocals to: {vocals_wav}")
        print(f"Saved separated drums to: {drums_wav}")

        title = ["Predicted Vocals", "Predicted Drums"]
        stems = [vocals_audio, drums_audio]

        plt.figure(figsize=(12, 6))
        for i in range(2):
            plt.subplot(2, 1, i + 1)
            plt.title(title[i])
            if hasattr(librosa.display, 'waveshow'):
                librosa.display.waveshow(stems[i], sr=config.sampling_rate)
            else:
                plt.plot(stems[i])
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

        plot_path = os.path.join(config.output_dir, 'separation_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved waveform visualization to: {plot_path}")

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)