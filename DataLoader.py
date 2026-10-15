import tensorflow as tf
import numpy as np
import librosa

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, sampling_rate, dim, batch_size=6, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices of the current batch
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        x, y = self.data_generation(batch_indices)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load(self, idx):
        """Load input and target processed audio."""
        item = self.data[int(idx)]

        mix_audio = item['mix'].audio_data
        if mix_audio.ndim > 1 and mix_audio.shape[0] > 1:
            mix_audio = np.mean(mix_audio, axis=0)
        else:
            mix_audio = mix_audio.squeeze()

        vocals_audio = item['sources']['vocals'].audio_data
        if vocals_audio.ndim > 1 and vocals_audio.shape[0] > 1:
            vocals_audio = np.mean(vocals_audio, axis=0)
        else:
            vocals_audio = vocals_audio.squeeze()

        drums_audio = item['sources']['drums'].audio_data
        if drums_audio.ndim > 1 and drums_audio.shape[0] > 1:
            drums_audio = np.mean(drums_audio, axis=0)
        else:
            drums_audio = drums_audio.squeeze()

        # Resample to target sampling rate
        input_resampled = librosa.resample(mix_audio, orig_sr=item['mix'].sample_rate, target_sr=self.sampling_rate)
        vocals_resampled = librosa.resample(vocals_audio, orig_sr=item['mix'].sample_rate, target_sr=self.sampling_rate)
        drums_resampled = librosa.resample(drums_audio, orig_sr=item['mix'].sample_rate, target_sr=self.sampling_rate)

        # Consistent relative peak normalization
        mix_peak = np.max(np.abs(input_resampled)) + 1e-7
        input_resampled = np.clip(input_resampled / mix_peak, -1.0, 1.0)
        vocals_resampled = np.clip(vocals_resampled / mix_peak, -1.0, 1.0)
        drums_resampled = np.clip(drums_resampled / mix_peak, -1.0, 1.0)

        # Fix length to target dimension
        input_fixed = librosa.util.fix_length(input_resampled, size=self.dim)
        vocals_fixed = librosa.util.fix_length(vocals_resampled, size=self.dim)
        drums_fixed = librosa.util.fix_length(drums_resampled, size=self.dim)

        input_fixed = np.expand_dims(input_fixed, axis=1)
        vocals_fixed = np.expand_dims(vocals_fixed, axis=1)
        drums_fixed = np.expand_dims(drums_fixed, axis=1)

        return input_fixed, vocals_fixed, drums_fixed

    def data_generation(self, batch_indices):
        current_batch_size = len(batch_indices)
        x = np.empty((current_batch_size, self.dim, 1), dtype=np.float32)
        y0 = np.empty((current_batch_size, self.dim, 1), dtype=np.float32)
        y1 = np.empty((current_batch_size, self.dim, 1), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            inp, voc, drm = self.load(idx)
            x[i] = inp
            y0[i] = voc
            y1[i] = drm

        return x, [y0, y1]