"""
Data augmentation techniques for audio classification.
Implements time stretching, pitch shifting, and SpecAugment.
"""

import numpy as np
import librosa


class AudioAugmenter:
    """
    Implements various audio augmentation techniques.
    """

    def __init__(self, sample_rate=22050):
        """
        Initialize the AudioAugmenter.

        Args:
            sample_rate (int): Sample rate of audio files
        """
        self.sample_rate = sample_rate

    def time_stretch(self, audio, rate=None):
        """
        Apply time stretching to audio signal.

        Args:
            audio (np.ndarray): Audio signal
            rate (float): Stretching rate. If None, randomly sample from [0.8, 1.2]

        Returns:
            np.ndarray: Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)

        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched

    def pitch_shift(self, audio, n_steps=None):
        """
        Apply pitch shifting to audio signal.

        Args:
            audio (np.ndarray): Audio signal
            n_steps (float): Number of semitones to shift. If None, randomly sample from [-2, 2]

        Returns:
            np.ndarray: Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)

        shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
        return shifted

    def add_noise(self, audio, noise_factor=None):
        """
        Add Gaussian noise to audio signal.

        Args:
            audio (np.ndarray): Audio signal
            noise_factor (float): Noise level. If None, randomly sample from [0.001, 0.005]

        Returns:
            np.ndarray: Noisy audio
        """
        if noise_factor is None:
            noise_factor = np.random.uniform(0.001, 0.005)

        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented

    def spec_augment(self, spectrogram, time_mask_param=None, freq_mask_param=None, num_masks=2):
        """
        Apply SpecAugment to a spectrogram (time and frequency masking).

        Args:
            spectrogram (np.ndarray): Input spectrogram with shape (freq, time)
            time_mask_param (int): Maximum width of time mask. If None, uses 40
            freq_mask_param (int): Maximum width of frequency mask. If None, uses 15
            num_masks (int): Number of masks to apply (default: 2)

        Returns:
            np.ndarray: Augmented spectrogram
        """
        if time_mask_param is None:
            time_mask_param = 40
        if freq_mask_param is None:
            freq_mask_param = 15

        spec_aug = spectrogram.copy()
        freq_len, time_len = spec_aug.shape

        # Apply time masking
        for _ in range(num_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, time_len - t))
            spec_aug[:, t0:t0 + t] = spec_aug.mean()

        # Apply frequency masking
        for _ in range(num_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, freq_len - f))
            spec_aug[f0:f0 + f, :] = spec_aug.mean()

        return spec_aug

    def time_shift(self, audio, shift_max=None):
        """
        Apply random time shift to audio signal.

        Args:
            audio (np.ndarray): Audio signal
            shift_max (int): Maximum shift in samples. If None, uses 10% of audio length

        Returns:
            np.ndarray: Time-shifted audio
        """
        if shift_max is None:
            shift_max = int(len(audio) * 0.1)

        shift = np.random.randint(-shift_max, shift_max)
        shifted = np.roll(audio, shift)
        return shifted

    def augment_audio(self, audio, augmentations=None):
        """
        Apply multiple augmentations to audio signal.

        Args:
            audio (np.ndarray): Audio signal
            augmentations (list): List of augmentation names to apply.
                                Options: ['time_stretch', 'pitch_shift', 'noise', 'time_shift']
                                If None, randomly selects 1-2 augmentations

        Returns:
            np.ndarray: Augmented audio
        """
        if augmentations is None:
            # Randomly select 1-2 augmentations
            available = ['time_stretch', 'pitch_shift', 'noise', 'time_shift']
            num_augs = np.random.randint(1, 3)
            augmentations = np.random.choice(available, num_augs, replace=False)

        augmented = audio.copy()

        for aug in augmentations:
            if aug == 'time_stretch':
                augmented = self.time_stretch(augmented)
            elif aug == 'pitch_shift':
                augmented = self.pitch_shift(augmented)
            elif aug == 'noise':
                augmented = self.add_noise(augmented)
            elif aug == 'time_shift':
                augmented = self.time_shift(augmented)

        return augmented

    def augment_spectrogram(self, spectrogram, use_spec_augment=True):
        """
        Apply augmentation to spectrogram.

        Args:
            spectrogram (np.ndarray): Input spectrogram
            use_spec_augment (bool): Whether to apply SpecAugment

        Returns:
            np.ndarray: Augmented spectrogram
        """
        if use_spec_augment:
            return self.spec_augment(spectrogram)
        return spectrogram


class AugmentedDataGenerator:
    """
    Data generator that applies augmentation on-the-fly during training.
    """

    def __init__(self, features, labels, batch_size=32, augmenter=None, augment_prob=0.5):
        """
        Initialize the augmented data generator.

        Args:
            features (np.ndarray): Input features (audio or spectrograms)
            labels (np.ndarray): Corresponding labels
            batch_size (int): Batch size
            augmenter (AudioAugmenter): Augmenter instance
            augment_prob (float): Probability of applying augmentation to each sample
        """
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.augmenter = augmenter if augmenter else AudioAugmenter()
        self.augment_prob = augment_prob
        self.n_samples = len(features)
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def shuffle(self):
        """Shuffle the data indices."""
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        """
        Generate one batch of data.

        Args:
            idx (int): Batch index

        Returns:
            tuple: (batch_features, batch_labels)
        """
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        batch_features = self.features[batch_indices].copy()
        batch_labels = self.labels[batch_indices]

        # Apply augmentation with probability
        for i in range(len(batch_features)):
            if np.random.random() < self.augment_prob:
                # Augment spectrogram
                batch_features[i] = self.augmenter.augment_spectrogram(batch_features[i])

        return batch_features, batch_labels

    def generate(self):
        """
        Generator function for training.

        Yields:
            tuple: (batch_features, batch_labels)
        """
        while True:
            self.shuffle()
            for idx in range(len(self)):
                yield self.__getitem__(idx)


def main():
    """
    Test augmentation functions.
    """
    # Create sample audio
    sample_rate = 22050
    duration = 5
    audio = np.random.randn(sample_rate * duration)

    # Test augmenter
    augmenter = AudioAugmenter(sample_rate=sample_rate)

    print("Testing audio augmentations...")
    print(f"Original audio shape: {audio.shape}")

    # Test time stretch
    stretched = augmenter.time_stretch(audio, rate=1.2)
    print(f"Time stretched audio shape: {stretched.shape}")

    # Test pitch shift
    shifted = augmenter.pitch_shift(audio, n_steps=2)
    print(f"Pitch shifted audio shape: {shifted.shape}")

    # Test noise addition
    noisy = augmenter.add_noise(audio, noise_factor=0.005)
    print(f"Noisy audio shape: {noisy.shape}")

    # Test SpecAugment
    spectrogram = np.random.randn(128, 1292)
    spec_aug = augmenter.spec_augment(spectrogram)
    print(f"SpecAugment output shape: {spec_aug.shape}")

    print("\nAll augmentation tests passed!")


if __name__ == "__main__":
    main()
