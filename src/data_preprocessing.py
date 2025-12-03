"""
Data preprocessing pipeline for music genre classification.
Handles loading audio files, resampling, and feature extraction (mel-spectrograms and MFCCs).
"""

import os
import librosa
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import soundfile as sf


class AudioPreprocessor:
    """
    Preprocesses audio files for music genre classification.
    """

    def __init__(self, sample_rate=22050, duration=30, n_mels=128, n_mfcc=20):
        """
        Initialize the AudioPreprocessor.

        Args:
            sample_rate (int): Target sample rate for audio files (default: 22050 Hz)
            duration (int): Fixed duration for audio clips in seconds (default: 30)
            n_mels (int): Number of mel bands for mel-spectrogram (default: 128)
            n_mfcc (int): Number of MFCC coefficients (default: 20)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.samples = sample_rate * duration

    def load_audio(self, file_path):
        """
        Load and preprocess a single audio file.

        Args:
            file_path (str): Path to audio file

        Returns:
            np.ndarray: Preprocessed audio signal
        """
        try:
            # Load audio file with librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Pad or trim to fixed length
            if len(audio) < self.samples:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, self.samples - len(audio)), mode='constant')
            else:
                # Trim if too long
                audio = audio[:self.samples]

            return audio

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_mel_spectrogram(self, audio):
        """
        Extract mel-spectrogram from audio signal.

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: Mel-spectrogram with shape (n_mels, time_steps)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmax=8000
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio signal.

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: MFCC features with shape (n_mfcc, time_steps)
        """
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )

        return mfcc

    def normalize_features(self, features):
        """
        Normalize features to [-1, 1] range.

        Args:
            features (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Normalized features
        """
        # Min-max normalization to [-1, 1]
        min_val = np.min(features)
        max_val = np.max(features)

        if max_val - min_val > 0:
            normalized = 2 * (features - min_val) / (max_val - min_val) - 1
        else:
            normalized = features

        return normalized

    def process_single_file(self, file_path, extract_mel=True, extract_mfcc_features=True):
        """
        Process a single audio file and extract features.

        Args:
            file_path (str): Path to audio file
            extract_mel (bool): Whether to extract mel-spectrogram
            extract_mfcc_features (bool): Whether to extract MFCC features

        Returns:
            dict: Dictionary containing extracted features
        """
        # Load audio
        audio = self.load_audio(file_path)
        if audio is None:
            return None

        features = {}

        # Extract mel-spectrogram
        if extract_mel:
            mel_spec = self.extract_mel_spectrogram(audio)
            mel_spec = self.normalize_features(mel_spec)
            features['mel_spectrogram'] = mel_spec

        # Extract MFCC
        if extract_mfcc_features:
            mfcc = self.extract_mfcc(audio)
            mfcc = self.normalize_features(mfcc)
            features['mfcc'] = mfcc

        return features

    def process_dataset(self, audio_dir, metadata_file, output_dir, max_samples=None):
        """
        Process entire dataset and save extracted features.

        Args:
            audio_dir (str): Directory containing audio files
            metadata_file (str): Path to metadata CSV file
            output_dir (str): Directory to save processed features
            max_samples (int): Maximum number of samples to process (for testing)
        """
        # Load metadata
        if not os.path.exists(metadata_file):
            print(f"Metadata file not found: {metadata_file}")
            return

        metadata = pd.read_csv(metadata_file)

        # Limit samples if specified
        if max_samples:
            metadata = metadata.head(max_samples)

        print(f"Processing {len(metadata)} audio files...")

        mel_spectrograms = []
        mfccs = []
        labels = []
        file_ids = []

        # Process each file
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            file_id = row['track_id']
            genre = row['genre_top']

            # Construct file path (adjust based on FMA structure)
            file_path = os.path.join(audio_dir, f"{file_id:06d}.mp3")

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Extract features
            features = self.process_single_file(file_path)

            if features is not None:
                mel_spectrograms.append(features['mel_spectrogram'])
                mfccs.append(features['mfcc'])
                labels.append(genre)
                file_ids.append(file_id)

        # Convert to numpy arrays
        mel_spectrograms = np.array(mel_spectrograms)
        mfccs = np.array(mfccs)
        labels = np.array(labels)
        file_ids = np.array(file_ids)

        # Save processed features
        os.makedirs(output_dir, exist_ok=True)

        mel_data = {
            'features': mel_spectrograms,
            'labels': labels,
            'file_ids': file_ids
        }

        mfcc_data = {
            'features': mfccs,
            'labels': labels,
            'file_ids': file_ids
        }

        with open(os.path.join(output_dir, 'mel_features.pkl'), 'wb') as f:
            pickle.dump(mel_data, f)

        with open(os.path.join(output_dir, 'mfcc_features.pkl'), 'wb') as f:
            pickle.dump(mfcc_data, f)

        print(f"\nProcessing complete!")
        print(f"Mel-spectrograms shape: {mel_spectrograms.shape}")
        print(f"MFCCs shape: {mfccs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Saved to: {output_dir}")

        # Print genre distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nGenre distribution:")
        for genre, count in zip(unique, counts):
            print(f"  {genre}: {count}")


def main():
    """
    Main function for testing preprocessing pipeline.
    """
    # Example usage
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=30,
        n_mels=128,
        n_mfcc=20
    )

    # Test on small subset first
    audio_dir = "data/fma_small"
    metadata_file = "data/fma_metadata.csv"
    output_dir = "data/processed"

    # Process dataset (use max_samples for testing)
    preprocessor.process_dataset(
        audio_dir=audio_dir,
        metadata_file=metadata_file,
        output_dir=output_dir,
        max_samples=20  # Test with 20 samples first
    )


if __name__ == "__main__":
    main()
