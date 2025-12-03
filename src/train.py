"""
Training script for music genre classification models.
Supports CNN, LSTM, and Hybrid models with Wandb logging.
"""

import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb
from wandb.keras import WandbCallback

# Import model architectures
from models.cnn_model import create_cnn_model
from models.rnn_model import create_lstm_model, create_simple_lstm_model
from models.hybrid_model import create_hybrid_model, create_compact_hybrid_model


class MusicGenreTrainer:
    """
    Trainer class for music genre classification.
    """

    def __init__(self, model_type='cnn', data_path=None, config=None):
        """
        Initialize the trainer.

        Args:
            model_type (str): Type of model ('cnn', 'lstm', 'hybrid')
            data_path (str): Path to preprocessed data pickle file
            config (dict): Configuration dictionary
        """
        self.model_type = model_type
        self.data_path = data_path
        self.config = config or {}
        self.model = None
        self.label_encoder = None
        self.history = None

        # Set defaults
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.validation_split = self.config.get('validation_split', 0.15)
        self.test_split = self.config.get('test_split', 0.15)

    def load_data(self):
        """
        Load and prepare data for training.

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"Loading data from {self.data_path}...")

        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        features = data['features']
        labels = data['labels']

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)

        print(f"Classes: {self.label_encoder.classes_}")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels_encoded,
            test_size=self.test_split,
            stratify=labels_encoded,
            random_state=42
        )

        # Second split: separate train and validation
        val_size = self.validation_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=42
        )

        # Reshape data based on model type
        if self.model_type == 'cnn' or self.model_type == 'hybrid':
            # CNN expects (samples, height, width, channels)
            if len(X_train.shape) == 3:
                X_train = X_train[..., np.newaxis]
                X_val = X_val[..., np.newaxis]
                X_test = X_test[..., np.newaxis]

        print(f"\nData splits:")
        print(f"Train: {X_train.shape}, {y_train.shape}")
        print(f"Val: {X_val.shape}, {y_val.shape}")
        print(f"Test: {X_test.shape}, {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_shape, num_classes):
        """
        Build the specified model architecture.

        Args:
            input_shape (tuple): Input shape
            num_classes (int): Number of classes

        Returns:
            keras.Model: Compiled model
        """
        print(f"\nBuilding {self.model_type} model...")

        if self.model_type == 'cnn':
            model = create_cnn_model(
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=self.config.get('dropout_rate', 0.25),
                learning_rate=self.learning_rate
            )

        elif self.model_type == 'lstm':
            use_simple = self.config.get('use_simple_lstm', False)
            if use_simple:
                model = create_simple_lstm_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    lstm_units=self.config.get('lstm_units', 128),
                    dropout_rate=self.config.get('dropout_rate', 0.3),
                    learning_rate=self.learning_rate
                )
            else:
                model = create_lstm_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    lstm_units=self.config.get('lstm_units', 256),
                    dropout_rate=self.config.get('dropout_rate', 0.3),
                    learning_rate=self.learning_rate,
                    use_gru=self.config.get('use_gru', False)
                )

        elif self.model_type == 'hybrid':
            use_compact = self.config.get('use_compact', False)
            if use_compact:
                model = create_compact_hybrid_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    dropout_rate=self.config.get('dropout_rate', 0.3),
                    learning_rate=self.learning_rate
                )
            else:
                model = create_hybrid_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    cnn_filters=self.config.get('cnn_filters', [32, 64, 128]),
                    lstm_units=self.config.get('lstm_units', 256),
                    dropout_rate=self.config.get('dropout_rate', 0.3),
                    learning_rate=self.learning_rate,
                    use_gru=self.config.get('use_gru', False)
                )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = model
        return model

    def get_callbacks(self, checkpoint_dir='checkpoints'):
        """
        Get training callbacks.

        Args:
            checkpoint_dir (str): Directory to save checkpoints

        Returns:
            list: List of Keras callbacks
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks = []

        # Model checkpoint - save best model
        checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_type}_best.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # TensorBoard logging
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', self.model_type),
            histogram_freq=1
        )
        callbacks.append(tensorboard)

        # Wandb callback (if initialized)
        if wandb.run is not None:
            callbacks.append(WandbCallback(save_model=False))

        return callbacks

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            History: Training history
        """
        print(f"\nStarting training...")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")

        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:]
            num_classes = len(np.unique(y_train))
            self.build_model(input_shape, num_classes)

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating on test set...")

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return results

    def save_model(self, save_path):
        """Save the trained model."""
        self.model.save(save_path)
        print(f"Model saved to {save_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train music genre classification model')

    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'lstm', 'hybrid'],
                       help='Type of model to train')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data pickle file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--lstm_units', type=int, default=256,
                       help='Number of LSTM units')
    parser.add_argument('--use_gru', action='store_true',
                       help='Use GRU instead of LSTM')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='music-genre-classification',
                       help='Wandb project name')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of data for testing')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Initialize Wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )

    # Create config dictionary
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'lstm_units': args.lstm_units,
        'use_gru': args.use_gru
    }

    # Initialize trainer
    trainer = MusicGenreTrainer(
        model_type=args.model_type,
        data_path=args.data_path,
        config=config
    )

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()

    # Use subset if specified
    if args.subset:
        print(f"\nUsing subset of {args.subset} samples for testing...")
        X_train = X_train[:args.subset]
        y_train = y_train[:args.subset]
        X_val = X_val[:args.subset//5]
        y_val = y_val[:args.subset//5]

    # Build model
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    trainer.build_model(input_shape, num_classes)

    # Print model summary
    print("\nModel Summary:")
    trainer.model.summary()

    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    results = trainer.evaluate(X_test, y_test)

    # Log final results to wandb
    if args.use_wandb:
        wandb.log(results)
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
