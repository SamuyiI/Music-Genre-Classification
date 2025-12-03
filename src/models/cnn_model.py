"""
CNN model for music genre classification using mel-spectrograms.
Architecture: 4 convolutional blocks with batch normalization, max pooling, and dropout.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class MelSpectrogramCNN:
    """
    Convolutional Neural Network for mel-spectrogram based genre classification.
    """

    def __init__(self, input_shape, num_classes=8, dropout_rate=0.25):
        """
        Initialize the CNN model.

        Args:
            input_shape (tuple): Shape of input mel-spectrogram (height, width, channels)
            num_classes (int): Number of genre classes
            dropout_rate (float): Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        """
        Build the CNN architecture.

        Returns:
            keras.Model: Compiled CNN model
        """
        inputs = keras.Input(shape=self.input_shape, name='mel_spectrogram_input')

        # Block 1: Conv2D(32) -> BatchNorm -> MaxPool -> Dropout
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)

        # Block 2: Conv2D(64) -> BatchNorm -> MaxPool -> Dropout
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout2')(x)

        # Block 3: Conv2D(128) -> BatchNorm -> MaxPool -> Dropout
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout3')(x)

        # Block 4: Conv2D(256) -> BatchNorm -> MaxPool -> Dropout
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout4')(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout_dense')(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='MelSpectrogramCNN')

        self.model = model
        return model

    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            learning_rate (float): Learning rate for Adam optimizer
        """
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        """
        Get the compiled model.

        Returns:
            keras.Model: Compiled model
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()

    def get_feature_extractor(self, layer_name='global_avg_pool'):
        """
        Create a feature extractor model that outputs intermediate layer activations.

        Args:
            layer_name (str): Name of the layer to extract features from

        Returns:
            keras.Model: Feature extractor model
        """
        if self.model is None:
            self.build_model()

        layer = self.model.get_layer(layer_name)
        feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=layer.output,
            name='feature_extractor'
        )
        return feature_extractor


def create_cnn_model(input_shape=(128, 1292, 1), num_classes=8, dropout_rate=0.25, learning_rate=1e-4):
    """
    Helper function to create and compile a CNN model.

    Args:
        input_shape (tuple): Input shape for mel-spectrogram
        num_classes (int): Number of genre classes
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate

    Returns:
        keras.Model: Compiled CNN model
    """
    cnn = MelSpectrogramCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    cnn.compile_model(learning_rate=learning_rate)
    return cnn.get_model()


def main():
    """
    Test the CNN model architecture.
    """
    print("Building CNN model...")

    # Create model with default parameters
    cnn = MelSpectrogramCNN(
        input_shape=(128, 1292, 1),
        num_classes=8,
        dropout_rate=0.25
    )

    # Build and compile
    model = cnn.get_model()

    # Print summary
    print("\nModel Summary:")
    cnn.summary()

    # Test with dummy input
    import numpy as np
    dummy_input = np.random.randn(1, 128, 1292, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    print(f"\nTest prediction shape: {output.shape}")
    print(f"Test prediction (softmax probabilities): {output}")
    print(f"Sum of probabilities: {output.sum():.4f}")

    print("\nCNN model test passed!")


if __name__ == "__main__":
    main()
