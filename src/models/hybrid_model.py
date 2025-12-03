"""
Hybrid CNN-LSTM model for music genre classification.
Combines CNN's spatial feature extraction with LSTM's temporal modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class HybridCNNLSTM:
    """
    Hybrid model that uses CNN to extract features from spectrograms,
    then LSTM to model temporal patterns in those features.
    """

    def __init__(self, input_shape, num_classes=8, cnn_filters=[32, 64, 128],
                 lstm_units=256, dropout_rate=0.3, use_gru=False):
        """
        Initialize the Hybrid CNN-LSTM model.

        Args:
            input_shape (tuple): Shape of input mel-spectrogram (height, width, channels)
            num_classes (int): Number of genre classes
            cnn_filters (list): List of filter counts for CNN blocks
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            use_gru (bool): Whether to use GRU instead of LSTM
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_gru = use_gru
        self.model = None

    def build_model(self):
        """
        Build the Hybrid CNN-LSTM architecture.

        Architecture flow:
        1. CNN extracts spatial features from spectrogram
        2. Reshape CNN output to temporal sequence
        3. LSTM processes temporal patterns
        4. Dense classification head

        Returns:
            keras.Model: Compiled hybrid model
        """
        inputs = keras.Input(shape=self.input_shape, name='spectrogram_input')

        # CNN Feature Extraction
        x = inputs

        # CNN Blocks
        for i, filters in enumerate(self.cnn_filters):
            x = layers.Conv2D(
                filters, (3, 3),
                activation='relu',
                padding='same',
                name=f'conv{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn{i+1}')(x)
            x = layers.MaxPooling2D((2, 2), name=f'pool{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_cnn{i+1}')(x)

        # Reshape for LSTM: (batch, freq, time, channels) -> (batch, time, features)
        # We want to treat time as the sequence dimension
        # Current shape after CNN: (batch, freq', time', channels)

        # Get the shape
        shape = x.shape

        # Reshape: collapse frequency and channels into feature dimension
        # New shape: (batch, time', freq' * channels)
        x = layers.Permute((2, 1, 3), name='permute')(x)  # (batch, time', freq', channels)

        # Flatten the freq and channel dimensions
        time_steps = x.shape[1]
        features = x.shape[2] * x.shape[3]

        x = layers.Reshape((time_steps, features), name='reshape_for_lstm')(x)

        # Choose between LSTM and GRU
        if self.use_gru:
            rnn_layer = layers.GRU
            layer_name = 'gru'
        else:
            rnn_layer = layers.LSTM
            layer_name = 'lstm'

        # Bidirectional RNN layers
        x = layers.Bidirectional(
            rnn_layer(self.lstm_units, return_sequences=True, name=f'{layer_name}1'),
            name=f'bidirectional_{layer_name}1'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_lstm1')(x)

        x = layers.Bidirectional(
            rnn_layer(self.lstm_units, return_sequences=False, name=f'{layer_name}2'),
            name=f'bidirectional_{layer_name}2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_lstm2')(x)

        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout_dense')(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # Create model
        model_name = f'Hybrid_CNN_{"GRU" if self.use_gru else "LSTM"}'
        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

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

    def get_feature_extractor(self, layer_name='dense1'):
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


class CompactHybridModel:
    """
    A more compact hybrid model for faster training.
    """

    def __init__(self, input_shape, num_classes=8, dropout_rate=0.3):
        """
        Initialize compact hybrid model.

        Args:
            input_shape (tuple): Shape of input mel-spectrogram
            num_classes (int): Number of genre classes
            dropout_rate (float): Dropout rate
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        """Build a compact hybrid architecture."""
        inputs = keras.Input(shape=self.input_shape, name='spectrogram_input')

        # Compact CNN: 3 blocks
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Reshape for LSTM
        x = layers.Permute((2, 1, 3))(x)
        time_steps = x.shape[1]
        features = x.shape[2] * x.shape[3]
        x = layers.Reshape((time_steps, features))(x)

        # Single Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Classification head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='CompactHybrid')
        self.model = model
        return model

    def compile_model(self, learning_rate=1e-4):
        """Compile the model."""
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        """Get the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()


def create_hybrid_model(input_shape=(128, 1292, 1), num_classes=8,
                       cnn_filters=[32, 64, 128], lstm_units=256,
                       dropout_rate=0.3, learning_rate=1e-4, use_gru=False):
    """
    Helper function to create and compile a hybrid CNN-LSTM model.

    Args:
        input_shape (tuple): Input shape for mel-spectrogram
        num_classes (int): Number of genre classes
        cnn_filters (list): List of filter counts for CNN blocks
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate
        use_gru (bool): Whether to use GRU instead of LSTM

    Returns:
        keras.Model: Compiled hybrid model
    """
    hybrid = HybridCNNLSTM(
        input_shape=input_shape,
        num_classes=num_classes,
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        use_gru=use_gru
    )
    hybrid.compile_model(learning_rate=learning_rate)
    return hybrid.get_model()


def create_compact_hybrid_model(input_shape=(128, 1292, 1), num_classes=8,
                                dropout_rate=0.3, learning_rate=1e-4):
    """
    Helper function to create a compact hybrid model.

    Args:
        input_shape (tuple): Input shape for mel-spectrogram
        num_classes (int): Number of genre classes
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate

    Returns:
        keras.Model: Compiled compact hybrid model
    """
    hybrid = CompactHybridModel(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    hybrid.compile_model(learning_rate=learning_rate)
    return hybrid.get_model()


def main():
    """
    Test the Hybrid CNN-LSTM model architecture.
    """
    print("Building Hybrid CNN-LSTM model...")

    # Create model with default parameters
    hybrid = HybridCNNLSTM(
        input_shape=(128, 1292, 1),
        num_classes=8,
        cnn_filters=[32, 64, 128],
        lstm_units=256,
        dropout_rate=0.3,
        use_gru=False
    )

    # Build and compile
    model = hybrid.get_model()

    # Print summary
    print("\nModel Summary:")
    hybrid.summary()

    # Test with dummy input
    import numpy as np
    dummy_input = np.random.randn(1, 128, 1292, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    print(f"\nTest prediction shape: {output.shape}")
    print(f"Test prediction (softmax probabilities): {output}")
    print(f"Sum of probabilities: {output.sum():.4f}")

    print("\n" + "="*50)
    print("Testing Compact Hybrid model...")
    print("="*50)

    # Test compact variant
    compact = CompactHybridModel(
        input_shape=(128, 1292, 1),
        num_classes=8,
        dropout_rate=0.3
    )

    compact_model = compact.get_model()
    print("\nCompact Hybrid Model Summary:")
    compact.summary()

    print("\nHybrid model tests passed!")


if __name__ == "__main__":
    main()
