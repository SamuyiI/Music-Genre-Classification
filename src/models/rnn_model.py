"""
RNN/LSTM model for music genre classification using MFCC features.
Architecture: Bidirectional LSTM layers for temporal modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class MFCCLSTMClassifier:
    """
    LSTM-based classifier for MFCC temporal sequences.
    """

    def __init__(self, input_shape, num_classes=8, lstm_units=256, dropout_rate=0.3, use_gru=False):
        """
        Initialize the LSTM model.

        Args:
            input_shape (tuple): Shape of input MFCC features (time_steps, n_mfcc)
            num_classes (int): Number of genre classes
            lstm_units (int): Number of LSTM units per layer
            dropout_rate (float): Dropout rate for regularization
            use_gru (bool): Whether to use GRU instead of LSTM (faster training)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_gru = use_gru
        self.model = None

    def build_model(self):
        """
        Build the LSTM architecture.

        Returns:
            keras.Model: Compiled LSTM model
        """
        inputs = keras.Input(shape=self.input_shape, name='mfcc_input')

        # Choose between LSTM and GRU
        if self.use_gru:
            rnn_layer = layers.GRU
            layer_name_prefix = 'gru'
        else:
            rnn_layer = layers.LSTM
            layer_name_prefix = 'lstm'

        # First Bidirectional RNN layer (return sequences for stacking)
        x = layers.Bidirectional(
            rnn_layer(self.lstm_units, return_sequences=True, name=f'{layer_name_prefix}1'),
            name=f'bidirectional_{layer_name_prefix}1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)

        # Second Bidirectional RNN layer (return final output only)
        x = layers.Bidirectional(
            rnn_layer(self.lstm_units, return_sequences=False, name=f'{layer_name_prefix}2'),
            name=f'bidirectional_{layer_name_prefix}2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout2')(x)

        # Dense layer
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout_dense')(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # Create model
        model_name = f'MFCC{"GRU" if self.use_gru else "LSTM"}Classifier'
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


class SimpleLSTM:
    """
    Simpler LSTM model with fewer parameters for faster training.
    """

    def __init__(self, input_shape, num_classes=8, lstm_units=128, dropout_rate=0.3):
        """
        Initialize the simple LSTM model.

        Args:
            input_shape (tuple): Shape of input MFCC features (time_steps, n_mfcc)
            num_classes (int): Number of genre classes
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        """
        Build a simpler LSTM architecture with single layer.

        Returns:
            keras.Model: Compiled LSTM model
        """
        inputs = keras.Input(shape=self.input_shape, name='mfcc_input')

        # Single Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, name='lstm1'),
            name='bidirectional_lstm1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout1')(x)

        # Dense layer
        x = layers.Dense(256, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout_dense')(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='SimpleLSTM')

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


def create_lstm_model(input_shape, num_classes=8, lstm_units=256,
                      dropout_rate=0.3, learning_rate=1e-4, use_gru=False):
    """
    Helper function to create and compile an LSTM model.

    Args:
        input_shape (tuple): Input shape for MFCC features (time_steps, n_mfcc)
        num_classes (int): Number of genre classes
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate
        use_gru (bool): Whether to use GRU instead of LSTM

    Returns:
        keras.Model: Compiled LSTM model
    """
    lstm = MFCCLSTMClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        use_gru=use_gru
    )
    lstm.compile_model(learning_rate=learning_rate)
    return lstm.get_model()


def create_simple_lstm_model(input_shape, num_classes=8, lstm_units=128,
                             dropout_rate=0.3, learning_rate=1e-4):
    """
    Helper function to create a simple LSTM model.

    Args:
        input_shape (tuple): Input shape for MFCC features
        num_classes (int): Number of genre classes
        lstm_units (int): Number of LSTM units
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate

    Returns:
        keras.Model: Compiled simple LSTM model
    """
    lstm = SimpleLSTM(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    lstm.compile_model(learning_rate=learning_rate)
    return lstm.get_model()


def main():
    """
    Test the LSTM model architecture.
    """
    print("Building LSTM model...")

    # Create model with default parameters
    # MFCC features: (time_steps, n_mfcc) = (~1300, 20)
    lstm = MFCCLSTMClassifier(
        input_shape=(1300, 20),
        num_classes=8,
        lstm_units=256,
        dropout_rate=0.3,
        use_gru=False
    )

    # Build and compile
    model = lstm.get_model()

    # Print summary
    print("\nModel Summary:")
    lstm.summary()

    # Test with dummy input
    import numpy as np
    dummy_input = np.random.randn(1, 1300, 20).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    print(f"\nTest prediction shape: {output.shape}")
    print(f"Test prediction (softmax probabilities): {output}")
    print(f"Sum of probabilities: {output.sum():.4f}")

    print("\n" + "="*50)
    print("Testing GRU variant...")
    print("="*50)

    # Test GRU variant
    gru = MFCCLSTMClassifier(
        input_shape=(1300, 20),
        num_classes=8,
        lstm_units=256,
        dropout_rate=0.3,
        use_gru=True
    )

    gru_model = gru.get_model()
    print("\nGRU Model Summary:")
    gru.summary()

    print("\nLSTM/GRU model tests passed!")


if __name__ == "__main__":
    main()
