"""
Generate confusion matrices for trained models.
This script loads saved models and generates confusion matrices on the test set.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


def load_test_data(data_path, test_split=0.15, validation_split=0.15):
    """
    Load and prepare test data.

    Args:
        data_path: Path to preprocessed data pickle file
        test_split: Test set split ratio
        validation_split: Validation set split ratio

    Returns:
        tuple: (X_test, y_test, label_encoder)
    """
    print(f"Loading data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data (must match training splits)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels_encoded,
        test_size=test_split,
        stratify=labels_encoded,
        random_state=42
    )

    print(f"Test set shape: {X_test.shape}")
    print(f"Classes: {label_encoder.classes_}")

    return X_test, y_test, label_encoder


def generate_confusion_matrix(model_path, X_test, y_test, label_encoder,
                              model_name, output_dir, needs_reshape=False):
    """
    Generate and save confusion matrix for a model.

    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder
        model_name: Name of the model
        output_dir: Directory to save figure
        needs_reshape: Whether to add channel dimension for CNN models
    """
    print(f"\n{'='*60}")
    print(f"Generating confusion matrix for {model_name}")
    print(f"{'='*60}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Skipping this model...")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    # Reshape if needed (for CNN/Hybrid models)
    X_test_input = X_test
    if needs_reshape and len(X_test.shape) == 3:
        X_test_input = X_test[..., np.newaxis]

    # Get predictions
    print("Making predictions...")
    y_pred_probs = model.predict(X_test_input, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=10)

    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'{model_name.lower()}_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")

    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def main():
    """Main function to generate all confusion matrices."""

    # Create output directory
    output_dir = 'results/figures/confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)

    # Model configurations
    models = [
        {
            'name': 'CNN',
            'checkpoint': 'checkpoints/cnn_best.h5',
            'data': 'data/processed/mel_features.pkl',
            'needs_reshape': True
        },
        {
            'name': 'LSTM',
            'checkpoint': 'checkpoints/lstm_best.h5',
            'data': 'data/processed/mfcc_features.pkl',
            'needs_reshape': False
        },
        {
            'name': 'Hybrid',
            'checkpoint': 'checkpoints/hybrid_best.h5',
            'data': 'data/processed/mel_features.pkl',
            'needs_reshape': True
        }
    ]

    # Generate confusion matrix for each model
    for model_config in models:
        # Load test data
        X_test, y_test, label_encoder = load_test_data(model_config['data'])

        # Generate confusion matrix
        generate_confusion_matrix(
            model_path=model_config['checkpoint'],
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            model_name=model_config['name'],
            output_dir=output_dir,
            needs_reshape=model_config['needs_reshape']
        )

    print(f"\n{'='*60}")
    print("All confusion matrices generated successfully!")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
