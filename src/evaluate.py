"""
Evaluation script for trained models.
Computes accuracy, F1-score, precision, recall, and generates predictions.
"""

import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class ModelEvaluator:
    """
    Evaluator class for trained models.
    """

    def __init__(self, model_path, data_path):
        """
        Initialize the evaluator.

        Args:
            model_path (str): Path to saved model
            data_path (str): Path to test data
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.label_encoder = None

    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return self.model

    def load_data(self):
        """
        Load test data.

        Returns:
            tuple: (X_test, y_test, labels)
        """
        print(f"Loading data from {self.data_path}...")

        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        features = data['features']
        labels = data['labels']

        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Reshape for CNN if needed
        if len(features.shape) == 3:
            features = features[..., np.newaxis]

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels_encoded.shape}")
        print(f"Classes: {self.label_encoder.classes_}")

        return features, labels_encoded, labels

    def predict(self, X):
        """
        Make predictions on data.

        Args:
            X: Input features

        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            self.load_model()

        print("Making predictions...")
        probabilities = self.model.predict(X, verbose=1)
        predictions = np.argmax(probabilities, axis=1)

        return predictions, probabilities

    def evaluate(self, y_true, y_pred):
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            dict: Dictionary of metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        # Macro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        metrics = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        return metrics

    def print_results(self, metrics):
        """
        Print evaluation results.

        Args:
            metrics (dict): Dictionary of metrics
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")

        print(f"\nWeighted Averages:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.4f}")

        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 65)

        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name:<15} "
                  f"{metrics['precision_per_class'][i]:<12.4f} "
                  f"{metrics['recall_per_class'][i]:<12.4f} "
                  f"{metrics['f1_per_class'][i]:<12.4f} "
                  f"{metrics['support_per_class'][i]:<10}")

    def save_results(self, metrics, output_dir='results'):
        """
        Save evaluation results to files.

        Args:
            metrics (dict): Dictionary of metrics
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save overall metrics
        overall_metrics = {
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_weighted': metrics['f1_weighted']
        }

        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)

        # Save per-class metrics
        per_class_metrics = pd.DataFrame({
            'class': self.label_encoder.classes_,
            'precision': metrics['precision_per_class'],
            'recall': metrics['recall_per_class'],
            'f1_score': metrics['f1_per_class'],
            'support': metrics['support_per_class']
        })

        per_class_metrics.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)

        print(f"\nResults saved to {output_dir}/")

    def get_classification_report(self, y_true, y_pred):
        """
        Generate sklearn classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            str: Classification report
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data pickle file')
    parser.add_argument('--output_dir', type=str, default='results/metrics',
                       help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path
    )

    # Load model and data
    evaluator.load_model()
    X_test, y_test, labels = evaluator.load_data()

    # Make predictions
    y_pred, probabilities = evaluator.predict(X_test)

    # Compute metrics
    metrics = evaluator.evaluate(y_test, y_pred)

    # Print results
    evaluator.print_results(metrics)

    # Print classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(evaluator.get_classification_report(y_test, y_pred))

    # Save results
    evaluator.save_results(metrics, output_dir=args.output_dir)

    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'true_class': evaluator.label_encoder.inverse_transform(y_test),
            'predicted_class': evaluator.label_encoder.inverse_transform(y_pred)
        })

        # Add probability columns
        for i, class_name in enumerate(evaluator.label_encoder.classes_):
            predictions_df[f'prob_{class_name}'] = probabilities[:, i]

        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\nPredictions saved to {predictions_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
