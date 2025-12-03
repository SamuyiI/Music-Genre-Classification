"""
Confusion matrix visualization and analysis.
Helps identify which genre pairs are most commonly confused.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os


class ConfusionMatrixAnalyzer:
    """
    Analyzer for confusion matrix visualization and interpretation.
    """

    def __init__(self, y_true, y_pred, class_names):
        """
        Initialize confusion matrix analyzer.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list): List of class names
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Compute confusion matrices
        self.cm = confusion_matrix(y_true, y_pred)
        self.cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

    def plot_confusion_matrix(self, normalize=False, save_path=None, show=True,
                              figsize=(10, 8), cmap='Blues'):
        """
        Plot confusion matrix.

        Args:
            normalize (bool): Whether to normalize by true label counts
            save_path (str): Path to save figure
            show (bool): Whether to display figure
            figsize (tuple): Figure size
            cmap (str): Colormap name
        """
        cm = self.cm_normalized if normalize else self.cm

        plt.figure(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
            square=True,
            linewidths=0.5
        )

        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')

        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_both_matrices(self, save_path=None, show=True, figsize=(20, 8)):
        """
        Plot both raw and normalized confusion matrices side by side.

        Args:
            save_path (str): Path to save figure
            show (bool): Whether to display figure
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw confusion matrix
        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0],
            cbar=True,
            square=True,
            linewidths=0.5
        )
        axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[0].set_title('Raw Counts', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

        # Normalized confusion matrix
        sns.heatmap(
            self.cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1],
            cbar=True,
            square=True,
            linewidths=0.5
        )
        axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[1].set_title('Normalized (by True Label)', fontsize=14, fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

        plt.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrices to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_most_confused_pairs(self, top_k=5):
        """
        Get the most commonly confused genre pairs.

        Args:
            top_k (int): Number of top confused pairs to return

        Returns:
            list: List of tuples (true_label, pred_label, count, percentage)
        """
        confused_pairs = []

        # Iterate through off-diagonal elements
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j:  # Skip diagonal (correct predictions)
                    count = self.cm[i, j]
                    percentage = self.cm_normalized[i, j] * 100

                    if count > 0:
                        confused_pairs.append((
                            self.class_names[i],
                            self.class_names[j],
                            count,
                            percentage
                        ))

        # Sort by count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        return confused_pairs[:top_k]

    def print_confusion_analysis(self):
        """
        Print detailed confusion analysis.
        """
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)

        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        print(f"{'Genre':<15} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
        print("-" * 50)

        for i, class_name in enumerate(self.class_names):
            correct = self.cm[i, i]
            total = self.cm[i, :].sum()
            accuracy = correct / total if total > 0 else 0

            print(f"{class_name:<15} {accuracy:>9.2%} {correct:>9} {total:>9}")

        # Most confused pairs
        print("\n" + "="*80)
        print("Most Confused Genre Pairs:")
        print("="*80)

        confused_pairs = self.get_most_confused_pairs(top_k=10)

        print(f"\n{'True Genre':<15} {'→ Predicted As':<15} {'Count':<10} {'% of True':<12}")
        print("-" * 55)

        for true_label, pred_label, count, percentage in confused_pairs:
            print(f"{true_label:<15} → {pred_label:<15} {count:<10} {percentage:>10.1f}%")

    def save_confusion_data(self, output_dir='results/tables'):
        """
        Save confusion matrix data to CSV files.

        Args:
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save raw confusion matrix
        cm_df = pd.DataFrame(
            self.cm,
            index=self.class_names,
            columns=self.class_names
        )
        cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix_raw.csv'))

        # Save normalized confusion matrix
        cm_norm_df = pd.DataFrame(
            self.cm_normalized,
            index=self.class_names,
            columns=self.class_names
        )
        cm_norm_df.to_csv(os.path.join(output_dir, 'confusion_matrix_normalized.csv'))

        # Save confused pairs
        confused_pairs = self.get_most_confused_pairs(top_k=20)
        confused_df = pd.DataFrame(
            confused_pairs,
            columns=['True_Genre', 'Predicted_As', 'Count', 'Percentage']
        )
        confused_df.to_csv(os.path.join(output_dir, 'confused_pairs.csv'), index=False)

        print(f"\nConfusion data saved to {output_dir}/")

    def plot_per_class_accuracy(self, save_path=None, show=True, figsize=(12, 6)):
        """
        Plot per-class accuracy bar chart.

        Args:
            save_path (str): Path to save figure
            show (bool): Whether to display figure
            figsize (tuple): Figure size
        """
        # Compute per-class accuracy
        accuracies = []
        for i in range(self.n_classes):
            correct = self.cm[i, i]
            total = self.cm[i, :].sum()
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy * 100)

        # Create bar plot
        plt.figure(figsize=figsize)
        colors = sns.color_palette("husl", self.n_classes)

        bars = plt.bar(self.class_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        plt.xlabel('Genre', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Genre Classification Accuracy', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved per-class accuracy to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def analyze_confusion_matrix(y_true, y_pred, class_names, output_dir='results'):
    """
    Perform complete confusion matrix analysis.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (list): Class names
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)

    # Initialize analyzer
    analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, class_names)

    # Plot confusion matrices
    analyzer.plot_both_matrices(
        save_path=os.path.join(output_dir, 'figures', 'confusion_matrix.png'),
        show=False
    )

    # Plot per-class accuracy
    analyzer.plot_per_class_accuracy(
        save_path=os.path.join(output_dir, 'figures', 'per_class_accuracy.png'),
        show=False
    )

    # Print analysis
    analyzer.print_confusion_analysis()

    # Save data
    analyzer.save_confusion_data(output_dir=os.path.join(output_dir, 'tables'))


def main():
    """
    Test confusion matrix analyzer.
    """
    print("Testing confusion matrix analyzer...")

    # Create dummy data
    n_samples = 1000
    n_classes = 8

    np.random.seed(42)
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()

    # Add some confusion
    confusion_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    for idx in confusion_indices:
        y_pred[idx] = (y_pred[idx] + np.random.randint(1, n_classes)) % n_classes

    class_names = [f"Genre_{i}" for i in range(n_classes)]

    # Analyze
    analyze_confusion_matrix(y_true, y_pred, class_names, output_dir='test_results')

    print("\nConfusion matrix test complete!")


if __name__ == "__main__":
    main()
