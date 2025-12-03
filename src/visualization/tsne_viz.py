"""
t-SNE visualization for genre embeddings.
Visualizes how the model clusters different genres in the learned feature space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os


class TSNEVisualizer:
    """
    Visualizes genre embeddings using t-SNE dimensionality reduction.
    """

    def __init__(self, perplexity=30, n_iter=1000, random_state=42):
        """
        Initialize t-SNE visualizer.

        Args:
            perplexity (int): t-SNE perplexity parameter
            n_iter (int): Number of iterations for optimization
            random_state (int): Random seed
        """
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state

    def extract_embeddings(self, model, X, layer_name=None):
        """
        Extract embeddings from a specific layer of the model.

        Args:
            model: Keras model
            X: Input data
            layer_name (str): Name of layer to extract from.
                            If None, uses penultimate layer.

        Returns:
            np.ndarray: Extracted embeddings
        """
        from tensorflow import keras

        # Find penultimate dense layer if not specified
        if layer_name is None:
            dense_layers = [layer for layer in model.layers
                           if isinstance(layer, keras.layers.Dense)]
            if len(dense_layers) >= 2:
                layer_name = dense_layers[-2].name
            else:
                layer_name = dense_layers[-1].name

        print(f"Extracting embeddings from layer: {layer_name}")

        # Create feature extractor
        feature_extractor = keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )

        # Extract embeddings
        embeddings = feature_extractor.predict(X, verbose=1)

        print(f"Embeddings shape: {embeddings.shape}")

        return embeddings

    def compute_tsne(self, embeddings):
        """
        Compute t-SNE projection.

        Args:
            embeddings (np.ndarray): High-dimensional embeddings

        Returns:
            np.ndarray: 2D t-SNE projection
        """
        print(f"Computing t-SNE with perplexity={self.perplexity}...")

        tsne = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=1
        )

        embeddings_2d = tsne.fit_transform(embeddings)

        print("t-SNE computation complete!")

        return embeddings_2d

    def visualize(self, embeddings_2d, labels, class_names,
                 save_path=None, show=True, figsize=(12, 10)):
        """
        Visualize t-SNE embeddings.

        Args:
            embeddings_2d (np.ndarray): 2D embeddings from t-SNE
            labels (np.ndarray): Class labels
            class_names (list): List of class names
            save_path (str): Path to save figure
            show (bool): Whether to display figure
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)

        # Create color palette
        n_classes = len(class_names)
        palette = sns.color_palette("husl", n_classes)

        # Plot each class
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[palette[i]],
                label=class_name,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )

        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE Visualization of Genre Embeddings', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved t-SNE visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_with_predictions(self, embeddings_2d, labels, predictions,
                                   class_names, save_path=None, show=True,
                                   figsize=(12, 10)):
        """
        Visualize t-SNE embeddings with correct/incorrect predictions highlighted.

        Args:
            embeddings_2d (np.ndarray): 2D embeddings
            labels (np.ndarray): True labels
            predictions (np.ndarray): Predicted labels
            class_names (list): List of class names
            save_path (str): Path to save figure
            show (bool): Whether to display figure
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)

        # Separate correct and incorrect predictions
        correct_mask = labels == predictions
        incorrect_mask = ~correct_mask

        # Create color map for genres
        n_classes = len(class_names)
        palette = sns.color_palette("husl", n_classes)
        colors = [palette[label] for label in labels]

        # Plot incorrect predictions with X marker
        if np.any(incorrect_mask):
            plt.scatter(
                embeddings_2d[incorrect_mask, 0],
                embeddings_2d[incorrect_mask, 1],
                c=np.array(colors)[incorrect_mask],
                marker='x',
                s=100,
                alpha=0.8,
                linewidths=2,
                label='Incorrect'
            )

        # Plot correct predictions with circle marker
        if np.any(correct_mask):
            plt.scatter(
                embeddings_2d[correct_mask, 0],
                embeddings_2d[correct_mask, 1],
                c=np.array(colors)[correct_mask],
                marker='o',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
                label='Correct'
            )

        # Create genre legend
        for i, class_name in enumerate(class_names):
            plt.scatter([], [], c=[palette[i]], label=class_name, s=50)

        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('t-SNE: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def create_tsne_visualization(model, X, y, label_encoder, output_dir='results/figures',
                              layer_name=None, perplexity=30):
    """
    Create and save t-SNE visualization.

    Args:
        model: Trained model
        X: Input data
        y: Labels
        label_encoder: Label encoder
        output_dir (str): Output directory
        layer_name (str): Layer to extract embeddings from
        perplexity (int): t-SNE perplexity
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize visualizer
    visualizer = TSNEVisualizer(perplexity=perplexity)

    # Extract embeddings
    embeddings = visualizer.extract_embeddings(model, X, layer_name)

    # Compute t-SNE
    embeddings_2d = visualizer.compute_tsne(embeddings)

    # Visualize
    save_path = os.path.join(output_dir, 'tsne_embeddings.png')
    visualizer.visualize(
        embeddings_2d,
        y,
        label_encoder.classes_,
        save_path=save_path,
        show=False
    )

    # Also visualize with predictions
    predictions = model.predict(X, verbose=1)
    predictions = np.argmax(predictions, axis=1)

    save_path_pred = os.path.join(output_dir, 'tsne_embeddings_with_predictions.png')
    visualizer.visualize_with_predictions(
        embeddings_2d,
        y,
        predictions,
        label_encoder.classes_,
        save_path=save_path_pred,
        show=False
    )

    # Save embeddings
    embeddings_save_path = os.path.join(output_dir, 'tsne_embeddings.npz')
    np.savez(
        embeddings_save_path,
        embeddings_2d=embeddings_2d,
        labels=y,
        predictions=predictions
    )
    print(f"Saved t-SNE embeddings to {embeddings_save_path}")


def visualize_genre_clusters(embeddings_2d, labels, class_names, save_path=None):
    """
    Visualize genre clusters with density contours.

    Args:
        embeddings_2d (np.ndarray): 2D embeddings
        labels (np.ndarray): Labels
        class_names (list): Class names
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Create overall palette
    palette = sns.color_palette("husl", len(class_names))

    for i, class_name in enumerate(class_names):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot all genres in gray
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c='lightgray',
            alpha=0.3,
            s=20
        )

        # Highlight current genre
        mask = labels == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[palette[i]],
            alpha=0.8,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

        ax.set_title(class_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved genre clusters to {save_path}")

    plt.show()


def main():
    """
    Test t-SNE visualization.
    """
    print("Testing t-SNE visualization...")

    # Create dummy data
    n_samples = 200
    n_features = 512
    n_classes = 8

    embeddings = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)
    class_names = [f"Genre_{i}" for i in range(n_classes)]

    # Initialize visualizer
    visualizer = TSNEVisualizer(perplexity=30, n_iter=300)

    # Compute t-SNE
    embeddings_2d = visualizer.compute_tsne(embeddings)

    # Visualize
    visualizer.visualize(
        embeddings_2d,
        labels,
        class_names,
        save_path='test_tsne.png',
        show=False
    )

    print("t-SNE test complete!")


if __name__ == "__main__":
    main()
