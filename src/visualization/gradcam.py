"""
Grad-CAM visualization for CNN models.
Visualizes which parts of the spectrogram the model focuses on for predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """

    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.

        Args:
            model (keras.Model): Trained CNN model
            layer_name (str): Name of convolutional layer to visualize.
                            If None, uses the last conv layer.
        """
        self.model = model

        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    layer_name = layer.name
                    break

        self.layer_name = layer_name
        print(f"Using layer: {self.layer_name}")

    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap for the given image.

        Args:
            img_array (np.ndarray): Input image with shape (1, H, W, C)
            class_idx (int): Target class index. If None, uses predicted class.
            eps (float): Small value to prevent division by zero

        Returns:
            np.ndarray: Heatmap with shape (H, W)
        """
        # Create a model that outputs both predictions and conv layer activations
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            # If class_idx not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            # Get the score for the target class
            class_channel = predictions[:, class_idx]

        # Compute gradients of class score w.r.t. conv layer output
        grads = tape.gradient(class_channel, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cm.jet):
        """
        Overlay heatmap on original image.

        Args:
            heatmap (np.ndarray): Grad-CAM heatmap
            original_img (np.ndarray): Original input image
            alpha (float): Transparency of heatmap overlay
            colormap: Matplotlib colormap

        Returns:
            np.ndarray: Overlayed image
        """
        # Resize heatmap to match original image size
        heatmap_resized = np.array(
            tf.image.resize(heatmap[..., np.newaxis], original_img.shape[:2])
        ).squeeze()

        # Apply colormap to heatmap
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]

        # Normalize original image to [0, 1]
        if original_img.ndim == 3 and original_img.shape[2] == 1:
            original_img = original_img.squeeze()

        if original_img.max() > 1:
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

        # Convert grayscale to RGB if needed
        if original_img.ndim == 2:
            original_img = np.stack([original_img] * 3, axis=-1)

        # Overlay heatmap on original image
        overlayed = heatmap_colored * alpha + original_img * (1 - alpha)

        return overlayed

    def visualize(self, img_array, class_idx=None, save_path=None, show=True):
        """
        Visualize Grad-CAM for an input image.

        Args:
            img_array (np.ndarray): Input image
            class_idx (int): Target class index
            save_path (str): Path to save visualization
            show (bool): Whether to display the plot
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(img_array, class_idx)

        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        pred_prob = predictions[0][pred_class]

        # Prepare original image
        original_img = img_array[0]

        # Create overlay
        overlayed = self.overlay_heatmap(heatmap, original_img)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if original_img.ndim == 3 and original_img.shape[2] == 1:
            axes[0].imshow(original_img.squeeze(), cmap='viridis')
        else:
            axes[0].imshow(original_img)
        axes[0].set_title('Original Spectrogram')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title(f'Overlay\nPredicted class: {pred_class} (prob: {pred_prob:.3f})')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_multiple(self, images, class_indices=None, class_names=None,
                          output_dir='results/figures/gradcam', prefix='gradcam'):
        """
        Visualize Grad-CAM for multiple images.

        Args:
            images (np.ndarray): Array of images
            class_indices (list): List of target class indices
            class_names (list): List of class names
            output_dir (str): Output directory
            prefix (str): File prefix for saved images
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(images):
            img_array = np.expand_dims(img, axis=0)

            class_idx = class_indices[i] if class_indices else None
            class_name = class_names[class_idx] if class_names and class_idx is not None else ''

            save_path = os.path.join(output_dir, f'{prefix}_{i}_{class_name}.png')

            self.visualize(
                img_array,
                class_idx=class_idx,
                save_path=save_path,
                show=False
            )

        print(f"\nSaved {len(images)} Grad-CAM visualizations to {output_dir}/")


def visualize_genre_patterns(model, X_test, y_test, label_encoder,
                             samples_per_genre=1, output_dir='results/figures/gradcam'):
    """
    Visualize Grad-CAM for samples from each genre.

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        label_encoder: Label encoder for class names
        samples_per_genre (int): Number of samples per genre
        output_dir (str): Output directory
    """
    gradcam = GradCAM(model)

    # Get samples for each genre
    for class_idx, class_name in enumerate(label_encoder.classes_):
        # Find samples of this genre
        genre_indices = np.where(y_test == class_idx)[0]

        if len(genre_indices) == 0:
            continue

        # Select random samples
        selected_indices = np.random.choice(
            genre_indices,
            min(samples_per_genre, len(genre_indices)),
            replace=False
        )

        for i, idx in enumerate(selected_indices):
            img_array = np.expand_dims(X_test[idx], axis=0)

            save_path = os.path.join(
                output_dir,
                f'gradcam_{class_name}_{i}.png'
            )

            gradcam.visualize(
                img_array,
                class_idx=class_idx,
                save_path=save_path,
                show=False
            )

    print(f"\nGrad-CAM visualizations saved to {output_dir}/")


def main():
    """
    Test Grad-CAM visualization.
    """
    print("Testing Grad-CAM visualization...")

    # Create dummy model
    from ..models.cnn_model import create_cnn_model

    model = create_cnn_model(input_shape=(128, 1292, 1), num_classes=8)

    # Create dummy data
    dummy_img = np.random.randn(1, 128, 1292, 1).astype(np.float32)

    # Initialize Grad-CAM
    gradcam = GradCAM(model)

    # Visualize
    gradcam.visualize(dummy_img, save_path='test_gradcam.png', show=False)

    print("Grad-CAM test complete!")


if __name__ == "__main__":
    main()
