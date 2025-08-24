import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


def reduce_image_dimensions(
    images: np.ndarray, 
    target_dim: int, 
    random_state: int = 42,
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, SparseRandomProjection, MinMaxScaler]:
    """
    Apply dimension reduction to a batch of images using sparse random projection.
    
    Args:
        images: Input images array of shape (n_samples, height, width) or (n_samples, height, width, channels)
        target_dim: Target number of dimensions to reduce to
        random_state: Random seed for reproducible results
        feature_range: Range for min-max scaling (default: (0, 1))
    
    Returns:
        Tuple containing:
        - reduced_images: Dimension-reduced images of shape (n_samples, target_dim)
        - projector: Fitted SparseRandomProjection transformer (for future transforms)
        - scaler: Fitted MinMaxScaler (for future transforms)
    
    Raises:
        ValueError: If target_dim is larger than the original feature space
    """
    if len(images.shape) < 2:
        raise ValueError(f"Images array must have at least 2 dimensions, got {len(images.shape)}")
    
    # Get original dimensions
    n_samples = images.shape[0]
    original_features = np.prod(images.shape[1:])  # Height * Width * Channels
    
    if target_dim > original_features:
        raise ValueError(
            f"Target dimension ({target_dim}) cannot be larger than original "
            f"feature space ({original_features})"
        )
    
    print(f"Original image shape: {images.shape}")
    print(f"Original feature space: {original_features}")
    print(f"Target dimensions: {target_dim}")
    
    # Flatten the images for dimension reduction
    flattened_images = images.reshape(n_samples, -1)
    print(f"Flattened shape: {flattened_images.shape}")
    
    # Apply sparse random projection
    random_projection = SparseRandomProjection(
        n_components=target_dim, 
        random_state=random_state
    )
    reduced_images = random_projection.fit_transform(flattened_images)
    print(f"After random projection: {reduced_images.shape}")
    
    # Apply min-max scaling to normalize the reduced features
    scaler = MinMaxScaler(feature_range=feature_range)
    reduced_images = scaler.fit_transform(reduced_images)
    print(f"After scaling: {reduced_images.shape}")
    
    return reduced_images, random_projection, scaler


def transform_new_images(
    images: np.ndarray,
    projector: SparseRandomProjection,
    scaler: MinMaxScaler
) -> np.ndarray:
    """
    Apply pre-fitted dimension reduction to new images.
    
    Args:
        images: New images to transform
        projector: Pre-fitted SparseRandomProjection transformer
        scaler: Pre-fitted MinMaxScaler
    
    Returns:
        Dimension-reduced and scaled images
    """
    # Flatten the images
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)
    
    # Apply the pre-fitted transformations
    reduced_images = projector.transform(flattened_images)
    scaled_images = scaler.transform(reduced_images)
    
    return scaled_images


def inverse_transform_images(
    reduced_images: np.ndarray,
    original_shape: Tuple[int, ...],
    projector: SparseRandomProjection,
    scaler: MinMaxScaler
) -> np.ndarray:
    """
    Attempt to reconstruct images from dimension-reduced representation.
    Note: This is an approximation since random projection is not exactly invertible.
    
    Args:
        reduced_images: Dimension-reduced images
        original_shape: Original shape of images (n_samples, height, width, ...)
        projector: SparseRandomProjection transformer used for reduction
        scaler: MinMaxScaler used for scaling
    
    Returns:
        Reconstructed images (approximation)
    """
    # Inverse scale
    unscaled = scaler.inverse_transform(reduced_images)
    
    # Pseudo-inverse projection (approximation)
    projection_matrix = projector.components_
    pseudo_inverse = np.linalg.pinv(projection_matrix)
    reconstructed_flat = unscaled @ pseudo_inverse.T
    
    # Reshape back to original image shape
    reconstructed_images = reconstructed_flat.reshape(original_shape)
    
    return reconstructed_images


# Example usage and testing
if __name__ == "__main__":
    # Create sample image data for testing
    print("Testing dimension reduction with sample data...")
    
    # Simulate MNIST-like data: 100 samples of 28x28 grayscale images
    sample_images = np.random.rand(100, 28, 28)
    target_dimensions = 50
    
    print(f"\n=== Testing with MNIST-like data ===")
    reduced, proj, scaler = reduce_image_dimensions(sample_images, target_dimensions)
    
    print(f"Compression ratio: {np.prod(sample_images.shape[1:]) / target_dimensions:.2f}x")
    print(f"Reduced data range: [{reduced.min():.3f}, {reduced.max():.3f}]")
    
    # Test with new data
    print(f"\n=== Testing transform of new data ===")
    new_images = np.random.rand(10, 28, 28)
    new_reduced = transform_new_images(new_images, proj, scaler)
    print(f"New data reduced shape: {new_reduced.shape}")
    
    # Test reconstruction
    print(f"\n=== Testing reconstruction ===")
    reconstructed = inverse_transform_images(
        reduced[:5], sample_images[:5].shape, proj, scaler
    )
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    mse = np.mean((sample_images[:5] - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    print(f"\n=== Testing with CIFAR-like data ===")
    # Simulate CIFAR-like data: 50 samples of 32x32x3 RGB images
    cifar_images = np.random.rand(50, 32, 32, 3)
    cifar_target = 100
    
    cifar_reduced, cifar_proj, cifar_scaler = reduce_image_dimensions(cifar_images, cifar_target)
    print(f"CIFAR compression ratio: {np.prod(cifar_images.shape[1:]) / cifar_target:.2f}x")
