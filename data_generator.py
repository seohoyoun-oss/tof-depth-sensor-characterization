"""
data_generator.py
AI-Augmented Depth Sensor Characterization Tool
Seoho Youn | Portfolio Project | May 2026

Generates synthetic Time-of-Flight (ToF) depth sensor data with realistic
physics-based noise models for characterization and validation workflows.

Noise sources implemented:
    - Shot noise (Poisson photon statistics)
    - Ambient light interference (additive Gaussian)
"""

import numpy as np
import matplotlib.pyplot as plt


def add_shot_noise(depth_true: np.ndarray, photon_density: float) -> tuple:
    """
    Add shot noise to a true depth map using inverse-square-law photon statistics.

    In a Time-of-Flight sensor, the number of photons received from a surface
    at distance d scales as 1/d^2 (inverse square law). Shot noise follows
    Poisson statistics, so its standard deviation equals the square root of
    the signal level. This means distant surfaces are noisier than close ones.

    Args:
        depth_true (np.ndarray): True depth values in meters. Shape: (H, W).
        photon_density (float): Photon count per unit area at 1 meter reference distance.

    Returns:
        depth_noisy (np.ndarray): Depth map with shot noise added.
        shot_noise_std (np.ndarray): Noise standard deviation at each pixel.
    """
    if np.any(depth_true <= 0):
        raise ValueError("depth_true must contain only positive values (distance in meters).")
    if photon_density <= 0:
        raise ValueError("photon_density must be positive.")

    signal = photon_density / depth_true**2         # Inverse square law
    shot_noise_std = np.sqrt(signal) / signal       # Relative noise (dimensionless → meters)
    noise = np.random.normal(0, shot_noise_std, depth_true.shape)
    depth_noisy = depth_true + noise
    return depth_noisy, shot_noise_std


def add_ambient_light_noise(depth_true: np.ndarray, ambient_std: float = 0.008) -> np.ndarray:
    """
    Add ambient light interference as additive Gaussian noise.

    Ambient light (sunlight, room lighting) enters the sensor receiver and
    creates an additive background that does not correlate with the ToF signal.
    Unlike shot noise, ambient noise is spatially uniform and does not scale
    with distance. Typical value: 8 mm standard deviation at room light levels.

    Args:
        depth_true (np.ndarray): True depth values in meters.
        ambient_std (float): Standard deviation of ambient noise in meters. Default: 0.008 m (8 mm).

    Returns:
        np.ndarray: Depth map with ambient light noise added.
    """
    ambient_noise = np.random.normal(0, ambient_std, depth_true.shape)
    return depth_true + ambient_noise


def generate_synthetic_depth_map(height: int = 64, width: int = 64,
                                  min_depth: float = 0.5, max_depth: float = 5.0) -> np.ndarray:
    """
    Generate a simple synthetic depth map with a smooth gradient.

    Args:
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        min_depth (float): Minimum depth in meters.
        max_depth (float): Maximum depth in meters.

    Returns:
        np.ndarray: Synthetic depth map of shape (height, width).
    """
    depth = np.linspace(min_depth, max_depth, height * width).reshape(height, width)
    return depth


if __name__ == "__main__":
    # --- Quick demo run ---
    depth_true = generate_synthetic_depth_map()

    depth_shot, shot_std = add_shot_noise(depth_true, photon_density=1000)
    depth_ambient = add_ambient_light_noise(depth_true, ambient_std=0.008)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(depth_true, cmap="viridis")
    axes[0].set_title("True Depth (m)")
    axes[1].imshow(depth_shot, cmap="viridis")
    axes[1].set_title("With Shot Noise")
    axes[2].imshow(depth_ambient, cmap="viridis")
    axes[2].set_title("With Ambient Light Noise")
    plt.tight_layout()
    plt.savefig("noise_comparison.png", dpi=150)
    plt.show()
    print("Saved: noise_comparison.png")