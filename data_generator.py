"""
data_generator.py
AI-Augmented Depth Sensor Characterization Tool
Seoho Youn | Portfolio Project | May 2026

Generates synthetic Time-of-Flight (ToF) depth sensor data with realistic
physics-based noise models for characterization and validation workflows.

Noise sources implemented:
    - Shot noise (Poisson photon statistics)
    - Ambient light interference (additive Gaussian)

===========================================================================
DISCLAIMER
===========================================================================

All sensor parameters, noise figures, scene configurations, and rule-of-thumb
values in this file are illustrative estimates chosen solely for educational
and portfolio demonstration purposes. They are not derived from, representative
of, or associated with any real product, commercial sensor, proprietary
specification, or employer. Any resemblance to actual product parameters is
coincidental. This tool is not intended for use in product design, hardware
acceptance testing, or any commercial application.

===========================================================================
ASSUMPTIONS
===========================================================================

Physical / Optical
------------------
1.  Inverse-square-law illumination: received signal ∝ 1/d².
    Requires a point-like source and no atmospheric absorption — valid
    for short-range (<10 m) indoor use; breaks down outdoors or in fog.

2.  Lambertian (diffuse), uniform-albedo surfaces: all pixels reflect the
    same fraction of incident light. Dark (low-albedo) materials absorb more
    light and return fewer photons, increasing shot noise. Specular materials
    violate the Lambertian assumption: signal is highly angle-dependent and
    can range from near-zero (off-axis) to sensor saturation (at the specular
    angle), neither of which this model captures.

3.  Normal incidence only: the cos(θ) drop-off in signal at oblique
    angles is not modeled. Grazing-angle surfaces are treated identically
    to face-on surfaces.

4.  Single-bounce illumination: inter-reflections and indirect lighting
    paths are ignored.

Noise Model
-----------
5.  Poisson shot noise approximated as Gaussian: valid when signal >> 1
    (large-N limit of the central limit theorem). At the default
    photon_density=50,000 the minimum signal at max range is
    50,000/5² = 2,000 photons, so the approximation holds well.
    It breaks down at very low light levels or extremely long range.

6.  Depth noise is derived via direct linear error propagation from
    photon-count noise. In a real continuous-wave ToF sensor, depth is
    recovered from the phase of a demodulated sinusoid; the true noise
    transfer function involves the modulation frequency and integration
    time. This model collapses that chain into a single photon-density
    parameter.

7.  Shot noise and ambient noise are statistically independent and
    simply additive. In reality, ambient photons raise the total photon
    count and therefore also amplify shot noise; this coupling is
    neglected here.

8.  Ambient noise is spatially uniform and distance-independent. Real
    ambient light varies spatially (e.g., near a window vs. a corner)
    and the resulting photon noise does weakly depend on ambient
    intensity.

9.  Pixel noise is spatially uncorrelated (white noise). Real sensors
    exhibit spatially correlated noise from readout electronics and
    optical cross-talk.

Scene Model
-----------
10. The synthetic depth map is a smooth linear gradient with no depth
    discontinuities. Real scenes have sharp edges, which produce
    mixed-pixel / flying-pixel artifacts in ToF that are not modeled.

11. No geometric projection or lens distortion: depth values are
    arranged on a flat Cartesian grid. Actual ToF images require a
    camera model to convert pixel coordinates to 3-D points.

Not Modeled (known limitations)
--------------------------------
12. Depth aliasing / range wrapping: unambiguous range = c / (2·f_mod).
    Targets beyond this distance wrap and appear at a shorter range.

13. Fixed-pattern noise: pixel-to-pixel sensitivity variation and dark
    current offsets are absent.

14. Motion blur: the scene is assumed static within one exposure frame.

15. Temperature drift: sensitivity and offset shifts with sensor
    temperature are not included.

===========================================================================
CAVEATS FOR USE
===========================================================================

C1. Noise levels are a lower bound, not a specification.
    Assumptions 1–4 together mean the model always gives the best-case
    optical scenario: perfectly diffuse surfaces, perpendicular incidence,
    no scattering, no inter-reflections. Any real deviation (dark
    material, oblique angle, fog) returns fewer photons and produces
    higher noise than this generator predicts. Treat output noise figures
    as optimistic estimates.

C2. photon_density_ph_at_1m cannot be read directly off a datasheet.
    Assumption 6 collapses laser power, modulation frequency, integration
    time, and detector quantum efficiency into a single scalar. You cannot
    set this parameter from a sensor datasheet without first calibrating
    the mapping. Use the noise-at-distance rule of thumb in the parameter
    block (std ≈ depth / √photon_density) to back-calculate a value that
    matches measured noise on your target hardware.

C3. Combined shot + ambient noise is an underestimate in bright light.
    Assumption 7 ignores the coupling: ambient photons increase the total
    detected signal and therefore amplify shot noise further. In outdoor
    or near-window conditions the true noise will exceed the sum of the
    two independently modeled contributions.

C4. Extreme-range or low-photon-density outputs carry unreliable tail
    statistics. Assumption 5's Gaussian approximation matches Poisson
    well in the bulk but underestimates the rate of large outlier errors.
    If photon_density is lowered below ~5,000 or max_depth_m is pushed
    beyond ~8 m (signal < ~80 photons), consider switching to
    np.random.poisson for shot noise sampling.

C5. Do not use this data alone to validate edge-aware or geometric
    algorithms. Assumptions 10–11 produce a featureless gradient with no
    depth edges, no lens distortion, and no perspective foreshortening.
    Algorithms that handle object boundaries, mixed pixels, or 3-D
    back-projection will see none of their critical failure modes in this
    data.

C6. Results are not reproducible between runs without a fixed random
    seed. Call np.random.seed(<value>) before generate_synthetic_depth_map
    if deterministic output is required (e.g., for unit tests or
    benchmarking comparisons).

C7. This generator is appropriate for early-stage noise sensitivity
    analysis and algorithm prototyping. It is not a substitute for
    real sensor data in final validation, characterization reports,
    or hardware acceptance testing.
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SENSOR PARAMETERS
# Edit these to match your target sensor / scene configuration.
# =============================================================================

# --- Scene geometry ---
sensor_height_px      = 240        # Sensor vertical resolution (pixels)
sensor_width_px       = 320        # Sensor horizontal resolution (pixels)
min_depth_m           = 0.3        # Nearest measurable distance
max_depth_m           = 5.0        # Farthest measurable distance

# --- Optical model ---
photon_density_ph_at_1m = 50_000   # Photon count at 1 m reference; scales as 1/d²
                                   # ~50 k → ~4.5 mm std at 1 m (consumer ToF)
                                   # ~250 k → ~2 mm std at 1 m (industrial ToF)
                                   # ~10 k  → ~10 mm std at 1 m (low-cost ToF)

# --- Environmental noise ---
ambient_noise_std_m   = 0.012      # Ambient light interference std (12 mm = typical indoor)
                                   # ~0.005 m → low-light lab
                                   # ~0.030 m → bright outdoor

# =============================================================================


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

    signal = photon_density / depth_true**2         # photons received: N ∝ 1/d² (inverse-square law)
    shot_noise_std = np.sqrt(signal) / signal       # σ_shot = √N / N = 1/√N; in meters: d/√N
    noise = np.random.normal(0, shot_noise_std, depth_true.shape)  # Gaussian approximation to Poisson (Assumption 5)
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
    ambient_noise = np.random.normal(0, ambient_std, depth_true.shape)  # spatially uniform, distance-independent (Assumption 8)
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
    depth = np.linspace(min_depth, max_depth, height * width).reshape(height, width)  # smooth ramp; no edges or occlusions (Assumption 10)
    return depth


def generate_scene_with_step(height: int, width: int,
                              near_depth: float = 1.0, far_depth: float = 3.5) -> np.ndarray:
    """
    Generate a depth map with a vertical step edge at the image midpoint.

    The left half is set to near_depth and the right half to far_depth,
    creating a sharp depth discontinuity. In a real ToF sensor, pixels whose
    footprint straddles this edge receive return photons from both planes
    simultaneously and report a blended depth — the "flying-pixel" artifact.
    This generator does not model that mixing (see Caveat C5); the step scene
    illustrates the spatial geometry where flying pixels occur in hardware.

    Args:
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        near_depth (float): Depth of the foreground (left half) in meters.
        far_depth (float): Depth of the background (right half) in meters.

    Returns:
        np.ndarray: Step-edge depth map of shape (height, width).
    """
    depth = np.full((height, width), far_depth, dtype=float)  # start with the far (background) plane
    depth[:, :width // 2] = near_depth                        # overwrite left half with the near (foreground) plane
    return depth


def add_combined_noise(depth_true: np.ndarray, photon_density: float,
                       ambient_std: float) -> tuple:
    """
    Apply shot noise and ambient light noise together in a single call.

    The two sources are statistically independent (Assumption 7), so their
    variances add: σ_combined² = σ_shot² + σ_ambient². This function draws
    one noisy realization with both sources applied.

    Args:
        depth_true (np.ndarray): True depth values in meters.
        photon_density (float): Photon count at 1 m reference distance.
        ambient_std (float): Standard deviation of ambient noise in meters.

    Returns:
        depth_combined (np.ndarray): Depth map with both noise sources applied.
        shot_noise_std (np.ndarray): Per-pixel shot noise standard deviation.
    """
    depth_after_shot, shot_noise_std = add_shot_noise(depth_true, photon_density)  # first draw: shot noise
    depth_combined = add_ambient_light_noise(depth_after_shot, ambient_std)         # second draw: ambient noise on top (statistically independent, Assumption 7)
    return depth_combined, shot_noise_std


if __name__ == "__main__":
    # Fix random state so every run produces identical figures and table values.
    # Remove or change this seed to explore different noise realizations.
    np.random.seed(42)

    # ------------------------------------------------------------------ #
    # 1.  Gradient scene — apply all three noise models                   #
    # ------------------------------------------------------------------ #

    # Ground-truth depth map: smooth ramp from min to max depth.
    # Used for all characterization plots because it densely samples the
    # full depth range in a single image.
    depth_true = generate_synthetic_depth_map(
        height=sensor_height_px,
        width=sensor_width_px,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )

    # Apply each noise model independently so their individual effects can
    # be compared side-by-side in Figure 1 and in the summary table.
    # The second return value (per-pixel std) is discarded here because the
    # characterization plot recomputes it analytically over a dense range.
    depth_shot, _     = add_shot_noise(depth_true, photon_density=photon_density_ph_at_1m)
    depth_ambient     = add_ambient_light_noise(depth_true, ambient_std=ambient_noise_std_m)
    depth_combined, _ = add_combined_noise(depth_true, photon_density_ph_at_1m, ambient_noise_std_m)

    # ------------------------------------------------------------------ #
    # 2.  Figure 1: Four-panel noise comparison (2×2)                     #
    #                                                                      #
    #     Side-by-side view of the three noise models against ground       #
    #     truth. All panels share the same color scale (vmin/vmax) so      #
    #     brightness differences reflect actual noise, not auto-scaling.   #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Shared depth color scale — keeps all panels visually comparable.
    vmin, vmax = min_depth_m, max_depth_m

    # Build the panel list as (axis, data array, title) tuples so the
    # same imshow / colorbar code runs once for every panel.
    panels = [
        (axes[0, 0], depth_true,     "True Depth"),
        (axes[0, 1], depth_shot,     "Shot Noise Only"),
        (axes[1, 0], depth_ambient,  "Ambient Light Noise Only"),
        (axes[1, 1], depth_combined, "Combined Noise"),
    ]
    for ax, data, title in panels:
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        fig.colorbar(im, ax=ax, label="Depth (m)")
    plt.tight_layout()
    plt.savefig("noise_comparison.png", dpi=150)
    plt.show()
    print("Saved: noise_comparison.png")

    # ------------------------------------------------------------------ #
    # 3.  Figure 2: Noise characterization — empirical vs. theoretical    #
    #                                                                      #
    #     This is the core sensor-characterization plot: it shows how      #
    #     noise grows with distance and compares the measured scatter       #
    #     against the analytical model.                                    #
    #                                                                      #
    #     Method: pixels are sorted into N_BINS depth slices; the std of   #
    #     the depth error within each slice is the empirical noise at       #
    #     that depth. Theoretical curves are computed directly from the     #
    #     noise model equations.                                            #
    # ------------------------------------------------------------------ #

    # 60 depth slices over the 0.3–5.0 m range → ~78 mm per bin.
    # At 240×320 = 76,800 pixels, each bin contains ~1,280 samples —
    # enough for a stable std estimate from a single noise realization.
    N_BINS = 60

    bin_width   = (max_depth_m - min_depth_m) / N_BINS  # meters per bin; reused as histogram window below
    bin_edges   = np.linspace(min_depth_m, max_depth_m, N_BINS + 1)  # N+1 fence-post values
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])             # midpoint of each bin (x-axis)

    depth_flat = depth_true.ravel()                          # flatten to 1-D for boolean bin indexing
    resid_flat = (depth_combined - depth_true).ravel()       # signed error: positive = overestimate

    # Per-bin empirical std: std of all residuals whose true depth falls in that bin.
    empirical_std = np.array([
        resid_flat[(depth_flat >= bin_edges[i]) & (depth_flat < bin_edges[i + 1])].std()
        for i in range(N_BINS)
    ])

    # Dense x-axis for smooth theoretical curves (500 pts vs 60 bin centers).
    d_th        = np.linspace(min_depth_m, max_depth_m, 500)
    shot_th     = d_th / np.sqrt(photon_density_ph_at_1m)   # σ_shot = d/√N — grows linearly with distance
    ambient_th  = np.full_like(d_th, ambient_noise_std_m)   # σ_ambient — constant, distance-independent
    combined_th = np.sqrt(shot_th**2 + ambient_th**2)        # RSS combination of independent sources (Assumption 7)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(bin_centers, empirical_std * 1e3, s=18, zorder=3,
               label="Empirical std (combined, one realization)")
    ax.plot(d_th, shot_th    * 1e3, "--", lw=1.5, label="Theoretical: shot noise  σ = d / √N")
    ax.plot(d_th, ambient_th * 1e3, ":",  lw=1.5, label=f"Theoretical: ambient noise  σ = {ambient_noise_std_m*1e3:.0f} mm")
    ax.plot(d_th, combined_th* 1e3, "-",  lw=2,   label="Theoretical: combined  √(σ_shot² + σ_amb²)")
    ax.set_xlabel("True Depth (m)")
    ax.set_ylabel("Noise Std Dev (mm)")
    ax.set_title("ToF Depth Noise Characterization: Empirical vs. Theoretical")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("noise_characterization.png", dpi=150)
    plt.show()
    print("Saved: noise_characterization.png")

    # ------------------------------------------------------------------ #
    # 4.  Figure 3: Residual histograms at three representative depths    #
    #                                                                      #
    #     Validates Assumption 5 (Poisson ≈ Gaussian in large-N regime):  #
    #     if the approximation holds, each histogram should be             #
    #     well-described by the fitted Gaussian, and its σ should match    #
    #     the theoretical prediction (red dashed lines).                   #
    # ------------------------------------------------------------------ #

    # Near / mid / far range — brackets the full depth span to show how the
    # distribution widens as signal weakens with increasing distance.
    TARGET_DEPTHS = [1.0, 2.5, 4.5]   # metres

    # Reuse bin_width as the pixel-selection half-window so the sample
    # density per histogram matches the characterization plot resolution.
    half_window = bin_width

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, d_target in zip(axes, TARGET_DEPTHS):
        # Select pixels whose true depth is within ±half_window of the target.
        mask = np.abs(depth_true - d_target) < half_window

        # Convert to mm for human-readable axis labels.
        residuals_mm = (depth_combined - depth_true)[mask] * 1e3

        ax.hist(residuals_mm, bins=35, density=True, alpha=0.65, label="Empirical")

        # Gaussian fit: mu ≈ 0 confirms the noise model is unbiased;
        # sigma should be close to the theoretical combined std below.
        mu, sigma = residuals_mm.mean(), residuals_mm.std()
        x_fit = np.linspace(residuals_mm.min(), residuals_mm.max(), 300)  # dense x for smooth curve
        ax.plot(x_fit,
                np.exp(-0.5 * ((x_fit - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)),
                lw=2, label=f"Gaussian fit  σ = {sigma:.1f} mm")

        # Theoretical ±1σ lines derived from the same formula as combined_th above.
        sigma_th = np.sqrt((d_target / np.sqrt(photon_density_ph_at_1m))**2
                           + ambient_noise_std_m**2) * 1e3
        ax.axvline(-sigma_th, color="red", linestyle="--", alpha=0.8)
        ax.axvline( sigma_th, color="red", linestyle="--", alpha=0.8,
                   label=f"Theoretical ±σ = {sigma_th:.1f} mm")

        ax.set_title(f"Depth ≈ {d_target:.1f} m  (n = {mask.sum():,} px)")
        ax.set_xlabel("Depth Error (mm)")
        ax.set_ylabel("Probability Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Noise Residual Distributions — Validating the Gaussian Approximation",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("residual_histograms.png", dpi=150)
    plt.show()
    print("Saved: residual_histograms.png")

    # ------------------------------------------------------------------ #
    # 5.  Figure 4: Step-edge scene                                        #
    #                                                                      #
    #     A foreground plane (1 m) beside a background plane (3.5 m)      #
    #     creates a hard depth discontinuity. The red line marks where     #
    #     real ToF sensors exhibit "flying-pixel" artifacts: boundary       #
    #     pixels straddle both planes, receive mixed return photons, and    #
    #     report a blended depth between the two surfaces. This model       #
    #     does not simulate that mixing — noise is applied independently    #
    #     per pixel — so the artifact is absent here (see Caveat C5).      #
    # ------------------------------------------------------------------ #

    # Step scene: left half at 1 m (near), right half at 3.5 m (far).
    depth_step = generate_scene_with_step(sensor_height_px, sensor_width_px,
                                          near_depth=1.0, far_depth=3.5)

    # Apply combined noise; the boundary pixels remain sharp in this model.
    depth_step_combined, _ = add_combined_noise(depth_step,
                                                photon_density_ph_at_1m, ambient_noise_std_m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in [
        (axes[0], depth_step,          "Step Edge — True Depth"),
        (axes[1], depth_step_combined, "Step Edge — Combined Noise\n"
                                       "(real sensors: flying pixels appear at boundary)"),
    ]:
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.axvline(sensor_width_px // 2, color="red", linestyle="--", alpha=0.8,
                   label="Depth discontinuity")
        ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, label="Depth (m)")
    plt.tight_layout()
    plt.savefig("step_edge_scene.png", dpi=150)
    plt.show()
    print("Saved: step_edge_scene.png")

    # ------------------------------------------------------------------ #
    # 6.  Summary table: RMSE at discrete depths                          #
    #                                                                      #
    #     RMSE = √mean(error²) — a single number summarising noise         #
    #     magnitude at each depth. For zero-bias noise (as here) RMSE ≈ σ. #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 68)
    print(f"  {'Depth':>6}  |  {'Shot RMSE':>11}  |  {'Ambient RMSE':>13}  |  {'Combined RMSE':>14}")
    print("-" * 68)
    for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
        # Reuse half_window to select the same pixel slice as the histograms.
        mask = np.abs(depth_true - d) < half_window
        if not mask.any():
            continue
        # Compute RMSE separately for each noise model over the selected pixels.
        rmse_shot     = np.sqrt(np.mean((depth_shot     - depth_true)[mask] ** 2)) * 1e3
        rmse_ambient  = np.sqrt(np.mean((depth_ambient  - depth_true)[mask] ** 2)) * 1e3
        rmse_combined = np.sqrt(np.mean((depth_combined - depth_true)[mask] ** 2)) * 1e3
        print(f"  {d:>5.1f} m  |  {rmse_shot:>8.1f} mm  |  {rmse_ambient:>10.1f} mm  |  {rmse_combined:>11.1f} mm")
    print("=" * 68)