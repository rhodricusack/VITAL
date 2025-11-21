import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os

def load_and_split_mosaic(image_path, grid_size=4, border_width=2):
    """
    Load a mosaic image and split it into individual patches, removing black borders.
    
    The mosaic has a structure like:
    [2px border][64px patch][2px border][64px patch]...[2px border]
    
    Args:
        image_path: Path to the mosaic image
        grid_size: Size of the grid (default 4x4 = 16 patches)
        border_width: Width of black borders between patches (default 2 pixels)
    
    Returns:
        List of 16 individual patch images as numpy arrays (without borders)
    """
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Get dimensions
    height, width = img_array.shape[:2]
    
    # Calculate patch size
    # Formula: total = (grid_size + 1) * border + grid_size * patch_size
    # Solving for patch_size: patch_size = (total - (grid_size + 1) * border) / grid_size
    patch_height = (height - (grid_size + 1) * border_width) // grid_size
    patch_width = (width - (grid_size + 1) * border_width) // grid_size
    
    # Split into patches, skipping borders
    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate start position, accounting for borders
            # Each patch starts after: initial border + (row * patch_size) + (row * border)
            y_start = border_width + row * (patch_height + border_width)
            y_end = y_start + patch_height
            x_start = border_width + col * (patch_width + border_width)
            x_end = x_start + patch_width
            
            patch = img_array[y_start:y_end, x_start:x_end]
            patches.append(patch)
    
    return patches

def calculate_power_spectrum(image):
    """
    Calculate the 2D power spectrum of an image.
    
    Args:
        image: Input image as numpy array (can be RGB or grayscale)
    
    Returns:
        power_spectrum: 2D power spectrum
        radial_profile: 1D radially averaged power spectrum
        frequencies: Frequency bins for radial profile
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image
    
    # Compute 2D FFT
    fft = np.fft.fft2(image_gray)
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_shifted) ** 2
    
    # Compute radially averaged power spectrum
    radial_profile, frequencies = radial_average(power_spectrum)
    
    return power_spectrum, radial_profile, frequencies

def radial_average(data):
    """
    Compute radially averaged profile of 2D data.
    
    Args:
        data: 2D array
    
    Returns:
        radial_profile: 1D radially averaged profile
        radial_bins: Radial distance bins
    """
    center = np.array(data.shape) // 2
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    
    radial_bins = np.arange(len(radial_profile))
    
    return radial_profile, radial_bins

def bootstrap_sem(data, n_bootstrap=1000, confidence=95):
    """
    Calculate bootstrap standard error and confidence intervals.
    
    Args:
        data: 2D array where rows are samples and columns are features
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for intervals (default 95%)
    
    Returns:
        bootstrap_sem: Standard error of the mean from bootstrap
        ci_lower: Lower confidence interval
        ci_upper: Upper confidence interval
    """
    n_samples = data.shape[0]
    bootstrap_means = []
    
    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[indices]
        bootstrap_means.append(np.mean(bootstrap_sample, axis=0))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate bootstrap SEM
    bootstrap_sem = np.std(bootstrap_means, axis=0)
    
    # Calculate confidence intervals
    alpha = (100 - confidence) / 2
    ci_lower = np.percentile(bootstrap_means, alpha, axis=0)
    ci_upper = np.percentile(bootstrap_means, 100 - alpha, axis=0)
    
    return bootstrap_sem, ci_lower, ci_upper

def calculate_jpeg_complexity(image, quality=95):
    """
    Calculate JPEG compression ratio as a measure of visual complexity.
    Higher compression ratio = lower complexity (more compressible).
    
    Args:
        image: Input image as numpy array
        quality: JPEG quality (default 95 for minimal loss)
    
    Returns:
        compression_ratio: Original size / compressed size
        compressed_size: Size in bytes of compressed image
    """
    from io import BytesIO
    
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        pil_img = Image.fromarray(image, mode='RGB')
    
    # Get uncompressed size
    original_size = image.nbytes
    
    # Compress to JPEG
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    compressed_size = buffer.tell()
    
    compression_ratio = original_size / compressed_size
    
    return compression_ratio, compressed_size

def calculate_wavelet_energy(image, wavelet='db4', levels=3):
    """
    Calculate energy in different wavelet subbands.
    Captures multi-scale, multi-orientation features.
    
    Args:
        image: Input image as numpy array
        wavelet: Wavelet type (default 'db4' - Daubechies 4)
        levels: Number of decomposition levels
    
    Returns:
        Dictionary with energy in each subband
    """
    import pywt
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image_gray, wavelet, level=levels)
    
    # Calculate energy in each subband
    energy = {}
    
    # Approximation coefficients (lowest frequency)
    energy['approximation'] = np.sum(coeffs[0] ** 2)
    
    # Detail coefficients (horizontal, vertical, diagonal)
    for level in range(1, len(coeffs)):
        cH, cV, cD = coeffs[level]
        energy[f'horizontal_level_{level}'] = np.sum(cH ** 2)
        energy[f'vertical_level_{level}'] = np.sum(cV ** 2)
        energy[f'diagonal_level_{level}'] = np.sum(cD ** 2)
    
    # Calculate total energy
    total_energy = sum(energy.values())
    
    # Normalize to get energy distribution
    energy_distribution = {k: v/total_energy for k, v in energy.items()}
    
    return energy_distribution, total_energy

def calculate_edge_density(image, threshold=30):
    """
    Calculate edge density using Sobel operator.
    Measures local contrast and sharpness.
    
    Args:
        image: Input image as numpy array
        threshold: Edge strength threshold
    
    Returns:
        edge_density: Proportion of pixels classified as edges
        mean_edge_strength: Average edge magnitude
    """
    from scipy import ndimage
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image
    
    # Sobel filters
    sx = ndimage.sobel(image_gray, axis=0)
    sy = ndimage.sobel(image_gray, axis=1)
    edge_magnitude = np.sqrt(sx**2 + sy**2)
    
    # Edge density (proportion above threshold)
    edge_density = np.mean(edge_magnitude > threshold)
    
    # Mean edge strength
    mean_edge_strength = np.mean(edge_magnitude)
    
    return edge_density, mean_edge_strength

def calculate_entropy(image):
    """
    Calculate Shannon entropy as a measure of information content.
    Higher entropy = more randomness/complexity.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        entropy: Shannon entropy value
    """
    from scipy.stats import entropy as scipy_entropy
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image
    
    # Calculate histogram
    hist, _ = np.histogram(image_gray.ravel(), bins=256, range=(0, 256), density=True)
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Calculate entropy
    ent = scipy_entropy(hist, base=2)
    
    return ent

def analyze_patch_complexity(image):
    """
    Comprehensive complexity analysis of a single patch.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with various complexity measures
    """
    results = {}
    
    # JPEG compression
    compression_ratio, compressed_size = calculate_jpeg_complexity(image)
    results['jpeg_compression_ratio'] = compression_ratio
    results['jpeg_compressed_size'] = compressed_size
    
    # Wavelet energy
    wavelet_dist, total_energy = calculate_wavelet_energy(image)
    results['wavelet_energy_distribution'] = wavelet_dist
    results['wavelet_total_energy'] = total_energy
    
    # Edge density
    edge_density, edge_strength = calculate_edge_density(image)
    results['edge_density'] = edge_density
    results['mean_edge_strength'] = edge_strength
    
    # Entropy
    ent = calculate_entropy(image)
    results['entropy'] = ent
    
    return results

def aggregate_complexity_measures(complexity_list, use_bootstrap=True, n_bootstrap=1000):
    """
    Aggregate complexity measures across multiple patches.
    
    Args:
        complexity_list: List of complexity dictionaries from analyze_patch_complexity
        use_bootstrap: If True, use bootstrap SEM; if False, use traditional SEM
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary with mean and SEM of each measure
    """
    summary = {}
    
    # Simple scalar measures
    scalar_keys = ['jpeg_compression_ratio', 'jpeg_compressed_size', 
                   'wavelet_total_energy', 'edge_density', 
                   'mean_edge_strength', 'entropy']
    
    for key in scalar_keys:
        values = np.array([c[key] for c in complexity_list])
        summary[f'{key}_mean'] = np.mean(values)
        
        if use_bootstrap:
            # Bootstrap SEM
            boot_sem, ci_lower, ci_upper = bootstrap_sem(values.reshape(-1, 1), n_bootstrap=n_bootstrap)
            summary[f'{key}_sem'] = boot_sem[0]
            summary[f'{key}_ci_lower'] = ci_lower[0]
            summary[f'{key}_ci_upper'] = ci_upper[0]
        else:
            # Traditional SEM
            summary[f'{key}_sem'] = np.std(values) / np.sqrt(len(values))
        
        # Keep std for backwards compatibility
        summary[f'{key}_std'] = np.std(values)
    
    # Wavelet energy distribution (aggregate across patches)
    wavelet_keys = list(complexity_list[0]['wavelet_energy_distribution'].keys())
    for wkey in wavelet_keys:
        values = np.array([c['wavelet_energy_distribution'][wkey] for c in complexity_list])
        summary[f'wavelet_{wkey}_mean'] = np.mean(values)
        
        if use_bootstrap:
            boot_sem, ci_lower, ci_upper = bootstrap_sem(values.reshape(-1, 1), n_bootstrap=n_bootstrap)
            summary[f'wavelet_{wkey}_sem'] = boot_sem[0]
            summary[f'wavelet_{wkey}_ci_lower'] = ci_lower[0]
            summary[f'wavelet_{wkey}_ci_upper'] = ci_upper[0]
        else:
            summary[f'wavelet_{wkey}_sem'] = np.std(values) / np.sqrt(len(values))
        
        summary[f'wavelet_{wkey}_std'] = np.std(values)
    
    return summary

def analyze_neuron_patches(base_dir, neuron_idx, output_dir=None):
    """
    Analyze power spectra for all patches of a specific neuron.
    
    Args:
        base_dir: Base directory (e.g., 'resnet50/neuron_layer4.2')
        neuron_idx: Neuron index
        output_dir: Optional output directory for saving results
    
    Returns:
        Dictionary containing patches, power spectra, and radial profiles
    """
    # Load the mosaic image
    image_path = Path(base_dir) / str(neuron_idx) / "patches.png"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading {image_path}")
    
    # Split into individual patches
    patches = load_and_split_mosaic(image_path)
    
    # Calculate power spectra and complexity measures for all patches
    power_spectra = []
    radial_profiles = []
    complexity_measures = []
    frequencies = None
    
    for i, patch in enumerate(patches):
        ps, rp, freq = calculate_power_spectrum(patch)
        power_spectra.append(ps)
        radial_profiles.append(rp)
        if frequencies is None:
            frequencies = freq
        
        # Calculate complexity measures
        complexity = analyze_patch_complexity(patch)
        complexity_measures.append(complexity)
    
    # Calculate mean and std of radial profiles
    radial_profiles = np.array(radial_profiles)
    mean_profile = np.mean(radial_profiles, axis=0)
    std_profile = np.std(radial_profiles, axis=0)
    
    # Aggregate complexity measures
    complexity_summary = aggregate_complexity_measures(complexity_measures)
    
    results = {
        'patches': patches,
        'power_spectra': power_spectra,
        'radial_profiles': radial_profiles,
        'mean_profile': mean_profile,
        'std_profile': std_profile,
        'frequencies': frequencies,
        'complexity_measures': complexity_measures,
        'complexity_summary': complexity_summary
    }
    
    # Save or plot results if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_results(results, neuron_idx, output_dir)
    
    return results

def save_results(results, neuron_idx, output_dir):
    """
    Save power spectrum analysis results.
    """
    # Plot individual power spectra as a grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, (ax, ps) in enumerate(zip(axes.flat, results['power_spectra'])):
        # Plot log power spectrum
        ax.imshow(np.log10(ps + 1), cmap='hot')
        ax.set_title(f'Patch {i}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Power Spectra - Neuron {neuron_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'power_spectra_grid_n{neuron_idx}.png'), dpi=150)
    plt.close()
    
    # Plot radial profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    frequencies = results['frequencies']
    
    # Plot individual profiles (semi-transparent)
    for i, rp in enumerate(results['radial_profiles']):
        ax.loglog(frequencies[1:], rp[1:], alpha=0.2, color='gray')
    
    # Plot mean profile
    ax.loglog(frequencies[1:], results['mean_profile'][1:], 'b-', linewidth=2, label='Mean')
    
    # Plot error bands
    ax.fill_between(frequencies[1:], 
                     results['mean_profile'][1:] - results['std_profile'][1:],
                     results['mean_profile'][1:] + results['std_profile'][1:],
                     alpha=0.3, color='blue', label='±1 std')
    
    ax.set_xlabel('Spatial Frequency (cycles/image)')
    ax.set_ylabel('Power')
    ax.set_title(f'Radially Averaged Power Spectra - Neuron {neuron_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'radial_profile_n{neuron_idx}.png'), dpi=150)
    plt.close()
    
    print(f"Saved results to {output_dir}")

def compare_neurons_spectra(base_dir, neuron_indices, output_path='spectral_comparison.png'):
    """
    Compare radial power spectra across multiple neurons.
    
    Args:
        base_dir: Base directory (e.g., 'resnet50/neuron_layer4.2')
        neuron_indices: List of neuron indices to compare
        output_path: Path to save comparison plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for neuron_idx in neuron_indices:
        results = analyze_neuron_patches(base_dir, neuron_idx)
        frequencies = results['frequencies']
        mean_profile = results['mean_profile']
        
        ax.loglog(frequencies[1:], mean_profile[1:], linewidth=2, label=f'Neuron {neuron_idx}')
    
    ax.set_xlabel('Spatial Frequency (cycles/image)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectra Comparison Across Neurons')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison to {output_path}")

def average_model_spectra(base_dir, neuron_indices=None, max_neurons=None):
    """
    Average radial power spectra across all neurons in a model.
    
    Args:
        base_dir: Base directory (e.g., 'resnet50/neuron_layer4.2')
        neuron_indices: Optional list of specific neuron indices to include
        max_neurons: Optional maximum number of neurons to process
    
    Returns:
        Dictionary with averaged spectra and metadata
    """
    from compare_models import get_sorted_neuron_indices
    
    # Get all available neurons if not specified
    if neuron_indices is None:
        neuron_indices = get_sorted_neuron_indices(base_dir)
        if max_neurons:
            neuron_indices = neuron_indices[:max_neurons]
    
    print(f"Processing {len(neuron_indices)} neurons from {base_dir}")
    
    all_profiles = []
    frequencies = None
    
    for neuron_idx in neuron_indices:
        try:
            results = analyze_neuron_patches(base_dir, neuron_idx)
            # Each neuron has 16 patches, we want all of them
            all_profiles.extend(results['radial_profiles'])
            if frequencies is None:
                frequencies = results['frequencies']
        except FileNotFoundError:
            print(f"Skipping neuron {neuron_idx} - not found")
            continue
    
    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    std_profile = np.std(all_profiles, axis=0)
    
    # Calculate bootstrap SEM and confidence intervals
    boot_sem, ci_lower, ci_upper = bootstrap_sem(all_profiles, n_bootstrap=1000)
    
    # Also calculate traditional SEM for comparison
    traditional_sem = std_profile / np.sqrt(len(all_profiles))
    
    return {
        'mean_profile': mean_profile,
        'std_profile': std_profile,
        'sem_profile': boot_sem,  # Use bootstrap SEM as default
        'traditional_sem': traditional_sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'frequencies': frequencies,
        'n_patches': len(all_profiles),
        'n_neurons': len(neuron_indices)
    }

def compare_models_averaged_spectra(model_dirs, model_names=None, neuron_indices=None, 
                                   max_neurons=None, output_path='model_averaged_spectra.png'):
    """
    Compare averaged power spectra across models (averaging over all neurons and patches).
    Uses a continuous colormap based on g values extracted from model names.
    
    Args:
        model_dirs: List of model directories
        model_names: Optional list of model names for labels (e.g., ['g=0', 'g=0.5', 'g=1'])
        neuron_indices: Optional list of specific neuron indices to use (same for all models)
        max_neurons: Optional maximum number of neurons to process per model
        output_path: Path to save comparison plot
    """
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    # Extract g values from model names
    import re
    g_values = []
    for name in model_names:
        # Try to match with 'g' prefix first, then without
        match = re.search(r'g[=_]?(\d+(?:\.\d+)?(?:pt\d+)?)', name, re.IGNORECASE)
        if not match:
            # Try matching just numbers with optional 'pt' separator
            match = re.search(r'^(\d+(?:pt\d+)?)$', name)
        
        if match:
            g_str = match.group(1).replace('pt', '.')
            g_values.append(float(g_str))
        else:
            g_values.append(0.0)  # Default to 0 if no g value found
    
    print(f"\nExtracted g values: {list(zip(model_names, g_values))}")
    print(f"g_values range: {min(g_values)} to {max(g_values)}")
    
    # Create colormap - use Spectral for nice red-yellow-blue diverging colors
    cmap = plt.cm.get_cmap('Spectral', len(g_values))
    # Map g_values to indices
    g_to_idx = {g: i for i, g in enumerate(sorted(set(g_values)))}
    colors = {g: cmap(g_to_idx[g]) for g in g_values}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_dir, model_name, g_val in zip(model_dirs, model_names, g_values):
        print(f"\nProcessing {model_name} (g={g_val})...")
        results = average_model_spectra(model_dir, neuron_indices, max_neurons)
        
        frequencies = results['frequencies']
        mean_profile = results['mean_profile']
        sem_profile = results['sem_profile']
        
        color = colors[g_val]
        
        # Plot mean line
        ax.loglog(frequencies[1:], mean_profile[1:], linewidth=2, color=color,
                 label=f"{model_name} (n={results['n_patches']})")
        
        # Plot SEM as shaded region
        ax.fill_between(frequencies[1:],
                        mean_profile[1:] - sem_profile[1:],
                        mean_profile[1:] + sem_profile[1:],
                        color=color, alpha=0.2)
    
    ax.set_xlabel('Spatial Frequency (cycles/image)')
    ax.set_ylabel('Power')
    ax.set_title('Averaged Power Spectra Across Models')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved comparison to {output_path}")

def compare_models_spectra(model_dirs, neuron_idx, model_names=None, output_path='model_spectral_comparison.png'):
    """
    Compare power spectra for the same neuron across different models.
    
    Args:
        model_dirs: List of model directories
        neuron_idx: Neuron index to compare
        model_names: Optional list of model names for labels
        output_path: Path to save comparison plot
    """
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_dir, model_name in zip(model_dirs, model_names):
        try:
            results = analyze_neuron_patches(model_dir, neuron_idx)
            frequencies = results['frequencies']
            mean_profile = results['mean_profile']
            
            ax.loglog(frequencies[1:], mean_profile[1:], linewidth=2, label=model_name)
        except FileNotFoundError:
            print(f"Skipping {model_name} - neuron {neuron_idx} not found")
    
    ax.set_xlabel('Spatial Frequency (cycles/image)')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Spectra - Neuron {neuron_idx} Across Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison to {output_path}")

def calculate_imagenet_baseline(imagenet_dir, n_images=1000, patch_size=64, patches_per_image=16):
    """
    Calculate baseline power spectrum from random ImageNet patches.
    
    Args:
        imagenet_dir: Path to ImageNet directory (e.g., 'train' or 'test')
        n_images: Number of random images to sample
        patch_size: Size of patches to extract
        patches_per_image: Number of random patches per image
    
    Returns:
        Dictionary with baseline spectrum statistics
    """
    from torchvision import transforms
    import torch.nn.functional as F
    import glob
    import random
    
    # Find all image files
    image_files = []
    for ext in ['*.JPEG', '*.jpg', '*.png']:
        image_files.extend(glob.glob(os.path.join(imagenet_dir, '**', ext), recursive=True))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {imagenet_dir}")
    
    print(f"Found {len(image_files)} images in {imagenet_dir}")
    
    # Sample random images
    sampled_images = random.sample(image_files, min(n_images, len(image_files)))
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    all_profiles = []
    frequencies = None
    
    print(f"Extracting patches from {len(sampled_images)} images...")
    for img_path in sampled_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            # Extract random patches
            stride = patch_size // 2
            patches = F.unfold(img_tensor, kernel_size=patch_size, stride=stride)
            patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
            
            # Sample random patches
            n_available = patches.shape[0]
            if n_available > 0:
                indices = random.sample(range(n_available), min(patches_per_image, n_available))
                for idx in indices:
                    patch_np = patches[idx].permute(1, 2, 0).numpy() * 255
                    patch_np = patch_np.astype(np.uint8)
                    
                    _, rp, freq = calculate_power_spectrum(patch_np)
                    all_profiles.append(rp)
                    if frequencies is None:
                        frequencies = freq
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    all_profiles = np.array(all_profiles)
    mean_profile = np.mean(all_profiles, axis=0)
    std_profile = np.std(all_profiles, axis=0)
    sem_profile = std_profile / np.sqrt(len(all_profiles))
    
    print(f"Calculated baseline from {len(all_profiles)} patches")
    
    return {
        'mean_profile': mean_profile,
        'std_profile': std_profile,
        'sem_profile': sem_profile,
        'frequencies': frequencies,
        'n_patches': len(all_profiles)
    }

def save_baseline(baseline, output_path='imagenet_baseline_spectrum.npz'):
    """Save baseline spectrum to file."""
    np.savez(output_path, **baseline)
    print(f"Saved baseline to {output_path}")

def load_baseline(baseline_path='imagenet_baseline_spectrum.npz'):
    """Load baseline spectrum from file."""
    data = np.load(baseline_path)
    return {key: data[key] for key in data.keys()}

def compare_models_relative_to_baseline(model_dirs, baseline, model_names=None, 
                                       neuron_indices=None, max_neurons=None,
                                       output_path='model_relative_spectra.png'):
    """
    Compare model spectra relative to ImageNet baseline.
    
    Args:
        model_dirs: List of model directories
        baseline: Baseline spectrum dictionary (from calculate_imagenet_baseline)
        model_names: Optional list of model names
        neuron_indices: Optional specific neurons to include
        max_neurons: Optional max number of neurons per model
        output_path: Path to save plot
    """
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    baseline_mean = baseline['mean_profile']
    baseline_sem = baseline['sem_profile']
    baseline_freq = baseline['frequencies']
    
    # Get first model to determine target frequency array
    results_first = average_model_spectra(model_dirs[0], neuron_indices, max_neurons)
    target_freq = results_first['frequencies']
    
    # Interpolate baseline to match model frequencies if needed
    if len(baseline_freq) != len(target_freq):
        from scipy.interpolate import interp1d
        print(f"Interpolating baseline from {len(baseline_freq)} to {len(target_freq)} points")
        
        # Interpolate in log space for power spectra
        interp_mean = interp1d(baseline_freq, np.log10(baseline_mean + 1e-10), 
                              kind='linear', fill_value='extrapolate')
        interp_sem = interp1d(baseline_freq, np.log10(baseline_sem + 1e-10), 
                             kind='linear', fill_value='extrapolate')
        
        baseline_mean = 10 ** interp_mean(target_freq)
        baseline_sem = 10 ** interp_sem(target_freq)
    
    frequencies = target_freq
    
    # Extract g values from model names for colormap
    import re
    g_values = []
    for name in model_names:
        # Try to match with 'g' prefix first, then without
        match = re.search(r'g[=_]?(\d+(?:\.\d+)?(?:pt\d+)?)', name, re.IGNORECASE)
        if not match:
            # Try matching just numbers with optional 'pt' separator
            match = re.search(r'^(\d+(?:pt\d+)?)$', name)
        
        if match:
            g_str = match.group(1).replace('pt', '.')
            g_values.append(float(g_str))
        else:
            g_values.append(0.0)  # Default to 0 if no g value found
    
    print(f"\nExtracted g values: {list(zip(model_names, g_values))}")
    print(f"g_values range: {min(g_values)} to {max(g_values)}")
    
    # Create colormap - use Spectral for nice red-yellow-blue diverging colors
    cmap = plt.cm.get_cmap('Spectral', len(g_values))
    # Map g_values to indices
    g_to_idx = {g: i for i, g in enumerate(sorted(set(g_values)))}
    colors = {g: cmap(g_to_idx[g]) for g in g_values}
    
    # Plot 1: Absolute power spectra
    ax1.loglog(frequencies[1:], baseline_mean[1:], 'k--', linewidth=2, 
               label=f"ImageNet baseline (n={baseline['n_patches']})")
    ax1.fill_between(frequencies[1:],
                     baseline_mean[1:] - baseline_sem[1:],
                     baseline_mean[1:] + baseline_sem[1:],
                     color='gray', alpha=0.2)
    
    for model_dir, model_name, g_val in zip(model_dirs, model_names, g_values):
        print(f"\nProcessing {model_name} (g={g_val})...")
        results = average_model_spectra(model_dir, neuron_indices, max_neurons)
        
        mean_profile = results['mean_profile']
        sem_profile = results['sem_profile']
        
        color = colors[g_val]
        
        ax1.loglog(frequencies[1:], mean_profile[1:], linewidth=2, color=color,
                   label=f"{model_name} (n={results['n_patches']})")
        ax1.fill_between(frequencies[1:],
                        mean_profile[1:] - sem_profile[1:],
                        mean_profile[1:] + sem_profile[1:],
                        color=color, alpha=0.2)
    
    ax1.set_xlabel('Spatial Frequency (cycles/image)')
    ax1.set_ylabel('Power')
    ax1.set_title('Absolute Power Spectra')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative to baseline (ratio)
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='ImageNet baseline')
    
    for model_dir, model_name, g_val in zip(model_dirs, model_names, g_values):
        results = average_model_spectra(model_dir, neuron_indices, max_neurons)
        
        mean_profile = results['mean_profile']
        relative_profile = mean_profile / baseline_mean
        
        color = colors[g_val]
        ax2.semilogx(frequencies[1:], relative_profile[1:], linewidth=2, color=color, label=model_name)
    
    ax2.set_xlabel('Spatial Frequency (cycles/image)')
    ax2.set_ylabel('Power Ratio (Model / Baseline)')
    ax2.set_title('Power Spectra Relative to ImageNet Baseline')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved comparison to {output_path}")

def compare_models_complexity(model_dirs, model_names=None, neuron_indices=None, 
                              max_neurons=None, output_path='model_complexity_comparison.png'):
    """
    Compare complexity measures across models.
    
    Args:
        model_dirs: List of model directories
        model_names: Optional list of model names
        neuron_indices: Optional specific neurons to include
        max_neurons: Optional max number of neurons per model
        output_path: Path to save plot
    """
    from compare_models import get_sorted_neuron_indices, extract_g_value
    
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    # Extract g values for x-axis
    g_values = []
    for name in model_names:
        g_str = extract_g_value(name)
        try:
            g_val = float(g_str)
        except (ValueError, TypeError):
            g_val = 0.0
        g_values.append(g_val)
    
    # Collect complexity measures for each model
    model_complexities = {}
    
    for model_dir, model_name, g_val in zip(model_dirs, model_names, g_values):
        print(f"\nProcessing {model_name} (g={g_val})...")
        
        # Get neuron indices
        if neuron_indices is None:
            indices = get_sorted_neuron_indices(model_dir)
            if max_neurons:
                indices = indices[:max_neurons]
        else:
            indices = neuron_indices
        
        # Collect all complexity measures
        all_measures = []
        for neuron_idx in indices:
            try:
                results = analyze_neuron_patches(model_dir, neuron_idx)
                all_measures.extend(results['complexity_measures'])
            except FileNotFoundError:
                continue
        
        # Aggregate across all patches and neurons
        if all_measures:
            agg = aggregate_complexity_measures(all_measures)
            model_complexities[g_val] = agg
    
    # Sort by g value for plotting
    g_sorted = sorted(model_complexities.keys())
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: JPEG Compression Ratio (higher = more compressible = simpler)
    vals = [model_complexities[g]['jpeg_compression_ratio_mean'] for g in g_sorted]
    errs = [model_complexities[g]['jpeg_compression_ratio_sem'] for g in g_sorted]
    axes[0].errorbar(g_sorted, vals, yerr=errs, marker='o', linewidth=2, capsize=5)
    axes[0].set_xlabel('Gaussian blur (σ)')
    axes[0].set_ylabel('JPEG Compression Ratio')
    axes[0].set_title('Compressibility\n(higher = simpler)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Entropy (higher = more complex/random)
    vals = [model_complexities[g]['entropy_mean'] for g in g_sorted]
    errs = [model_complexities[g]['entropy_sem'] for g in g_sorted]
    axes[1].errorbar(g_sorted, vals, yerr=errs, marker='o', linewidth=2, capsize=5)
    axes[1].set_xlabel('Gaussian blur (σ)')
    axes[1].set_ylabel('Shannon Entropy (bits)')
    axes[1].set_title('Information Content\n(higher = more complex)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Edge Density (higher = more edges/detail)
    vals = [model_complexities[g]['edge_density_mean'] for g in g_sorted]
    errs = [model_complexities[g]['edge_density_sem'] for g in g_sorted]
    axes[2].errorbar(g_sorted, vals, yerr=errs, marker='o', linewidth=2, capsize=5)
    axes[2].set_xlabel('Gaussian blur (σ)')
    axes[2].set_ylabel('Edge Density')
    axes[2].set_title('Edge Density\n(higher = sharper)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Mean Edge Strength
    vals = [model_complexities[g]['mean_edge_strength_mean'] for g in g_sorted]
    errs = [model_complexities[g]['mean_edge_strength_sem'] for g in g_sorted]
    axes[3].errorbar(g_sorted, vals, yerr=errs, marker='o', linewidth=2, capsize=5)
    axes[3].set_xlabel('Gaussian blur (σ)')
    axes[3].set_ylabel('Mean Edge Strength')
    axes[3].set_title('Average Edge Magnitude')
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Wavelet Energy Distribution (stacked bar chart)
    # Show energy in high frequency details vs approximation
    detail_keys = [k for k in model_complexities[g_sorted[0]].keys() 
                   if 'wavelet_' in k and '_mean' in k and 'approximation' not in k]
    approx_key = 'wavelet_approximation_mean'
    
    detail_vals = []
    approx_vals = []
    for g in g_sorted:
        detail_sum = sum(model_complexities[g][k] for k in detail_keys if 'horizontal' in k or 'vertical' in k or 'diagonal' in k)
        detail_vals.append(detail_sum)
        approx_vals.append(model_complexities[g][approx_key])
    
    axes[4].plot(g_sorted, detail_vals, marker='o', label='High frequency details', linewidth=2)
    axes[4].plot(g_sorted, approx_vals, marker='s', label='Low frequency (approximation)', linewidth=2)
    axes[4].set_xlabel('Gaussian blur (σ)')
    axes[4].set_ylabel('Normalized Wavelet Energy')
    axes[4].set_title('Wavelet Energy Distribution')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Wavelet Total Energy
    vals = [model_complexities[g]['wavelet_total_energy_mean'] for g in g_sorted]
    errs = [model_complexities[g]['wavelet_total_energy_sem'] for g in g_sorted]
    axes[5].errorbar(g_sorted, vals, yerr=errs, marker='o', linewidth=2, capsize=5)
    axes[5].set_xlabel('Gaussian blur (σ)')
    axes[5].set_ylabel('Total Wavelet Energy')
    axes[5].set_title('Total Energy')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved complexity comparison to {output_path}")
    
    return model_complexities

# Example usage
if __name__ == "__main__": 
    # Example 5: Calculate ImageNet baseline and compare models relative to it
    # First, calculate or load baseline
    baseline_path = 'inner_fvis/spectra_analysis/imagenet_baseline.npz'
    
    if not os.path.exists(baseline_path):
        # Calculate baseline from ImageNet (only need to do this once)
        baseline = calculate_imagenet_baseline(
            imagenet_dir='inner_fvis/train',  # Adjust path to your ImageNet directory
            n_images=500,
            patch_size=64,
            patches_per_image=16
        )
        save_baseline(baseline, baseline_path)
    else:
        # Load previously calculated baseline
        baseline = load_baseline(baseline_path)
        print(f"Loaded baseline with {baseline['n_patches']} patches")
    
    # Compare models relative to baseline
    glist = ['0', '0pt5', '1', '1pt5', '2', '3', '4', '6']
    for layer in ['2.2','3.2', '4.2']:
        # compare_models_averaged_spectra(
        #     model_dirs=[f'inner_fvis/patch_results/resnet50_BV_g{g}For60_E60/neuron_layer{layer}' for g in glist],
        #     model_names=glist,
        #     output_path=f'inner_fvis/spectra_analysis/models_averaged_comparison_layer{layer}.png'
        # )
        
        # compare_models_relative_to_baseline(
        #     model_dirs=[f'inner_fvis/patch_results/resnet50_BV_g{g}For60_E60/neuron_layer{layer}' for g in glist],
        #     baseline=baseline,
        #     model_names=glist,
        #     output_path=f'inner_fvis/spectra_analysis/models_relative_to_baseline_layer{layer}.png'
        # )

        compare_models_complexity(
            model_dirs=[f'inner_fvis/patch_results/resnet50_BV_g{g}For60_E60/neuron_layer{layer}' for g in glist],
            model_names=glist,
            output_path=f'inner_fvis/spectra_analysis/models_complexity_comparison_layer{layer}.png'
        )