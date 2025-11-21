import math
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def load_patch_image(base_dir, neuron_idx):
    """
    Given base_dir like 'resnet50/neuron_layer4.2', load: base_dir/<idx>/patches.png
    """
    p = Path(base_dir) / str(neuron_idx) / "patches.png"
    if not p.exists():
        raise FileNotFoundError(f"Image missing: {p}")
    return Image.open(p)

def get_sorted_neuron_indices(base_dir):
    """
    Find all integer subdirectories inside the model directory.
    """
    dirs = [d for d in os.listdir(base_dir) if d.isdigit() and (Path(base_dir) / d / "patches.png").exists()]
    return sorted(map(int, dirs))

def compare_models(model_dirs, out_dir="comparisons", model_names=None):
    """
    For each neuron index present in ALL models, generate a side-by-side comparison figure.
    
    Args:
        model_dirs: List of model directories (e.g., ["resnet50/neuron_layer4.2", "resnet50_blurry_6/neuron_layer4.2"])
        out_dir: Output directory for comparison images
        model_names: Optional list of model names. If None, will use parent directory names.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get indices for all models
    all_indices = [set(get_sorted_neuron_indices(d)) for d in model_dirs]
    
    # Find common indices across all models
    common_indices = sorted(set.intersection(*all_indices))
    
    # Generate model names
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    num_models = len(model_dirs)
    print(f"Found {len(common_indices)} common neuron indices across {num_models} models.")

    for i in common_indices:
        # Load all images
        images = [load_patch_image(d, i) for d in model_dirs]
        
        # Create subplots - arrange in a row
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        
        # Handle single model case
        if num_models == 1:
            axes = [axes]
        
        for idx, (ax, img, name) in enumerate(zip(axes, images, model_names)):
            ax.imshow(img)
            ax.set_title(f"{name} — neuron {i}")
            ax.axis("off")

        out_path = os.path.join(out_dir, f"compare_{i}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {out_path}")

def extract_g_value(model_name):
    """
    Extract the value after 'g' from model name.
    E.g., 'resnet50_BV_g2For60_E60' -> '2'
          'resnet50_BV_g0pt5For60_E60' -> '0.5'
          'resnet50_gauss_6' -> '6'
    """
    import re
    # Look for pattern like g2, g0pt5, g6, etc.
    match = re.search(r'_g(\d+(?:pt\d+)?)', model_name)
    if match:
        g_val = match.group(1)
        # Replace 'pt' with '.'
        g_val = g_val.replace('pt', '.')
        return g_val
    # Fallback: look for gauss_X pattern
    match = re.search(r'gauss_(\d+)', model_name)
    if match:
        return match.group(1)
    return model_name  # Return original if no pattern found

def composite_grid(model_dirs, out_path="comparison_grid.png", model_names=None):
    """
    Creates one large grid with models in rows and neurons in columns:
        [model1 neuron0] [model1 neuron1] ...
        [model2 neuron0] [model2 neuron1] ...
        [model3 neuron0] [model3 neuron1] ...
        ...
    
    Args:
        model_dirs: List of model directories
        out_path: Output path for the grid image
        model_names: Optional list of model names. If None, will use parent directory names.
    """
    # Get indices for all models
    all_indices = [set(get_sorted_neuron_indices(d)) for d in model_dirs]
    
    # Find common indices across all models
    common_indices = sorted(set.intersection(*all_indices))
    
    # Generate model names
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    # Extract g values for row labels
    row_labels = [extract_g_value(name) for name in model_names]
    
    num_models = len(model_dirs)
    num_neurons = len(common_indices)
    
    # Create grid: rows=models, cols=neurons
    fig, axes = plt.subplots(num_models, num_neurons, figsize=(num_neurons * 2, num_models * 2))

    # Handle edge cases for axes indexing
    if num_models == 1 and num_neurons == 1:
        axes = [[axes]]
    elif num_models == 1:
        axes = [axes]
    elif num_neurons == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (model_dir, model_name, row_label) in enumerate(zip(model_dirs, model_names, row_labels)):
        for col_idx, neuron_idx in enumerate(common_indices):
            img = load_patch_image(model_dir, neuron_idx)
            
            ax = axes[row_idx][col_idx]
            ax.imshow(img)
            ax.axis("off")
            
            # Add column labels (neuron indices) at the top
            if row_idx == 0:
                ax.set_title(f"n={neuron_idx}", fontsize=10, pad=5)
            
            # Add row labels (g values) on the left
            if col_idx == 0:
                ax.text(-0.05, 0.5, row_label, transform=ax.transAxes, 
                       fontsize=10, rotation=90, ha='center', va='center')

    plt.tight_layout()
    
    # Add overall y-axis label (to the left of the g numbers)
    fig.text(-0.01, 0.5, 'Gaussian smoothing (pixels)', ha='center', va='center', 
             rotation=90, fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    # Add overall x-axis label (at the top, above the neuron numbers)
    fig.text(0.5, 1.02, 'Neuron number', ha='center', va='top', 
             fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved grid: {out_path}")

def select_representative_neurons(model_dir, n_neurons=8, max_neurons=None):
    """
    Select neurons whose frequency spectra are most similar to the model average.
    
    Args:
        model_dir: Model directory (e.g., 'resnet50/neuron_layer4.2')
        n_neurons: Number of representative neurons to select
        max_neurons: Optional maximum number of neurons to consider
    
    Returns:
        List of neuron indices sorted by similarity to average spectrum
    """
    import sys
    import numpy as np
    
    # Import from analyze_patch_spectra
    sys.path.insert(0, os.path.dirname(__file__))
    from analyze_patch_spectra import analyze_neuron_patches, average_model_spectra
    
    # Get available neurons
    neuron_indices = get_sorted_neuron_indices(model_dir)
    if max_neurons:
        neuron_indices = neuron_indices[:max_neurons]
    
    print(f"Analyzing {len(neuron_indices)} neurons from {model_dir}...")
    
    # Get the average spectrum for the model
    model_avg = average_model_spectra(model_dir, neuron_indices=neuron_indices)
    avg_profile = model_avg['mean_profile']
    
    # Calculate similarity for each neuron
    similarities = []
    valid_neurons = []
    
    for neuron_idx in neuron_indices:
        try:
            results = analyze_neuron_patches(model_dir, neuron_idx)
            neuron_mean = results['mean_profile']
            
            # Calculate correlation in log space (better for power spectra)
            log_avg = np.log10(avg_profile[1:] + 1e-10)
            log_neuron = np.log10(neuron_mean[1:] + 1e-10)
            
            # Pearson correlation
            correlation = np.corrcoef(log_avg, log_neuron)[0, 1]
            
            similarities.append(correlation)
            valid_neurons.append(neuron_idx)
            
        except Exception as e:
            print(f"Skipping neuron {neuron_idx}: {e}")
            continue
    
    # Sort by similarity (highest correlation first)
    sorted_indices = np.argsort(similarities)[::-1]
    representative_neurons = [valid_neurons[i] for i in sorted_indices[:n_neurons]]
    representative_similarities = [similarities[i] for i in sorted_indices[:n_neurons]]
    
    print(f"Selected {len(representative_neurons)} representative neurons:")
    for neuron, sim in zip(representative_neurons, representative_similarities):
        print(f"  Neuron {neuron}: correlation = {sim:.3f}")
    
    return representative_neurons

def composite_grid_representative(model_dirs, n_neurons=8, out_path="comparison_grid_representative.png", 
                                 model_names=None, max_neurons=None):
    """
    Creates a grid with models in rows and representative neurons in columns.
    Representative neurons are selected based on similarity to model average spectrum.
    
    Args:
        model_dirs: List of model directories
        n_neurons: Number of representative neurons to select per model
        out_path: Output path for the grid image
        model_names: Optional list of model names
        max_neurons: Optional maximum number of neurons to consider when selecting
    """
    if model_names is None:
        model_names = [Path(d).parent.name for d in model_dirs]
    
    # Extract g values for row labels
    row_labels = [extract_g_value(name) for name in model_names]
    
    num_models = len(model_dirs)
    
    # Select representative neurons for each model
    all_representative_neurons = []
    for model_dir, model_name in zip(model_dirs, model_names):
        print(f"\nSelecting representative neurons for {model_name}...")
        rep_neurons = select_representative_neurons(model_dir, n_neurons=n_neurons, max_neurons=max_neurons)
        all_representative_neurons.append(rep_neurons)
    
    # Create grid: rows=models, cols=neurons
    fig, axes = plt.subplots(num_models, n_neurons, figsize=(n_neurons * 2, num_models * 2))

    # Handle edge cases for axes indexing
    if num_models == 1 and n_neurons == 1:
        axes = [[axes]]
    elif num_models == 1:
        axes = [axes]
    elif n_neurons == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (model_dir, model_name, row_label, rep_neurons) in enumerate(
        zip(model_dirs, model_names, row_labels, all_representative_neurons)
    ):
        for col_idx, neuron_idx in enumerate(rep_neurons):
            try:
                img = load_patch_image(model_dir, neuron_idx)
                
                ax = axes[row_idx][col_idx]
                ax.imshow(img)
                ax.axis("off")
                
                # Add column labels (neuron indices) at the top
                if row_idx == 0:
                    ax.set_title(f"n{neuron_idx}", fontsize=10, pad=5)
                
                # Add row labels (g values) on the left
                if col_idx == 0:
                    ax.text(-0.1, 0.5, f"σ={row_label}", transform=ax.transAxes,
                           fontsize=10, va='center', ha='right', rotation=0)
            
            except FileNotFoundError:
                print(f"Warning: Image not found for {model_name}, neuron {neuron_idx}")
                ax = axes[row_idx][col_idx]
                ax.axis("off")
    
    # Add overall y-axis label (to the left of the g numbers)
    fig.text(-0.01, 0.5, 'Gaussian smoothing (pixels)', ha='center', va='center', 
             rotation=90, fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    # Add overall x-axis label (at the top, above the neuron numbers)
    fig.text(0.5, 1.02, 'Representative neurons (most similar to model average)', ha='center', va='top', 
             fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved representative grid: {out_path}")

# Example usage for 2 models (backward compatible):
# compare_models(
#     model_dirs=["resnet50/neuron_layer4.2", "resnet50_blurry_6/neuron_layer4.2"],
#     out_dir="pairwise_comparisons",
#     model_names=["resnet50", "resnet50_blurry_6"]
# )

# Example usage for 3+ models:
# compare_models(
#     model_dirs=[
#         "resnet50/neuron_layer4.2",
#         "resnet50_blurry_6/neuron_layer4.2",
#         "resnet50_BV_g0For60_E60/neuron_layer4.2"
#     ],
#     out_dir="multi_model_comparisons",
#     model_names=["resnet50", "blurry_6", "BV_g0"]
# )

# Optional big grid
if __name__ == "__main__":

    composite_grid(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer3.2"
        ],
        out_path="inner_fvis/visualization_grids/all_neurons_grid_layer3.2.png"
    )

    composite_grid(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer4.2"
        ],
        out_path="inner_fvis/visualization_grids/all_neurons_grid_layer4.2.png"
    )
    
    composite_grid(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer2.2",
        ],
        out_path="inner_fvis/visualization_grids/all_neurons_grid_layer2.2.png"
    )
    # Create representative neuron grids (8 neurons most similar to model average)
    composite_grid_representative(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer3.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer3.2"
        ],
        n_neurons=8,
        out_path="inner_fvis/visualization_grids/representative_neurons_grid_layer3.2.png",
        max_neurons=50  # Only consider first 50 neurons for speed
    )
    
    composite_grid_representative(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer4.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer4.2"
        ],
        n_neurons=8,
        out_path="inner_fvis/visualization_grids/representative_neurons_grid_layer4.2.png",
        max_neurons=50
    )

    composite_grid_representative(
        model_dirs=[
            "inner_fvis/patch_results/resnet50_BV_g0For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g0pt5For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g1For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g1pt5For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g2For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g3For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g4For60_E60/neuron_layer2.2",
            "inner_fvis/patch_results/resnet50_BV_g6For60_E60/neuron_layer2.2",
        ],
        n_neurons=8,
        out_path="inner_fvis/visualization_grids/representative_neurons_grid_layer2.2.png",
        max_neurons=50
    )
