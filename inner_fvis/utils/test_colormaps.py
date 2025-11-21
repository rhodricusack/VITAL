"""
Visualize different colormap options for the frequency spectra plots.
This helps you choose which colormap works best.
"""

import matplotlib.pyplot as plt
import numpy as np

# Your g values
g_values = [0, 0.5, 1, 2, 3, 4, 6]
n_colors = len(g_values)

# Different colormap options
colormap_options = {
    'tab10': 'Categorical (10 distinct colors) - RECOMMENDED',
    'Set1': 'Categorical (9 distinct colors)',
    'Dark2': 'Categorical (8 muted colors)',
    'tab20': 'Categorical (20 distinct colors)',
    'turbo': 'Continuous (rainbow-like, high contrast)',
    'Spectral': 'Diverging (red-yellow-blue)',
    'rainbow': 'Continuous (classic rainbow)',
    'coolwarm': 'Diverging (blue-red)',
    'viridis': 'Continuous (purple-yellow) - CURRENT'
}

fig, axes = plt.subplots(len(colormap_options), 1, figsize=(10, 12))

for ax, (cmap_name, description) in zip(axes, colormap_options.items()):
    # Get colormap
    cmap = plt.cm.get_cmap(cmap_name, n_colors)
    
    # Create a color map for each g value
    g_to_idx = {g: i for i, g in enumerate(sorted(set(g_values)))}
    colors = [cmap(g_to_idx[g]) for g in g_values]
    
    # Draw color bars
    for i, (g, color) in enumerate(zip(g_values, colors)):
        ax.barh(0, 1, left=i, color=color, edgecolor='black', linewidth=1)
        ax.text(i + 0.5, -0.15, f'g={g}', ha='center', va='top', fontsize=10)
    
    ax.set_xlim(0, n_colors)
    ax.set_ylim(-0.3, 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'{cmap_name}: {description}', fontsize=11, loc='left', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

plt.suptitle('Colormap Options for 7 Models\n(showing colors for g=0, 0.5, 1, 2, 3, 4, 6)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('inner_fvis/colormap_comparison.png', dpi=150, bbox_inches='tight')
print("Saved colormap comparison to: inner_fvis/colormap_comparison.png")
plt.show()
