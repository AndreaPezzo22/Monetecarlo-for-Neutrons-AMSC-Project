import json
import argparse
import os
import matplotlib.pyplot as plt

def draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color, label, seen_labels):
    edges = [
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_max, y_min], [z_min, z_min]),
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_max, y_min], [z_max, z_max]),
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max])
    ]
    
    # Ensure the material name only appears once in the legend
    plot_label = label if label not in seen_labels else ""
    if label:
        seen_labels.add(label)

    for i, (x, y, z) in enumerate(edges):
        ax.plot(x, y, z, color=color, label=plot_label if i == 0 else "")

def main():
    parser = argparse.ArgumentParser(description="Launch a 3D view of the simulation domain.")
    parser.add_argument("config", help="Configuration name (e.g., multilayered_wall)")
    args = parser.parse_args()

    clean_name = args.config.replace(".json", "")
    config_path = f"{clean_name}.json"

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    seen_labels = set()
    
    material_colors = {
        'iron': 'blue',
        'water': 'cyan',
        'boron': 'green',
        'vacuum': 'gray'
    }
    fallback_colors = ['orange', 'purple', 'magenta', 'yellow', 'brown']
    fallback_idx = 0

    # Dictionaries to track global boundaries for the fixed aspect ratio
    bounds = {'x': [], 'y': [], 'z': []}

    def update_bounds(xmin, xmax, ymin, ymax, zmin, zmax):
        bounds['x'].extend([xmin, xmax])
        bounds['y'].extend([ymin, ymax])
        bounds['z'].extend([zmin, zmax])

    # Draw the Source region
    src = config.get('source')
    if src:
        update_bounds(src['min_x'], src['max_x'], src['min_y'], src['max_y'], src['min_z'], src['max_z'])
        draw_box(ax, src['min_x'], src['max_x'], src['min_y'], src['max_y'], 
                 src['min_z'], src['max_z'], color='red', label='source', seen_labels=seen_labels)

    # Draw the Material regions
    for region in config.get('regions', []):
        mat_name = region.get('material_name', 'unknown')
        
        if mat_name in material_colors:
            color = material_colors[mat_name]
        else:
            color = fallback_colors[fallback_idx % len(fallback_colors)]
            material_colors[mat_name] = color
            fallback_idx += 1

        update_bounds(region['min_x'], region['max_x'], region['min_y'], region['max_y'], region['min_z'], region['max_z'])
        draw_box(ax, region['min_x'], region['max_x'], region['min_y'], region['max_y'], 
                 region['min_z'], region['max_z'], color=color, label=mat_name, seen_labels=seen_labels)

    # Enforce strict 1:1:1 aspect ratio by squaring the axis limits
    if bounds['x']:
        x_min, x_max = min(bounds['x']), max(bounds['x'])
        y_min, y_max = min(bounds['y']), max(bounds['y'])
        z_min, z_max = min(bounds['z']), max(bounds['z'])
        
        # Find the midpoints of each axis
        mid_x = (x_max + x_min) / 2.0
        mid_y = (y_max + y_min) / 2.0
        mid_z = (z_max + z_min) / 2.0
        
        # Find the maximum range among all 3 axes
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Set all axes to the exact same range size, centered on their midpoints
        half_range = max_range / 2.0
        ax.set_xlim(mid_x - half_range, mid_x + half_range)
        ax.set_ylim(mid_y - half_range, mid_y + half_range)
        ax.set_zlim(mid_z - half_range, mid_z + half_range)
        
        # Force the physical drawing box to be a perfect cube
        ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Simulation Domain: {clean_name}")

    default_mat = config.get('default', 'Not specified')
    ax.text2D(0.05, 0.95, f"Default Background: {default_mat}", transform=ax.transAxes)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
