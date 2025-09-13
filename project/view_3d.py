#!/usr/bin/env python3
"""
Simple 3D viewer for Gaussian Splatting PLY files using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import struct

def read_ply_file(file_path: Path):
    """Read PLY file and extract vertex data"""
    
    print(f"Reading PLY file: {file_path}")
    
    vertices = []
    colors = []
    scales = []
    rotations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find header information
    vertex_count = 0
    in_header = True
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if in_header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                in_header = False
                data_start = i + 1
                break
    
    print(f"Found {vertex_count} vertices")
    
    # Parse vertex data
    for i in range(data_start, data_start + vertex_count):
        parts = lines[i].strip().split()
        if len(parts) >= 13:  # Ensure we have all required fields
            # Position (x, y, z)
            pos = [float(parts[0]), float(parts[1]), float(parts[2])]
            vertices.append(pos)
            
            # Color (r, g, b)
            color = [int(parts[3]), int(parts[4]), int(parts[5])]
            colors.append(color)
            
            # Scale (scale_0, scale_1, scale_2)
            scale = [float(parts[7]), float(parts[8]), float(parts[9])]
            scales.append(scale)
            
            # Rotation (rot_0, rot_1, rot_2, rot_3)
            rot = [float(parts[10]), float(parts[11]), float(parts[12]), float(parts[13])]
            rotations.append(rot)
    
    return np.array(vertices), np.array(colors), np.array(scales), np.array(rotations)

def visualize_3d_model(vertices, colors, scales, title="3D Gaussian Splat Model"):
    """Create 3D visualization of the Gaussian Splat model"""
    
    print(f"Creating 3D visualization with {len(vertices)} points...")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to 0-1 range
    colors_normalized = colors / 255.0
    
    # Create scatter plot
    scatter = ax.scatter(
        vertices[:, 0], 
        vertices[:, 1], 
        vertices[:, 2],
        c=colors_normalized,
        s=20,  # Point size
        alpha=0.6,
        edgecolors='none'
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        vertices[:,0].max() - vertices[:,0].min(),
        vertices[:,1].max() - vertices[:,1].min(),
        vertices[:,2].max() - vertices[:,2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:,0].max() + vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max() + vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max() + vertices[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add statistics text
    stats_text = f"""
Statistics:
• Points: {len(vertices):,}
• Position range: {vertices.min():.3f} to {vertices.max():.3f}
• Average scale: {scales.mean():.3f}
• Color range: {colors.min()} to {colors.max()}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=8, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def save_visualization(fig, output_path: Path):
    """Save the visualization to an image file"""
    
    print(f"Saving visualization to {output_path}")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="3D Gaussian Splat model viewer")
    parser.add_argument("--ply_file", type=str, default="outputs/point_cloud.ply", 
                       help="Path to PLY file")
    parser.add_argument("--output_image", type=str, default="outputs/3d_visualization.png",
                       help="Path to save visualization image")
    parser.add_argument("--show_interactive", action="store_true",
                       help="Show interactive 3D plot")
    
    args = parser.parse_args()
    
    ply_path = Path(args.ply_file)
    if not ply_path.exists():
        print(f"Error: PLY file not found at {ply_path}")
        return
    
    try:
        # Read PLY file
        vertices, colors, scales, rotations = read_ply_file(ply_path)
        
        # Create visualization
        fig, ax = visualize_3d_model(vertices, colors, scales, 
                                   title=f"3D Gaussian Splat Model - {ply_path.name}")
        
        # Save visualization
        output_path = Path(args.output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_visualization(fig, output_path)
        
        # Show interactive plot if requested
        if args.show_interactive:
            print("🖱️  Showing interactive 3D plot...")
            print("   - Use mouse to rotate, zoom, and pan")
            print("   - Close the window to continue")
            plt.show()
        
        print("3D visualization completed!")
        print(f"Image saved to: {output_path}")
        print(f"Model contains {len(vertices):,} Gaussian points")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
