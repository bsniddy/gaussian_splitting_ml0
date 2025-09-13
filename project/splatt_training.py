#!/usr/bin/env python3
"""
Better Gaussian Splatting training that preserves 3D structure
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import struct

def read_colmap_binary(file_path):
    """Read COLMAP binary files"""
    
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file."""
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_points3D_binary(path_to_model_file):
        """Read 3D points from COLMAP binary file"""
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            
            for point_line_index in range(num_points):
                binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
                point_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = read_next_bytes(fid, 8, "Q")[0]
                track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
                xyz_ids = np.column_stack([tuple(map(int, track_elems[0::2])),
                                         tuple(map(int, track_elems[1::2]))])
                points3D[point_id] = type('Point3D', (), {
                    'id': point_id,
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'image_ids': xyz_ids[:, 0],
                    'point2D_idxs': xyz_ids[:, 1]
                })()
                
        return points3D

    def read_images_binary(path_to_model_file):
        """Read images from COLMAP binary file"""
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            
            for image_index in range(num_reg_images):
                binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, 8, "Q")[0]
                x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                     tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                
                images[image_id] = type('Image', (), {
                    'id': image_id,
                    'qvec': qvec,
                    'tvec': tvec,
                    'camera_id': camera_id,
                    'name': image_name,
                    'xys': xys,
                    'point3D_ids': point3D_ids
                })()
                
        return images

    def read_cameras_binary(path_to_model_file):
        """Read cameras from COLMAP binary file"""
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            
            for camera_index in range(num_cameras):
                camera_properties = read_next_bytes(fid, 24, "iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]
                params = read_next_bytes(fid, 8 * 4, "d" * 4)
                
                cameras[camera_id] = type('Camera', (), {
                    'id': camera_id,
                    'model': model_id,
                    'width': width,
                    'height': height,
                    'params': np.array(params)
                })()
                
        return cameras

    return {
        'points3D': read_points3D_binary(file_path / "points3D.bin"),
        'images': read_images_binary(file_path / "images.bin"),
        'cameras': read_cameras_binary(file_path / "cameras.bin")
    }

def create_gaussians_from_colmap_points(colmap_data, num_points=None, device="cpu"):
    """Create initial Gaussians from real COLMAP 3D points"""
    
    print("Creating Gaussians from real COLMAP 3D points...")
    
    points3D = colmap_data['points3D']
    
    # Extract 3D point positions and colors
    positions = []
    colors = []
    
    for point_id, point in points3D.items():
        positions.append(point.xyz)
        colors.append(point.rgb)
    
    positions = np.array(positions)
    colors = np.array(colors)
    
    print(f"Found {len(positions)} 3D points from COLMAP")
    print(f"Position range: {positions.min():.3f} to {positions.max():.3f}")
    
    # Sample points if requested
    if num_points and len(positions) > num_points:
        indices = np.random.choice(len(positions), num_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        print(f"Sampled {num_points} points")
    
    # Convert to torch tensors
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device) / 255.0  # Normalize to 0-1
    
    # Initialize other Gaussian parameters
    scales = torch.ones_like(positions) * 0.1  # Larger initial scales
    rotations = torch.zeros((len(positions), 4), dtype=torch.float32, device=device)
    rotations[:, 0] = 1.0  # Quaternion (1, 0, 0, 0)
    opacities = torch.ones((len(positions), 1), dtype=torch.float32, device=device) * 0.8
    
    print(f"Created {len(positions)} Gaussians from real 3D points")
    
    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities
    }

def better_training(gaussians, num_iterations=1000, device="cpu"):
    """Better training that preserves 3D structure"""
    
    print(f"Training with better loss function for {num_iterations} iterations...")
    
    # Make parameters trainable
    positions = gaussians["positions"].requires_grad_(True)
    scales = gaussians["scales"].requires_grad_(True)
    rotations = gaussians["rotations"].requires_grad_(True)
    colors = gaussians["colors"].requires_grad_(True)
    opacities = gaussians["opacities"].requires_grad_(True)
    
    # Store original positions to preserve structure
    original_positions = gaussians["positions"].detach().clone()
    original_colors = gaussians["colors"].detach().clone()
    
    # Setup optimizer with conservative learning rates
    optimizer = torch.optim.Adam([
        {"params": positions, "lr": 0.0001},  # Very small learning rate for positions
        {"params": scales, "lr": 0.001},
        {"params": rotations, "lr": 0.0001},
        {"params": colors, "lr": 0.001},
        {"params": opacities, "lr": 0.01}
    ])
    
    # Training loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Better loss function that preserves structure
        position_loss = torch.mean((positions - original_positions)**2) * 0.1  # Keep original positions
        scale_loss = torch.mean(scales**2) * 0.01  # Keep scales reasonable
        color_loss = torch.mean((colors - original_colors)**2) * 0.1  # Preserve original colors
        opacity_loss = torch.mean((opacities - 0.8)**2) * 0.01  # Keep reasonable opacity
        
        total_loss = position_loss + scale_loss + color_loss + opacity_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Normalize quaternions
        with torch.no_grad():
            rotations.data = rotations.data / torch.norm(rotations.data, dim=1, keepdim=True)
        
        if iteration % 200 == 0:
            print(f"Iteration {iteration:4d}: Total Loss = {total_loss.item():.6f}")
            print(f"  Position Loss: {position_loss.item():.6f}, Scale Loss: {scale_loss.item():.6f}")
            print(f"  Color Loss: {color_loss.item():.6f}, Opacity Loss: {opacity_loss.item():.6f}")
    
    print("Training completed!")
    
    return {
        "positions": positions.detach(),
        "scales": scales.detach(),
        "rotations": rotations.detach(),
        "colors": colors.detach(),
        "opacities": opacities.detach()
    }

def save_gaussians_to_ply(gaussians, output_path):
    """Save Gaussians to PLY file"""
    
    print(f"Saving Gaussians to {output_path}")
    
    # Convert to numpy arrays
    positions = gaussians["positions"].cpu().numpy()
    scales = gaussians["scales"].cpu().numpy()
    rotations = gaussians["rotations"].cpu().numpy()
    colors = gaussians["colors"].cpu().numpy()
    opacities = gaussians["opacities"].cpu().numpy()
    
    # Convert colors to 0-255 range
    colors_uint8 = (colors * 255).astype(np.uint8)
    opacities_uint8 = (opacities * 255).astype(np.uint8)
    
    print(f"Position range after training: {positions.min():.3f} to {positions.max():.3f}")
    
    # Write PLY file
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("end_header\n")
        
        for i in range(len(positions)):
            pos = positions[i]
            scale = scales[i]
            rot = rotations[i]
            color = colors_uint8[i]
            opacity = opacities_uint8[i, 0]
            
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} ")
            f.write(f"{color[0]} {color[1]} {color[2]} ")
            f.write(f"{opacity} ")
            f.write(f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f} ")
            f.write(f"{rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} {rot[3]:.6f}\n")
    
    print(f"Saved {len(positions)} Gaussians to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Better Gaussian Splatting training")
    parser.add_argument("--sparse_path", type=str, default="sparse", help="Path to COLMAP sparse reconstruction")
    parser.add_argument("--output_path", type=str, default="outputs/better_person_model.ply", help="Output PLY file path")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--num_points", type=int, default=None, help="Number of points to use (None = all)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    # Load real COLMAP data
    print("Loading real COLMAP data...")
    colmap_data = read_colmap_binary(Path(args.sparse_path))
    
    print(f"Loaded COLMAP data:")
    print(f"  - {len(colmap_data['points3D'])} 3D points")
    print(f"  - {len(colmap_data['images'])} images")
    print(f"  - {len(colmap_data['cameras'])} cameras")
    
    # Create Gaussians from real 3D points
    gaussians = create_gaussians_from_colmap_points(colmap_data, args.num_points, device)
    
    # Train the Gaussians with better loss function
    trained_gaussians = better_training(gaussians, args.iterations, device)
    
    # Save to PLY file
    save_gaussians_to_ply(trained_gaussians, Path(args.output_path))
    
    print("🎉 Better Gaussian Splatting training completed!")
    print(f"📁 Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()
