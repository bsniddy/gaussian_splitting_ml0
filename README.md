# Gaussian Splatting 3D Reconstruction

A Python implementation for creating 3D Gaussian Splats from multi-view images using COLMAP and gsplat.

## Dependencies

- Python 3.13.5
- PyTorch 2.8.0
- gsplat 1.5.3
- OpenCV 4.12.0
- COLMAP 3.12.5
- matplotlib 3.10.6

## Installation

```bash
# Clone and navigate to project
cd "Gaussian Splatting"

# Activate virtual environment
source gsplat_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install COLMAP (macOS)
brew install colmap
```

## Usage

### 1. Data Preparation
Create your own image dataset and place input images in `project/images/` directory. Minimum 50 images recommended for quality reconstruction.

### 2. COLMAP Processing
```bash
cd project
colmap feature_extractor --database_path sparse/database.db --image_path images/
colmap exhaustive_matcher --database_path sparse/database.db
colmap mapper --database_path sparse/database.db --image_path images --output_path sparse/
```

### 3. Training
```bash
python splatt_training.py --iterations 2000 --num_points 20000
```

### 4. Visualization
```bash
python view_3d.py --ply_file outputs/model.ply --show_interactive
```

## Project Structure

```
project/
├── images/              # Input images (JPG/PNG). Make your own image dataset.
├── sparse/              # COLMAP reconstruction data
│   ├── cameras.bin      # Camera parameters
│   ├── images.bin       # Image poses
│   └── points3D.bin     # 3D point cloud
├── outputs/             # Generated models (.ply)
|   |__ screenshots      # Screenshots of dif. angle outputs
├── splatt_training.py   # Training script
└── view_3d.py          # Visualization script
```

## Parameters

### Training Options
- `--iterations`: Number of training iterations (default: 1000)
- `--num_points`: Number of Gaussian primitives (default: all available)
- `--output_path`: Output PLY file path
- `--device`: Compute device (cpu/cuda)

### Visualization Options
- `--ply_file`: Path to PLY file
- `--output_image`: Save visualization as PNG
- `--show_interactive`: Display interactive 3D viewer

## Output Format

Models are saved as PLY files containing:
- 3D positions (x, y, z)
- RGB colors (0-255)
- Alpha values (0-255)
- Scale parameters (3D)
- Rotation quaternions (4D)

## Technical Details

The pipeline implements:
1. SIFT feature extraction and matching via COLMAP
2. Bundle adjustment for camera pose estimation
3. Sparse 3D point cloud reconstruction
4. Gaussian primitive initialization from 3D points
5. Optimization using Adam with structure-preserving loss
6. PLY export for visualization and further processing

## Performance Notes

- Training time scales linearly with iterations and point count
- Memory usage depends on number of Gaussian primitives
- Interactive visualization requires matplotlib backend
- COLMAP processing time increases quadratically with image count
