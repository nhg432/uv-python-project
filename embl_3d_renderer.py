"""
EMBL OME-Zarr Explorer and 3D Renderer
Direct HTTP access to EMBL culture collections dataset
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from pathlib import Path
import requests
import json
import fsspec
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

def explore_embl_dataset():
    """Explore the EMBL dataset structure"""
    
    base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
    
    print("EMBL OME-Zarr Dataset Explorer")
    print("=" * 50)
    print(f"Dataset: {base_url}")
    
    # Check available scales
    print("\nChecking available resolution scales...")
    available_scales = []
    
    for scale in range(10):  # Check scales 0-9
        scale_url = f"{base_url}/{scale}"
        try:
            response = requests.head(scale_url + "/.zarray", timeout=10)
            if response.status_code == 200:
                available_scales.append(scale)
                print(f"✓ Scale {scale} available")
            else:
                break
        except:
            break
    
    print(f"Available scales: {available_scales}")
    
    # Try to load data from the highest available scale (lowest resolution)
    if available_scales:
        target_scale = max(available_scales)  # Start with lowest resolution
        print(f"\nAttempting to load scale {target_scale} data...")
        
        try:
            # Create HTTP mapper
            scale_url = f"{base_url}/{target_scale}"
            http_mapper = fsspec.get_mapper(scale_url)
            
            # Open zarr array
            zarr_array = zarr.open(http_mapper, mode='r')
            
            print(f"✓ Successfully opened zarr array!")
            print(f"Shape: {zarr_array.shape}")
            print(f"Dtype: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            
            # Load array metadata
            try:
                array_info_url = f"{scale_url}/.zarray"
                response = requests.get(array_info_url)
                if response.status_code == 200:
                    array_metadata = json.loads(response.text)
                    print(f"Array metadata: {json.dumps(array_metadata, indent=2)}")
            except Exception as e:
                print(f"Could not load array metadata: {e}")
            
            # Determine data dimensions
            shape = zarr_array.shape
            print(f"\nAnalyzing data dimensions: {shape}")
            
            if len(shape) == 4:
                print("4D data detected: likely CZYX (Channel, Z, Y, X)")
                c, z, y, x = shape
                print(f"Channels: {c}, Z-slices: {z}, Height: {y}, Width: {x}")
            elif len(shape) == 3:
                print("3D data detected: likely ZYX (Z, Y, X)")
                z, y, x = shape
                print(f"Z-slices: {z}, Height: {y}, Width: {x}")
            elif len(shape) == 5:
                print("5D data detected: likely TCZYX (Time, Channel, Z, Y, X)")
                t, c, z, y, x = shape
                print(f"Time points: {t}, Channels: {c}, Z-slices: {z}, Height: {y}, Width: {x}")
            
            # Load a small sample for visualization
            print(f"\nLoading sample data...")
            
            try:
                if len(shape) == 4:  # CZYX
                    # Take first channel, central z-slices, and subsample XY
                    z_center = shape[1] // 2
                    z_range = slice(max(0, z_center-5), min(shape[1], z_center+5))
                    y_range = slice(0, min(shape[2], 100))
                    x_range = slice(0, min(shape[3], 100))
                    
                    sample_data = zarr_array[0, z_range, y_range, x_range]
                    
                elif len(shape) == 3:  # ZYX
                    # Take central z-slices and subsample XY
                    z_center = shape[0] // 2
                    z_range = slice(max(0, z_center-5), min(shape[0], z_center+5))
                    y_range = slice(0, min(shape[1], 100))
                    x_range = slice(0, min(shape[2], 100))
                    
                    sample_data = zarr_array[z_range, y_range, x_range]
                    
                elif len(shape) == 5:  # TCZYX
                    # Take first time point, first channel, central z-slices
                    z_center = shape[2] // 2
                    z_range = slice(max(0, z_center-5), min(shape[2], z_center+5))
                    y_range = slice(0, min(shape[3], 100))
                    x_range = slice(0, min(shape[4], 100))
                    
                    sample_data = zarr_array[0, 0, z_range, y_range, x_range]
                else:
                    print(f"Unsupported shape: {shape}")
                    return None
                
                print(f"✓ Loaded sample data with shape: {sample_data.shape}")
                print(f"Data range: {sample_data.min()} to {sample_data.max()}")
                print(f"Data type: {sample_data.dtype}")
                
                return sample_data, zarr_array
                
            except Exception as e:
                print(f"Error loading sample data: {e}")
                return None, zarr_array
                
        except Exception as e:
            print(f"Error opening zarr array: {e}")
            return None, None
    
    return None, None

def create_3d_visualizations(data_array):
    """Create comprehensive 3D visualizations"""
    
    if data_array is None:
        print("No data available for visualization")
        return
    
    print(f"\nCreating 3D visualizations from data shape: {data_array.shape}")
    
    # Ensure we have 3D data for visualization
    if len(data_array.shape) == 4:
        # Take first channel if CZYX
        volume = data_array[0]
    elif len(data_array.shape) == 3:
        # Already ZYX
        volume = data_array
    else:
        print(f"Cannot visualize data with shape: {data_array.shape}")
        return
    
    print(f"Processing 3D volume with shape: {volume.shape}")
    
    # Normalize data
    volume = volume.astype(np.float32)
    if volume.max() > volume.min():
        volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # 1. Create slice visualization
    print("1. Creating slice visualization...")
    create_slice_plots(volume)
    
    # 2. Create 3D scatter plot
    print("2. Creating 3D scatter plot...")
    create_3d_scatter(volume)
    
    # 3. Create isosurface if data is suitable
    print("3. Creating isosurface...")
    create_isosurface(volume)

def create_slice_plots(volume):
    """Create 2D slice plots"""
    try:
        z, y, x = volume.shape
        
        # Get central slices
        z_center = z // 2
        y_center = y // 2
        x_center = x // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # XY slice (top view)
        xy_slice = volume[z_center]
        im1 = axes[0, 0].imshow(xy_slice, cmap='viridis', aspect='equal')
        axes[0, 0].set_title(f'XY Slice (Z={z_center})')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # XZ slice (side view)
        xz_slice = volume[:, y_center, :]
        im2 = axes[0, 1].imshow(xz_slice, cmap='viridis', aspect='equal')
        axes[0, 1].set_title(f'XZ Slice (Y={y_center})')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # YZ slice (front view)
        yz_slice = volume[:, :, x_center]
        im3 = axes[1, 0].imshow(yz_slice, cmap='viridis', aspect='equal')
        axes[1, 0].set_title(f'YZ Slice (X={x_center})')
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Maximum intensity projection
        max_proj = np.max(volume, axis=0)
        im4 = axes[1, 1].imshow(max_proj, cmap='viridis', aspect='equal')
        axes[1, 1].set_title('Maximum Intensity Projection')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('embl_bmcc122_slices.png', dpi=150, bbox_inches='tight')
        print("✓ Slice plots saved as 'embl_bmcc122_slices.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating slice plots: {e}")

def create_3d_scatter(volume, threshold=0.5, max_points=10000):
    """Create 3D scatter plot of significant voxels"""
    try:
        # Find voxels above threshold
        mask = volume > threshold
        z_coords, y_coords, x_coords = np.where(mask)
        values = volume[mask]
        
        # Subsample if too many points
        if len(z_coords) > max_points:
            indices = np.random.choice(len(z_coords), max_points, replace=False)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
            values = values[indices]
        
        print(f"Creating 3D scatter with {len(z_coords)} points")
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=values,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Intensity")
            ),
            text=[f'Intensity: {v:.3f}' for v in values],
            name='Data Points'
        ))
        
        fig.update_layout(
            title='3D Visualization: EMBL bmcc122_pfa_cetn-tub-dna',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        # Save and show
        fig.write_html('embl_bmcc122_3d_scatter.html')
        print("✓ 3D scatter plot saved as 'embl_bmcc122_3d_scatter.html'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating 3D scatter plot: {e}")

def create_isosurface(volume, threshold=0.4):
    """Create isosurface using marching cubes"""
    try:
        # Apply smoothing
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(volume, sigma=1)
        
        # Generate isosurface
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, level=threshold, spacing=(1, 1, 1)
        )
        
        print(f"Generated isosurface with {len(verts)} vertices and {len(faces)} faces")
        
        # Create mesh plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=values,
                colorscale='Plasma',
                opacity=0.8,
                name='Isosurface',
                lighting=dict(ambient=0.3, diffuse=0.8, specular=0.5),
                lightposition=dict(x=100, y=100, z=100)
            )
        ])
        
        fig.update_layout(
            title='3D Isosurface: EMBL bmcc122_pfa_cetn-tub-dna',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700
        )
        
        # Save and show
        fig.write_html('embl_bmcc122_isosurface.html')
        print("✓ Isosurface plot saved as 'embl_bmcc122_isosurface.html'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating isosurface: {e}")

def main():
    """Main execution"""
    # Explore dataset
    sample_data, zarr_array = explore_embl_dataset()
    
    if sample_data is not None:
        # Create visualizations
        create_3d_visualizations(sample_data)
        
        print("\n" + "="*50)
        print("3D Visualization Complete!")
        print("Generated files:")
        print("- embl_bmcc122_slices.png")
        print("- embl_bmcc122_3d_scatter.html")
        print("- embl_bmcc122_isosurface.html")
        print("\nOpen the HTML files in a web browser for interactive 3D exploration!")
    else:
        print("Could not load data for visualization")

if __name__ == "__main__":
    main()
