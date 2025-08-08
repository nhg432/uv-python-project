"""
EMBL Dataset Path Discovery
Systematically explore all possible paths in the OME-Zarr dataset
"""

import requests
import json
import zarr
import fsspec
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage
import re
from urllib.parse import urljoin

def discover_dataset_paths():
    """Systematically discover all accessible paths in the dataset"""
    
    base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
    
    print("EMBL Dataset Path Discovery")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    
    # Try to get a listing of the bucket contents (if possible)
    print("\n1. Attempting to discover dataset structure...")
    
    # Common OME-Zarr patterns to try
    patterns_to_try = []
    
    # Resolution levels (0-9)
    for i in range(10):
        patterns_to_try.append(str(i))
    
    # Common OME-Zarr subdirectories
    common_paths = [
        'data', 'image', 'pyramid', 'resolution_0', 'level_0',
        '0/0', '0/1', '1/0', 'pyramid/0', 'pyramid/1',
        'multiscales/0', 'multiscales/1'
    ]
    patterns_to_try.extend(common_paths)
    
    # Letter combinations (some datasets use letters)
    for letter in 'abcdefgh':
        patterns_to_try.append(letter)
    
    print(f"Testing {len(patterns_to_try)} potential paths...")
    
    found_paths = []
    
    for pattern in patterns_to_try:
        test_url = f"{base_url}/{pattern}"
        
        # First check if there's a .zarray file (indicates an array)
        zarray_url = f"{test_url}/.zarray"
        
        try:
            response = requests.head(zarray_url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Found array at: {pattern}")
                found_paths.append((pattern, 'array'))
                
                # Get array metadata
                try:
                    meta_response = requests.get(zarray_url, timeout=5)
                    if meta_response.status_code == 200:
                        array_meta = json.loads(meta_response.text)
                        print(f"  Shape: {array_meta.get('shape', 'unknown')}")
                        print(f"  Dtype: {array_meta.get('dtype', 'unknown')}")
                        print(f"  Chunks: {array_meta.get('chunks', 'unknown')}")
                except:
                    pass
                    
        except:
            pass
        
        # Also check for .zgroup (indicates a group)
        zgroup_url = f"{test_url}/.zgroup"
        try:
            response = requests.head(zgroup_url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Found group at: {pattern}")
                found_paths.append((pattern, 'group'))
        except:
            pass
    
    print(f"\nFound {len(found_paths)} accessible paths:")
    for path, path_type in found_paths:
        print(f"  {path} ({path_type})")
    
    # Try to load data from the first available array
    for path, path_type in found_paths:
        if path_type == 'array':
            print(f"\nAttempting to load data from: {path}")
            success = try_load_array_data(base_url, path)
            if success:
                return True
    
    return False

def try_load_array_data(base_url, path):
    """Try to load and visualize data from a specific path"""
    
    try:
        array_url = f"{base_url}/{path}"
        print(f"Loading array from: {array_url}")
        
        # Create HTTP mapper and open array
        http_mapper = fsspec.get_mapper(array_url)
        zarr_array = zarr.open(http_mapper, mode='r')
        
        print(f"✓ Successfully opened array!")
        print(f"Shape: {zarr_array.shape}")
        print(f"Dtype: {zarr_array.dtype}")
        print(f"Chunks: {zarr_array.chunks}")
        
        # Determine sample size based on array shape
        shape = zarr_array.shape
        
        # Calculate appropriate sample size
        max_dim_size = 64  # Maximum size per dimension for sampling
        
        if len(shape) == 5:  # TCZYX
            t, c, z, y, x = shape
            sample_slices = (
                slice(0, 1),  # First time point
                slice(0, 1),  # First channel
                slice(0, min(z, max_dim_size)),
                slice(0, min(y, max_dim_size)),
                slice(0, min(x, max_dim_size))
            )
        elif len(shape) == 4:  # CZYX
            c, z, y, x = shape
            sample_slices = (
                slice(0, 1),  # First channel
                slice(0, min(z, max_dim_size)),
                slice(0, min(y, max_dim_size)),
                slice(0, min(x, max_dim_size))
            )
        elif len(shape) == 3:  # ZYX
            z, y, x = shape
            sample_slices = (
                slice(0, min(z, max_dim_size)),
                slice(0, min(y, max_dim_size)),
                slice(0, min(x, max_dim_size))
            )
        else:
            print(f"Unsupported array shape: {shape}")
            return False
        
        print(f"Loading sample with slices: {sample_slices}")
        
        # Load the sample data
        sample_data = zarr_array[sample_slices]
        
        print(f"✓ Sample loaded! Shape: {sample_data.shape}")
        print(f"Data range: {sample_data.min()} to {sample_data.max()}")
        print(f"Data type: {sample_data.dtype}")
        
        # Convert to 3D for visualization
        if len(sample_data.shape) == 5:
            volume = sample_data[0, 0]  # Remove time and channel dimensions
        elif len(sample_data.shape) == 4:
            volume = sample_data[0]  # Remove channel dimension
        else:
            volume = sample_data
        
        if len(volume.shape) != 3:
            print(f"Cannot create 3D visualization from shape: {volume.shape}")
            return False
        
        # Normalize the volume
        volume = volume.astype(np.float32)
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Create all visualizations
        print("\nCreating visualizations...")
        create_all_visualizations(volume, path)
        
        return True
        
    except Exception as e:
        print(f"Error loading array from {path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_all_visualizations(volume, dataset_name):
    """Create comprehensive 3D visualizations"""
    
    print(f"Creating visualizations for volume shape: {volume.shape}")
    
    # 1. 2D Slice Views
    print("1. Creating 2D slice views...")
    create_slice_views(volume, dataset_name)
    
    # 2. 3D Scatter Plot
    print("2. Creating 3D scatter plot...")
    create_interactive_scatter(volume, dataset_name)
    
    # 3. Isosurface
    print("3. Creating isosurface...")
    create_mesh_surface(volume, dataset_name)

def create_slice_views(volume, name):
    """Create comprehensive 2D slice views"""
    try:
        z, y, x = volume.shape
        
        # Create a comprehensive slice figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Row 1: XY slices at different Z levels
        z_levels = [z//4, z//2, 3*z//4]
        for i, z_level in enumerate(z_levels):
            im = axes[0, i].imshow(volume[z_level], cmap='viridis', aspect='equal')
            axes[0, i].set_title(f'XY Slice (Z={z_level}/{z})')
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Y')
            plt.colorbar(im, ax=axes[0, i])
        
        # Row 2: XZ slices at different Y levels
        y_levels = [y//4, y//2, 3*y//4]
        for i, y_level in enumerate(y_levels):
            im = axes[1, i].imshow(volume[:, y_level, :], cmap='viridis', aspect='equal')
            axes[1, i].set_title(f'XZ Slice (Y={y_level}/{y})')
            axes[1, i].set_xlabel('X')
            axes[1, i].set_ylabel('Z')
            plt.colorbar(im, ax=axes[1, i])
        
        # Row 3: YZ slices at different X levels
        x_levels = [x//4, x//2, 3*x//4]
        for i, x_level in enumerate(x_levels):
            im = axes[2, i].imshow(volume[:, :, x_level], cmap='viridis', aspect='equal')
            axes[2, i].set_title(f'YZ Slice (X={x_level}/{x})')
            axes[2, i].set_xlabel('Y')
            axes[2, i].set_ylabel('Z')
            plt.colorbar(im, ax=axes[2, i])
        
        plt.suptitle(f'Comprehensive Slice Views: EMBL {name}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        filename = f'embl_{name}_comprehensive_slices.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Comprehensive slice views saved as '{filename}'")
        plt.show()
        
        # Also create projections
        create_projections(volume, name)
        
    except Exception as e:
        print(f"Error creating slice views: {e}")

def create_projections(volume, name):
    """Create maximum intensity projections"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Maximum projections along each axis
        xy_proj = np.max(volume, axis=0)
        xz_proj = np.max(volume, axis=1)
        yz_proj = np.max(volume, axis=2)
        
        im1 = axes[0].imshow(xy_proj, cmap='hot', aspect='equal')
        axes[0].set_title('Maximum Intensity Projection (XY)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(xz_proj, cmap='hot', aspect='equal')
        axes[1].set_title('Maximum Intensity Projection (XZ)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(yz_proj, cmap='hot', aspect='equal')
        axes[2].set_title('Maximum Intensity Projection (YZ)')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f'Maximum Intensity Projections: EMBL {name}', fontsize=14)
        plt.tight_layout()
        
        filename = f'embl_{name}_projections.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Projections saved as '{filename}'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating projections: {e}")

def create_interactive_scatter(volume, name, threshold=0.5, max_points=8000):
    """Create interactive 3D scatter plot"""
    try:
        # Find significant voxels
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
        
        print(f"Creating interactive 3D scatter with {len(z_coords)} points")
        
        # Create the scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=5,
                color=values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Intensity",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2
                ),
                line=dict(width=0)
            ),
            text=[f'Position: ({x}, {y}, {z})<br>Intensity: {v:.3f}' 
                  for x, y, z, v in zip(x_coords, y_coords, z_coords, values)],
            hovertemplate='%{text}<extra></extra>',
            name='Data Points'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Interactive 3D Visualization: EMBL {name}',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Z (slices)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.3, y=1.3, z=1.3),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgb(240, 240, 240)'
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        filename = f'embl_{name}_interactive_3d.html'
        fig.write_html(filename)
        print(f"✓ Interactive 3D plot saved as '{filename}'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating interactive scatter: {e}")

def create_mesh_surface(volume, name, threshold=0.4):
    """Create 3D mesh surface using marching cubes"""
    try:
        print(f"Generating 3D surface mesh (threshold={threshold})...")
        
        # Apply gaussian smoothing
        smoothed = ndimage.gaussian_filter(volume, sigma=1.5)
        
        # Generate mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, level=threshold, spacing=(1, 1, 1)
        )
        
        print(f"Generated mesh: {len(verts)} vertices, {len(faces)} faces")
        
        # Create the mesh plot
        fig = go.Figure()
        
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=values,
            colorscale='Plasma',
            opacity=0.9,
            name='3D Surface',
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                specular=0.6,
                roughness=0.1,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=100
            ),
            hovertemplate='Surface Point<br>' +
                         'X: %{x}<br>' +
                         'Y: %{y}<br>' +
                         'Z: %{z}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'3D Surface Mesh: EMBL {name} (Isosurface at {threshold})',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Z (slices)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgb(230, 230, 230)',
                xaxis=dict(showbackground=True, backgroundcolor='rgb(200, 200, 200)'),
                yaxis=dict(showbackground=True, backgroundcolor='rgb(200, 200, 200)'),
                zaxis=dict(showbackground=True, backgroundcolor='rgb(200, 200, 200)')
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        filename = f'embl_{name}_surface_mesh.html'
        fig.write_html(filename)
        print(f"✓ 3D surface mesh saved as '{filename}'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating mesh surface: {e}")

def main():
    """Main execution"""
    success = discover_dataset_paths()
    
    if success:
        print("\n" + "="*60)
        print("🎉 3D VISUALIZATION SUCCESSFUL! 🎉")
        print("\nGenerated files:")
        print("- *_comprehensive_slices.png - Detailed 2D slice views")
        print("- *_projections.png - Maximum intensity projections")
        print("- *_interactive_3d.html - Interactive 3D scatter plot")
        print("- *_surface_mesh.html - 3D surface mesh rendering")
        print("\nOpen the HTML files in a web browser for interactive exploration!")
        print("The 3D renderings show the cellular structures from the EMBL dataset.")
    else:
        print("\nCould not successfully load and visualize the dataset")
        print("The EMBL OME-Zarr structure may be different than expected")

if __name__ == "__main__":
    main()
