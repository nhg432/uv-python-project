"""
EMBL Dataset Structure Explorer
Investigate the exact structure of the OME-Zarr dataset
"""

import requests
import json
import zarr
import fsspec
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage

def explore_dataset_structure():
    """Thoroughly explore the dataset structure"""
    
    base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
    
    print("EMBL Dataset Structure Explorer")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    
    # First, check what metadata is available
    print("\n1. Checking metadata files...")
    metadata_files = ['.zattrs', '.zgroup', '.zmetadata']
    
    for meta_file in metadata_files:
        url = f"{base_url}/{meta_file}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✓ {meta_file}: {response.text[:200]}...")
                if meta_file == '.zattrs':
                    try:
                        attrs = json.loads(response.text)
                        print(f"   Attributes: {json.dumps(attrs, indent=2)}")
                    except:
                        pass
            else:
                print(f"✗ {meta_file}: Status {response.status_code}")
        except Exception as e:
            print(f"✗ {meta_file}: {e}")
    
    # Check for common OME-Zarr structures
    print("\n2. Exploring directory structure...")
    
    # Try to access the zarr group directly
    try:
        print("Attempting direct zarr access...")
        http_mapper = fsspec.get_mapper(base_url)
        zarr_group = zarr.open(http_mapper, mode='r')
        
        print(f"✓ Zarr group opened successfully!")
        print(f"Group info: {zarr_group.info}")
        
        if hasattr(zarr_group, 'attrs'):
            print(f"Group attributes: {dict(zarr_group.attrs)}")
        
        # List all keys in the group
        all_keys = list(zarr_group.keys())
        array_keys = list(zarr_group.array_keys()) if hasattr(zarr_group, 'array_keys') else []
        group_keys = list(zarr_group.group_keys()) if hasattr(zarr_group, 'group_keys') else []
        
        print(f"All keys: {all_keys}")
        print(f"Array keys: {array_keys}")
        print(f"Group keys: {group_keys}")
        
        # Investigate each key
        for key in all_keys:
            print(f"\nInvestigating key: '{key}'")
            try:
                item = zarr_group[key]
                if hasattr(item, 'shape'):  # It's an array
                    print(f"  Array - Shape: {item.shape}, Dtype: {item.dtype}")
                    if hasattr(item, 'chunks'):
                        print(f"  Chunks: {item.chunks}")
                    
                    # Try to load a small sample
                    try:
                        if len(item.shape) >= 3:
                            sample_size = min(10, min(item.shape))
                            if len(item.shape) == 3:
                                sample = item[:sample_size, :sample_size, :sample_size]
                            elif len(item.shape) == 4:
                                sample = item[0, :sample_size, :sample_size, :sample_size]
                            elif len(item.shape) == 5:
                                sample = item[0, 0, :sample_size, :sample_size, :sample_size]
                            else:
                                sample = item[(slice(0, sample_size),) * len(item.shape)]
                            
                            print(f"  Sample loaded: {sample.shape}, range: {sample.min():.3f} to {sample.max():.3f}")
                            return item, key  # Return the first suitable array
                            
                    except Exception as e:
                        print(f"  Could not load sample: {e}")
                        
                elif hasattr(item, 'keys'):  # It's a group
                    subkeys = list(item.keys())
                    print(f"  Group with {len(subkeys)} items: {subkeys[:5]}{'...' if len(subkeys) > 5 else ''}")
                    
                    # Check if this group contains arrays
                    for subkey in subkeys[:3]:  # Check first 3 subkeys
                        try:
                            subitem = item[subkey]
                            if hasattr(subitem, 'shape'):
                                print(f"    {subkey}: Array {subitem.shape}")
                                return subitem, f"{key}/{subkey}"
                        except:
                            pass
                            
            except Exception as e:
                print(f"  Error accessing '{key}': {e}")
        
        return zarr_group, 'root'
        
    except Exception as e:
        print(f"Direct zarr access failed: {e}")
    
    # Try numeric keys (common OME-Zarr pattern)
    print("\n3. Trying numeric scale keys...")
    for scale in range(6):
        url = f"{base_url}/{scale}"
        try:
            print(f"Checking scale {scale}...")
            scale_mapper = fsspec.get_mapper(url)
            scale_array = zarr.open(scale_mapper, mode='r')
            
            print(f"✓ Scale {scale} found!")
            print(f"  Shape: {scale_array.shape}")
            print(f"  Dtype: {scale_array.dtype}")
            print(f"  Chunks: {scale_array.chunks}")
            
            return scale_array, f"scale_{scale}"
            
        except Exception as e:
            print(f"Scale {scale} not accessible: {e}")
    
    # Try common OME-Zarr paths
    print("\n4. Trying common OME-Zarr paths...")
    common_paths = ['data', 'image', '0/0', 'pyramid/0', 'resolution_0']
    
    for path in common_paths:
        url = f"{base_url}/{path}"
        try:
            print(f"Checking path: {path}")
            path_mapper = fsspec.get_mapper(url)
            path_array = zarr.open(path_mapper, mode='r')
            
            print(f"✓ Path '{path}' found!")
            print(f"  Shape: {path_array.shape}")
            print(f"  Dtype: {path_array.dtype}")
            
            return path_array, path
            
        except Exception as e:
            print(f"Path '{path}' not accessible: {e}")
    
    print("\nCould not find accessible data arrays")
    return None, None

def load_and_visualize_data(zarr_array, array_name):
    """Load data and create comprehensive visualizations"""
    
    if zarr_array is None:
        print("No data available for visualization")
        return
    
    print(f"\n" + "="*60)
    print(f"VISUALIZING DATA FROM: {array_name}")
    print(f"Array shape: {zarr_array.shape}")
    print(f"Array dtype: {zarr_array.dtype}")
    
    try:
        # Determine appropriate sample size based on array dimensions
        shape = zarr_array.shape
        
        if len(shape) == 5:  # TCZYX
            print("5D data detected: TCZYX format")
            t, c, z, y, x = shape
            print(f"Time points: {t}, Channels: {c}, Z-slices: {z}, Height: {y}, Width: {x}")
            
            # Take first time point, first channel, central z-slices
            z_sample = min(20, z)
            z_start = max(0, (z - z_sample) // 2)
            y_sample = min(100, y)
            x_sample = min(100, x)
            
            data = zarr_array[0, 0, z_start:z_start+z_sample, :y_sample, :x_sample]
            
        elif len(shape) == 4:  # CZYX
            print("4D data detected: CZYX format")
            c, z, y, x = shape
            print(f"Channels: {c}, Z-slices: {z}, Height: {y}, Width: {x}")
            
            # Take first channel, central z-slices
            z_sample = min(20, z)
            z_start = max(0, (z - z_sample) // 2)
            y_sample = min(100, y)
            x_sample = min(100, x)
            
            data = zarr_array[0, z_start:z_start+z_sample, :y_sample, :x_sample]
            
        elif len(shape) == 3:  # ZYX
            print("3D data detected: ZYX format")
            z, y, x = shape
            print(f"Z-slices: {z}, Height: {y}, Width: {x}")
            
            # Take central z-slices
            z_sample = min(20, z)
            z_start = max(0, (z - z_sample) // 2)
            y_sample = min(100, y)
            x_sample = min(100, x)
            
            data = zarr_array[z_start:z_start+z_sample, :y_sample, :x_sample]
            
        else:
            print(f"Unsupported data shape: {shape}")
            return
        
        print(f"Loading sample data with shape: {data.shape}")
        
        # Convert to numpy array
        data_array = np.array(data)
        print(f"Loaded data shape: {data_array.shape}")
        print(f"Data range: {data_array.min()} to {data_array.max()}")
        print(f"Data type: {data_array.dtype}")
        
        # Normalize data
        if data_array.max() > data_array.min():
            data_normalized = (data_array - data_array.min()) / (data_array.max() - data_array.min())
        else:
            data_normalized = data_array
        
        # Create visualizations
        create_comprehensive_visualizations(data_normalized, array_name)
        
    except Exception as e:
        print(f"Error loading and visualizing data: {e}")
        import traceback
        traceback.print_exc()

def create_comprehensive_visualizations(volume, name):
    """Create all visualization types"""
    
    print(f"\nCreating comprehensive visualizations for: {name}")
    print(f"Volume shape: {volume.shape}")
    
    # 1. 2D Slice Plots
    print("1. Creating 2D slice plots...")
    create_slice_plots(volume, name)
    
    # 2. 3D Scatter Plot
    print("2. Creating 3D scatter plot...")
    create_3d_scatter_plot(volume, name)
    
    # 3. Isosurface
    print("3. Creating isosurface...")
    create_isosurface_plot(volume, name)
    
    # 4. Volume Rendering (if plotly supports it)
    print("4. Creating volume rendering...")
    create_volume_rendering(volume, name)

def create_slice_plots(volume, name):
    """Create 2D slice visualization"""
    try:
        if len(volume.shape) != 3:
            print(f"Slice plots require 3D data, got {len(volume.shape)}D")
            return
        
        z, y, x = volume.shape
        z_center = z // 2
        y_center = y // 2
        x_center = x // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # XY slice (top view)
        xy_slice = volume[z_center, :, :]
        im1 = axes[0, 0].imshow(xy_slice, cmap='viridis', aspect='equal')
        axes[0, 0].set_title(f'XY Slice (Z={z_center}/{z})')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # XZ slice (side view)
        xz_slice = volume[:, y_center, :]
        im2 = axes[0, 1].imshow(xz_slice, cmap='viridis', aspect='equal')
        axes[0, 1].set_title(f'XZ Slice (Y={y_center}/{y})')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # YZ slice (front view)
        yz_slice = volume[:, :, x_center]
        im3 = axes[1, 0].imshow(yz_slice, cmap='viridis', aspect='equal')
        axes[1, 0].set_title(f'YZ Slice (X={x_center}/{x})')
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
        
        plt.suptitle(f'2D Slices: EMBL {name}', fontsize=16)
        plt.tight_layout()
        
        filename = f'embl_{name.replace("/", "_")}_slices.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Slice plots saved as '{filename}'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating slice plots: {e}")

def create_3d_scatter_plot(volume, name, threshold=0.4, max_points=5000):
    """Create 3D scatter plot"""
    try:
        if len(volume.shape) != 3:
            print(f"3D scatter requires 3D data, got {len(volume.shape)}D")
            return
        
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
        
        print(f"Creating 3D scatter with {len(z_coords)} points (threshold={threshold})")
        
        # Create interactive 3D plot
        fig = go.Figure(data=go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Intensity"),
                line=dict(width=0)
            ),
            text=[f'({x}, {y}, {z})<br>Intensity: {v:.3f}' 
                  for x, y, z, v in zip(x_coords, y_coords, z_coords, values)],
            hovertemplate='%{text}<extra></extra>',
            name='Data Points'
        ))
        
        fig.update_layout(
            title=f'3D Scatter Plot: EMBL {name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            width=1000,
            height=800
        )
        
        filename = f'embl_{name.replace("/", "_")}_3d_scatter.html'
        fig.write_html(filename)
        print(f"✓ 3D scatter plot saved as '{filename}'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating 3D scatter plot: {e}")

def create_isosurface_plot(volume, name, threshold=0.3):
    """Create isosurface using marching cubes"""
    try:
        if len(volume.shape) != 3:
            print(f"Isosurface requires 3D data, got {len(volume.shape)}D")
            return
        
        # Smooth the data
        smoothed = ndimage.gaussian_filter(volume, sigma=1.0)
        
        # Generate isosurface
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, level=threshold, spacing=(1, 1, 1)
        )
        
        print(f"Generated isosurface: {len(verts)} vertices, {len(faces)} faces")
        
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
                opacity=0.9,
                name='Isosurface',
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    specular=0.6,
                    roughness=0.2
                ),
                lightposition=dict(x=100, y=100, z=100)
            )
        ])
        
        fig.update_layout(
            title=f'3D Isosurface: EMBL {name} (threshold={threshold})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        filename = f'embl_{name.replace("/", "_")}_isosurface.html'
        fig.write_html(filename)
        print(f"✓ Isosurface plot saved as '{filename}'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating isosurface: {e}")

def create_volume_rendering(volume, name):
    """Create volume rendering"""
    try:
        if len(volume.shape) != 3:
            print(f"Volume rendering requires 3D data, got {len(volume.shape)}D")
            return
        
        # Downsample for performance
        step = max(1, min(volume.shape) // 64)
        volume_ds = volume[::step, ::step, ::step]
        
        print(f"Creating volume rendering (downsampled to {volume_ds.shape})")
        
        # Create coordinate grids
        z, y, x = np.mgrid[0:volume_ds.shape[0], 0:volume_ds.shape[1], 0:volume_ds.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume_ds.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=15,
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        fig.update_layout(
            title=f'Volume Rendering: EMBL {name}',
            scene=dict(
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            width=1000,
            height=800
        )
        
        filename = f'embl_{name.replace("/", "_")}_volume.html'
        fig.write_html(filename)
        print(f"✓ Volume rendering saved as '{filename}'")
        fig.show()
        
    except Exception as e:
        print(f"Error creating volume rendering: {e}")

def main():
    """Main execution"""
    # Explore the dataset structure
    zarr_array, array_name = explore_dataset_structure()
    
    if zarr_array is not None:
        # Load and visualize the data
        load_and_visualize_data(zarr_array, array_name)
        
        print("\n" + "="*60)
        print("🎉 3D VISUALIZATION COMPLETE! 🎉")
        print("Generated files:")
        print(f"- embl_{array_name.replace('/', '_')}_slices.png")
        print(f"- embl_{array_name.replace('/', '_')}_3d_scatter.html")
        print(f"- embl_{array_name.replace('/', '_')}_isosurface.html")
        print(f"- embl_{array_name.replace('/', '_')}_volume.html")
        print("\nOpen the HTML files in a web browser for interactive 3D exploration!")
        
    else:
        print("Could not access any data from the EMBL dataset")

if __name__ == "__main__":
    main()
