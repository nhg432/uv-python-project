"""
Enhanced EMBL OME-Zarr Viewer with HTTP Access
Handles EMBL culture collections OME-Zarr data via HTTP
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from skimage import measure, filters
from pathlib import Path
import requests
import json
from urllib.parse import urljoin
import fsspec
import warnings
warnings.filterwarnings('ignore')

class EMBLHTTPZarrViewer:
    def __init__(self, dataset_name):
        """Initialize with EMBL dataset name"""
        self.dataset_name = dataset_name
        self.base_url = f"https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/{dataset_name}"
        self.data = None
        
    def list_zarr_structure(self):
        """List the structure of the zarr dataset"""
        try:
            # Try to access via HTTP
            http_mapper = fsspec.get_mapper(self.base_url, trust_env=True)
            
            # Open zarr store
            zarr_group = zarr.open(http_mapper, mode='r')
            
            print(f"Zarr group structure:")
            print(f"Group info: {zarr_group.info}")
            print(f"Available arrays: {list(zarr_group.array_keys())}")
            print(f"Attributes: {dict(zarr_group.attrs)}")
            
            # Check for multiscale structure
            for key in zarr_group.array_keys():
                array = zarr_group[key]
                print(f"Array '{key}': shape={array.shape}, dtype={array.dtype}")
                
            return zarr_group
            
        except Exception as e:
            print(f"HTTP Zarr access error: {e}")
            return None
    
    def try_direct_http_access(self):
        """Try accessing zarr metadata via direct HTTP"""
        try:
            # Common OME-Zarr metadata files
            metadata_urls = [
                f"{self.base_url}/.zattrs",
                f"{self.base_url}/.zgroup", 
                f"{self.base_url}/.zmetadata",
                f"{self.base_url}/0/.zarray",  # Scale 0 metadata
                f"{self.base_url}/1/.zarray",  # Scale 1 metadata
            ]
            
            for url in metadata_urls:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        print(f"✓ Found: {url}")
                        if url.endswith('.zattrs'):
                            metadata = json.loads(response.text)
                            print(f"Metadata: {json.dumps(metadata, indent=2)}")
                        elif url.endswith('.zarray'):
                            array_info = json.loads(response.text)
                            print(f"Array info: {json.dumps(array_info, indent=2)}")
                    else:
                        print(f"✗ Not found: {url} (status: {response.status_code})")
                except Exception as e:
                    print(f"✗ Error accessing {url}: {e}")
                    
        except Exception as e:
            print(f"Direct HTTP access error: {e}")
    
    def load_sample_data(self, scale_level=2, max_size=256):
        """Load a sample of the data at a specific scale level"""
        try:
            # Try accessing at different scale levels (lower resolution)
            scale_url = f"{self.base_url}/{scale_level}"
            
            print(f"Attempting to load scale {scale_level} from: {scale_url}")
            
            # Try HTTP mapper
            http_mapper = fsspec.get_mapper(scale_url)
            
            # Open as zarr array
            zarr_array = zarr.open(http_mapper, mode='r')
            
            print(f"Successfully opened zarr array:")
            print(f"Shape: {zarr_array.shape}")
            print(f"Dtype: {zarr_array.dtype}")
            print(f"Chunks: {zarr_array.chunks}")
            
            # Load a sample of the data
            if len(zarr_array.shape) == 4:  # CZYX
                c, z, y, x = zarr_array.shape
                # Take central slice and subsample
                z_center = z // 2
                y_sample = slice(0, min(y, max_size))
                x_sample = slice(0, min(x, max_size))
                
                sample_data = zarr_array[:, z_center-10:z_center+10, y_sample, x_sample]
                
            elif len(zarr_array.shape) == 3:  # ZYX
                z, y, x = zarr_array.shape
                y_sample = slice(0, min(y, max_size))
                x_sample = slice(0, min(x, max_size))
                
                sample_data = zarr_array[:, y_sample, x_sample]
                
            else:
                # Take first max_size elements along each dimension
                slices = tuple(slice(0, min(s, max_size)) for s in zarr_array.shape)
                sample_data = zarr_array[slices]
            
            print(f"Loaded sample data shape: {sample_data.shape}")
            return sample_data
            
        except Exception as e:
            print(f"Sample data loading error: {e}")
            return None
    
    def create_3d_volume_plot(self, data_array, channel=0):
        """Create 3D volume visualization"""
        try:
            # Handle different data shapes
            if len(data_array.shape) == 4:  # CZYX
                volume = data_array[channel]
            elif len(data_array.shape) == 3:  # ZYX
                volume = data_array
            else:
                print(f"Unsupported data shape: {data_array.shape}")
                return None
            
            print(f"Creating 3D plot from volume shape: {volume.shape}")
            
            # Normalize data
            volume = volume.astype(np.float32)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            
            # Create coordinate arrays
            z, y, x = np.meshgrid(
                np.arange(volume.shape[0]),
                np.arange(volume.shape[1]),
                np.arange(volume.shape[2]),
                indexing='ij'
            )
            
            # Flatten arrays
            z_flat = z.flatten()
            y_flat = y.flatten()
            x_flat = x.flatten()
            values_flat = volume.flatten()
            
            # Filter to show only significant values
            threshold = 0.2
            mask = values_flat > threshold
            
            # Create 3D scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=x_flat[mask],
                y=y_flat[mask], 
                z=z_flat[mask],
                mode='markers',
                marker=dict(
                    size=2,
                    color=values_flat[mask],
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="Intensity")
                ),
                name='3D Data Points'
            ))
            
            fig.update_layout(
                title=f'3D Volume: EMBL {self.dataset_name}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y', 
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                width=900,
                height=700
            )
            
            return fig
            
        except Exception as e:
            print(f"3D plot creation error: {e}")
            return None
    
    def create_isosurface_plot(self, data_array, channel=0, threshold=0.3):
        """Create isosurface visualization using marching cubes"""
        try:
            # Handle different data shapes
            if len(data_array.shape) == 4:  # CZYX
                volume = data_array[channel]
            elif len(data_array.shape) == 3:  # ZYX  
                volume = data_array
            else:
                print(f"Unsupported data shape: {data_array.shape}")
                return None
            
            print(f"Creating isosurface from volume shape: {volume.shape}")
            
            # Normalize volume
            volume = volume.astype(np.float32)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            
            # Apply some smoothing
            from scipy import ndimage
            volume = ndimage.gaussian_filter(volume, sigma=1)
            
            # Generate isosurface using marching cubes
            verts, faces, normals, values = measure.marching_cubes(
                volume, level=threshold, spacing=(1, 1, 1)
            )
            
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
                    name='Isosurface'
                )
            ])
            
            fig.update_layout(
                title=f'3D Isosurface: EMBL {self.dataset_name}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z', 
                    aspectmode='cube',
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2)
                    )
                ),
                width=900,
                height=700
            )
            
            return fig
            
        except Exception as e:
            print(f"Isosurface creation error: {e}")
            return None
    
    def create_slice_visualization(self, data_array, channel=0):
        """Create multi-slice visualization"""
        try:
            # Handle different data shapes
            if len(data_array.shape) == 4:  # CZYX
                volume = data_array[channel]
            elif len(data_array.shape) == 3:  # ZYX
                volume = data_array
            else:
                print(f"Unsupported data shape: {data_array.shape}")
                return None
            
            # Get central slices
            z_center = volume.shape[0] // 2
            y_center = volume.shape[1] // 2
            x_center = volume.shape[2] // 2
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # XY slice (Z-projection)
            xy_slice = volume[z_center]
            axes[0, 0].imshow(xy_slice, cmap='viridis')
            axes[0, 0].set_title(f'XY Slice (Z={z_center})')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            
            # XZ slice (Y-projection)
            xz_slice = volume[:, y_center, :]
            axes[0, 1].imshow(xz_slice, cmap='viridis')
            axes[0, 1].set_title(f'XZ Slice (Y={y_center})')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Z')
            
            # YZ slice (X-projection)
            yz_slice = volume[:, :, x_center]
            axes[1, 0].imshow(yz_slice, cmap='viridis')
            axes[1, 0].set_title(f'YZ Slice (X={x_center})')
            axes[1, 0].set_xlabel('Y')
            axes[1, 0].set_ylabel('Z')
            
            # Max projection
            max_proj = np.max(volume, axis=0)
            im = axes[1, 1].imshow(max_proj, cmap='viridis')
            axes[1, 1].set_title('Maximum Intensity Projection')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(f'embl_{self.dataset_name}_slices.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Slice visualization error: {e}")

def main():
    """Main execution"""
    print("Enhanced EMBL OME-Zarr Viewer")
    print("=" * 50)
    
    # EMBL dataset
    dataset_name = "bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
    
    viewer = EMBLHTTPZarrViewer(dataset_name)
    
    print(f"Dataset: {dataset_name}")
    print(f"Base URL: {viewer.base_url}")
    
    # Try direct HTTP access first
    print("\n1. Checking direct HTTP access...")
    viewer.try_direct_http_access()
    
    # Try to list zarr structure
    print("\n2. Attempting to access zarr structure...")
    zarr_group = viewer.list_zarr_structure()
    
    if zarr_group is None:
        print("\n3. Trying to load sample data...")
        # Try different scale levels
        for scale in [3, 2, 1, 0]:
            print(f"\nTrying scale level {scale}...")
            data = viewer.load_sample_data(scale_level=scale, max_size=128)
            
            if data is not None:
                print(f"✓ Successfully loaded data at scale {scale}")
                
                # Create visualizations
                print("\n4. Creating slice visualization...")
                viewer.create_slice_visualization(data)
                
                print("\n5. Creating 3D volume plot...")
                fig_volume = viewer.create_3d_volume_plot(data)
                if fig_volume:
                    fig_volume.write_html(f"embl_{dataset_name}_volume.html")
                    print(f"✓ Volume plot saved to embl_{dataset_name}_volume.html")
                    fig_volume.show()
                
                print("\n6. Creating isosurface plot...")
                fig_iso = viewer.create_isosurface_plot(data, threshold=0.4)
                if fig_iso:
                    fig_iso.write_html(f"embl_{dataset_name}_isosurface.html")
                    print(f"✓ Isosurface plot saved to embl_{dataset_name}_isosurface.html")
                    fig_iso.show()
                
                break
            else:
                print(f"✗ Could not load data at scale {scale}")
        else:
            print("Could not load data at any scale level")
    else:
        print("✓ Successfully accessed zarr structure!")
        # Process the full zarr group
        arrays = list(zarr_group.array_keys())
        if arrays:
            array_name = arrays[0]  # Use first array
            print(f"Processing array: {array_name}")
            
            # Load a subset of the data
            array_data = zarr_group[array_name]
            
            # Take a smaller sample for visualization
            if len(array_data.shape) == 4:  # CZYX
                sample = array_data[:, ::4, ::4, ::4]  # Downsample by 4
            else:
                sample = array_data[::4, ::4, ::4]  # Downsample by 4
                
            print(f"Sample shape: {sample.shape}")
            
            # Create visualizations
            viewer.create_slice_visualization(sample)
            
            fig_volume = viewer.create_3d_volume_plot(sample)
            if fig_volume:
                fig_volume.write_html(f"embl_{dataset_name}_full_volume.html")
                fig_volume.show()
            
            fig_iso = viewer.create_isosurface_plot(sample)
            if fig_iso:
                fig_iso.write_html(f"embl_{dataset_name}_full_isosurface.html")
                fig_iso.show()

if __name__ == "__main__":
    main()
