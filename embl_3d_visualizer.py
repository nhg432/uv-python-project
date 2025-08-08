"""
EMBL OME-Zarr 3D Visualizer
Accesses and renders 3D data from EMBL culture collections
Dataset: bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from skimage import measure, filters
from pathlib import Path
import requests
import json
from urllib.parse import urlparse
import fsspec
import xarray as xr

class EMBLZarrVisualizer:
    def __init__(self, base_url):
        """Initialize with EMBL S3 dataset URL"""
        self.base_url = base_url
        self.s3_path = self._convert_to_s3_path(base_url)
        self.metadata = None
        self.data = None
        
    def _convert_to_s3_path(self, url):
        """Convert EMBL console URL to S3 path"""
        # Extract the S3 path from the console URL
        if "culture-collections/data" in url:
            parts = url.split("culture-collections/data")
            if len(parts) > 1:
                s3_path = "s3://culture-collections/data" + parts[1].replace("%2F", "/")
                return s3_path.rstrip("/")
        return None
    
    def load_metadata(self):
        """Load OME-Zarr metadata"""
        try:
            # Try direct S3 access
            if self.s3_path:
                print(f"Accessing S3 path: {self.s3_path}")
                fs = fsspec.filesystem('s3', anon=True)
                
                # Check if path exists and list contents
                try:
                    contents = fs.ls(self.s3_path)
                    print(f"Found {len(contents)} items in dataset")
                    for item in contents[:10]:  # Show first 10 items
                        print(f"  - {item}")
                    
                    # Try to load zarr metadata
                    zarr_store = fs.get_mapper(self.s3_path)
                    zarr_group = zarr.open(zarr_store, mode='r')
                    
                    print(f"Zarr group info: {zarr_group.info}")
                    print(f"Available arrays: {list(zarr_group.array_keys())}")
                    
                    # Try to access OME metadata
                    if hasattr(zarr_group, 'attrs') and 'omero' in zarr_group.attrs:
                        self.metadata = zarr_group.attrs['omero']
                        print("Found OME metadata")
                    
                    return zarr_group
                    
                except Exception as e:
                    print(f"S3 access failed: {e}")
                    return None
                    
        except Exception as e:
            print(f"Metadata loading error: {e}")
            return None
    
    def download_sample_data(self, output_dir="data"):
        """Download a sample of the dataset for local processing"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            if self.s3_path:
                fs = fsspec.filesystem('s3', anon=True)
                
                # Download metadata files
                metadata_files = ['.zattrs', '.zgroup', '.zmetadata']
                for meta_file in metadata_files:
                    try:
                        meta_path = f"{self.s3_path}/{meta_file}"
                        if fs.exists(meta_path):
                            local_path = output_path / meta_file
                            fs.download(meta_path, str(local_path))
                            print(f"Downloaded {meta_file}")
                    except Exception as e:
                        print(f"Could not download {meta_file}: {e}")
                
                # Try to download scale 0 data (highest resolution)
                scale_dirs = ['0', '1', '2']  # Common OME-Zarr scale levels
                for scale in scale_dirs:
                    try:
                        scale_path = f"{self.s3_path}/{scale}"
                        if fs.exists(scale_path):
                            local_scale_path = output_path / scale
                            print(f"Downloading scale {scale} data...")
                            fs.download(scale_path, str(local_scale_path), recursive=True)
                            print(f"Downloaded scale {scale}")
                            break
                    except Exception as e:
                        print(f"Could not download scale {scale}: {e}")
                        continue
                        
            return output_path
            
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def load_data_xarray(self):
        """Load data using xarray for better metadata handling"""
        try:
            if self.s3_path:
                # Use xarray to open the zarr dataset
                import xarray as xr
                
                # Configure fsspec for S3 access
                fs = fsspec.filesystem('s3', anon=True)
                store = fs.get_mapper(self.s3_path)
                
                # Open with xarray
                ds = xr.open_zarr(store, consolidated=False)
                print(f"Dataset dimensions: {ds.dims}")
                print(f"Dataset variables: {list(ds.data_vars)}")
                print(f"Dataset coordinates: {list(ds.coords)}")
                
                return ds
                
        except Exception as e:
            print(f"XArray loading error: {e}")
            return None
    
    def create_3d_visualization(self, data_array, channel=0, sample_factor=2):
        """Create 3D visualization using plotly"""
        try:
            # Sample data to reduce memory usage
            if len(data_array.shape) == 4:  # CZYX format
                volume = data_array[channel, ::sample_factor, ::sample_factor, ::sample_factor]
            elif len(data_array.shape) == 3:  # ZYX format
                volume = data_array[::sample_factor, ::sample_factor, ::sample_factor]
            else:
                print(f"Unexpected data shape: {data_array.shape}")
                return None
            
            print(f"Processing volume of shape: {volume.shape}")
            
            # Normalize data
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            
            # Create isosurface
            threshold = 0.3  # Adjust based on data
            
            # Generate mesh using marching cubes
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    volume, level=threshold, spacing=(1, 1, 1)
                )
                
                # Create 3D mesh plot
                fig = go.Figure(data=[
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        intensity=values,
                        colorscale='Viridis',
                        opacity=0.7,
                        name='3D Surface'
                    )
                ])
                
                fig.update_layout(
                    title=f'3D Visualization: EMBL bmcc122_pfa_cetn-tub-dna',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='cube'
                    ),
                    width=800,
                    height=600
                )
                
                return fig
                
            except Exception as e:
                print(f"Marching cubes failed: {e}")
                
                # Fallback: Create volume rendering
                fig = go.Figure(data=go.Volume(
                    x=np.arange(volume.shape[2]),
                    y=np.arange(volume.shape[1]),
                    z=np.arange(volume.shape[0]),
                    value=volume.flatten(),
                    isomin=0.1,
                    isomax=0.8,
                    opacity=0.1,
                    surface_count=17,
                    colorscale='Viridis'
                ))
                
                fig.update_layout(
                    title='3D Volume Rendering: EMBL bmcc122_pfa_cetn-tub-dna',
                    scene=dict(aspectmode='cube')
                )
                
                return fig
                
        except Exception as e:
            print(f"3D visualization error: {e}")
            return None
    
    def create_napari_view(self, data_array):
        """Create interactive 3D view using napari"""
        try:
            import napari
            
            # Create napari viewer
            viewer = napari.Viewer()
            
            if len(data_array.shape) == 4:  # CZYX format
                for c in range(data_array.shape[0]):
                    viewer.add_image(
                        data_array[c], 
                        name=f'Channel {c}',
                        colormap='viridis',
                        blending='additive'
                    )
            else:
                viewer.add_image(data_array, name='Volume', colormap='viridis')
            
            print("Napari viewer opened. Close the window to continue.")
            napari.run()
            
        except Exception as e:
            print(f"Napari visualization error: {e}")

def main():
    """Main execution function"""
    print("EMBL OME-Zarr 3D Visualizer")
    print("=" * 50)
    
    # EMBL dataset URL
    dataset_url = "https://console.s3.embl.de/browser/culture-collections/data%2Fsingle_volumes%2Fimages%2Fome-zarr%2Fbmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr%2F"
    
    # Initialize visualizer
    visualizer = EMBLZarrVisualizer(dataset_url)
    
    print(f"S3 Path: {visualizer.s3_path}")
    
    # Try different approaches to load data
    print("\n1. Attempting to load metadata...")
    zarr_group = visualizer.load_metadata()
    
    if zarr_group is None:
        print("\n2. Attempting XArray approach...")
        dataset = visualizer.load_data_xarray()
        
        if dataset is not None:
            # Extract data from xarray dataset
            var_names = list(dataset.data_vars)
            if var_names:
                data_var = dataset[var_names[0]]
                data_array = data_var.values
                
                print(f"Data shape: {data_array.shape}")
                print(f"Data type: {data_array.dtype}")
                
                # Create 3D visualization
                print("\n3. Creating 3D visualization...")
                fig = visualizer.create_3d_visualization(data_array)
                
                if fig:
                    # Save interactive plot
                    output_file = "embl_3d_visualization.html"
                    fig.write_html(output_file)
                    print(f"3D visualization saved to: {output_file}")
                    
                    # Show plot
                    fig.show()
                
                # Optional: Open in napari for interactive exploration
                print("\n4. Opening in napari for interactive exploration...")
                try:
                    visualizer.create_napari_view(data_array)
                except Exception as e:
                    print(f"Napari not available: {e}")
                    
        else:
            print("Could not load dataset with any method.")
            
            # Fallback: Try to download sample data
            print("\n3. Attempting to download sample data...")
            local_path = visualizer.download_sample_data()
            
            if local_path:
                print(f"Sample data downloaded to: {local_path}")
                
                # Try to load local data
                try:
                    local_zarr = zarr.open(str(local_path), mode='r')
                    print(f"Local zarr info: {local_zarr.info}")
                    
                    # Find arrays in the local data
                    if hasattr(local_zarr, 'array_keys'):
                        arrays = list(local_zarr.array_keys())
                        if arrays:
                            array_name = arrays[0]
                            data_array = local_zarr[array_name][:]
                            
                            print(f"Loaded array '{array_name}' with shape: {data_array.shape}")
                            
                            # Create visualization
                            fig = visualizer.create_3d_visualization(data_array)
                            if fig:
                                fig.write_html("embl_3d_local.html")
                                fig.show()
                                
                except Exception as e:
                    print(f"Local data loading error: {e}")
    else:
        print("Successfully loaded zarr group!")
        
        # Extract data from zarr group
        arrays = list(zarr_group.array_keys())
        if arrays:
            array_name = arrays[0]
            data_array = zarr_group[array_name][:]
            
            print(f"Loaded array '{array_name}' with shape: {data_array.shape}")
            
            # Create 3D visualization
            print("\n3. Creating 3D visualization...")
            fig = visualizer.create_3d_visualization(data_array)
            
            if fig:
                fig.write_html("embl_3d_direct.html")
                fig.show()
                
                # Open in napari
                try:
                    visualizer.create_napari_view(data_array)
                except Exception as e:
                    print(f"Napari not available: {e}")

if __name__ == "__main__":
    main()
