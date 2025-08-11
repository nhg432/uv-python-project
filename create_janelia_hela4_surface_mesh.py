#!/usr/bin/env python3
"""
Janelia COSEM HeLa-4 Dataset 3D Surface Mesh Visualization
This script downloads and creates 3D surface mesh renderings from the Janelia COSEM dataset:
- Dataset: jrc_hela-4 from Quilt Data
- Multi-channel cellular imaging data
- Creates interactive 3D surface meshes
"""

import zarr
import numpy as np
import plotly.graph_objects as go
import fsspec
import os
import requests
from skimage import measure
from scipy import ndimage
import json

def explore_janelia_dataset():
    """Explore the Janelia COSEM HeLa-4 dataset structure"""
    try:
        print("🔍 Exploring Janelia COSEM HeLa-4 dataset...")
        
        # First, try to access the S3 bucket to see structure
        try:
            fs = fsspec.filesystem('s3', anon=True)
            bucket_path = "janelia-cosem-datasets/jrc_hela-4"
            
            contents = fs.ls(f"s3://{bucket_path}")
            print(f"✅ Found S3 contents:")
            for item in contents:
                print(f"   📁 {item}")
            
            # Now try to access the Zarr dataset
            zarr_path = "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr"
            print(f"\n🔗 Trying to open Zarr: {zarr_path}")
            
            zarr_array = zarr.open_array(zarr_path, mode='r')
            print(f"✅ Successfully opened Zarr array!")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            
            return zarr_path, zarr_array
            
        except Exception as e:
            print(f"❌ Error with S3/Zarr access: {e}")
        
        # Try alternative Zarr paths
        zarr_alternatives = [
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/0",
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1"
        ]
        
        for zarr_path in zarr_alternatives:
            try:
                print(f"🔗 Trying Zarr path: {zarr_path}")
                zarr_array = zarr.open_array(zarr_path, mode='r')
                print(f"✅ Success with: {zarr_path}")
                print(f"Shape: {zarr_array.shape}")
                print(f"Data type: {zarr_array.dtype}")
                return zarr_path, zarr_array
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        return None, None
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        return None, None

def try_alternative_access():
    """Try alternative methods to access the dataset"""
    try:
        print("\n🔄 Trying direct access to specific data paths...")
        
        # Based on the structure we found, try direct paths
        direct_paths = [
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16",
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16/s0",
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16/s1",
            "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16/s2",
        ]
        
        for path in direct_paths:
            try:
                print(f"🔗 Trying direct path: {path}")
                zarr_array = zarr.open_array(path, mode='r')
                print(f"✅ SUCCESS! Found data at: {path}")
                print(f"   Shape: {zarr_array.shape}")
                print(f"   Data type: {zarr_array.dtype}")
                print(f"   Chunks: {zarr_array.chunks}")
                return path, zarr_array
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}...")
                continue
        
        # Try using zarr.open_group to navigate the hierarchy
        try:
            print(f"\n� Trying hierarchical group access...")
            
            # Open the root group
            mapper = fsspec.get_mapper("s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr")
            root_group = zarr.open_group(mapper, mode='r')
            print(f"✅ Opened root group!")
            print(f"   Root keys: {list(root_group.keys())}")
            
            # Navigate to recon-1
            if 'recon-1' in root_group:
                recon_group = root_group['recon-1']
                print(f"   Recon-1 keys: {list(recon_group.keys())}")
                
                # Navigate to em
                if 'em' in recon_group:
                    em_group = recon_group['em']
                    print(f"   EM keys: {list(em_group.keys())}")
                    
                    # Try to access fibsem-uint16
                    if 'fibsem-uint16' in em_group:
                        fibsem_group = em_group['fibsem-uint16']
                        print(f"   Fibsem keys: {list(fibsem_group.keys())}")
                        
                        # Try different resolution levels
                        for res_key in ['s0', 's1', 's2', 's3']:
                            if res_key in fibsem_group:
                                try:
                                    zarr_array = fibsem_group[res_key]
                                    print(f"✅ SUCCESS! Found array at recon-1/em/fibsem-uint16/{res_key}")
                                    print(f"   Shape: {zarr_array.shape}")
                                    print(f"   Data type: {zarr_array.dtype}")
                                    print(f"   Chunks: {zarr_array.chunks}")
                                    return f"recon-1/em/fibsem-uint16/{res_key}", zarr_array
                                except Exception as e:
                                    print(f"   ❌ Could not access {res_key}: {e}")
                                    continue
            
        except Exception as e:
            print(f"❌ Hierarchical access failed: {e}")
        
        return None, None
        
    except Exception as e:
        print(f"❌ Alternative access failed: {e}")
        return None, None

def download_sample_data(source, zarr_array, sample_size=(200, 200, 200)):
    """Download a sample of the dataset for processing"""
    try:
        print(f"\n📥 Downloading sample data (size: {sample_size})...")
        
        if zarr_array is None:
            print("❌ No valid zarr array provided")
            return None
            
        print(f"📊 Full dataset shape: {zarr_array.shape}")
        print(f"📊 Data type: {zarr_array.dtype}")
        
        # Determine how to slice the data based on shape
        if len(zarr_array.shape) == 3:  # Simple 3D array (Z, Y, X)
            z, y, x = zarr_array.shape
            z_end = min(sample_size[0], z)
            y_end = min(sample_size[1], y)
            x_end = min(sample_size[2], x)
            
            print(f"📦 Extracting 3D region: Z[0:{z_end}], Y[0:{y_end}], X[0:{x_end}]")
            sample_data = np.array(zarr_array[0:z_end, 0:y_end, 0:x_end])
            
        elif len(zarr_array.shape) == 4:  # 4D array (likely C, Z, Y, X)
            c, z, y, x = zarr_array.shape
            z_end = min(sample_size[0], z)
            y_end = min(sample_size[1], y)
            x_end = min(sample_size[2], x)
            
            print(f"📦 Extracting 4D region from channel 0: Z[0:{z_end}], Y[0:{y_end}], X[0:{x_end}]")
            sample_data = np.array(zarr_array[0, 0:z_end, 0:y_end, 0:x_end])
            
        elif len(zarr_array.shape) == 5:  # 5D array (likely T, C, Z, Y, X)
            t, c, z, y, x = zarr_array.shape
            z_end = min(sample_size[0], z)
            y_end = min(sample_size[1], y)
            x_end = min(sample_size[2], x)
            
            print(f"📦 Extracting 5D region from T=0, C=0: Z[0:{z_end}], Y[0:{y_end}], X[0:{x_end}]")
            sample_data = np.array(zarr_array[0, 0, 0:z_end, 0:y_end, 0:x_end])
        
        else:
            print(f"❌ Unsupported array shape: {zarr_array.shape}")
            return None
            
        print(f"✅ Downloaded sample data: {sample_data.shape}")
        print(f"📈 Data range: {np.min(sample_data):.3f} to {np.max(sample_data):.3f}")
        print(f"📊 Data type: {sample_data.dtype}")
        
        return sample_data
            
    except Exception as e:
        print(f"❌ Error downloading sample data: {e}")
        return None

def create_janelia_surface_mesh(data, threshold=0.1, color='cyan', opacity=0.5):
    """Create a 3D surface mesh from Janelia COSEM data"""
    try:
        print(f"🔧 Creating surface mesh from Janelia data...")
        
        # Normalize data to 0-1 range
        data_normalized = (data - data.min()) / (data.max() - data.min())
        print(f"📊 Normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
        
        # Apply Gaussian smoothing
        print(f"🔄 Applying smoothing...")
        smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.0)
        
        # Create isosurface using marching cubes
        print(f"🔄 Generating mesh with threshold {threshold}...")
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ Generated {len(verts):,} vertices, {len(faces):,} faces")
        
        # Apply Z-axis flattening
        verts_flattened = verts.copy()
        verts_flattened[:, 0] = verts_flattened[:, 0] * 0.3  # Flatten Z-axis
        
        print(f"🔄 Z-flattening applied: {verts[:, 0].min():.1f}-{verts[:, 0].max():.1f} → {verts_flattened[:, 0].min():.1f}-{verts_flattened[:, 0].max():.1f}")
        
        # Create the mesh
        mesh = go.Mesh3d(
            x=verts_flattened[:, 2],  # X coordinates
            y=verts_flattened[:, 1],  # Y coordinates  
            z=verts_flattened[:, 0],  # Z coordinates (flattened)
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name="Janelia COSEM HeLa-4 Surface",
            showscale=False,
            hovertemplate="<b>Janelia COSEM HeLa-4</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold}<br>" +
                         "<extra></extra>",
            lighting=dict(
                ambient=0.3,
                diffuse=0.7,
                specular=0.4,
                roughness=0.3,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=100, z=100)
        )
        
        return mesh
        
    except Exception as e:
        print(f"❌ Error creating surface mesh: {e}")
        return None

def create_janelia_visualization(data):
    """Create the final 3D visualization"""
    try:
        print(f"🎨 Creating Janelia COSEM visualization...")
        
        # Create the figure
        fig = go.Figure()
        
        # Create surface mesh
        mesh = create_janelia_surface_mesh(data, threshold=0.15, color='cyan', opacity=0.6)
        
        if mesh:
            fig.add_trace(mesh)
            print(f"✅ Added surface mesh")
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Janelia COSEM HeLa-4 3D Surface Mesh Visualization<br>" +
                       "<sub>Interactive 3D cellular structure rendering with Z-axis flattening</sub>",
                'x': 0.5,
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)", 
                zaxis_title="Z (slices)",
                camera=dict(
                    eye=dict(x=3.0, y=3.0, z=3.0),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(
                    showbackground=True, 
                    backgroundcolor="rgb(250, 250, 250)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    showbackground=True, 
                    backgroundcolor="rgb(250, 250, 250)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    showbackground=True, 
                    backgroundcolor="rgb(250, 250, 250)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3),
                bgcolor='black'
            ),
            width=1600,
            height=1200,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            ),
            annotations=[
                dict(
                    text="Janelia COSEM HeLa-4 dataset 3D surface mesh visualization<br>" +
                         f"Data shape: {data.shape}<br>" +
                         f"Threshold: 0.15 | Z-axis flattened to 30%<br>" +
                         "Source: Janelia Research Campus COSEM",
                    x=0.02, y=0.02,
                    xref="paper", yref="paper",
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def main():
    """Main execution function"""
    print("🧬 Janelia COSEM HeLa-4 Dataset 3D Surface Mesh Visualization")
    print("=" * 80)
    
    try:
        # Step 1: Explore dataset
        source, data_source = explore_janelia_dataset()
        
        if source is None:
            print("\n🔄 Trying alternative access methods...")
            source, data_source = try_alternative_access()
        
        if source is None:
            print("\n❌ Could not access the Janelia COSEM dataset")
            print("💡 The dataset might require authentication or be in a different format")
            print("💡 Please check the URL and ensure the dataset is publicly accessible")
            return
        
        # Step 2: Download sample data
        sample_data = download_sample_data(source, data_source)
        if sample_data is None:
            return
        
        # Step 3: Create visualization
        fig = create_janelia_visualization(sample_data)
        if fig is None:
            return
        
        # Step 4: Save visualization
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "janelia_hela4_surface_mesh.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Janelia COSEM HeLa-4 visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This visualization shows:")
        print(f"   🔬 Janelia COSEM HeLa-4 cellular structures")
        print(f"   📊 Data shape: {sample_data.shape}")
        print(f"   🎨 3D surface mesh with cyan coloring")
        print(f"   📏 Z-axis flattened to 30% for enhanced viewing")
        print(f"   ⚡ Interactive 3D exploration with rotation and zoom")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
