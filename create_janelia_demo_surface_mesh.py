#!/usr/bin/env python3
"""
Janelia COSEM HeLa-4 Dataset 3D Surface Mesh Visualization (N5 Format)
This script attempts to access the N5 format version of the dataset
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

def try_n5_access():
    """Try to access the N5 format dataset"""
    try:
        print("🔍 Trying N5 format access...")
        
        # Try to access via different methods
        n5_path = "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.n5"
        
        try:
            # Method 1: Direct zarr access to N5
            print(f"🔗 Trying direct N5 access: {n5_path}")
            zarr_array = zarr.open_array(n5_path, mode='r')
            print(f"✅ Success with N5!")
            print(f"Shape: {zarr_array.shape}")
            print(f"Data type: {zarr_array.dtype}")
            return n5_path, zarr_array
        except Exception as e:
            print(f"   ❌ Direct N5 failed: {e}")
        
        # Method 2: Try with specific paths in N5
        n5_subpaths = [
            f"{n5_path}/em/fibsem-uint16/s0",
            f"{n5_path}/em/fibsem-uint16/s1", 
            f"{n5_path}/em/fibsem-uint16/s2",
            f"{n5_path}/recon-1/em/fibsem-uint16/s0",
            f"{n5_path}/recon-1/em/fibsem-uint16/s1",
            f"{n5_path}/recon-1/em/fibsem-uint16/s2"
        ]
        
        for subpath in n5_subpaths:
            try:
                print(f"🔗 Trying N5 subpath: {subpath}")
                zarr_array = zarr.open_array(subpath, mode='r')
                print(f"✅ Success with N5 subpath!")
                print(f"Shape: {zarr_array.shape}")
                print(f"Data type: {zarr_array.dtype}")
                return subpath, zarr_array
            except Exception as e:
                print(f"   ❌ N5 subpath failed: {str(e)[:50]}...")
                continue
        
        # Method 3: Try to explore N5 structure
        try:
            print(f"🔗 Exploring N5 structure...")
            fs = fsspec.filesystem('s3', anon=True)
            n5_contents = fs.ls("janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.n5")
            print(f"✅ N5 contents:")
            for item in n5_contents[:10]:
                print(f"   📁 {item}")
        except Exception as e:
            print(f"   ❌ N5 exploration failed: {e}")
        
        return None, None
        
    except Exception as e:
        print(f"❌ N5 access failed: {e}")
        return None, None

def try_direct_download():
    """Try to download a small sample directly using requests"""
    try:
        print("🔍 Trying direct HTTP access...")
        
        # Try to access the Neuroglancer format or find downloadable samples
        base_url = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-4"
        
        # Check if we can access the thumbnail first
        thumbnail_url = f"{base_url}/thumbnail.jpg"
        try:
            response = requests.head(thumbnail_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ HTTP access works! Thumbnail accessible.")
            else:
                print(f"❌ HTTP access issue. Status: {response.status_code}")
                return None, None
        except Exception as e:
            print(f"❌ HTTP test failed: {e}")
            return None, None
        
        # Try to create a dummy dataset for demonstration
        print("🔧 Creating demonstration dataset...")
        
        # Create a synthetic dataset that resembles cellular structures
        z, y, x = 100, 200, 200
        synthetic_data = np.zeros((z, y, x), dtype=np.uint16)
        
        # Add some cellular-like structures
        # Create a nucleus-like structure
        center_z, center_y, center_x = z//2, y//2, x//2
        for zi in range(z):
            for yi in range(y):
                for xi in range(x):
                    # Distance from center
                    dist = np.sqrt((zi-center_z)**2 + (yi-center_y)**2 + (xi-center_x)**2)
                    if dist < 30:  # Nuclear region
                        synthetic_data[zi, yi, xi] = int(65535 * (1 - dist/30))
                    elif dist < 50:  # Cytoplasm
                        synthetic_data[zi, yi, xi] = int(32767 * (1 - (dist-30)/20))
        
        # Add some organelle-like structures
        np.random.seed(42)
        for _ in range(20):
            oz = np.random.randint(10, z-10)
            oy = np.random.randint(10, y-10) 
            ox = np.random.randint(10, x-10)
            for zi in range(max(0, oz-5), min(z, oz+5)):
                for yi in range(max(0, oy-5), min(y, oy+5)):
                    for xi in range(max(0, ox-5), min(x, ox+5)):
                        synthetic_data[zi, yi, xi] = max(synthetic_data[zi, yi, xi], 40000)
        
        print(f"✅ Created synthetic dataset: {synthetic_data.shape}")
        print(f"📈 Data range: {np.min(synthetic_data)} to {np.max(synthetic_data)}")
        
        return "synthetic", synthetic_data
        
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        return None, None

def create_janelia_surface_mesh(data, threshold=0.15, color='cyan', opacity=0.6):
    """Create a 3D surface mesh from data"""
    try:
        print(f"🔧 Creating surface mesh...")
        
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
            name="Janelia COSEM-like Surface",
            showscale=False,
            hovertemplate="<b>Cellular Structure</b><br>" +
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

def create_visualization(data, data_source):
    """Create the final 3D visualization"""
    try:
        print(f"🎨 Creating visualization...")
        
        # Create the figure
        fig = go.Figure()
        
        # Create surface mesh
        mesh = create_janelia_surface_mesh(data, threshold=0.2, color='lightblue', opacity=0.7)
        
        if mesh:
            fig.add_trace(mesh)
            print(f"✅ Added surface mesh")
        
        # Determine title based on data source
        if data_source == "synthetic":
            title_text = "Synthetic Cellular Structure 3D Surface Mesh<br><sub>Demonstration of 3D rendering techniques (Nuclear and organelle-like structures)</sub>"
            source_text = "Synthetic demonstration data"
        else:
            title_text = "Janelia COSEM HeLa-4 3D Surface Mesh Visualization<br><sub>Interactive 3D cellular structure rendering</sub>"
            source_text = f"Source: {data_source}"
        
        # Update layout
        fig.update_layout(
            title={
                'text': title_text,
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
            annotations=[
                dict(
                    text=f"3D surface mesh visualization<br>" +
                         f"Data shape: {data.shape}<br>" +
                         f"Threshold: 0.2 | Z-axis flattened to 30%<br>" +
                         source_text,
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
    print("🧬 Janelia COSEM HeLa-4 Dataset Access Attempt")
    print("=" * 60)
    
    try:
        # Try N5 format first
        source, data = try_n5_access()
        
        if data is None:
            # Try direct download/access
            source, data = try_direct_download()
        
        if data is None:
            print("\n❌ Could not access the Janelia COSEM dataset")
            print("💡 The dataset might require special authentication or access permissions")
            return
        
        # Create visualization
        fig = create_visualization(data, source)
        if fig is None:
            return
        
        # Save visualization
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "janelia_hela4_surface_mesh.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This visualization shows:")
        print(f"   🔬 Cellular structures (nuclear and organelle-like)")
        print(f"   📊 Data shape: {data.shape}")
        print(f"   🎨 3D surface mesh with light blue coloring")
        print(f"   📏 Z-axis flattened to 30% for enhanced viewing")
        print(f"   ⚡ Interactive 3D exploration")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
