#!/usr/bin/env python3
"""
Janelia COSEM HeLa-4 Dataset Download and 3D Surface Mesh Visualization
Downloads data from Quilt Data platform and creates interactive 3D visualization
"""

import numpy as np
import plotly.graph_objects as go
import requests
import zarr
import fsspec
import os
from skimage import measure
from scipy import ndimage
import json
from urllib.parse import urlparse
import time

def download_from_quilt():
    """Download data from Quilt Data platform using direct S3 access"""
    try:
        print("🔍 Accessing Janelia COSEM HeLa-4 dataset from Quilt Data...")
        
        # Base S3 path for the dataset
        base_s3_path = "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16"
        
        # Create filesystem
        fs = fsspec.filesystem('s3', anon=True)
        
        # Try different resolution levels
        resolution_levels = ['s0', 's1', 's2', 's3', 's4', 's5']
        
        for level in resolution_levels:
            try:
                zarr_path = f"{base_s3_path}/{level}"
                print(f"🔗 Trying resolution level: {level}")
                
                # Check if this level exists
                try:
                    level_contents = fs.ls(f"janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16/{level}")
                    print(f"   ✅ Level {level} exists with {len(level_contents)} items")
                except:
                    print(f"   ❌ Level {level} not accessible")
                    continue
                
                # Try to open as zarr array
                try:
                    zarr_array = zarr.open_array(zarr_path, mode='r')
                    print(f"   ✅ Successfully opened zarr array!")
                    print(f"   📊 Shape: {zarr_array.shape}")
                    print(f"   📊 Data type: {zarr_array.dtype}")
                    print(f"   📊 Chunks: {zarr_array.chunks}")
                    
                    return zarr_array, level
                    
                except Exception as e:
                    print(f"   ❌ Zarr open failed: {str(e)[:100]}...")
                    
                    # Try alternative zarr access methods
                    try:
                        # Method 2: Use fsspec mapping
                        mapper = fsspec.get_mapper(zarr_path)
                        zarr_array = zarr.open_array(mapper, mode='r')
                        print(f"   ✅ Success with fsspec mapper!")
                        print(f"   📊 Shape: {zarr_array.shape}")
                        return zarr_array, level
                    except Exception as e2:
                        print(f"   ❌ Mapper method failed: {str(e2)[:50]}...")
                        continue
            
            except Exception as e:
                print(f"   ❌ Level {level} failed: {str(e)[:50]}...")
                continue
        
        print("❌ Could not access any resolution level")
        return None, None
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None, None

def try_alternative_s3_access():
    """Try alternative S3 access methods"""
    try:
        print("🔄 Trying alternative S3 access methods...")
        
        # Method 1: Direct S3 without zarr wrapper
        fs = fsspec.filesystem('s3', anon=True)
        
        # Explore the structure
        base_path = "janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16"
        
        try:
            contents = fs.ls(base_path)
            print(f"✅ Base path contents ({len(contents)} items):")
            for item in contents[:10]:
                print(f"   📁 {item}")
            
            # Look for resolution levels
            for item in contents:
                if item.endswith('/s0') or item.endswith('/s1') or item.endswith('/s2'):
                    level_name = item.split('/')[-1]
                    print(f"\n🔍 Exploring level: {level_name}")
                    
                    try:
                        level_contents = fs.ls(item)
                        print(f"   📁 Level contents ({len(level_contents)} items):")
                        for subitem in level_contents[:5]:
                            print(f"      📄 {subitem}")
                        
                        # Try to read some chunks directly
                        for subitem in level_contents:
                            if subitem.endswith('.0.0.0') or 'chunk' in subitem:
                                try:
                                    print(f"   🔗 Trying to read: {subitem}")
                                    chunk_data = fs.cat(subitem)
                                    print(f"   ✅ Read {len(chunk_data)} bytes")
                                    
                                    # This indicates we can access the data
                                    # Now try to reconstruct the zarr array
                                    zarr_path = f"s3://{item}"
                                    try:
                                        zarr_array = zarr.open_array(zarr_path, mode='r')
                                        print(f"   ✅ Zarr array opened successfully!")
                                        return zarr_array, level_name
                                    except Exception as ze:
                                        print(f"   ❌ Zarr reconstruction failed: {ze}")
                                        
                                except Exception as ce:
                                    print(f"   ❌ Chunk read failed: {ce}")
                                    continue
                    
                    except Exception as le:
                        print(f"   ❌ Level exploration failed: {le}")
                        continue
        
        except Exception as e:
            print(f"❌ Base path exploration failed: {e}")
        
        return None, None
        
    except Exception as e:
        print(f"❌ Alternative access failed: {e}")
        return None, None

def create_sample_from_metadata():
    """Create a sample dataset if direct access fails, using available metadata"""
    try:
        print("🔧 Creating sample dataset based on HeLa-4 specifications...")
        
        # Based on Janelia COSEM typical dimensions for HeLa cells
        # These are realistic dimensions for EM data
        z, y, x = 150, 400, 400  # Approximate HeLa cell dimensions
        
        print(f"📊 Creating dataset with shape: {(z, y, x)}")
        
        # Create realistic cellular structures
        data = np.zeros((z, y, x), dtype=np.uint16)
        
        # Create nucleus (spherical, centered)
        center_z, center_y, center_x = z//2, y//2, x//2
        nucleus_radius = 40
        
        for zi in range(z):
            for yi in range(y):
                for xi in range(x):
                    # Distance from center
                    dist = np.sqrt((zi-center_z)**2 + (yi-center_y)**2 + (xi-center_x)**2)
                    
                    # Nuclear region (high intensity)
                    if dist < nucleus_radius:
                        intensity = int(65535 * (1 - dist/nucleus_radius) * 0.8)
                        data[zi, yi, xi] = max(data[zi, yi, xi], intensity)
                    
                    # Cytoplasmic region (medium intensity)
                    elif dist < nucleus_radius + 30:
                        cytoplasm_dist = dist - nucleus_radius
                        intensity = int(32767 * (1 - cytoplasm_dist/30) * 0.6)
                        data[zi, yi, xi] = max(data[zi, yi, xi], intensity)
        
        # Add organelle-like structures (mitochondria, ER, etc.)
        np.random.seed(42)  # For reproducible results
        
        # Mitochondria-like structures
        for _ in range(30):
            mito_z = np.random.randint(20, z-20)
            mito_y = np.random.randint(50, y-50)
            mito_x = np.random.randint(50, x-50)
            
            # Skip if too close to nucleus
            if np.sqrt((mito_z-center_z)**2 + (mito_y-center_y)**2 + (mito_x-center_x)**2) < nucleus_radius + 10:
                continue
            
            # Create elongated mitochondria
            for dz in range(-3, 4):
                for dy in range(-8, 9):
                    for dx in range(-3, 4):
                        zi, yi, xi = mito_z + dz, mito_y + dy, mito_x + dx
                        if 0 <= zi < z and 0 <= yi < y and 0 <= xi < x:
                            if abs(dz) <= 2 and abs(dy) <= 6 and abs(dx) <= 2:
                                data[zi, yi, xi] = max(data[zi, yi, xi], 45000)
        
        # Add ER-like network
        for _ in range(50):
            er_z = np.random.randint(10, z-10)
            er_y = np.random.randint(30, y-30)
            er_x = np.random.randint(30, x-30)
            
            # Create tubular ER structures
            for i in range(20):
                offset_y = np.random.randint(-15, 16)
                offset_x = np.random.randint(-15, 16)
                zi, yi, xi = er_z, er_y + offset_y, er_x + offset_x
                
                if 0 <= zi < z and 0 <= yi < y and 0 <= xi < x:
                    data[zi, yi, xi] = max(data[zi, yi, xi], 30000)
        
        print(f"✅ Created realistic HeLa-4-like dataset")
        print(f"📈 Data range: {np.min(data)} to {np.max(data)}")
        print(f"📊 Non-zero pixels: {np.count_nonzero(data):,} ({100*np.count_nonzero(data)/data.size:.1f}%)")
        
        return data, "hela4_synthetic"
        
    except Exception as e:
        print(f"❌ Sample creation failed: {e}")
        return None, None

def create_surface_mesh(data, threshold=0.3, color='lightcoral', opacity=0.8):
    """Create 3D surface mesh from imaging data"""
    try:
        print(f"🔧 Creating 3D surface mesh...")
        
        # Normalize data
        data_normalized = (data - data.min()) / (data.max() - data.min())
        print(f"📊 Normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
        
        # Apply Gaussian smoothing for better surface quality
        print(f"🔄 Applying Gaussian smoothing...")
        smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.5)
        
        # Generate surface mesh using marching cubes
        print(f"🔄 Generating surface mesh (threshold: {threshold})...")
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data,
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ Generated mesh: {len(verts):,} vertices, {len(faces):,} faces")
        
        # Apply scaling for better visualization
        verts_scaled = verts.copy()
        verts_scaled[:, 0] = verts_scaled[:, 0] * 0.4  # Compress Z-axis
        
        # Create the 3D mesh
        mesh = go.Mesh3d(
            x=verts_scaled[:, 2],  # X coordinates
            y=verts_scaled[:, 1],  # Y coordinates
            z=verts_scaled[:, 0],  # Z coordinates (compressed)
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name="HeLa-4 Cellular Structure",
            showscale=False,
            hovertemplate="<b>HeLa-4 Cell</b><br>" +
                         "X: %{x:.1f}<br>" +
                         "Y: %{y:.1f}<br>" +
                         "Z: %{z:.1f}<br>" +
                         f"Threshold: {threshold}<br>" +
                         "<extra></extra>",
            lighting=dict(
                ambient=0.4,
                diffuse=0.6,
                specular=0.5,
                roughness=0.2,
                fresnel=0.3
            ),
            lightposition=dict(x=100, y=100, z=100)
        )
        
        return mesh
        
    except Exception as e:
        print(f"❌ Surface mesh creation failed: {e}")
        return None

def create_visualization(data, data_source):
    """Create final 3D visualization"""
    try:
        print(f"🎨 Creating 3D visualization...")
        
        fig = go.Figure()
        
        # Create main surface mesh
        mesh = create_surface_mesh(data, threshold=0.3, color='lightcoral', opacity=0.8)
        if mesh:
            fig.add_trace(mesh)
        
        # Add a second mesh with different threshold for more detail
        mesh2 = create_surface_mesh(data, threshold=0.5, color='lightblue', opacity=0.6)
        if mesh2:
            fig.add_trace(mesh2)
        
        # Determine title based on source
        if "synthetic" in data_source:
            title_text = "HeLa-4 Synthetic Cellular Structure 3D Visualization<br><sub>Realistic synthetic dataset based on Janelia COSEM specifications</sub>"
            source_text = "Synthetic HeLa-4-like data"
        else:
            title_text = "Janelia COSEM HeLa-4 3D Surface Mesh Visualization<br><sub>Downloaded from Quilt Data Platform</sub>"
            source_text = f"Janelia COSEM - {data_source}"
        
        # Configure layout
        fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'font': {'size': 18, 'color': 'darkblue'}
            },
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                zaxis_title="Z (μm)",
                camera=dict(
                    eye=dict(x=2.5, y=2.5, z=2.5),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    zerolinecolor="white"
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.4),
                bgcolor='black'
            ),
            width=1600,
            height=1200,
            annotations=[
                dict(
                    text=f"Janelia COSEM HeLa-4 Cell Visualization<br>" +
                         f"Dataset shape: {data.shape}<br>" +
                         f"Dual threshold rendering (0.3 & 0.5)<br>" +
                         f"Source: {source_text}<br>" +
                         f"Z-axis compressed to 40% for enhanced viewing",
                    x=0.02, y=0.02,
                    xref="paper", yref="paper",
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=12, color='white'),
                    bgcolor="rgba(0,0,0,0.8)",
                    bordercolor="white",
                    borderwidth=1
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        return None

def main():
    """Main execution function"""
    print("🧬 Janelia COSEM HeLa-4 Dataset Download and Visualization")
    print("=" * 70)
    
    try:
        # Try to download from Quilt Data
        data, source = download_from_quilt()
        
        if data is None:
            # Try alternative access methods
            data, source = try_alternative_s3_access()
        
        if data is None:
            # Create synthetic data as demonstration
            print("\n⚠️  Direct dataset access failed - creating synthetic demonstration")
            data, source = create_sample_from_metadata()
        
        if data is None:
            print("❌ Could not access or create dataset")
            return
        
        # Sample the data if it's too large
        if hasattr(data, 'shape') and np.prod(data.shape) > 50_000_000:  # 50M pixels
            print(f"🔄 Dataset is large {data.shape}, sampling for performance...")
            # Take a central crop
            z, y, x = data.shape
            crop_z = min(z, 200)
            crop_y = min(y, 400)
            crop_x = min(x, 400)
            
            start_z = (z - crop_z) // 2
            start_y = (y - crop_y) // 2
            start_x = (x - crop_x) // 2
            
            if hasattr(data, '__getitem__'):
                data = data[start_z:start_z+crop_z, start_y:start_y+crop_y, start_x:start_x+crop_x]
            else:
                data = np.array(data[start_z:start_z+crop_z, start_y:start_y+crop_y, start_x:start_x+crop_x])
            
            print(f"✅ Cropped to: {data.shape}")
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            print("🔄 Converting to numpy array...")
            data = np.array(data)
        
        # Create visualization
        fig = create_visualization(data, source)
        if fig is None:
            return
        
        # Save the visualization
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "janelia_hela4_cosem_mesh.html")
        fig.write_html(output_file)
        
        print(f"\n✅ 3D Visualization Complete!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Dataset shape: {data.shape}")
        print(f"📈 Data range: {np.min(data)} to {np.max(data)}")
        print(f"🎨 Features: Dual-threshold surface mesh with realistic cellular structures")
        print(f"🔬 Source: {source}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
