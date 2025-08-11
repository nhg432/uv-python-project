#!/usr/bin/env python3
"""
AWS S3 Download and 3D Surface Mesh Visualization for Janelia COSEM HeLa-4
Downloads data using AWS CLI and creates interactive 3D visualization
"""

import subprocess
import os
import numpy as np
import plotly.graph_objects as go
import zarr
from skimage import measure
from scipy import ndimage
import json
import sys
from pathlib import Path

def setup_aws_cli():
    """Check if AWS CLI is available and install if needed"""
    try:
        print("🔍 Checking AWS CLI availability...")
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ AWS CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ AWS CLI not found")
            return False
    except FileNotFoundError:
        print("❌ AWS CLI not installed")
        return False

def install_awscli():
    """Install AWS CLI using pip"""
    try:
        print("🔧 Installing AWS CLI...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'awscli'], check=True)
        print("✅ AWS CLI installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install AWS CLI: {e}")
        return False

def configure_aws_anonymous():
    """Configure AWS for anonymous access"""
    try:
        print("🔧 Configuring AWS for anonymous access...")
        
        # Create AWS config directory if it doesn't exist
        aws_dir = Path.home() / '.aws'
        aws_dir.mkdir(exist_ok=True)
        
        # Create config file for anonymous access
        config_content = """[default]
region = us-east-1
output = json

[profile anonymous]
region = us-east-1
output = json
"""
        
        config_file = aws_dir / 'config'
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Create empty credentials file (for anonymous access)
        credentials_content = """[default]
aws_access_key_id = 
aws_secret_access_key = 
"""
        
        credentials_file = aws_dir / 'credentials'
        with open(credentials_file, 'w') as f:
            f.write(credentials_content)
        
        print("✅ AWS configured for anonymous access")
        return True
        
    except Exception as e:
        print(f"❌ AWS configuration failed: {e}")
        return False

def download_s3_data(s3_path, local_path):
    """Download data from S3 using AWS CLI"""
    try:
        print(f"📥 Downloading data from: {s3_path}")
        print(f"📁 Local destination: {local_path}")
        
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        # Try different AWS CLI approaches
        commands_to_try = [
            # Method 1: Standard sync with no-sign-request
            ['aws', 's3', 'sync', s3_path, local_path, '--no-sign-request'],
            # Method 2: Copy with no-sign-request
            ['aws', 's3', 'cp', s3_path, local_path, '--recursive', '--no-sign-request'],
            # Method 3: Using anonymous profile
            ['aws', 's3', 'sync', s3_path, local_path, '--profile', 'anonymous'],
        ]
        
        for i, cmd in enumerate(commands_to_try, 1):
            try:
                print(f"🔄 Attempt {i}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ Download successful!")
                    print(f"📊 Output: {result.stdout[:500]}...")
                    return True
                else:
                    print(f"❌ Attempt {i} failed:")
                    print(f"   Error: {result.stderr[:200]}...")
                    
            except subprocess.TimeoutExpired:
                print(f"⏱️ Attempt {i} timed out")
                continue
            except Exception as e:
                print(f"❌ Attempt {i} exception: {e}")
                continue
        
        print("❌ All download attempts failed")
        return False
        
    except Exception as e:
        print(f"❌ Download setup failed: {e}")
        return False

def explore_downloaded_data(local_path):
    """Explore the downloaded data structure"""
    try:
        print(f"🔍 Exploring downloaded data in: {local_path}")
        
        if not os.path.exists(local_path):
            print(f"❌ Local path does not exist: {local_path}")
            return None
        
        # List contents
        contents = list(os.listdir(local_path))
        print(f"📁 Found {len(contents)} items:")
        for item in contents[:10]:
            item_path = os.path.join(local_path, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                size = os.path.getsize(item_path)
                print(f"   📄 {item} ({size:,} bytes)")
        
        # Look for zarr arrays
        zarr_candidates = []
        for root, dirs, files in os.walk(local_path):
            if '.zarray' in files:
                zarr_candidates.append(root)
        
        print(f"🔍 Found {len(zarr_candidates)} potential zarr arrays:")
        for candidate in zarr_candidates:
            print(f"   📦 {candidate}")
        
        return zarr_candidates
        
    except Exception as e:
        print(f"❌ Data exploration failed: {e}")
        return None

def load_zarr_data(zarr_path):
    """Load zarr data from local path"""
    try:
        print(f"📊 Loading zarr data from: {zarr_path}")
        
        # Try to open as zarr array
        zarr_array = zarr.open_array(zarr_path, mode='r')
        
        print(f"✅ Zarr array loaded successfully!")
        print(f"   Shape: {zarr_array.shape}")
        print(f"   Data type: {zarr_array.dtype}")
        print(f"   Chunks: {zarr_array.chunks}")
        
        # Check if we can read some data
        if len(zarr_array.shape) >= 3:
            # Take a small sample to test reading
            sample_shape = tuple(min(s, 50) for s in zarr_array.shape)
            sample_slices = tuple(slice(0, s) for s in sample_shape)
            
            print(f"🔍 Reading sample data: {sample_shape}")
            sample_data = zarr_array[sample_slices]
            
            print(f"✅ Sample read successful!")
            print(f"   Sample range: {np.min(sample_data)} to {np.max(sample_data)}")
            print(f"   Non-zero pixels: {np.count_nonzero(sample_data):,}")
            
            return zarr_array
        else:
            print(f"❌ Unexpected array dimensions: {zarr_array.shape}")
            return None
            
    except Exception as e:
        print(f"❌ Zarr loading failed: {e}")
        return None

def process_data_for_visualization(zarr_array, max_size=(200, 400, 400)):
    """Process zarr data for 3D visualization"""
    try:
        print(f"🔄 Processing data for visualization...")
        print(f"   Original shape: {zarr_array.shape}")
        print(f"   Max target size: {max_size}")
        
        # Determine sampling strategy
        shape = zarr_array.shape
        if len(shape) != 3:
            print(f"❌ Expected 3D data, got {len(shape)}D")
            return None
        
        # Calculate subsampling factors
        factors = [max(1, s // m) for s, m in zip(shape, max_size)]
        print(f"   Subsampling factors: {factors}")
        
        # Create slice objects for subsampling
        slices = tuple(slice(0, s, f) for s, f in zip(shape, factors))
        
        print(f"🔄 Reading subsampled data...")
        data = zarr_array[slices]
        
        print(f"✅ Data processed successfully!")
        print(f"   Final shape: {data.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Data range: {np.min(data)} to {np.max(data)}")
        print(f"   Memory usage: {data.nbytes / (1024**2):.1f} MB")
        
        return np.array(data)
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return None

def create_surface_mesh(data, threshold=0.3, color='lightcoral', opacity=0.8):
    """Create 3D surface mesh from data"""
    try:
        print(f"🎨 Creating 3D surface mesh...")
        
        # Normalize data to 0-1 range
        data_normalized = (data - data.min()) / (data.max() - data.min())
        print(f"📊 Normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
        
        # Apply Gaussian smoothing for better surface quality
        print(f"🔄 Applying Gaussian smoothing...")
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
        
        # Apply Z-axis scaling for better visualization
        verts_scaled = verts.copy()
        verts_scaled[:, 0] = verts_scaled[:, 0] * 0.5  # Compress Z-axis
        
        # Create the mesh
        mesh = go.Mesh3d(
            x=verts_scaled[:, 2],  # X coordinates
            y=verts_scaled[:, 1],  # Y coordinates  
            z=verts_scaled[:, 0],  # Z coordinates (compressed)
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"Janelia HeLa-4 Surface (threshold {threshold})",
            showscale=False,
            hovertemplate="<b>Janelia COSEM HeLa-4</b><br>" +
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
        print(f"❌ Error creating surface mesh: {e}")
        return None

def create_visualization(data, data_source="Downloaded Janelia COSEM"):
    """Create the final 3D visualization"""
    try:
        print(f"🎨 Creating comprehensive 3D visualization...")
        
        fig = go.Figure()
        
        # Create multiple threshold levels for comprehensive view
        thresholds = [0.2, 0.4, 0.6]
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        opacities = [0.7, 0.6, 0.5]
        
        mesh_count = 0
        for threshold, color, opacity in zip(thresholds, colors, opacities):
            mesh = create_surface_mesh(data, threshold=threshold, color=color, opacity=opacity)
            if mesh:
                fig.add_trace(mesh)
                mesh_count += 1
        
        if mesh_count == 0:
            print("❌ No meshes were created successfully")
            return None
        
        # Configure layout
        fig.update_layout(
            title={
                'text': "Janelia COSEM HeLa-4 3D Surface Mesh Visualization<br><sub>Downloaded from AWS S3 - Multi-threshold rendering</sub>",
                'x': 0.5,
                'font': {'size': 18, 'color': 'darkblue'}
            },
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)", 
                zaxis_title="Z (slices)",
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
                aspectratio=dict(x=1, y=1, z=0.5),
                bgcolor='black'
            ),
            width=1600,
            height=1200,
            annotations=[
                dict(
                    text=f"Janelia COSEM HeLa-4 Cell Visualization<br>" +
                         f"Data shape: {data.shape}<br>" +
                         f"Multi-threshold rendering (0.2, 0.4, 0.6)<br>" +
                         f"Source: {data_source}<br>" +
                         f"Z-axis compressed to 50% for enhanced viewing",
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
        print(f"❌ Error creating visualization: {e}")
        return None

def main():
    """Main execution function"""
    print("🧬 Janelia COSEM HeLa-4 AWS S3 Download and Visualization")
    print("=" * 70)
    
    try:
        # Step 1: Setup AWS CLI
        if not setup_aws_cli():
            if not install_awscli():
                print("❌ Could not setup AWS CLI")
                return
        
        # Step 2: Configure AWS for anonymous access
        if not configure_aws_anonymous():
            print("❌ Could not configure AWS")
            return
        
        # Step 3: Download data using AWS CLI
        s3_path = "s3://janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16/"
        local_path = "./fibsem-uint16"
        
        print(f"\n📥 Downloading data...")
        if not download_s3_data(s3_path, local_path):
            print("❌ Data download failed")
            return
        
        # Step 4: Explore downloaded data
        zarr_candidates = explore_downloaded_data(local_path)
        if not zarr_candidates:
            print("❌ No zarr arrays found in downloaded data")
            return
        
        # Step 5: Load zarr data
        data = None
        for candidate in zarr_candidates:
            data_array = load_zarr_data(candidate)
            if data_array is not None:
                # Process data for visualization
                data = process_data_for_visualization(data_array)
                if data is not None:
                    break
        
        if data is None:
            print("❌ Could not load any zarr data")
            return
        
        # Step 6: Create 3D visualization
        fig = create_visualization(data, "AWS S3 Downloaded")
        if fig is None:
            return
        
        # Step 7: Save visualization
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "janelia_s3_downloaded_mesh.html")
        fig.write_html(output_file)
        
        print(f"\n✅ 3D Visualization Complete!")
        print(f"📁 File created: {output_file}")
        print(f"📊 Dataset shape: {data.shape}")
        print(f"📈 Data range: {np.min(data)} to {np.max(data)}")
        print(f"🎨 Features: Multi-threshold surface mesh with AWS S3 downloaded data")
        print(f"🔬 Source: Janelia COSEM HeLa-4 (AWS S3)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
