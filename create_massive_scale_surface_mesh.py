#!/usr/bin/env python3
"""
Massive Scale EMBL OME-Zarr Multi-Channel 3D Surface Mesh Visualization
This script creates 3D surface mesh renderings at massive scale:
- Region: 1000x1000x1000 voxels 
- Tubulin (Channel 1) - Green surface mesh (threshold 0.5 - original)
- DNA (Channel 2) - Blue surface mesh (threshold 0.4 - original)
- Centrin (Channel 0) - Red surface mesh (threshold 0.3 - original)
"""

import zarr
import numpy as np
import plotly.graph_objects as go
import fsspec
import os
from skimage import measure
from scipy import ndimage

def load_embl_dataset():
    """Load EMBL OME-Zarr dataset from S3"""
    try:
        # Direct HTTP access to EMBL S3 data
        base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
        
        print(f"🔗 Connecting to EMBL dataset...")
        
        # Use HTTP filesystem
        fs = fsspec.filesystem('http')
        
        # Try to access the highest resolution data (0/0)
        zarr_path = f"{base_url}/0/0"
        
        print(f"📥 Loading array from: {zarr_path}")
        
        # Open the zarr array
        zarr_array = zarr.open_array(zarr_path, mode='r')
        
        print(f"✅ Successfully loaded array!")
        print(f"Shape: {zarr_array.shape}")
        print(f"Data type: {zarr_array.dtype}")
        
        return zarr_array
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def extract_massive_sample_data_all_channels(zarr_array, sample_size=1000):
    """Extract MASSIVE sample data from zarr array for all channels - 1000x1000x1000"""
    try:
        print(f"\n📊 Extracting MASSIVE sample data for all channels...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full array shape: {zarr_array.shape}")
        
        # For massive scale, we need to be more strategic about sampling
        # Let's take a large region but ensure we don't exceed dataset bounds
        
        # Calculate bounds for massive region
        z_max_size = min(sample_size, z)
        y_max_size = min(sample_size, y) 
        x_max_size = min(sample_size, x)
        
        # Start from center and expand as much as possible
        z_center, y_center, x_center = z // 2, y // 2, x // 2
        
        z_half = z_max_size // 2
        y_half = y_max_size // 2
        x_half = x_max_size // 2
        
        # Define bounds ensuring we don't go out of dataset bounds
        z_start = max(0, z_center - z_half)
        z_end = min(z, z_start + z_max_size)
        y_start = max(0, y_center - y_half)
        y_end = min(y, y_start + y_max_size)
        x_start = max(0, x_center - x_half)
        x_end = min(x, x_start + x_max_size)
        
        # Adjust if we hit boundaries
        if z_end - z_start < z_max_size:
            z_start = max(0, z_end - z_max_size)
        if y_end - y_start < y_max_size:
            y_start = max(0, y_end - y_max_size)
        if x_end - x_start < x_max_size:
            x_start = max(0, x_end - x_max_size)
            
        actual_z_size = z_end - z_start
        actual_y_size = y_end - y_start
        actual_x_size = x_end - x_start
        
        print(f"📦 Extracting MASSIVE region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        print(f"📏 MASSIVE dimensions: {actual_z_size} x {actual_y_size} x {actual_x_size}")
        print(f"🎯 Target was: {sample_size}x{sample_size}x{sample_size}")
        
        # Extract data for all three channels - MASSIVE REGION
        print(f"🔄 Loading Centrin channel (this may take a moment)...")
        centrin_data = zarr_array[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 0
        
        print(f"🔄 Loading Tubulin channel (this may take a moment)...")
        tubulin_data = zarr_array[0, 1, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 1
        
        print(f"🔄 Loading DNA channel (this may take a moment)...")
        dna_data = zarr_array[0, 2, z_start:z_end, y_start:y_end, x_start:x_end]       # Channel 2
        
        total_voxels = actual_z_size * actual_y_size * actual_x_size
        print(f"✅ All channels extracted! MASSIVE Shape: {centrin_data.shape}")
        print(f"📊 Total voxels processed: {total_voxels:,}")
        print(f"📈 Centrin range: {np.min(centrin_data):.3f} to {np.max(centrin_data):.3f}")
        print(f"📈 Tubulin range: {np.min(tubulin_data):.3f} to {np.max(tubulin_data):.3f}")
        print(f"📈 DNA range: {np.min(dna_data):.3f} to {np.max(dna_data):.3f}")
        
        return centrin_data, tubulin_data, dna_data
        
    except Exception as e:
        print(f"❌ Error extracting massive sample: {e}")
        return None, None, None

def create_surface_mesh_original_thresholds(data, channel_name, threshold, color, opacity=0.4):
    """Create a 3D surface mesh from massive volumetric data with ultra-low thresholds"""
    print(f"🔧 Creating surface mesh for {channel_name} at massive scale...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Apply moderate Gaussian smoothing for massive data
    print(f"🔄 Applying smoothing to {channel_name} (this may take time for massive data)...")
    smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.5)
    
    try:
        # Create isosurface using marching cubes with ULTRA-LOW thresholds
        print(f"🔄 Generating mesh for {channel_name} with ultra-low threshold {threshold}...")
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ {channel_name}: Generated {len(verts):,} vertices, {len(faces):,} faces (massive scale)")
        
        # Create the mesh with optimized settings for massive data
        mesh = go.Mesh3d(
            x=verts[:, 2],  # X coordinates
            y=verts[:, 1],  # Y coordinates  
            z=verts[:, 0],  # Z coordinates
            i=faces[:, 0],  # Triangle vertex indices
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"{channel_name} Massive Scale Surface",
            showscale=False,
            hovertemplate=f"<b>{channel_name} Massive Scale Surface</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold} (ultra-low)<br>" +
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
        print(f"⚠️ Could not create surface mesh for {channel_name}: {e}")
        print(f"   Data shape: {data.shape}, threshold: {threshold}")
        print(f"   Data range: {data.min():.3f} to {data.max():.3f}")
        return None

def create_massive_scale_surface_mesh_visualization(centrin_data, tubulin_data, dna_data):
    """Create massive scale 3D surface mesh visualization with original thresholds"""
    print(f"🎨 Creating massive scale surface mesh visualization...")
    
    # Create the figure
    fig = go.Figure()
    
    # Create surface meshes for each channel with ULTRA-LOW thresholds
    print(f"\n🔴 Processing Centrin channel at massive scale...")
    centrin_mesh = create_surface_mesh_original_thresholds(
        centrin_data, "Centrin", threshold=0.03, color='red', opacity=0.4  # Ultra-low threshold
    )
    
    print(f"\n🟢 Processing Tubulin channel at massive scale...")
    tubulin_mesh = create_surface_mesh_original_thresholds(
        tubulin_data, "Tubulin", threshold=0.03, color='lime', opacity=0.3  # Ultra-low threshold
    )
    
    print(f"\n🔵 Processing DNA channel at massive scale...")
    dna_mesh = create_surface_mesh_original_thresholds(
        dna_data, "DNA", threshold=0.03, color='dodgerblue', opacity=0.4  # Ultra-low threshold
    )
    
    # Add meshes to figure
    meshes_added = 0
    total_vertices = 0
    total_faces = 0
    
    if centrin_mesh:
        fig.add_trace(centrin_mesh)
        meshes_added += 1
        # Extract vertex count (approximate)
        centrin_count = len(centrin_mesh.x) if hasattr(centrin_mesh, 'x') else 0
        total_vertices += centrin_count
        print(f"✅ Added massive-scale Centrin surface mesh ({centrin_count:,} vertices)")
    
    if tubulin_mesh:
        fig.add_trace(tubulin_mesh)
        meshes_added += 1
        tubulin_count = len(tubulin_mesh.x) if hasattr(tubulin_mesh, 'x') else 0
        total_vertices += tubulin_count
        print(f"✅ Added massive-scale Tubulin surface mesh ({tubulin_count:,} vertices)")
    
    if dna_mesh:
        fig.add_trace(dna_mesh)
        meshes_added += 1
        dna_count = len(dna_mesh.x) if hasattr(dna_mesh, 'x') else 0
        total_vertices += dna_count
        print(f"✅ Added massive-scale DNA surface mesh ({dna_count:,} vertices)")
    
    print(f"📊 Total massive-scale surface meshes created: {meshes_added}")
    print(f"📊 Approximate total vertices: {total_vertices:,}")
    
    # Update layout for massive scale surface mesh visualization
    fig.update_layout(
        title={
            'text': "Massive Scale Ultra-Detail Surface Mesh: EMBL bmcc122 Multi-Channel 3D<br>" +
                   "<sub>🔴 Centrin (0.03) | 🟢 Tubulin (0.03) | 🔵 DNA (0.03) - Full dataset, ultra-low thresholds</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)", 
            zaxis_title="Z (slices)",
            camera=dict(
                eye=dict(x=3.0, y=3.0, z=3.0),  # Much further back for massive view
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
            aspectmode='cube',
            bgcolor='black'
        ),
        width=1600,  # Larger for massive scale
        height=1200,  # Larger for massive scale
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"Massive scale ultra-detail surface mesh (full dataset)<br>" +
                     f"Meshes generated: {meshes_added}/3 channels<br>" +
                     f"Ultra-low thresholds (0.03) reveal maximum detail<br>" +
                     f"Complete cellular architecture at unprecedented scale and sensitivity",
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

def main():
    """Main execution function"""
    print("🧬 EMBL OME-Zarr MASSIVE SCALE Ultra-Detail Surface Mesh Multi-Channel 3D Visualization")
    print("=" * 95)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract massive scale data for all channels
        centrin_data, tubulin_data, dna_data = extract_massive_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create massive scale combined surface mesh visualization
        fig = create_massive_scale_surface_mesh_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save massive scale surface mesh visualization
        output_file = os.path.join(output_dir, "surface_mesh_massive_scale.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Massive scale ultra-detail surface mesh visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This massive scale ultra-detail surface mesh visualization shows:")
        print(f"   🔴 Centrin (0.03 threshold): Complete centriole structures with maximum sensitivity")
        print(f"   🟢 Tubulin (0.03 threshold): Entire microtubule network with ultra-fine detail")
        print(f"   🔵 DNA (0.03 threshold): Full nuclear organization with maximum resolution")
        print(f"   📏 Massive scale: Full dataset (223×860×806 voxels)")
        print(f"   🔬 Ultra-low thresholds (0.03) reveal finest possible detail")
        print(f"   🌍 Shows complete cellular context with maximum sensitivity")
        print(f"   ⚡ May have extremely high polygon counts - ultimate detail visualization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
