#!/usr/bin/env python3
"""
Ultra-Enhanced Surface Mesh EMBL OME-Zarr Multi-Channel 3D Visualization
This script creates 3D surface mesh renderings with ultra-low thresholds for maximum detail:
- Tubulin (Channel 1) - Green surface mesh (threshold 0.05)
- DNA (Channel 2) - Blue surface mesh (threshold 0.1)
- Centrin (Channel 0) - Red surface mesh (threshold 0.08)
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

def extract_sample_data_all_channels(zarr_array, sample_size=140):
    """Extract larger sample data from zarr array for all channels to show more of the cell"""
    try:
        print(f"\n📊 Extracting EXPANDED sample data for all channels...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full array shape: {zarr_array.shape}")
        
        # Calculate center region for sampling - EXPANDED
        z_center, y_center, x_center = z // 2, y // 2, x // 2
        half_size = sample_size // 2
        
        # Define bounds
        z_start = max(0, z_center - half_size)
        z_end = min(z, z_center + half_size)
        y_start = max(0, y_center - half_size)
        y_end = min(y, y_center + half_size)
        x_start = max(0, x_center - half_size)
        x_end = min(x, x_center + half_size)
        
        print(f"📦 Extracting EXPANDED region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        print(f"📏 Expanded dimensions: {z_end-z_start} x {y_end-y_start} x {x_end-x_start} (was 80x80x80)")
        
        # Extract data for all three channels - LARGER REGION
        centrin_data = zarr_array[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 0
        tubulin_data = zarr_array[0, 1, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 1
        dna_data = zarr_array[0, 2, z_start:z_end, y_start:y_end, x_start:x_end]       # Channel 2
        
        print(f"✅ All channels extracted! EXPANDED Shape: {centrin_data.shape}")
        print(f"📈 Centrin range: {np.min(centrin_data):.3f} to {np.max(centrin_data):.3f}")
        print(f"📈 Tubulin range: {np.min(tubulin_data):.3f} to {np.max(tubulin_data):.3f}")
        print(f"📈 DNA range: {np.min(dna_data):.3f} to {np.max(dna_data):.3f}")
        
        return centrin_data, tubulin_data, dna_data
        
    except Exception as e:
        print(f"❌ Error extracting sample: {e}")
        return None, None, None

def create_surface_mesh(data, channel_name, threshold, color, opacity=0.3):
    """Create a 3D surface mesh from volumetric data with ultra-high sensitivity"""
    print(f"🔧 Creating ultra-detailed surface mesh for {channel_name}...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Apply lighter Gaussian smoothing to preserve more detail
    smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=0.7)
    
    try:
        # Create isosurface using marching cubes with ultra-low threshold
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ {channel_name}: Generated {len(verts)} vertices, {len(faces)} faces (ultra-detail)")
        
        # Create the mesh with enhanced lighting for ultra-detail
        mesh = go.Mesh3d(
            x=verts[:, 2],  # X coordinates
            y=verts[:, 1],  # Y coordinates  
            z=verts[:, 0],  # Z coordinates
            i=faces[:, 0],  # Triangle vertex indices
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"{channel_name} Ultra-Detail Surface",
            showscale=False,
            hovertemplate=f"<b>{channel_name} Ultra-Detail Surface</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold} (ultra-low)<br>" +
                         "<extra></extra>",
            lighting=dict(
                ambient=0.5,
                diffuse=0.9,
                specular=0.6,
                roughness=0.1,
                fresnel=0.3
            ),
            lightposition=dict(x=100, y=100, z=100)
        )
        
        return mesh
        
    except Exception as e:
        print(f"⚠️ Could not create surface mesh for {channel_name}: {e}")
        print(f"   Data shape: {data.shape}, threshold: {threshold}")
        print(f"   Data range: {data.min():.3f} to {data.max():.3f}")
        return None

def create_ultra_detailed_surface_mesh_visualization(centrin_data, tubulin_data, dna_data):
    """Create ultra-detailed combined 3D surface mesh visualization"""
    print(f"🎨 Creating ultra-detailed surface mesh visualization...")
    
    # Create the figure
    fig = go.Figure()
    
    # Create surface meshes for each channel with ULTRA-LOW thresholds
    print(f"\n🔴 Processing Centrin channel with ultra-low threshold...")
    centrin_mesh = create_surface_mesh(
        centrin_data, "Centrin", threshold=0.08, color='red', opacity=0.5  # Much lower than 0.15
    )
    
    print(f"\n🟢 Processing Tubulin channel with ultra-low threshold...")
    tubulin_mesh = create_surface_mesh(
        tubulin_data, "Tubulin", threshold=0.05, color='lime', opacity=0.3  # Much lower than 0.1
    )
    
    print(f"\n🔵 Processing DNA channel with ultra-low threshold...")
    dna_mesh = create_surface_mesh(
        dna_data, "DNA", threshold=0.1, color='dodgerblue', opacity=0.4  # Lower than 0.2
    )
    
    # Add meshes to figure
    meshes_added = 0
    
    if centrin_mesh:
        fig.add_trace(centrin_mesh)
        meshes_added += 1
        print(f"✅ Added ultra-detailed Centrin surface mesh")
    
    if tubulin_mesh:
        fig.add_trace(tubulin_mesh)
        meshes_added += 1
        print(f"✅ Added ultra-detailed Tubulin surface mesh")
    
    if dna_mesh:
        fig.add_trace(dna_mesh)
        meshes_added += 1
        print(f"✅ Added ultra-detailed DNA surface mesh")
    
    print(f"📊 Total ultra-detailed surface meshes created: {meshes_added}")
    
    # Update layout for ultra-detailed surface mesh visualization
    fig.update_layout(
        title={
            'text': "Expanded Ultra-Detail Surface Mesh: EMBL bmcc122 Multi-Channel 3D<br>" +
                   "<sub>🔴 Centrin (0.08) | 🟢 Tubulin (0.05) | 🔵 DNA (0.1) - Expanded 140x140x140 region</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)", 
            zaxis_title="Z (slices)",
            camera=dict(
                eye=dict(x=2.5, y=2.5, z=2.5),  # Further back for expanded view
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
        width=1400,
        height=1000,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"Expanded ultra-detailed surface mesh (140x140x140 region)<br>" +
                     f"Meshes generated: {meshes_added}/3 channels<br>" +
                     "Maximum sensitivity + expanded scale reveals more cellular context<br>" +
                     "Shows broader cellular architecture and organelle relationships",
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
    print("🧬 EMBL OME-Zarr Expanded Ultra-Detailed Surface Mesh Multi-Channel 3D Visualization")
    print("=" * 85)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract all channel sample data - EXPANDED REGION
        centrin_data, tubulin_data, dna_data = extract_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create ultra-detailed combined surface mesh visualization
        fig = create_ultra_detailed_surface_mesh_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ultra-detailed surface mesh visualization
        output_file = os.path.join(output_dir, "surface_mesh_ultra_detail.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Expanded ultra-detailed surface mesh visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This expanded ultra-detailed surface mesh visualization shows:")
        print(f"   🔴 Centrin (0.08 threshold): Maximum centriole detail across expanded region")
        print(f"   🟢 Tubulin (0.05 threshold): Complete microtubule network in broader context")
        print(f"   🔵 DNA (0.1 threshold): Extended nuclear architecture and organization")
        print(f"   📏 Expanded scale: 140x140x140 voxels (vs 80x80x80) shows more cellular context")
        print(f"   ✨ Ultra-low thresholds + expanded view = comprehensive cellular architecture")
        print(f"   🎯 Reveals broader organelle relationships and spatial organization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
