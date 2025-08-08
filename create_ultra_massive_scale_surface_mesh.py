#!/usr/bin/env python3
"""
Ultra-Massive Scale EMBL OME-Zarr Multi-Channel 3D Surface Mesh Visualization
This script creates 3D surface mesh renderings at ultra-massive scale:
- Region: Up to 2000x2000x2000 voxels 
- Tubulin (Channel 1) - Purple surface mesh (threshold 0.1)
- DNA (Channel 2) - Blue surface mesh (threshold 0.1)
- Centrin (Channel 0) - Yellow surface mesh (threshold 0.4)
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

def extract_ultra_massive_sample_data_all_channels(zarr_array, x_size=2200, y_size=2200, z_size=1500):
    """Extract ULTRA-MASSIVE sample data from zarr array for all channels - Custom X:2200, Y:2200, Z:1500"""
    try:
        print(f"\n📊 Extracting ULTRA-MASSIVE sample data for all channels...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full array shape: {zarr_array.shape}")
        
        # For ultra-massive scale with custom dimensions
        # Let's take the largest possible region from the dataset
        
        # Calculate bounds for ultra-massive region with custom dimensions
        z_max_size = min(z_size, z)
        y_max_size = min(y_size, y) 
        x_max_size = min(x_size, x)
        
        # Start from beginning and take as much as possible (full dataset likely)
        z_start = 0
        z_end = z_max_size
        y_start = 0
        y_end = y_max_size
        x_start = 0
        x_end = x_max_size
        
        # Ensure we don't exceed dataset bounds
        z_end = min(z_end, z)
        y_end = min(y_end, y)
        x_end = min(x_end, x)
            
        actual_z_size = z_end - z_start
        actual_y_size = y_end - y_start
        actual_x_size = x_end - x_start
        
        print(f"📦 Extracting ULTRA-MASSIVE region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        print(f"📏 ULTRA-MASSIVE custom dimensions: Z:{actual_z_size} x Y:{actual_y_size} x X:{actual_x_size}")
        print(f"🎯 Target was: Z:{z_size} x Y:{y_size} x X:{x_size}")
        
        # Extract data for all three channels - ULTRA-MASSIVE REGION
        print(f"🔄 Loading Centrin channel (ultra-massive scale, this will take time)...")
        centrin_data = zarr_array[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 0
        
        print(f"🔄 Loading Tubulin channel (ultra-massive scale, this will take time)...")
        tubulin_data = zarr_array[0, 1, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 1
        
        print(f"🔄 Loading DNA channel (ultra-massive scale, this will take time)...")
        dna_data = zarr_array[0, 2, z_start:z_end, y_start:y_end, x_start:x_end]       # Channel 2
        
        total_voxels = actual_z_size * actual_y_size * actual_x_size
        print(f"✅ All channels extracted! ULTRA-MASSIVE Shape: {centrin_data.shape}")
        print(f"📊 Total voxels processed: {total_voxels:,}")
        print(f"📈 Centrin range: {np.min(centrin_data):.3f} to {np.max(centrin_data):.3f}")
        print(f"📈 Tubulin range: {np.min(tubulin_data):.3f} to {np.max(tubulin_data):.3f}")
        print(f"📈 DNA range: {np.min(dna_data):.3f} to {np.max(dna_data):.3f}")
        
        return centrin_data, tubulin_data, dna_data
        
    except Exception as e:
        print(f"❌ Error extracting ultra-massive sample: {e}")
        return None, None, None

def create_surface_mesh_custom_thresholds(data, channel_name, threshold, color, opacity=0.4):
    """Create a 3D surface mesh from ultra-massive volumetric data with custom thresholds"""
    print(f"🔧 Creating surface mesh for {channel_name} at ultra-massive scale...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    print(f"📊 {channel_name} normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
    
    # Apply moderate Gaussian smoothing for ultra-massive data
    print(f"🔄 Applying smoothing to {channel_name} (this will take significant time for ultra-massive data)...")
    smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.5)
    
    try:
        # Create isosurface using marching cubes with CUSTOM thresholds
        print(f"🔄 Generating mesh for {channel_name} with custom threshold {threshold}...")
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ {channel_name}: Generated {len(verts):,} vertices, {len(faces):,} faces (ultra-massive scale)")
        
        # Create the mesh with optimized settings for ultra-massive data
        mesh = go.Mesh3d(
            x=verts[:, 2],  # X coordinates
            y=verts[:, 1],  # Y coordinates  
            z=verts[:, 0],  # Z coordinates
            i=faces[:, 0],  # Triangle vertex indices
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"{channel_name} Ultra-Massive Scale Surface",
            showscale=False,
            hovertemplate=f"<b>{channel_name} Ultra-Massive Scale Surface</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold} (custom)<br>" +
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
        print(f"   Smoothed data range: {smoothed_data.min():.3f} to {smoothed_data.max():.3f}")
        print(f"   Original data range: {data.min():.3f} to {data.max():.3f}")
        return None

def create_ultra_massive_scale_surface_mesh_visualization(centrin_data, tubulin_data, dna_data):
    """Create ultra-massive scale 3D surface mesh visualization with custom colors and thresholds"""
    print(f"🎨 Creating ultra-massive scale surface mesh visualization...")
    
    # Create the figure
    fig = go.Figure()
    
    # Create surface meshes for each channel with UPDATED thresholds and colors
    print(f"\n🟡 Processing Centrin channel at ultra-massive scale...")
    centrin_mesh = create_surface_mesh_custom_thresholds(
        centrin_data, "Centrin", threshold=0.1, color='gold', opacity=0.7  # Bright gold, reduced to 0.1 threshold
    )
    
    print(f"\n🟣 Processing Tubulin channel at ultra-massive scale...")
    tubulin_mesh = create_surface_mesh_custom_thresholds(
        tubulin_data, "Tubulin", threshold=0.1, color='purple', opacity=0.4  # Purple, kept at 0.1 threshold
    )
    
    print(f"\n🔵 Processing DNA channel at ultra-massive scale...")
    dna_mesh = create_surface_mesh_custom_thresholds(
        dna_data, "DNA", threshold=0.005, color='blue', opacity=0.5  # Blue, reduced to 0.005 threshold for maximum detail
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
        print(f"✅ Added ultra-massive-scale Centrin surface mesh ({centrin_count:,} vertices)")
    
    if tubulin_mesh:
        fig.add_trace(tubulin_mesh)
        meshes_added += 1
        tubulin_count = len(tubulin_mesh.x) if hasattr(tubulin_mesh, 'x') else 0
        total_vertices += tubulin_count
        print(f"✅ Added ultra-massive-scale Tubulin surface mesh ({tubulin_count:,} vertices)")
    
    if dna_mesh:
        fig.add_trace(dna_mesh)
        meshes_added += 1
        dna_count = len(dna_mesh.x) if hasattr(dna_mesh, 'x') else 0
        total_vertices += dna_count
        print(f"✅ Added ultra-massive-scale DNA surface mesh ({dna_count:,} vertices)")
    
    print(f"📊 Total ultra-massive-scale surface meshes created: {meshes_added}")
    print(f"📊 Approximate total vertices: {total_vertices:,}")
    
    # Update layout for ultra-massive scale surface mesh visualization
    fig.update_layout(
        title={
            'text': "Ultra-Massive Scale Enhanced Custom Surface Mesh: EMBL bmcc122 Multi-Channel 3D<br>" +
                   "<sub>🟡 Centrin (0.1) Bright Gold | 🟣 Tubulin (0.1) Purple | 🔵 DNA (0.005) Blue - Up to 2200³ voxels</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)", 
            zaxis_title="Z (slices)",
            camera=dict(
                eye=dict(x=3.5, y=3.5, z=3.5),  # Much further back for ultra-massive view
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
            aspectmode='cube',
            bgcolor='black'
        ),
        width=1800,  # Even larger for ultra-massive scale
        height=1400,  # Even larger for ultra-massive scale
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"Ultra-massive scale enhanced custom surface mesh (up to 2200³ voxels)<br>" +
                     f"Meshes generated: {meshes_added}/3 channels<br>" +
                     f"High-detail thresholds: Bright Gold Centrin (0.1), Purple Tubulin (0.1), Blue DNA (0.005)<br>" +
                     f"Shows complete cellular architecture with maximum detail sensitivity",
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
    print("🧬 EMBL OME-Zarr ULTRA-MASSIVE SCALE Custom Surface Mesh Multi-Channel 3D Visualization")
    print("=" * 100)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract ultra-massive scale data for all channels
        centrin_data, tubulin_data, dna_data = extract_ultra_massive_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create ultra-massive scale combined surface mesh visualization
        fig = create_ultra_massive_scale_surface_mesh_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ultra-massive scale surface mesh visualization
        output_file = os.path.join(output_dir, "surface_mesh_massive_scale.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Ultra-massive scale enhanced custom surface mesh visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This ultra-massive scale enhanced custom surface mesh visualization shows:")
        print(f"   🟡 Centrin (0.1 threshold): Maximum centriole detail in bright gold")
        print(f"   🟣 Tubulin (0.1 threshold): Detailed microtubule network in vibrant purple")
        print(f"   🔵 DNA (0.005 threshold): Ultra-detailed nuclear structures in classic blue")
        print(f"   📏 Ultra-massive scale: Up to 2200x2200x2200 voxels")
        print(f"   🎨 High-detail thresholds for maximum structural visibility")
        print(f"   🌍 Shows complete cellular context with ultra-high sensitivity")
        print(f"   ⚡ Ultra-low thresholds reveal finest possible structural detail")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
