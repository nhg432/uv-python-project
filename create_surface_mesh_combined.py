#!/usr/bin/env python3
"""
Enhanced Surface Mesh EMBL OME-Zarr Multi-Channel 3D Visualization
This script creates 3D surface mesh renderings with enhanced sensitivity:
- Tubulin (Channel 1) - Green surface mesh (threshold 0.1)
- DNA (Channel 2) - Blue surface mesh (threshold 0.2)
- Centrin (Channel 0) - Red surface mesh (threshold 0.15)
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

def extract_sample_data_all_channels(zarr_array, sample_size=80):
    """Extract sample data from zarr array for all channels"""
    try:
        print(f"\n📊 Extracting sample data for all channels...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full array shape: {zarr_array.shape}")
        
        # Calculate center region for sampling
        z_center, y_center, x_center = z // 2, y // 2, x // 2
        half_size = sample_size // 2
        
        # Define bounds
        z_start = max(0, z_center - half_size)
        z_end = min(z, z_center + half_size)
        y_start = max(0, y_center - half_size)
        y_end = min(y, y_center + half_size)
        x_start = max(0, x_center - half_size)
        x_end = min(x, x_center + half_size)
        
        print(f"📦 Extracting region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        
        # Extract data for all three channels
        centrin_data = zarr_array[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 0
        tubulin_data = zarr_array[0, 1, z_start:z_end, y_start:y_end, x_start:x_end]  # Channel 1
        dna_data = zarr_array[0, 2, z_start:z_end, y_start:y_end, x_start:x_end]       # Channel 2
        
        print(f"✅ All channels extracted! Shape: {centrin_data.shape}")
        print(f"📈 Centrin range: {np.min(centrin_data):.3f} to {np.max(centrin_data):.3f}")
        print(f"📈 Tubulin range: {np.min(tubulin_data):.3f} to {np.max(tubulin_data):.3f}")
        print(f"📈 DNA range: {np.min(dna_data):.3f} to {np.max(dna_data):.3f}")
        
        return centrin_data, tubulin_data, dna_data
        
    except Exception as e:
        print(f"❌ Error extracting sample: {e}")
        return None, None, None

def create_surface_mesh(data, channel_name, threshold, color, opacity=0.3):
    """Create a 3D surface mesh from volumetric data"""
    print(f"🔧 Creating surface mesh for {channel_name}...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Apply Gaussian smoothing to reduce noise
    smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.0)
    
    try:
        # Create isosurface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        
        print(f"✅ {channel_name}: Generated {len(verts)} vertices, {len(faces)} faces")
        
        # Create the mesh
        mesh = go.Mesh3d(
            x=verts[:, 2],  # X coordinates
            y=verts[:, 1],  # Y coordinates  
            z=verts[:, 0],  # Z coordinates
            i=faces[:, 0],  # Triangle vertex indices
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"{channel_name} Surface",
            showscale=False,
            hovertemplate=f"<b>{channel_name} Surface</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold}<br>" +
                         "<extra></extra>",
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.5,
                roughness=0.2,
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

def create_combined_surface_mesh_visualization(centrin_data, tubulin_data, dna_data):
    """Create combined 3D surface mesh visualization"""
    print(f"🎨 Creating combined surface mesh visualization...")
    
    # Create the figure
    fig = go.Figure()
    
    # Create surface meshes for each channel with enhanced thresholds
    print(f"\n🔴 Processing Centrin channel...")
    centrin_mesh = create_surface_mesh(
        centrin_data, "Centrin", threshold=0.15, color='red', opacity=0.6
    )
    
    print(f"\n🟢 Processing Tubulin channel...")
    tubulin_mesh = create_surface_mesh(
        tubulin_data, "Tubulin", threshold=0.1, color='lime', opacity=0.4
    )
    
    print(f"\n🔵 Processing DNA channel...")
    dna_mesh = create_surface_mesh(
        dna_data, "DNA", threshold=0.2, color='dodgerblue', opacity=0.5
    )
    
    # Add meshes to figure
    meshes_added = 0
    
    if centrin_mesh:
        fig.add_trace(centrin_mesh)
        meshes_added += 1
        print(f"✅ Added Centrin surface mesh")
    
    if tubulin_mesh:
        fig.add_trace(tubulin_mesh)
        meshes_added += 1
        print(f"✅ Added Tubulin surface mesh")
    
    if dna_mesh:
        fig.add_trace(dna_mesh)
        meshes_added += 1
        print(f"✅ Added DNA surface mesh")
    
    print(f"📊 Total surface meshes created: {meshes_added}")
    
    # Update layout for surface mesh visualization
    fig.update_layout(
        title={
            'text': "Enhanced Surface Mesh: EMBL bmcc122 Multi-Channel 3D<br>" +
                   "<sub>🔴 Centrin (0.15) | 🟢 Tubulin (0.1) | 🔵 DNA (0.2) - Surface mesh rendering</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)", 
            zaxis_title="Z (slices)",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(
                showbackground=True, 
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                zerolinecolor="white"
            ),
            yaxis=dict(
                showbackground=True, 
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                zerolinecolor="white"
            ),
            zaxis=dict(
                showbackground=True, 
                backgroundcolor="rgb(230, 230, 230)",
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
                text=f"Surface mesh rendering with enhanced sensitivity<br>" +
                     f"Meshes generated: {meshes_added}/3 channels<br>" +
                     "Lower thresholds reveal detailed cellular architecture",
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
    print("🧬 EMBL OME-Zarr Enhanced Surface Mesh Multi-Channel 3D Visualization")
    print("=" * 75)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract all channel sample data
        centrin_data, tubulin_data, dna_data = extract_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create combined surface mesh visualization
        fig = create_combined_surface_mesh_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save surface mesh visualization
        output_file = os.path.join(output_dir, "surface_mesh_combined_enhanced.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Enhanced surface mesh visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This surface mesh visualization shows:")
        print(f"   🔴 Centrin (0.15 threshold): Centriole structure as red surface")
        print(f"   🟢 Tubulin (0.1 threshold): Microtubule network as green surface")
        print(f"   🔵 DNA (0.2 threshold): Nuclear structure as blue surface")
        print(f"   ✨ Continuous 3D surfaces reveal cellular architecture shape and volume")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
