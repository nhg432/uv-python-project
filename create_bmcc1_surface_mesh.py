#!/usr/bin/env python3
"""
BMCC1 EMBL OME-Zarr Multi-Channel 3D Surface Mesh Visualization
This script creates 3D surface mesh renderings for the BMCC1 dataset:
- Dataset: bmcc1_pfa_cetn-tub-dna_20231208_cl.ome.zarr
- Tubulin (Channel 1) - Purple surface mesh
- DNA (Channel 2) - Blue surface mesh
- Centrin (Channel 0) - Gold surface mesh
"""

import zarr
import numpy as np
import plotly.graph_objects as go
import fsspec
import os
from skimage import measure
from scipy import ndimage

def load_bmcc1_dataset():
    """Load BMCC1 OME-Zarr dataset from EMBL S3"""
    try:
        # Direct HTTP access to EMBL S3 data for BMCC1
        base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc1_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
        
        print(f"🔗 Connecting to BMCC1 dataset...")
        print(f"📍 Dataset: bmcc1_pfa_cetn-tub-dna_20231208_cl.ome.zarr")
        
        # Use HTTP filesystem
        fs = fsspec.filesystem('http')
        
        # Try to access the highest resolution data (0/0)
        zarr_path = f"{base_url}/0/0"
        
        print(f"📥 Loading array from: {zarr_path}")
        
        # Open the zarr array
        zarr_array = zarr.open_array(zarr_path, mode='r')
        
        print(f"✅ Successfully loaded BMCC1 array!")
        print(f"Shape: {zarr_array.shape}")
        print(f"Data type: {zarr_array.dtype}")
        
        return zarr_array
        
    except Exception as e:
        print(f"❌ Error loading BMCC1 dataset: {e}")
        return None

def extract_bmcc1_sample_data_all_channels(zarr_array, x_size=2200, y_size=2200, z_size=2200):
    """Extract sample data from BMCC1 zarr array for all channels - Enhanced scale 2200x2200x2200"""
    try:
        print(f"\n📊 Extracting BMCC1 sample data for all channels at enhanced scale...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full BMCC1 array shape: {zarr_array.shape}")
        
        # For BMCC1, use the full dataset as it's smaller (96, 330, 388)
        # But we'll scale the visualization to appear larger
        z_start = 0
        z_end = z  # Use full Z dimension
        y_start = 0
        y_end = y  # Use full Y dimension
        x_start = 0
        x_end = x  # Use full X dimension
            
        actual_z_size = z_end - z_start
        actual_y_size = y_end - y_start
        actual_x_size = x_end - x_start
        
        print(f"📦 Extracting BMCC1 FULL region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        print(f"📏 BMCC1 actual dimensions: Z:{actual_z_size} x Y:{actual_y_size} x X:{actual_x_size}")
        print(f"🎯 Enhanced rendering scale target: Z:{z_size} x Y:{y_size} x X:{x_size}")
        
        # Extract data for all three channels - Load in smaller chunks to avoid timeout
        print(f"🔄 Loading BMCC1 Centrin channel for enhanced scale rendering...")
        centrin_data = np.array(zarr_array[0, 0, :, :, :])  # Channel 0 - full dataset
        
        print(f"🔄 Loading BMCC1 Tubulin channel for enhanced scale rendering...")
        tubulin_data = np.array(zarr_array[0, 1, :, :, :])  # Channel 1 - full dataset
        
        print(f"🔄 Loading BMCC1 DNA channel for enhanced scale rendering...")
        dna_data = np.array(zarr_array[0, 2, :, :, :])       # Channel 2 - full dataset
        
        total_voxels = actual_z_size * actual_y_size * actual_x_size
        print(f"✅ All BMCC1 channels extracted! Shape: {centrin_data.shape}")
        print(f"📊 Total voxels processed: {total_voxels:,}")
        print(f"📈 BMCC1 Centrin range: {np.min(centrin_data):.3f} to {np.max(centrin_data):.3f}")
        print(f"📈 BMCC1 Tubulin range: {np.min(tubulin_data):.3f} to {np.max(tubulin_data):.3f}")
        print(f"📈 BMCC1 DNA range: {np.min(dna_data):.3f} to {np.max(dna_data):.3f}")
        
        return centrin_data, tubulin_data, dna_data
        
    except Exception as e:
        print(f"❌ Error extracting BMCC1 sample: {e}")
        return None, None, None

def create_bmcc1_surface_mesh(data, channel_name, threshold, color, opacity=0.4):
    """Create a 3D surface mesh from BMCC1 volumetric data with enhanced scale"""
    print(f"🔧 Creating BMCC1 enhanced scale surface mesh for {channel_name}...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    print(f"📊 BMCC1 {channel_name} normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
    
    # Apply Gaussian smoothing for better surface quality
    print(f"🔄 Applying enhanced smoothing to BMCC1 {channel_name}...")
    smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.2)  # Slightly reduced sigma for more detail
    
    try:
        # Create isosurface using marching cubes
        print(f"🔄 Generating BMCC1 enhanced mesh for {channel_name} with threshold {threshold}...")
        verts, faces, normals, values = measure.marching_cubes(
            smoothed_data, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0),  # Keep original spacing, scaling applied to coordinates
            allow_degenerate=False
        )
        
        print(f"✅ BMCC1 {channel_name}: Generated {len(verts):,} vertices, {len(faces):,} faces")
        
        # Apply enhanced scaling and Z-axis flattening
        verts_scaled = verts.copy()
        
        # Scale coordinates to enhance the rendering (simulate 2200x2200x2200 appearance)
        scale_factor_x = 2200 / data.shape[2]  # X scaling
        scale_factor_y = 2200 / data.shape[1]  # Y scaling  
        scale_factor_z = 2200 / data.shape[0]  # Z scaling
        
        verts_scaled[:, 0] = verts_scaled[:, 0] * scale_factor_z * 0.2  # Z with flattening
        verts_scaled[:, 1] = verts_scaled[:, 1] * scale_factor_y        # Y
        verts_scaled[:, 2] = verts_scaled[:, 2] * scale_factor_x        # X
        
        print(f"🔄 BMCC1 Enhanced scaling applied:")
        print(f"   📏 Scale factors: X={scale_factor_x:.1f}, Y={scale_factor_y:.1f}, Z={scale_factor_z:.1f}")
        print(f"   📐 Original coords: Z:{verts[:, 0].min():.1f}-{verts[:, 0].max():.1f}")
        print(f"   📐 Scaled coords: Z:{verts_scaled[:, 0].min():.1f}-{verts_scaled[:, 0].max():.1f}")
        
        # Create the mesh with enhanced properties
        mesh = go.Mesh3d(
            x=verts_scaled[:, 2],  # X coordinates (enhanced scale)
            y=verts_scaled[:, 1],  # Y coordinates (enhanced scale)
            z=verts_scaled[:, 0],  # Z coordinates (enhanced scale + flattened)
            i=faces[:, 0],  # Triangle vertex indices
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=f"BMCC1 {channel_name} Enhanced Scale Surface",
            showscale=False,
            hovertemplate=f"<b>BMCC1 {channel_name} Enhanced Scale</b><br>" +
                         "X: %{x}<br>" +
                         "Y: %{y}<br>" +
                         "Z: %{z}<br>" +
                         f"Threshold: {threshold}<br>" +
                         f"Enhanced Scale: 2200x2200x2200<br>" +
                         "<extra></extra>",
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,   # Increased diffuse for better visibility
                specular=0.5,  # Increased specular for enhanced appearance
                roughness=0.2, # Reduced roughness for smoother look
                fresnel=0.3
            ),
            lightposition=dict(x=150, y=150, z=150)  # Enhanced lighting position
        )
        
        return mesh
        
    except Exception as e:
        print(f"⚠️ Could not create BMCC1 enhanced surface mesh for {channel_name}: {e}")
        print(f"   Data shape: {data.shape}, threshold: {threshold}")
        print(f"   Smoothed data range: {smoothed_data.min():.3f} to {smoothed_data.max():.3f}")
        return None

def create_bmcc1_visualization(centrin_data, tubulin_data, dna_data):
    """Create BMCC1 3D surface mesh visualization with enhanced scale 2200x2200x2200"""
    print(f"🎨 Creating BMCC1 enhanced scale surface mesh visualization...")
    
    # Create the figure
    fig = go.Figure()
    
    # Create surface meshes for each channel with optimized thresholds for enhanced scale
    print(f"\n🟡 Processing BMCC1 Centrin channel at enhanced scale...")
    centrin_mesh = create_bmcc1_surface_mesh(
        centrin_data, "Centrin", threshold=0.08, color='gold', opacity=0.7
    )
    
    print(f"\n🟣 Processing BMCC1 Tubulin channel at enhanced scale...")
    tubulin_mesh = create_bmcc1_surface_mesh(
        tubulin_data, "Tubulin", threshold=0.1, color='purple', opacity=0.4
    )
    
    print(f"\n🔵 Processing BMCC1 DNA channel at enhanced scale...")
    dna_mesh = create_bmcc1_surface_mesh(
        dna_data, "DNA", threshold=0.2, color='blue', opacity=0.5  # Increased threshold to 0.2
    )
    
    # Add meshes to figure
    meshes_added = 0
    total_vertices = 0
    
    if centrin_mesh:
        fig.add_trace(centrin_mesh)
        meshes_added += 1
        centrin_count = len(centrin_mesh.x) if hasattr(centrin_mesh, 'x') else 0
        total_vertices += centrin_count
        print(f"✅ Added BMCC1 Centrin enhanced scale surface mesh ({centrin_count:,} vertices)")
    
    if tubulin_mesh:
        fig.add_trace(tubulin_mesh)
        meshes_added += 1
        tubulin_count = len(tubulin_mesh.x) if hasattr(tubulin_mesh, 'x') else 0
        total_vertices += tubulin_count
        print(f"✅ Added BMCC1 Tubulin enhanced scale surface mesh ({tubulin_count:,} vertices)")
    
    if dna_mesh:
        fig.add_trace(dna_mesh)
        meshes_added += 1
        dna_count = len(dna_mesh.x) if hasattr(dna_mesh, 'x') else 0
        total_vertices += dna_count
        print(f"✅ Added BMCC1 DNA enhanced scale surface mesh ({dna_count:,} vertices)")
    
    print(f"📊 Total BMCC1 enhanced scale surface meshes created: {meshes_added}")
    print(f"📊 Approximate total vertices: {total_vertices:,}")
    
    # Update layout for enhanced scale visualization
    fig.update_layout(
        title={
            'text': "BMCC1 Enhanced Scale 3D Surface Mesh: EMBL Multi-Channel Visualization<br>" +
                   "<sub>🟡 Centrin (0.08) Gold | 🟣 Tubulin (0.1) Purple | 🔵 DNA (0.2) Blue - Enhanced 2200³ Scale</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels - Enhanced Scale)",
            yaxis_title="Y (pixels - Enhanced Scale)", 
            zaxis_title="Z (slices - Enhanced Scale)",
            camera=dict(
                eye=dict(x=4.0, y=4.0, z=4.0),  # Further back for enhanced scale view
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
            aspectratio=dict(x=1, y=1, z=0.2),  # Enhanced flattened Z for better viewing
            bgcolor='black'
        ),
        width=1800,  # Larger for enhanced scale
        height=1400,  # Larger for enhanced scale
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"BMCC1 Enhanced Scale (2200³) surface mesh visualization<br>" +
                     f"Meshes generated: {meshes_added}/3 channels<br>" +
                     f"Enhanced thresholds: Gold Centrin (0.08), Purple Tubulin (0.1), Blue DNA (0.2)<br>" +
                     f"Original dataset: 96×330×388 → Enhanced scale: 2200×2200×2200<br>" +
                     f"Dataset: bmcc1_pfa_cetn-tub-dna_20231208_cl.ome.zarr",
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
    """Main execution function for BMCC1 visualization"""
    print("🧬 BMCC1 EMBL OME-Zarr Multi-Channel 3D Surface Mesh Visualization")
    print("=" * 80)
    
    try:
        # Load BMCC1 dataset
        zarr_array = load_bmcc1_dataset()
        if zarr_array is None:
            return
        
        # Extract data for all channels
        centrin_data, tubulin_data, dna_data = extract_bmcc1_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create visualization
        fig = create_bmcc1_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        output_file = os.path.join(output_dir, "bmcc1_surface_mesh.html")
        fig.write_html(output_file)
        
        print(f"\n✅ BMCC1 surface mesh visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This BMCC1 enhanced scale visualization shows:")
        print(f"   🟡 Centrin (0.08 threshold): Centriole structures in gold")
        print(f"   🟣 Tubulin (0.1 threshold): Microtubule network in purple")
        print(f"   🔵 DNA (0.2 threshold): Core nuclear structures in blue")
        print(f"   📏 Enhanced scale: 2200×2200×2200 with Z-axis flattening")
        print(f"   🎨 Optimized thresholds for BMCC1 dataset characteristics")
        print(f"   🌍 Complete cellular architecture at enhanced rendering scale")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
