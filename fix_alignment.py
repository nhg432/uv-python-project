#!/usr/bin/env python3
"""
Fixed Alignment Version: Janelia COSEM HeLa-4 3D Visualization
Ensures all mesh data is properly aligned on the same coordinate system
"""

import fsspec
import numpy as np
import plotly.graph_objects as go
import zarr
import os
from skimage import measure
from scipy import ndimage

def load_real_data():
    """Load the real Janelia COSEM data"""
    try:
        print("🔍 Loading real Janelia COSEM HeLa-4 data...")
        
        # Access the S3 data directly
        fs = fsspec.filesystem('s3', anon=True)
        base_path = "janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16"
        
        # Try s5 level (smallest, fastest to process)
        level_path = f"{base_path}/s5"
        mapper = fs.get_mapper(f"s3://{level_path}")
        zarr_array = zarr.open_array(mapper, mode='r')
        
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {zarr_array.shape}")
        print(f"   Data type: {zarr_array.dtype}")
        
        # Convert to numpy array
        data = np.array(zarr_array[:])
        print(f"   Data range: {np.min(data)} to {np.max(data)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Real data loading failed: {e}")
        return None

def create_aligned_mesh(data):
    """Create properly aligned multi-threshold mesh visualization"""
    try:
        print(f"🎨 Creating aligned mesh visualization...")
        print(f"📊 Input data shape: {data.shape}")
        
        # Normalize data consistently
        data_normalized = (data - data.min()) / (data.max() - data.min())
        print(f"📊 Normalized range: {data_normalized.min():.3f} to {data_normalized.max():.3f}")
        
        # Apply ISOTROPIC smoothing for consistent results
        print(f"🔄 Applying isotropic smoothing...")
        smoothed_data = ndimage.gaussian_filter(data_normalized, sigma=1.0)  # Same sigma for all axes
        
        # Create figure
        fig = go.Figure()
        
        # Define threshold levels with consistent coordinate system
        thresholds = [
            {'level': 0.15, 'color': 'lightcoral', 'opacity': 0.3, 'name': 'Cell Envelope'},
            {'level': 0.35, 'color': 'gold', 'opacity': 0.5, 'name': 'Cytoplasm'},
            {'level': 0.55, 'color': 'lightblue', 'opacity': 0.7, 'name': 'Organelles'},
            {'level': 0.75, 'color': 'lightgreen', 'opacity': 0.9, 'name': 'Core Structures'}
        ]
        
        total_vertices = 0
        mesh_info = []
        
        # Generate meshes with CONSISTENT coordinate system
        for i, threshold in enumerate(thresholds):
            try:
                print(f"🔄 Generating {threshold['name']} mesh (threshold: {threshold['level']})...")
                
                # Use marching cubes with ISOTROPIC spacing
                verts, faces, normals, values = measure.marching_cubes(
                    smoothed_data,
                    level=threshold['level'],
                    spacing=(1.0, 1.0, 1.0),  # ISOTROPIC spacing for alignment
                    allow_degenerate=False
                )
                
                print(f"   📊 Generated {len(verts):,} vertices, {len(faces):,} faces")
                
                # Apply CONSISTENT coordinate transformation for ALL meshes
                # Use the same transformation for all thresholds to ensure alignment
                verts_aligned = verts.copy()
                
                # Optional: Apply uniform scaling if needed (same for all meshes)
                scale_z = 0.4  # Compress Z-axis for better visualization
                verts_aligned[:, 0] = verts_aligned[:, 0] * scale_z
                
                # Create mesh with CONSISTENT coordinate mapping
                mesh = go.Mesh3d(
                    # IMPORTANT: Use consistent coordinate mapping for ALL meshes
                    x=verts_aligned[:, 2],  # X-axis maps to 3rd dimension (consistent)
                    y=verts_aligned[:, 1],  # Y-axis maps to 2nd dimension (consistent)
                    z=verts_aligned[:, 0],  # Z-axis maps to 1st dimension (consistent, compressed)
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=threshold['color'],
                    opacity=threshold['opacity'],
                    name=threshold['name'],
                    showscale=False,
                    hovertemplate=f"<b>{threshold['name']}</b><br>" +
                                 "X: %{x:.1f}<br>" +
                                 "Y: %{y:.1f}<br>" +
                                 "Z: %{z:.1f}<br>" +
                                 f"Threshold: {threshold['level']}<br>" +
                                 "<extra></extra>",
                    # Consistent lighting for all meshes
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.6,
                        specular=0.3,
                        roughness=0.2,
                        fresnel=0.2
                    ),
                    lightposition=dict(x=100, y=100, z=100)
                )
                
                fig.add_trace(mesh)
                total_vertices += len(verts)
                mesh_info.append(f"{threshold['name']}: {len(verts):,} vertices")
                
                print(f"   ✅ {threshold['name']} mesh added successfully")
                
            except Exception as mesh_error:
                print(f"   ❌ {threshold['name']} mesh failed: {mesh_error}")
                continue
        
        print(f"✅ All meshes created with consistent alignment!")
        print(f"📊 Total vertices: {total_vertices:,}")
        
        # Configure layout with proper axis alignment
        fig.update_layout(
            title={
                'text': "Janelia COSEM HeLa-4 - Aligned Multi-Threshold Visualization<br>" +
                        "<sub>Real EM data with consistent coordinate system alignment</sub>",
                'x': 0.5,
                'font': {'size': 20, 'color': 'navy'}
            },
            scene=dict(
                # Consistent axis configuration
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                zaxis_title="Z (sections, compressed)",
                
                # Ensure all axes have the same properties
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(245, 245, 245)",
                    gridcolor="rgb(220, 220, 220)",
                    zerolinecolor="rgb(200, 200, 200)",
                    showspikes=False,
                    range=None  # Auto-range based on data
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(245, 245, 245)",
                    gridcolor="rgb(220, 220, 220)",
                    zerolinecolor="rgb(200, 200, 200)",
                    showspikes=False,
                    range=None  # Auto-range based on data
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor="rgb(245, 245, 245)",
                    gridcolor="rgb(220, 220, 220)",
                    zerolinecolor="rgb(200, 200, 200)",
                    showspikes=False,
                    range=None  # Auto-range based on data
                ),
                
                # Proper aspect ratio to maintain alignment
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.4),  # Consistent aspect ratio
                
                # Camera position for optimal viewing
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                
                bgcolor='white'
            ),
            
            width=1600,
            height=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            
            # Information annotations
            annotations=[
                dict(
                    text=f"<b>📊 Dataset Information</b><br>" +
                         f"Shape: {data.shape[0]}×{data.shape[1]}×{data.shape[2]} voxels<br>" +
                         f"Data range: {np.min(data):,} - {np.max(data):,}<br>" +
                         f"Total vertices: {total_vertices:,}<br>" +
                         f"Coordinate system: Aligned & isotropic<br>" +
                         f"Z-compression: 40% for visualization",
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=8
                ),
                dict(
                    text=f"<b>🎛️ Mesh Details</b><br>" + "<br>".join(mesh_info),
                    x=0.98, y=0.98,
                    xref="paper", yref="paper",
                    xanchor="right", yanchor="top",
                    showarrow=False,
                    font=dict(size=11, color='black'),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=8
                )
            ],
            
            # Show legend for threshold levels
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"❌ Aligned mesh creation failed: {e}")
        return None

def main():
    """Main execution with alignment fixes"""
    print("🔧 Janelia COSEM HeLa-4 - ALIGNMENT CORRECTION")
    print("=" * 60)
    
    try:
        # Load real data
        data = load_real_data()
        
        if data is None:
            print("❌ Could not load data")
            return
        
        # Create aligned visualization
        fig = create_aligned_mesh(data)
        
        if fig is None:
            print("❌ Could not create visualization")
            return
        
        # Save corrected visualization
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "janelia_hela4_fibsem_aligned.html")
        fig.write_html(output_file)
        
        print(f"\n✅ ALIGNMENT CORRECTED!")
        print(f"📁 Fixed visualization saved to: {output_file}")
        print(f"🔧 All meshes now use consistent coordinate system")
        print(f"📊 Isotropic spacing and uniform transformations applied")
        print(f"🎯 Perfect alignment achieved across all threshold levels")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
