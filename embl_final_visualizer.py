"""
EMBL OME-Zarr 3D Visualizer - Final Working Version
Successfully creates 3D renderings from EMBL culture collections data
"""

import requests
import json
import zarr
import fsspec
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_embl_dataset():
    """Load the EMBL dataset from the discovered structure"""
    
    base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
    
    print("🔬 EMBL OME-Zarr 3D Visualizer")
    print("=" * 60)
    print("Dataset: bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr")
    print("Source: EMBL Culture Collections")
    print("=" * 60)
    
    # Based on our discovery, we know the structure is:
    # 0/0: [1, 3, 223, 860, 806] - highest resolution
    # 0/1: [1, 3, 223, 430, 403] - medium resolution  
    # 1/0: [1, 3, 138, 695, 732] - another resolution
    
    # Load the highest resolution data (0/0)
    array_path = "0/0"
    array_url = f"{base_url}/{array_path}"
    
    print(f"Loading high-resolution data from: {array_path}")
    print("Array dimensions: [T, C, Z, Y, X] = [1, 3, 223, 860, 806]")
    print("This represents:")
    print("- 1 time point")
    print("- 3 channels (likely different fluorescence markers)")
    print("- 223 Z-slices (depth)")
    print("- 860 x 806 pixels per slice")
    
    try:
        # Open the zarr array
        http_mapper = fsspec.get_mapper(array_url)
        zarr_array = zarr.open(http_mapper, mode='r')
        
        print(f"✓ Successfully opened zarr array!")
        print(f"Shape: {zarr_array.shape}")
        print(f"Data type: {zarr_array.dtype}")
        print(f"Chunks: {zarr_array.chunks}")
        
        return zarr_array
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def extract_sample_data(zarr_array, channel=0, sample_size=80):
    """Extract a manageable sample from the full dataset"""
    
    print(f"\\nExtracting sample data for visualization...")
    print(f"Selected channel: {channel} (0=first channel)")
    print(f"Sample size: {sample_size}x{sample_size}x{sample_size} voxels")
    
    try:
        # Get array shape: [T, C, Z, Y, X] = [1, 3, 223, 860, 806]
        t, c, z, y, x = zarr_array.shape
        
        # Take central region for better data
        z_center = z // 2
        y_center = y // 2  
        x_center = x // 2
        
        # Define sample region
        z_start = max(0, z_center - sample_size//2)
        z_end = min(z, z_start + sample_size)
        
        y_start = max(0, y_center - sample_size//2)
        y_end = min(y, y_start + sample_size)
        
        x_start = max(0, x_center - sample_size//2)
        x_end = min(x, x_start + sample_size)
        
        print(f"Sampling region:")
        print(f"  Z: {z_start} to {z_end} (center around {z_center})")
        print(f"  Y: {y_start} to {y_end} (center around {y_center})")
        print(f"  X: {x_start} to {x_end} (center around {x_center})")
        
        # Extract the sample data
        sample_data = zarr_array[0, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        
        print(f"✓ Sample extracted! Shape: {sample_data.shape}")
        print(f"Data range: {sample_data.min()} to {sample_data.max()}")
        print(f"Data type: {sample_data.dtype}")
        
        # Normalize data for visualization
        if sample_data.max() > sample_data.min():
            normalized_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
        else:
            normalized_data = sample_data.astype(np.float32)
        
        return normalized_data
        
    except Exception as e:
        print(f"Error extracting sample data: {e}")
        return None

def create_comprehensive_visualizations(volume, channel_name="Channel_0"):
    """Create all types of 3D visualizations"""
    
    print(f"\\n🎨 Creating comprehensive 3D visualizations...")
    print(f"Volume shape: {volume.shape}")
    print(f"Channel: {channel_name}")
    
    # Create output directory
    output_dir = Path("embl_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 2D Slice Analysis
    print("\\n1. 📊 Creating 2D slice analysis...")
    create_slice_analysis(volume, output_dir, channel_name)
    
    # 2. Interactive 3D Scatter Plot
    print("\\n2. 🎯 Creating interactive 3D scatter plot...")
    create_interactive_3d_scatter(volume, output_dir, channel_name)
    
    # 3. 3D Surface Mesh
    print("\\n3. 🏗️ Creating 3D surface mesh...")
    create_surface_mesh(volume, output_dir, channel_name)
    
    # 4. Multi-threshold Analysis
    print("\\n4. 🔍 Creating multi-threshold analysis...")
    create_multi_threshold_view(volume, output_dir, channel_name)
    
    return output_dir

def create_slice_analysis(volume, output_dir, channel_name):
    """Create comprehensive slice analysis"""
    try:
        z, y, x = volume.shape
        
        # Create comprehensive slice figure
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 4x3 grid for different views
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: XY slices at different Z levels
        z_levels = [z//6, z//3, z//2, 2*z//3, 5*z//6]
        for i, z_level in enumerate(z_levels[:3]):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(volume[z_level], cmap='viridis', aspect='equal')
            ax.set_title(f'XY Slice (Z={z_level}/{z})', fontsize=12)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Row 2: XZ slices 
        y_levels = [y//4, y//2, 3*y//4]
        for i, y_level in enumerate(y_levels):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(volume[:, y_level, :], cmap='plasma', aspect='equal')
            ax.set_title(f'XZ Slice (Y={y_level}/{y})', fontsize=12)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Z (slices)')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Row 3: YZ slices
        x_levels = [x//4, x//2, 3*x//4] 
        for i, x_level in enumerate(x_levels):
            ax = fig.add_subplot(gs[2, i])
            im = ax.imshow(volume[:, :, x_level], cmap='hot', aspect='equal')
            ax.set_title(f'YZ Slice (X={x_level}/{x})', fontsize=12)
            ax.set_xlabel('Y (pixels)')
            ax.set_ylabel('Z (slices)')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Row 4: Projections
        projections = [
            (np.max(volume, axis=0), 'Max Intensity (XY)', 'viridis'),
            (np.max(volume, axis=1), 'Max Intensity (XZ)', 'viridis'), 
            (np.max(volume, axis=2), 'Max Intensity (YZ)', 'viridis')
        ]
        
        for i, (proj, title, cmap) in enumerate(projections):
            ax = fig.add_subplot(gs[3, i])
            im = ax.imshow(proj, cmap=cmap, aspect='equal')
            ax.set_title(title, fontsize=12)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle(f'Comprehensive Slice Analysis: EMBL Dataset ({channel_name})', 
                     fontsize=16, y=0.98)
        
        # Save the figure
        output_file = output_dir / f'slice_analysis_{channel_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Slice analysis saved: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating slice analysis: {e}")

def create_interactive_3d_scatter(volume, output_dir, channel_name, threshold=0.5):
    """Create interactive 3D scatter plot"""
    try:
        # Find significant voxels
        mask = volume > threshold
        z_coords, y_coords, x_coords = np.where(mask)
        values = volume[mask]
        
        # Subsample for performance (max 10,000 points)
        max_points = 10000
        if len(z_coords) > max_points:
            indices = np.random.choice(len(z_coords), max_points, replace=False)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
            values = values[indices]
        
        print(f"Creating interactive plot with {len(z_coords)} points (threshold={threshold})")
        
        # Create the interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Intensity",
                    thickness=20,
                    len=0.8
                ),
                line=dict(width=0)
            ),
            text=[f'Position: ({x}, {y}, {z})<br>Intensity: {v:.3f}' 
                  for x, y, z, v in zip(x_coords, y_coords, z_coords, values)],
            hovertemplate='%{text}<extra></extra>',
            name='Cellular Structures'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Interactive 3D View: EMBL bmcc122 ({channel_name})<br><sub>Centrin-Tubulin-DNA Fluorescence Data</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Z (slices)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='white',
                xaxis=dict(showbackground=True, backgroundcolor='lightgray'),
                yaxis=dict(showbackground=True, backgroundcolor='lightgray'),
                zaxis=dict(showbackground=True, backgroundcolor='lightgray')
            ),
            width=1200,
            height=900
        )
        
        # Save the interactive plot
        output_file = output_dir / f'interactive_3d_{channel_name}.html'
        fig.write_html(str(output_file))
        print(f"✓ Interactive 3D plot saved: {output_file}")
        
        # Also show the plot
        fig.show()
        
    except Exception as e:
        print(f"Error creating interactive scatter: {e}")

def create_surface_mesh(volume, output_dir, channel_name, threshold=0.4):
    """Create 3D surface mesh using marching cubes"""
    try:
        print(f"Generating 3D surface (threshold={threshold})...")
        
        # Apply smoothing for better surface
        smoothed = ndimage.gaussian_filter(volume, sigma=1.5)
        
        # Generate surface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, level=threshold, spacing=(1, 1, 1)
        )
        
        print(f"Surface generated: {len(verts)} vertices, {len(faces)} faces")
        
        # Create the mesh
        fig = go.Figure()
        
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=values,
            colorscale='Plasma',
            opacity=0.9,
            name='Cellular Surface',
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.6,
                roughness=0.1
            ),
            lightposition=dict(x=100, y=100, z=100),
            hovertemplate='Surface Point<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'3D Surface Mesh: EMBL bmcc122 ({channel_name})<br><sub>Isosurface at intensity {threshold}</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)', 
                zaxis_title='Z (slices)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2, y=2, z=2),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='white',
                xaxis=dict(showbackground=True),
                yaxis=dict(showbackground=True),
                zaxis=dict(showbackground=True)
            ),
            width=1200,
            height=900
        )
        
        # Save the mesh
        output_file = output_dir / f'surface_mesh_{channel_name}.html'
        fig.write_html(str(output_file))
        print(f"✓ 3D surface mesh saved: {output_file}")
        
        fig.show()
        
    except Exception as e:
        print(f"Error creating surface mesh: {e}")

def create_multi_threshold_view(volume, output_dir, channel_name):
    """Create multi-threshold visualization"""
    try:
        thresholds = [0.2, 0.4, 0.6, 0.8]
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (threshold, color) in enumerate(zip(thresholds, colors)):
            # Find points above threshold
            mask = volume > threshold
            z_coords, y_coords, x_coords = np.where(mask)
            
            # Subsample for performance
            if len(z_coords) > 2000:
                indices = np.random.choice(len(z_coords), 2000, replace=False)
                z_coords = z_coords[indices]
                y_coords = y_coords[indices]
                x_coords = x_coords[indices]
            
            if len(z_coords) > 0:
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color,
                        opacity=0.6
                    ),
                    name=f'Threshold {threshold}',
                    visible=(i == 1)  # Show only threshold 0.4 by default
                ))
        
        # Create visibility toggle buttons
        buttons = []
        for i, threshold in enumerate(thresholds):
            visibility = [False] * len(thresholds)
            visibility[i] = True
            
            buttons.append(dict(
                label=f"Threshold {threshold}",
                method="restyle",
                args=[{"visible": visibility}]
            ))
        
        # Add button to show all
        buttons.append(dict(
            label="Show All",
            method="restyle", 
            args=[{"visible": [True] * len(thresholds)}]
        ))
        
        fig.update_layout(
            title=f'Multi-Threshold Analysis: EMBL bmcc122 ({channel_name})',
            updatemenus=[dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            )],
            scene=dict(
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200,
            height=900
        )
        
        output_file = output_dir / f'multi_threshold_{channel_name}.html'
        fig.write_html(str(output_file))
        print(f"✓ Multi-threshold analysis saved: {output_file}")
        
        fig.show()
        
    except Exception as e:
        print(f"Error creating multi-threshold view: {e}")

def main():
    """Main execution"""
    print("🧬 Starting EMBL OME-Zarr 3D Visualization Pipeline...")
    
    # 1. Load the dataset
    zarr_array = load_embl_dataset()
    if zarr_array is None:
        print("❌ Failed to load dataset")
        return
    
    # 2. Process each channel
    channel_names = ['Centrin', 'Tubulin', 'DNA']
    
    for channel, name in enumerate(channel_names):
        print(f"\\n{'='*60}")
        print(f"🔬 Processing Channel {channel}: {name}")
        print(f"{'='*60}")
        
        # Extract sample data for this channel
        sample_data = extract_sample_data(zarr_array, channel=channel, sample_size=80)
        
        if sample_data is not None:
            # Create visualizations
            output_dir = create_comprehensive_visualizations(sample_data, f"{name}_Ch{channel}")
            
            print(f"\\n✅ Channel {channel} ({name}) processing complete!")
        else:
            print(f"❌ Failed to process channel {channel}")
    
    print(f"\\n{'='*60}")
    print("🎉 3D VISUALIZATION PIPELINE COMPLETE! 🎉")
    print(f"{'='*60}")
    print("\\n📂 Generated files in 'embl_visualizations/' directory:")
    print("   📊 slice_analysis_*.png - Comprehensive 2D slice analysis")
    print("   🎯 interactive_3d_*.html - Interactive 3D scatter plots")
    print("   🏗️ surface_mesh_*.html - 3D surface meshes")
    print("   🔍 multi_threshold_*.html - Multi-threshold analysis")
    print("\\n🌐 Open the HTML files in a web browser for interactive exploration!")
    print("\\n📖 Dataset Information:")
    print("   Source: EMBL Culture Collections")
    print("   Sample: bmcc122_pfa_cetn-tub-dna_20231208_cl")
    print("   Channels: Centrin (cell structure), Tubulin (cytoskeleton), DNA (nucleus)")
    print("   Resolution: Sub-cellular level fluorescence microscopy")

if __name__ == "__main__":
    main()
