#!/usr/bin/env python3
"""
Create Tubulin Interactive 3D Visualization with Lower Threshold (0.2)
"""

import numpy as np
import zarr
import fsspec
import plotly.graph_objects as go
from pathlib import Path
import requests

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

def extract_sample_data(zarr_array, channel=1, sample_size=80):
    """Extract sample data from zarr array for Tubulin (channel 1)"""
    try:
        print(f"\\n📊 Extracting sample data for channel {channel} (Tubulin)...")
        
        # Get array dimensions [Time, Channel, Z, Y, X]
        t, c, z, y, x = zarr_array.shape
        print(f"Full array shape: {zarr_array.shape}")
        
        # Calculate sampling indices for center region
        z_start = max(0, (z - sample_size) // 2)
        z_end = min(z, z_start + sample_size)
        y_start = max(0, (y - sample_size) // 2)
        y_end = min(y, y_start + sample_size)
        x_start = max(0, (x - sample_size) // 2)
        x_end = min(x, x_start + sample_size)
        
        print(f"Sampling region: Z[{z_start}:{z_end}], Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        
        # Extract the sample (time=0, specific channel)
        sample_data = zarr_array[0, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        
        print(f"Sample shape: {sample_data.shape}")
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

def create_interactive_3d_scatter_low_threshold(volume, output_dir, channel_name, threshold=0.2):
    """Create interactive 3D scatter plot with lower threshold"""
    try:
        # Find significant voxels
        mask = volume > threshold
        z_coords, y_coords, x_coords = np.where(mask)
        values = volume[mask]
        
        # Subsample for performance (max 15,000 points for lower threshold)
        max_points = 15000
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
                size=3,  # Slightly smaller points due to more data
                color=values,
                colorscale='Viridis',
                opacity=0.7,  # Slightly more transparent
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
            name='Tubulin Network (Low Threshold)'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Interactive 3D View: EMBL bmcc122 ({channel_name}) - Enhanced Sensitivity<br><sub>Threshold: {threshold} (Lower threshold shows more microtubule details)</sub>',
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
            height=900,
            annotations=[
                dict(
                    text=f"Enhanced view with {len(z_coords)} data points<br>Lower threshold ({threshold}) reveals more microtubule structure",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ]
        )
        
        # Save the interactive plot
        output_file = output_dir / f'interactive_3d_{channel_name}_low_threshold.html'
        fig.write_html(str(output_file))
        print(f"✅ Enhanced interactive 3D plot saved: {output_file}")
        
        # Also show the plot
        fig.show()
        
        return output_file
        
    except Exception as e:
        print(f"Error creating interactive scatter: {e}")
        return None

def main():
    """Main execution"""
    print("🧬 Creating Enhanced Tubulin 3D Visualization (Low Threshold)...")
    
    # Create output directory
    output_dir = Path("embl_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load the dataset
    zarr_array = load_embl_dataset()
    if zarr_array is None:
        print("❌ Failed to load dataset")
        return
    
    # 2. Extract Tubulin data (channel 1)
    print(f"\\n{'='*60}")
    print(f"🔬 Processing Tubulin Channel with Enhanced Sensitivity")
    print(f"{'='*60}")
    
    # Extract sample data for Tubulin channel
    sample_data = extract_sample_data(zarr_array, channel=1, sample_size=80)
    
    if sample_data is not None:
        # Create enhanced visualization with lower threshold
        output_file = create_interactive_3d_scatter_low_threshold(
            sample_data, 
            output_dir, 
            "Tubulin_Ch1", 
            threshold=0.2
        )
        
        if output_file:
            print(f"\\n✅ Enhanced Tubulin visualization complete!")
            print(f"📁 File created: {output_file}")
            print("\\n🔍 This visualization shows:")
            print("   • More microtubule details with threshold 0.2 (vs 0.5)")
            print("   • Finer cytoskeletal structures")
            print("   • Weaker fluorescence signals")
            print("   • Enhanced view of the cellular transport network")
        else:
            print("❌ Failed to create enhanced visualization")
    else:
        print("❌ Failed to extract Tubulin data")

if __name__ == "__main__":
    main()
