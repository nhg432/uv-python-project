#!/usr/bin/env python3
"""
Enhanced EMBL OME-Zarr Tubulin Visualization with Ultra-Low Threshold (0.1)
This script creates a highly sensitive 3D visualization of Tubulin fluorescence data
with threshold 0.1 to reveal maximum microtubule detail.
"""

import zarr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fsspec
from scipy import ndimage
import os

def load_embl_dataset():
    """Load EMBL OME-Zarr dataset from S3"""
    try:
        # Direct HTTP access to EMBL S3 data
        base_url = "https://s3.embl.de/culture-collections/data/single_volumes/images/ome-zarr/bmcc122_pfa_cetn-tub-dna_20231208_cl.ome.zarr"
        
        print(f"� Connecting to EMBL dataset...")
        
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
        print(f"\n📊 Extracting sample data for channel {channel} (Tubulin)...")
        
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
        
        # Extract data for the specified channel
        sample_data = zarr_array[0, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        
        print(f"✅ Sample extracted! Shape: {sample_data.shape}")
        print(f"� Value range: {np.min(sample_data):.3f} to {np.max(sample_data):.3f}")
        
        return sample_data
        
    except Exception as e:
        print(f"❌ Error extracting sample: {e}")
        return None

def create_interactive_3d_scatter_ultra_low_threshold(data, threshold=0.1, max_points=20000):
    """Create ultra-sensitive 3D scatter plot with threshold 0.1"""
    print(f"🎨 Creating ultra-sensitive 3D visualization (threshold={threshold})...")
    
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Find points above threshold
    z_coords, y_coords, x_coords = np.where(data_normalized > threshold)
    intensities = data_normalized[z_coords, y_coords, x_coords]
    
    print(f"📍 Found {len(z_coords)} points above threshold {threshold}")
    
    # Sample points if too many (keep highest intensity points)
    if len(z_coords) > max_points:
        indices = np.argsort(intensities)[-max_points:]
        z_coords = z_coords[indices]
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
        intensities = intensities[indices]
        print(f"📉 Sampled to {max_points} highest intensity points")
    
    # Create color scale based on intensity
    colors = intensities
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=2,  # Even smaller markers for ultra-fine detail
            color=colors,
            colorscale='Viridis',
            opacity=0.6,  # Lower opacity for better visibility of dense structures
            colorbar=dict(title="Tubulin Intensity"),
            showscale=True
        ),
        text=[f"Position: ({x}, {y}, {z})<br>Intensity: {intensity:.3f}" 
              for x, y, z, intensity in zip(x_coords, y_coords, z_coords, intensities)],
        hovertemplate="<b>Tubulin Signal</b><br>" +
                      "X: %{x}<br>" +
                      "Y: %{y}<br>" +
                      "Z: %{z}<br>" +
                      "Intensity: %{marker.color:.3f}<br>" +
                      "<extra></extra>",
        name="Tubulin Microtubules"
    ))
    
    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': "Interactive 3D View: EMBL bmcc122 (Tubulin_Ch1) - Ultra-High Sensitivity<br>" +
                   "<sub>Threshold: 0.1 (Maximum microtubule detail resolution)</sub>",
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            zaxis_title="Z (slices)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(showbackground=True, backgroundcolor="lightgray"),
            yaxis=dict(showbackground=True, backgroundcolor="lightgray"),
            zaxis=dict(showbackground=True, backgroundcolor="lightgray"),
            aspectmode='cube',
            bgcolor='white'
        ),
        width=1200,
        height=900,
        annotations=[
            dict(
                text=f"Ultra-sensitive view with {len(z_coords)} data points<br>" +
                     "Maximum threshold (0.1) reveals finest microtubule details",
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    return fig

def main():
    """Main execution function"""
    print("🧬 EMBL OME-Zarr Ultra-Sensitive Tubulin Visualization")
    print("=" * 60)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract Tubulin channel sample
        tubulin_data = extract_sample_data(zarr_array, channel=1)
        if tubulin_data is None:
            return
        
        # Create ultra-sensitive 3D visualization
        fig = create_interactive_3d_scatter_ultra_low_threshold(tubulin_data, threshold=0.1, max_points=20000)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ultra-sensitive visualization
        output_file = os.path.join(output_dir, "interactive_3d_Tubulin_Ch1_ultra_low_threshold.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Ultra-sensitive Tubulin visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This ultra-sensitive visualization shows:")
        print(f"   • Maximum microtubule detail with threshold 0.1")
        print(f"   • Finest cytoskeletal structures")
        print(f"   • Even weaker fluorescence signals")
        print(f"   • Complete microtubule network architecture")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
