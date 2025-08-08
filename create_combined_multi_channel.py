#!/usr/bin/env python3
"""
Combined EMBL OME-Zarr Multi-Channel 3D Visualization
This script creates a comprehensive 3D visualization showing all three channels:
- Tubulin (Channel 1) - Green
- DNA (Channel 2) - Blue  
- Centrin (Channel 0) - Red
"""

import zarr
import numpy as np
import plotly.graph_objects as go
import fsspec
import os

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

def create_channel_scatter(data, channel_name, color, threshold=0.2, max_points=8000):
    """Create scatter points for a single channel"""
    # Normalize data to 0-1 range
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Find points above threshold
    z_coords, y_coords, x_coords = np.where(data_normalized > threshold)
    intensities = data_normalized[z_coords, y_coords, x_coords]
    
    print(f"📍 {channel_name}: Found {len(z_coords)} points above threshold {threshold}")
    
    # Sample points if too many (keep highest intensity points)
    if len(z_coords) > max_points:
        indices = np.argsort(intensities)[-max_points:]
        z_coords = z_coords[indices]
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
        intensities = intensities[indices]
        print(f"📉 {channel_name}: Sampled to {max_points} highest intensity points")
    
    return x_coords, y_coords, z_coords, intensities

def create_combined_3d_visualization(centrin_data, tubulin_data, dna_data):
    """Create combined 3D visualization of all three channels"""
    print(f"🎨 Creating combined multi-channel 3D visualization...")
    
    # Create scatter data for each channel with different thresholds for optimal visualization
    centrin_x, centrin_y, centrin_z, centrin_intensities = create_channel_scatter(
        centrin_data, "Centrin", "red", threshold=0.3, max_points=6000)
    
    tubulin_x, tubulin_y, tubulin_z, tubulin_intensities = create_channel_scatter(
        tubulin_data, "Tubulin", "green", threshold=0.2, max_points=8000)
    
    dna_x, dna_y, dna_z, dna_intensities = create_channel_scatter(
        dna_data, "DNA", "blue", threshold=0.4, max_points=6000)
    
    # Create the combined figure
    fig = go.Figure()
    
    # Add Centrin (red) - Centrioles
    fig.add_trace(go.Scatter3d(
        x=centrin_x,
        y=centrin_y,
        z=centrin_z,
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            opacity=0.8,
            symbol='circle'
        ),
        text=[f"Centrin<br>Position: ({x}, {y}, {z})<br>Intensity: {intensity:.3f}" 
              for x, y, z, intensity in zip(centrin_x, centrin_y, centrin_z, centrin_intensities)],
        hovertemplate="<b>Centrin (Centrioles)</b><br>" +
                      "X: %{x}<br>" +
                      "Y: %{y}<br>" +
                      "Z: %{z}<br>" +
                      "<extra></extra>",
        name="Centrin (Centrioles)",
        legendgroup="centrin"
    ))
    
    # Add Tubulin (green) - Microtubules
    fig.add_trace(go.Scatter3d(
        x=tubulin_x,
        y=tubulin_y,
        z=tubulin_z,
        mode='markers',
        marker=dict(
            size=3,
            color='lime',
            opacity=0.7,
            symbol='circle'
        ),
        text=[f"Tubulin<br>Position: ({x}, {y}, {z})<br>Intensity: {intensity:.3f}" 
              for x, y, z, intensity in zip(tubulin_x, tubulin_y, tubulin_z, tubulin_intensities)],
        hovertemplate="<b>Tubulin (Microtubules)</b><br>" +
                      "X: %{x}<br>" +
                      "Y: %{y}<br>" +
                      "Z: %{z}<br>" +
                      "<extra></extra>",
        name="Tubulin (Microtubules)",
        legendgroup="tubulin"
    ))
    
    # Add DNA (blue) - Nucleus
    fig.add_trace(go.Scatter3d(
        x=dna_x,
        y=dna_y,
        z=dna_z,
        mode='markers',
        marker=dict(
            size=3.5,
            color='dodgerblue',
            opacity=0.6,
            symbol='circle'
        ),
        text=[f"DNA<br>Position: ({x}, {y}, {z})<br>Intensity: {intensity:.3f}" 
              for x, y, z, intensity in zip(dna_x, dna_y, dna_z, dna_intensities)],
        hovertemplate="<b>DNA (Nucleus)</b><br>" +
                      "X: %{x}<br>" +
                      "Y: %{y}<br>" +
                      "Z: %{z}<br>" +
                      "<extra></extra>",
        name="DNA (Nucleus)",
        legendgroup="dna"
    ))
    
    # Update layout for the combined visualization
    fig.update_layout(
        title={
            'text': "Combined Multi-Channel 3D View: EMBL bmcc122<br>" +
                   "<sub>🔴 Centrin (Centrioles) | 🟢 Tubulin (Microtubules) | 🔵 DNA (Nucleus)</sub>",
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
            bgcolor='black'  # Dark background to make fluorescence colors pop
        ),
        width=1400,
        height=1000,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"Combined view: {len(centrin_x)} Centrin + {len(tubulin_x)} Tubulin + {len(dna_x)} DNA points<br>" +
                     "Interactive 3D cellular architecture visualization",
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                xanchor="left", yanchor="bottom",
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    return fig

def main():
    """Main execution function"""
    print("🧬 EMBL OME-Zarr Combined Multi-Channel 3D Visualization")
    print("=" * 60)
    
    try:
        # Load dataset
        zarr_array = load_embl_dataset()
        if zarr_array is None:
            return
        
        # Extract all channel sample data
        centrin_data, tubulin_data, dna_data = extract_sample_data_all_channels(zarr_array)
        if centrin_data is None:
            return
        
        # Create combined 3D visualization
        fig = create_combined_3d_visualization(centrin_data, tubulin_data, dna_data)
        
        # Create output directory
        output_dir = "embl_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined visualization
        output_file = os.path.join(output_dir, "interactive_3d_combined_all_channels.html")
        fig.write_html(output_file)
        
        print(f"\n✅ Combined multi-channel visualization complete!")
        print(f"📁 File created: {output_file}")
        print(f"\n🔍 This combined visualization shows:")
        print(f"   🔴 Centrin (Channel 0): Centrioles - cellular organization centers")
        print(f"   🟢 Tubulin (Channel 1): Microtubules - cytoskeletal network")
        print(f"   🔵 DNA (Channel 2): Nucleus - genetic material")
        print(f"   🎯 Complete cellular architecture in one interactive view")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
