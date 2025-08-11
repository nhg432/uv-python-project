#!/usr/bin/env python3
"""
Direct Zarr Chunk Reader for Janelia COSEM HeLa-4
Attempts to read individual zarr chunks and reconstruct a sample
"""

import fsspec
import numpy as np
import json
import struct
from skimage import measure
from scipy import ndimage
import plotly.graph_objects as go
import os

def read_zarr_metadata(fs, level_path):
    """Read zarr metadata to understand the array structure"""
    try:
        # Read .zarray metadata
        zarray_path = f"{level_path}/.zarray"
        zarray_content = fs.cat(zarray_path)
        zarray_info = json.loads(zarray_content.decode())
        
        print(f"📊 Zarr metadata:")
        print(f"   Shape: {zarray_info.get('shape', 'unknown')}")
        print(f"   Chunks: {zarray_info.get('chunks', 'unknown')}")
        print(f"   Data type: {zarray_info.get('dtype', 'unknown')}")
        print(f"   Compressor: {zarray_info.get('compressor', 'none')}")
        
        return zarray_info
    except Exception as e:
        print(f"❌ Could not read zarr metadata: {e}")
        return None

def try_direct_chunk_access():
    """Try to access zarr chunks directly"""
    try:
        print("🔍 Attempting direct zarr chunk access...")
        
        fs = fsspec.filesystem('s3', anon=True)
        base_path = "janelia-cosem-datasets/jrc_hela-4/jrc_hela-4.zarr/recon-1/em/fibsem-uint16"
        
        # Try different resolution levels
        for level in ['s5', 's4', 's3']:  # Start with smaller ones
            level_path = f"{base_path}/{level}"
            print(f"\n🔗 Trying level {level}...")
            
            # Read metadata
            metadata = read_zarr_metadata(fs, level_path)
            if not metadata:
                continue
            
            # Get chunk information
            shape = metadata.get('shape')
            chunks = metadata.get('chunks')
            dtype = metadata.get('dtype', '<u2')  # Default to uint16
            
            if not shape or not chunks:
                print(f"   ❌ Missing shape or chunk info")
                continue
            
            print(f"   📐 Array shape: {shape}")
            print(f"   📦 Chunk size: {chunks}")
            
            # List available chunks
            try:
                chunk_files = fs.ls(level_path)
                chunk_files = [f for f in chunk_files if not f.endswith('.zarray')]
                print(f"   📁 Found {len(chunk_files)} chunks")
                
                # Try to read a few chunks
                chunks_read = 0
                sample_data = []
                
                for chunk_file in chunk_files[:5]:  # Try first 5 chunks
                    try:
                        chunk_data = fs.cat(chunk_file)
                        print(f"   ✅ Read chunk {chunk_file.split('/')[-1]}: {len(chunk_data)} bytes")
                        
                        # Try to decompress if needed
                        if metadata.get('compressor'):
                            # This would require specific decompression logic
                            print(f"   ⚠️  Compressed data - skipping decompression")
                            continue
                        
                        # Convert bytes to numpy array
                        if dtype == '<u2':  # uint16 little endian
                            chunk_array = np.frombuffer(chunk_data, dtype=np.uint16)
                        elif dtype == '<u1':  # uint8
                            chunk_array = np.frombuffer(chunk_data, dtype=np.uint8)
                        else:
                            print(f"   ⚠️  Unknown dtype: {dtype}")
                            continue
                        
                        # Reshape to chunk dimensions
                        try:
                            chunk_array = chunk_array.reshape(chunks)
                            sample_data.append(chunk_array)
                            chunks_read += 1
                            print(f"   ✅ Successfully parsed chunk shape: {chunk_array.shape}")
                            
                            if chunks_read >= 3:  # We have enough for a sample
                                break
                                
                        except Exception as reshape_error:
                            print(f"   ❌ Reshape failed: {reshape_error}")
                            continue
                        
                    except Exception as chunk_error:
                        print(f"   ❌ Chunk read failed: {str(chunk_error)[:50]}...")
                        continue
                
                if sample_data:
                    print(f"   ✅ Successfully read {len(sample_data)} chunks!")
                    # Combine chunks into a sample volume
                    combined_data = np.stack(sample_data, axis=0)
                    print(f"   📊 Combined sample shape: {combined_data.shape}")
                    return combined_data, f"janelia_chunks_{level}"
                
            except Exception as list_error:
                print(f"   ❌ Chunk listing failed: {list_error}")
                continue
        
        return None, None
        
    except Exception as e:
        print(f"❌ Direct chunk access failed: {e}")
        return None, None

def create_janelia_demo_mesh(data, source_name):
    """Create mesh from successfully read data"""
    try:
        print(f"🎨 Creating mesh from {source_name}...")
        
        # Normalize data
        data_norm = (data - data.min()) / (data.max() - data.min())
        
        # Apply smoothing
        data_smooth = ndimage.gaussian_filter(data_norm, sigma=1.0)
        
        # Create surface mesh
        verts, faces, normals, values = measure.marching_cubes(
            data_smooth,
            level=0.3,
            spacing=(1.0, 1.0, 1.0)
        )
        
        print(f"✅ Mesh created: {len(verts):,} vertices")
        
        # Create plotly figure
        fig = go.Figure()
        
        mesh = go.Mesh3d(
            x=verts[:, 2],
            y=verts[:, 1], 
            z=verts[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='orange',
            opacity=0.7,
            name=f"Janelia HeLa-4 - {source_name}"
        )
        
        fig.add_trace(mesh)
        
        fig.update_layout(
            title=f"Janelia COSEM HeLa-4 Direct Chunk Access<br><sub>Source: {source_name}</sub>",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            width=1400,
            height=1000
        )
        
        return fig
        
    except Exception as e:
        print(f"❌ Mesh creation failed: {e}")
        return None

def main():
    """Main execution"""
    print("🧬 Janelia COSEM Direct Chunk Access Attempt")
    print("=" * 60)
    
    # Try direct chunk access
    data, source = try_direct_chunk_access()
    
    if data is not None:
        print(f"\n✅ Successfully accessed data from {source}")
        
        # Create visualization
        fig = create_janelia_demo_mesh(data, source)
        
        if fig:
            output_dir = "embl_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "janelia_direct_chunk_access.html")
            fig.write_html(output_file)
            print(f"✅ Saved visualization: {output_file}")
        
    else:
        print("\n❌ Could not access Janelia COSEM data via direct chunk reading")
        print("💡 The dataset may require specific authentication or access methods")

if __name__ == "__main__":
    main()
