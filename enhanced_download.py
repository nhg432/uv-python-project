#!/usr/bin/env python3
"""
Enhanced OpenOrganelle dataset downloader with better error handling
"""

import quilt3 as q3
import os
import subprocess
from pathlib import Path
import json

print("🔬 Enhanced OpenOrganelle Dataset Downloader")
print("=" * 50)

# Connect to bucket
b = q3.Bucket("s3://janelia-cosem-datasets")
output_dir = Path("./jrc_hela-2")
output_dir.mkdir(exist_ok=True)

print(f"📁 Download directory: {output_dir.absolute()}")

# Key files to download for jrc_hela-2 dataset
key_files = [
    # Zarr metadata files
    "jrc_hela-2/jrc_hela-2.zarr/.zattrs",
    "jrc_hela-2/jrc_hela-2.zarr/.zgroup",
    "jrc_hela-2/jrc_hela-2.zarr/0/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/0/.zattrs",
    "jrc_hela-2/jrc_hela-2.zarr/1/.zarray", 
    "jrc_hela-2/jrc_hela-2.zarr/1/.zattrs",
    "jrc_hela-2/jrc_hela-2.zarr/2/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/2/.zattrs",
    
    # Some actual data chunks (start small)
    "jrc_hela-2/jrc_hela-2.zarr/0/0.0.0",
    "jrc_hela-2/jrc_hela-2.zarr/1/0.0.0", 
    "jrc_hela-2/jrc_hela-2.zarr/2/0.0.0",
    
    # N5 format files (alternative format)
    "jrc_hela-2/jrc_hela-2.n5/attributes.json",
    
    # Sample data files
    "jrc_hela-2/jrc_hela-2_metadata.json",
]

print(f"\n⏬ Downloading key files...")
downloaded_files = []
failed_files = []

for file_key in key_files:
    try:
        print(f"   Downloading: {file_key}")
        local_path = output_dir / file_key.replace("jrc_hela-2/", "")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        b.fetch(file_key, str(local_path))
        
        if local_path.exists():
            size = local_path.stat().st_size
            size_mb = size / (1024 * 1024)
            downloaded_files.append((local_path, size))
            print(f"   ✅ Success: {size_mb:.3f} MB")
        
    except Exception as e:
        failed_files.append((file_key, str(e)))
        print(f"   ❌ Failed: {e}")
        continue

print(f"\n📊 Download Summary:")
print(f"   ✅ Downloaded: {len(downloaded_files)} files")
print(f"   ❌ Failed: {len(failed_files)} files")

if downloaded_files:
    print(f"\n📁 Downloaded files:")
    total_size = 0
    for file_path, size in downloaded_files:
        size_mb = size / (1024 * 1024)
        total_size += size
        rel_path = file_path.relative_to(output_dir)
        print(f"   {size_mb:8.3f} MB - {rel_path}")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\n📊 Total downloaded: {total_mb:.1f} MB")
    
    # Check if we have a valid Zarr dataset
    zarr_root = output_dir / "jrc_hela-2.zarr"
    if zarr_root.exists():
        print(f"\n🔍 Zarr dataset found: {zarr_root}")
        
        # Read zarr metadata
        zattrs_file = zarr_root / ".zattrs"
        if zattrs_file.exists():
            try:
                with open(zattrs_file, 'r') as f:
                    attrs = json.load(f)
                print(f"📋 Dataset attributes:")
                for key, value in attrs.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"   ⚠️  Error reading attributes: {e}")
        
        # Check available scales
        scales = []
        for i in range(10):  # Check for scales 0-9
            scale_dir = zarr_root / str(i)
            if scale_dir.exists():
                scales.append(i)
        
        if scales:
            print(f"🔢 Available scales: {scales}")
        
        # Now try to open in Fiji
        print(f"\n🚀 Opening Zarr dataset in Fiji...")
        
        fiji_exe = Path("fiji_new/fiji/fiji-windows-x64.exe")
        if fiji_exe.exists():
            try:
                # Launch Fiji first
                process = subprocess.Popen([str(fiji_exe)], 
                                         cwd=str(fiji_exe.parent))
                
                print("✅ Fiji launched!")
                print(f"\n📋 To open the Zarr dataset in Fiji:")
                print(f"1. In Fiji: Plugins → BigDataViewer → HDF5/N5/Zarr/OME-NGFF Viewer")
                print(f"2. Navigate to: {zarr_root.absolute()}")
                print(f"3. Select the folder: jrc_hela-2.zarr")
                print(f"\n💡 Alternative paths to try:")
                print(f"   - {output_dir.absolute()}")
                print(f"   - {zarr_root.absolute()}")
                
            except Exception as e:
                print(f"⚠️  Error launching Fiji: {e}")
        else:
            print(f"❌ Fiji not found at: {fiji_exe}")
    
    else:
        print(f"⚠️  No complete Zarr dataset found")
        print(f"💡 Try downloading more data or check available files")

else:
    print("❌ No files downloaded successfully")
    print("💡 Check your internet connection or try a different dataset")

print(f"\n🏁 Download completed!")
print(f"📁 Files saved to: {output_dir.absolute()}")
