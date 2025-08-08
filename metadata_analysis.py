#!/usr/bin/env python3
"""
Inspect downloaded metadata and try to find actual image data
"""

import json
from pathlib import Path
import quilt3 as q3

print("🔍 Analyzing Downloaded Metadata")
print("=" * 40)

# Check what we have downloaded
jrc_dir = Path("./jrc_hela-2")
print(f"📁 Dataset directory: {jrc_dir.absolute()}")

# Read the Zarr group metadata
zarr_group_file = jrc_dir / "jrc_hela-2.zarr" / ".zgroup"
if zarr_group_file.exists():
    print(f"\n📋 Zarr Group Metadata:")
    with open(zarr_group_file, 'r') as f:
        group_data = json.load(f)
        print(json.dumps(group_data, indent=2))

# Read the Zarr attributes
zarr_attrs_file = jrc_dir / "jrc_hela-2.zarr" / ".zattrs"
if zarr_attrs_file.exists():
    print(f"\n📋 Zarr Attributes:")
    with open(zarr_attrs_file, 'r') as f:
        attrs_data = json.load(f)
        print(json.dumps(attrs_data, indent=2))

# Read the N5 attributes
n5_attrs_file = jrc_dir / "jrc_hela-2.n5" / "attributes.json"
if n5_attrs_file.exists():
    print(f"\n📋 N5 Attributes:")
    with open(n5_attrs_file, 'r') as f:
        n5_data = json.load(f)
        print(json.dumps(n5_data, indent=2))

# Based on metadata, try to infer the correct structure and download more files
print(f"\n🔄 Attempting to download based on inferred structure...")

# Connect to bucket
b = q3.Bucket("s3://janelia-cosem-datasets")

# Common OpenOrganelle structures to try
structure_patterns = [
    # Raw EM data patterns
    "jrc_hela-2/jrc_hela-2.n5/setup0/timepoint0/s0/attributes.json",
    "jrc_hela-2/jrc_hela-2.n5/setup0/timepoint0/s1/attributes.json",
    "jrc_hela-2/jrc_hela-2.n5/setup0/timepoint0/s2/attributes.json",
    
    # Alternative N5 structure
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s0/attributes.json",
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s1/attributes.json", 
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s2/attributes.json",
    
    # Segmentation data
    "jrc_hela-2/jrc_hela-2.n5/volumes/labels/s0/attributes.json",
    "jrc_hela-2/jrc_hela-2.n5/volumes/labels/s1/attributes.json",
    
    # Zarr multiscale structure
    "jrc_hela-2/jrc_hela-2.zarr/em/fibsem-uint16/s0/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/em/fibsem-uint16/s1/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/em/fibsem-uint16/s2/.zarray",
    
    # Labels in Zarr
    "jrc_hela-2/jrc_hela-2.zarr/labels/mito/s0/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/labels/er/s0/.zarray",
    "jrc_hela-2/jrc_hela-2.zarr/labels/nucleus/s0/.zarray",
    
    # Try some actual data chunks
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s0/0/0/0",
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s1/0/0/0",
    "jrc_hela-2/jrc_hela-2.n5/volumes/raw/s2/0/0/0",
]

downloaded_new = 0
for pattern in structure_patterns:
    try:
        print(f"   Trying: {pattern}")
        local_path = jrc_dir / pattern.replace("jrc_hela-2/", "")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        b.fetch(pattern, str(local_path))
        
        if local_path.exists() and local_path.stat().st_size > 0:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Downloaded: {pattern} ({size_mb:.3f} MB)")
            downloaded_new += 1
            
            # If we found a good pattern, try to get more from the same structure
            if "s0" in pattern and size_mb > 0.001:
                print(f"      🔍 Found promising structure, exploring more...")
                
        else:
            print(f"   ⚠️  File empty: {pattern}")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        continue

print(f"\n📊 New Downloads: {downloaded_new} files")

# List everything we have now
print(f"\n📁 Current Dataset Contents:")
all_files = list(jrc_dir.rglob("*"))
data_files = [f for f in all_files if f.is_file()]

if data_files:
    total_size = 0
    for f in data_files:
        size = f.stat().st_size
        size_mb = size / (1024 * 1024)
        total_size += size
        rel_path = f.relative_to(jrc_dir)
        print(f"   {size_mb:8.3f} MB - {rel_path}")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\nTotal size: {total_mb:.1f} MB")
    
    # Now launch Fiji
    print(f"\n🚀 Launching Fiji to open the dataset...")
    import subprocess
    
    fiji_exe = Path("fiji_new/fiji/fiji-windows-x64.exe")
    if fiji_exe.exists():
        try:
            process = subprocess.Popen([str(fiji_exe)], cwd=str(fiji_exe.parent))
            print("✅ Fiji launched!")
            print(f"\n📋 In Fiji, try these locations:")
            print(f"1. Plugins → BigDataViewer → HDF5/N5/Zarr/OME-NGFF Viewer")
            print(f"2. Browse to: {jrc_dir.absolute()}")
            print(f"3. Select folders:")
            
            for folder in ["jrc_hela-2.zarr", "jrc_hela-2.n5"]:
                folder_path = jrc_dir / folder
                if folder_path.exists():
                    print(f"   - {folder}")
                    
        except Exception as e:
            print(f"Error launching Fiji: {e}")
    else:
        print("Fiji not found")
        
else:
    print("No files found")

print(f"\n🏁 Analysis complete!")
