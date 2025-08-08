#!/usr/bin/env python3
"""
Simple test script to download jrc_hela-2 dataset and open in Fiji
"""

import sys
import os
from pathlib import Path

print("🔬 OpenOrganelle Dataset Downloader")
print("=" * 40)

try:
    import quilt3 as q3
    print("✅ quilt3 imported successfully")
except ImportError as e:
    print(f"❌ Failed to import quilt3: {e}")
    sys.exit(1)

try:
    print("🌐 Connecting to OpenOrganelle bucket...")
    b = q3.Bucket("s3://janelia-cosem-datasets")
    print("✅ Connected successfully!")
    
    # List available datasets first
    print("\n📋 Checking available datasets...")
    try:
        # Try to list what's in the bucket
        print("Bucket connected, preparing download...")
        
        # Create output directory
        output_dir = Path("./jrc_hela-2")
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        if output_dir.exists():
            print("📂 Directory already exists, checking contents...")
            files = list(output_dir.rglob("*"))
            print(f"   Found {len(files)} existing files")
            if len(files) > 0:
                print("   Sample files:")
                for f in files[:5]:
                    if f.is_file():
                        size = f.stat().st_size / (1024*1024)
                        print(f"     {f.name} ({size:.1f} MB)")
        else:
            output_dir.mkdir(exist_ok=True)
            print("📁 Created output directory")
        
        # Start download
        print(f"\n⏬ Starting download of jrc_hela-2...")
        print("   (This may take several minutes...)")
        
        # Download with progress (destination must end with /)
        result = b.fetch("jrc_hela-2/", str(output_dir) + "/")
        
        print("✅ Download completed!")
        
        # Check what we downloaded
        downloaded_files = list(output_dir.rglob("*"))
        data_files = [f for f in downloaded_files if f.is_file()]
        
        print(f"\n📊 Downloaded {len(data_files)} files")
        
        # Show largest files
        if data_files:
            data_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            print("\n📁 Largest files:")
            for f in data_files[:10]:
                size_mb = f.stat().st_size / (1024*1024)
                rel_path = f.relative_to(output_dir)
                print(f"   {size_mb:8.1f} MB - {rel_path}")
        
        # Look for Zarr files
        zarr_files = [f for f in data_files if '.zarr' in str(f) or f.name in ['.zarray', '.zattrs']]
        if zarr_files:
            print(f"\n🔍 Found {len(zarr_files)} Zarr-related files")
            
            # Find zarr roots
            zarr_roots = set()
            for zf in zarr_files:
                if zf.name == '.zarray':
                    zarr_roots.add(zf.parent)
            
            print(f"🗂️  Found {len(zarr_roots)} Zarr datasets:")
            for root in zarr_roots:
                rel_path = root.relative_to(output_dir)
                print(f"   📦 {rel_path}")
        
        print(f"\n✅ Dataset ready at: {output_dir.absolute()}")
        
        # Now try to open in Fiji
        print("\n🚀 Attempting to open in Fiji...")
        
        fiji_exe = Path("fiji_new/fiji/fiji-windows-x64.exe")
        if fiji_exe.exists():
            import subprocess
            
            print(f"🔬 Launching Fiji: {fiji_exe.absolute()}")
            
            # Launch Fiji and let user open files manually
            try:
                process = subprocess.Popen([str(fiji_exe.absolute())], 
                                         cwd=str(fiji_exe.parent))
                print("✅ Fiji launched successfully!")
                print("\n📋 To open the dataset in Fiji:")
                print("1. In Fiji, go to: Plugins → BigDataViewer → HDF5/N5/Zarr/OME-NGFF Viewer")
                print(f"2. Navigate to: {output_dir.absolute()}")
                print("3. Select the Zarr dataset folder")
                print("\n💡 Alternative: File → Open for individual files")
                
            except Exception as e:
                print(f"⚠️  Error launching Fiji: {e}")
                print(f"💡 Try running manually: {fiji_exe.absolute()}")
        else:
            print(f"❌ Fiji not found at: {fiji_exe}")
            print("💡 Install Fiji first")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Error connecting to bucket: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 Script completed!")
