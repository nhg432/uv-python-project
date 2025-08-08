import quilt3 as q3
import os
import subprocess
from pathlib import Path
import time

print("🔬 Downloading jrc_hela-2 dataset from OpenOrganelle...")
print("=" * 60)

# Download the dataset
b = q3.Bucket("s3://janelia-cosem-datasets")

# Create output directory
output_dir = Path("./jrc_hela-2")
output_dir.mkdir(exist_ok=True)

print(f"📁 Downloading to: {output_dir.absolute()}")
print("⏳ First, let's see what's available...")

try:
    # Use alternative download method - target specific files directly
    print("\n� Using alternative download method...")
    print("📥 Downloading specific dataset files...")
    
    # Comprehensive list of files to try downloading
    target_files = [
        # Zarr metadata files
        "jrc_hela-2/jrc_hela-2.zarr/.zattrs",
        "jrc_hela-2/jrc_hela-2.zarr/.zgroup",
        
        # Multi-scale Zarr arrays
        "jrc_hela-2/jrc_hela-2.zarr/s0/.zarray",
        "jrc_hela-2/jrc_hela-2.zarr/s0/.zattrs", 
        "jrc_hela-2/jrc_hela-2.zarr/s1/.zarray",
        "jrc_hela-2/jrc_hela-2.zarr/s1/.zattrs",
        "jrc_hela-2/jrc_hela-2.zarr/s2/.zarray",
        "jrc_hela-2/jrc_hela-2.zarr/s2/.zattrs",
        
        # Alternative scale naming
        "jrc_hela-2/jrc_hela-2.zarr/0/.zarray",
        "jrc_hela-2/jrc_hela-2.zarr/0/.zattrs",
        "jrc_hela-2/jrc_hela-2.zarr/1/.zarray", 
        "jrc_hela-2/jrc_hela-2.zarr/1/.zattrs",
        "jrc_hela-2/jrc_hela-2.zarr/2/.zarray",
        "jrc_hela-2/jrc_hela-2.zarr/2/.zattrs",
        
        # Sample data chunks for different scales
        "jrc_hela-2/jrc_hela-2.zarr/s0/0/0/0",
        "jrc_hela-2/jrc_hela-2.zarr/s1/0/0/0",
        "jrc_hela-2/jrc_hela-2.zarr/s2/0/0/0",
        "jrc_hela-2/jrc_hela-2.zarr/0/0/0/0",
        "jrc_hela-2/jrc_hela-2.zarr/1/0/0/0",
        "jrc_hela-2/jrc_hela-2.zarr/2/0/0/0",
        
        # N5 format files
        "jrc_hela-2/jrc_hela-2.n5/attributes.json",
        "jrc_hela-2/jrc_hela-2.n5/s0/attributes.json",
        "jrc_hela-2/jrc_hela-2.n5/s1/attributes.json", 
        "jrc_hela-2/jrc_hela-2.n5/s2/attributes.json",
        
        # Possible sample data files
        "jrc_hela-2/jrc_hela-2_sample.tif",
        "jrc_hela-2/jrc_hela-2_thumbnail.png",
        "jrc_hela-2/metadata.json",
        "jrc_hela-2/jrc_hela-2_metadata.json",
        
        # OME-NGFF metadata
        "jrc_hela-2/jrc_hela-2.ome.zarr/.zattrs",
        "jrc_hela-2/jrc_hela-2.ome.zarr/.zgroup",
    ]
    
    downloaded_count = 0
    failed_count = 0
    
    for file_path in target_files:
        try:
            print(f"   Trying: {file_path}")
            local_path = output_dir / file_path.replace("jrc_hela-2/", "")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt download
            b.fetch(file_path, str(local_path))
            
            if local_path.exists() and local_path.stat().st_size > 0:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ Downloaded: {file_path} ({size_mb:.3f} MB)")
                downloaded_count += 1
            else:
                print(f"   ⚠️  File empty: {file_path}")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Failed: {file_path} - {e}")
            failed_count += 1
            continue
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {downloaded_count} files")
    print(f"   ❌ Failed downloads: {failed_count} files")
    
    if downloaded_count == 0:
        print("\n🔄 Trying alternative file structure patterns...")
        
        # Try different common patterns for OpenOrganelle datasets
        alt_patterns = [
            "jrc_hela-2/jrc_hela-2.zarr/0.0.0",
            "jrc_hela-2/jrc_hela-2.zarr/1.0.0", 
            "jrc_hela-2/jrc_hela-2.zarr/2.0.0",
            "jrc_hela-2/em/fibsem-uint16/s0/.zarray",
            "jrc_hela-2/em/fibsem-uint16/s0/.zattrs",
            "jrc_hela-2/labels/mito/s0/.zarray",
            "jrc_hela-2/labels/er/s0/.zarray",
        ]
        
        for pattern in alt_patterns:
            try:
                print(f"   Trying pattern: {pattern}")
                local_path = output_dir / pattern.replace("jrc_hela-2/", "")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                b.fetch(pattern, str(local_path))
                
                if local_path.exists() and local_path.stat().st_size > 0:
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ Success with pattern: {pattern} ({size_mb:.3f} MB)")
                    downloaded_count += 1
                    
            except Exception as e:
                print(f"   ❌ Pattern failed: {e}")
                continue
    
    # Explore what we downloaded
    print("\n📊 Dataset Contents:")
    print("-" * 40)
    
    downloaded_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = Path(root) / file
            file_size = file_path.stat().st_size
            downloaded_files.append((file_path, file_size))
    
    # Sort by size (largest first)
    downloaded_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Total files: {len(downloaded_files)}")
    print("\nLargest files:")
    for file_path, size in downloaded_files[:10]:  # Show top 10
        size_mb = size / (1024 * 1024)
        relative_path = file_path.relative_to(output_dir)
        print(f"  {size_mb:8.1f} MB - {relative_path}")
    
    # Look for Zarr files specifically
    zarr_files = []
    fiji_compatible = []
    
    for file_path, size in downloaded_files:
        ext = file_path.suffix.lower()
        if '.zarr' in str(file_path) or file_path.name == '.zarray':
            zarr_files.append(file_path)
        elif ext in ['.tif', '.tiff', '.h5', '.hdf5', '.n5']:
            fiji_compatible.append(file_path)
    
    print(f"\n🔍 Found {len(zarr_files)} Zarr-related files")
    print(f"🔍 Found {len(fiji_compatible)} Fiji-compatible files")
    
    # Now let's open the files in Fiji
    print("\n🚀 Opening in Fiji...")
    print("-" * 30)
    
    # Path to Fiji executable
    fiji_exe = Path("fiji_new/fiji/fiji-windows-x64.exe").absolute()
    
    if fiji_exe.exists():
        # Try to find the best file to open
        if zarr_files:
            # Look for main zarr dataset
            main_zarr = None
            for zf in zarr_files:
                if 'jrc_hela-2' in str(zf) and zf.name == '.zarray':
                    main_zarr = zf.parent
                    break
            
            if main_zarr:
                print(f"📁 Opening Zarr dataset: {main_zarr}")
                try:
                    # Use BigDataViewer for Zarr files
                    fiji_command = [
                        str(fiji_exe),
                        "-eval",
                        f'run("BigDataViewer", "open={main_zarr.as_posix()}");'
                    ]
                    
                    process = subprocess.Popen(fiji_command, 
                                             cwd=fiji_exe.parent,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
                    
                    print("✅ Fiji launched with Zarr dataset!")
                    
                except Exception as e:
                    print(f"⚠️  Error launching with BigDataViewer: {e}")
                    print("💡 Try opening manually in Fiji:")
                    print(f"   Plugins → BigDataViewer → HDF5/N5/Zarr/OME-NGFF Viewer")
                    print(f"   Select: {main_zarr}")
        
        elif fiji_compatible:
            # Open the largest compatible file
            largest_file = fiji_compatible[0]
            print(f"📁 Opening file: {largest_file}")
            
            try:
                fiji_command = [
                    str(fiji_exe),
                    "-eval",
                    f'open("{largest_file.as_posix()}");'
                ]
                
                process = subprocess.Popen(fiji_command,
                                         cwd=fiji_exe.parent,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                
                print("✅ Fiji launched with dataset!")
                
            except Exception as e:
                print(f"⚠️  Error launching Fiji: {e}")
        
        else:
            print("💡 No directly compatible files found. Opening dataset directory in Fiji:")
            try:
                fiji_command = [str(fiji_exe)]
                process = subprocess.Popen(fiji_command,
                                         cwd=fiji_exe.parent,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                
                print("✅ Fiji launched! Manual file opening required.")
                print(f"📁 Dataset location: {output_dir.absolute()}")
                print("💡 In Fiji, use File → Open or Plugins → BigDataViewer")
                
            except Exception as e:
                print(f"⚠️  Error launching Fiji: {e}")
    
    else:
        print(f"❌ Fiji not found at: {fiji_exe}")
        print("💡 Make sure Fiji is installed correctly")
    
    print(f"\n📁 Dataset downloaded to: {output_dir.absolute()}")
    print("🔬 Ready for analysis in Fiji!")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("💡 Check your internet connection and try again")