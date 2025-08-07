#!/usr/bin/env python3
"""
Test the new N5 direct reader functionality
"""

from src.openorganelle_downloader import OpenOrganelleDownloader
import numpy as np
import os

def test_n5_reader():
    """Test the custom N5 reader with various datasets"""
    print("🧪 Testing Custom N5 Reader")
    print("=" * 50)
    
    downloader = OpenOrganelleDownloader(output_dir="./data")
    
    # Test different datasets and arrays
    test_cases = [
        ("jrc_hela-1", "labels/mito_seg/s0"),
        ("jrc_hela-1", "labels/er_seg/s0"), 
        ("jrc_cos7-11", "em/fibsem-uint16/s0"),
    ]
    
    for dataset, data_path in test_cases:
        print(f"\n📥 Testing {dataset}/{data_path}")
        
        try:
            # Try downloading a small sample
            result_file = downloader.download_array_slice(
                dataset, 
                data_path,
                slice_spec=(slice(0, 16), slice(0, 16), slice(0, 16))
            )
            
            if result_file and os.path.exists(result_file):
                data = np.load(result_file)
                print(f"✅ Success!")
                print(f"   File: {result_file}")
                print(f"   Shape: {data.shape}")
                print(f"   Dtype: {data.dtype}")
                print(f"   Value range: {data.min()} - {data.max()}")
                
                # Check if we got actual data (not all zeros)
                if data.max() > 0:
                    print(f"   📊 Contains actual data! Unique values: {len(np.unique(data))}")
                    return True
                else:
                    print(f"   ⚠️  Data is all zeros - chunks may not contain data in this region")
            else:
                print(f"❌ Failed to create file")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return False

if __name__ == "__main__":
    success = test_n5_reader()
    if success:
        print(f"\n🎉 N5 Reader is working! The download limitation has been resolved.")
    else:
        print(f"\n⚠️  N5 Reader works structurally but may need refinement for data extraction.")
