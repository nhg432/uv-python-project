#!/usr/bin/env python3
"""
Test script to verify the resolution level fix
"""

from src.openorganelle_downloader import OpenOrganelleDownloader
import os

def test_download_with_resolution():
    """Test downloading with correct resolution level paths"""
    print("🧪 Testing download with resolution levels...")
    
    downloader = OpenOrganelleDownloader(output_dir="./data")
    
    # Test paths with resolution levels
    test_paths = [
        'labels/mito_seg/s0',
        'labels/er_seg/s0',
        'labels/nucleus_seg/s0'
    ]
    
    for data_path in test_paths:
        print(f"\n📥 Testing path: {data_path}")
        try:
            result = downloader.download_array_slice(
                'jrc_hela-1', 
                data_path,
                slice_spec=(slice(0, 16), slice(0, 16), slice(0, 16))  # Very small sample
            )
            
            if result and os.path.exists(result):
                print(f"✅ Success! Downloaded to: {result}")
                import numpy as np
                data = np.load(result)
                print(f"   Shape: {data.shape}, dtype: {data.dtype}")
                print(f"   Value range: {data.min()} - {data.max()}")
                return True  # Success!
            else:
                print(f"❌ Failed - no file created")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            continue
    
    print("❌ All test paths failed")
    return False

if __name__ == "__main__":
    success = test_download_with_resolution()
    print(f"\n🎯 Test result: {'SUCCESS' if success else 'FAILED'}")
