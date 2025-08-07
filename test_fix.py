#!/usr/bin/env python3
"""
Quick test script to verify our OpenOrganelle downloader fixes
"""

import sys
import os
from src.openorganelle_downloader import OpenOrganelleDownloader

def test_basic_functionality():
    """Test basic functionality without AttributeError"""
    print("Testing OpenOrganelle downloader...")
    
    # Initialize downloader
    downloader = OpenOrganelleDownloader()
    print("✅ Downloader initialized successfully")
    
    # Test listing datasets
    print("Testing dataset listing...")
    datasets = downloader.list_datasets()
    print(f"✅ Found {len(datasets)} datasets")
    
    # Test exploring a dataset structure
    test_dataset = "jrc_hela-1"
    print(f"\nTesting dataset exploration for {test_dataset}...")
    
    try:
        # Get dataset info
        info = downloader.get_dataset_info(test_dataset)
        print(f"✅ Dataset info retrieved: {info.get('name', 'Unknown')}")
        
        # List data types
        data_types = downloader.list_data_types(test_dataset)
        print(f"✅ Data types found: {data_types}")
        
        # Test array listing within a group
        if 'labels' in data_types:
            arrays = downloader._list_s3_arrays_in_group(
                f"{downloader.base_s3_url}/{test_dataset}/{test_dataset}.n5",
                'labels'
            )
            print(f"✅ Arrays in labels group: {len(arrays)} arrays")
            
            # Test getting array info for first array
            if arrays:
                test_array = f"labels/{arrays[0]}"
                array_info = downloader.get_array_info(test_dataset, test_array)
                if 'error' not in array_info:
                    print(f"✅ Array info retrieved for {test_array}")
                else:
                    print(f"⚠️  Array info had error: {array_info['error']}")
        
        print("\n🎉 All tests passed! No AttributeError for '_open_zarr_group'")
        return True
        
    except AttributeError as e:
        if '_open_zarr_group' in str(e):
            print(f"❌ AttributeError still present: {e}")
            return False
        else:
            print(f"❌ Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Other error occurred: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
