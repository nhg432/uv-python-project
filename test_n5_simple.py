#!/usr/bin/env python3
"""Simple test of N5 functionality"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openorganelle_downloader import OpenOrganelleDownloader

def test_n5_download():
    """Test N5 download functionality"""
    print('🧪 Testing N5 download functionality...')
    
    downloader = OpenOrganelleDownloader(output_dir='./data')
    
    # Test with a small slice of mitochondria segmentation data
    print('Downloading N5 data slice...')
    result = downloader.download_array_slice(
        'jrc_hela-1', 
        'labels/mito_seg/s0',
        slice_spec=(slice(0, 32), slice(512, 544), slice(10752, 10784))
    )
    
    if result and os.path.exists(result):
        data = np.load(result)
        print(f'✅ SUCCESS! Downloaded: {result}')
        print(f'   Shape: {data.shape}')
        print(f'   Dtype: {data.dtype}')
        print(f'   Min/Max: {data.min()}/{data.max()}')
        
        # Check if we have actual segmentation data
        unique_vals = np.unique(data)
        print(f'   Unique values: {len(unique_vals)}')
        
        if len(unique_vals) > 1:
            print(f'🎉 Contains segmentation data! Values: {unique_vals[:10]}')
        else:
            print(f'   Only single value: {unique_vals[0]}')
            
        return True
    else:
        print('❌ Download failed')
        return False

if __name__ == '__main__':
    success = test_n5_download()
    print(f'\nTest {"PASSED" if success else "FAILED"}')
