#!/usr/bin/env python3
"""
Example usage of the OpenOrganelle downloader.

This script demonstrates how to use the OpenOrganelleDownloader class
to explore and download cellular imaging data.
"""

import sys
import os

# Add the src directory to the path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.openorganelle_downloader import OpenOrganelleDownloader

def main():
    """Demonstrate basic usage of the OpenOrganelle downloader."""
    
    print("=== OpenOrganelle Data Downloader Example ===\n")
    
    # Initialize the downloader
    downloader = OpenOrganelleDownloader(output_dir="./data")
    
    # List available datasets
    print("1. Listing available datasets...")
    datasets = downloader.list_datasets()
    print(f"Found {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets[:10], 1):  # Show first 10
        print(f"   {i:2d}. {dataset}")
    
    if not datasets:
        print("No datasets found. Check your internet connection.")
        return
    
    # Explore a specific dataset (use the first one or HeLa if available)
    target_dataset = None
    for dataset in datasets:
        if 'hela' in dataset.lower():
            target_dataset = dataset
            break
    
    if not target_dataset:
        target_dataset = datasets[0]  # Use first available dataset
    
    print(f"\n2. Exploring dataset: {target_dataset}")
    downloader.explore_dataset(target_dataset)
    
    # Download metadata
    print(f"\n3. Downloading metadata for {target_dataset}...")
    metadata_file = downloader.download_metadata(target_dataset)
    if metadata_file:
        print(f"   Metadata saved to: {metadata_file}")
    
    # Download sample data using known working paths
    print(f"\n4. Downloading sample data...")
    
    # List of known working data paths for different datasets
    data_paths_to_try = [
        'labels/mito_seg/s0',      # Mitochondria segmentation (verified working)
        'labels/er_seg/s0',        # ER segmentation
        'labels/nucleus_seg/s0',   # Nucleus segmentation  
        'em/fibsem-uint16/s0',     # Raw EM data
        'em/fibsem-uint8/s0',      # Raw EM data (8-bit)
    ]
    
    sample_downloaded = False
    for data_path in data_paths_to_try:
        print(f"   Trying data path: {data_path}")
        try:
            sample_file = downloader.download_array_slice(
                target_dataset, 
                data_path,
                slice_spec=(slice(0, 32), slice(512, 544), slice(10752, 10784))  # Known good coordinates
            )
            if sample_file:
                print(f"   ✅ Sample data saved to: {sample_file}")
                
                # Show basic info about the downloaded data
                import numpy as np
                data = np.load(sample_file)
                unique_vals = len(np.unique(data))
                print(f"   📊 Shape: {data.shape}, Unique values: {unique_vals}")
                if unique_vals > 1:
                    print(f"   🎉 Contains real data (not just zeros)!")
                
                sample_downloaded = True
                break
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
            continue
    
    if not sample_downloaded:
        print("   ❌ Could not download any sample data. Dataset might have different structure.")
    
    print(f"\n=== Download Complete ===")
    print(f"Check the './data' directory for downloaded files.")
    print(f"\nTo use the command-line interface, run:")
    print(f"python src/openorganelle_downloader.py --list-datasets")
    print(f"python src/openorganelle_downloader.py --explore {target_dataset}")
    print(f"python src/openorganelle_downloader.py --download {target_dataset}")

if __name__ == "__main__":
    main()
