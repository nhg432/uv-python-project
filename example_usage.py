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
    
    # Download a small sample of EM data
    print(f"\n4. Downloading a sample of EM data...")
    try:
        # Try to download EM data
        sample_file = downloader.download_array_slice(
            target_dataset, 
            'em/fibsem-uint16/s0',  # Full resolution EM data
            slice_spec=(slice(0, 32), slice(0, 32), slice(0, 32))  # Small 32x32x32 cube
        )
        if sample_file:
            print(f"   Sample EM data saved to: {sample_file}")
    except Exception as e:
        print(f"   Could not download EM data: {e}")
        
        # Try alternative paths
        print("   Trying alternative data paths...")
        data_types = downloader.list_data_types(target_dataset)
        for data_type in data_types[:3]:  # Try first 3 data types
            try:
                # Get arrays in this data type
                info = downloader.get_dataset_info(target_dataset)
                arrays = info.get(f'{data_type}_arrays', [])
                if arrays:
                    array_path = f"{data_type}/{arrays[0]}"
                    print(f"   Trying: {array_path}")
                    sample_file = downloader.download_array_slice(
                        target_dataset, 
                        array_path,
                        slice_spec=(slice(0, 16), slice(0, 16), slice(0, 16))  # Even smaller sample
                    )
                    if sample_file:
                        print(f"   Sample data saved to: {sample_file}")
                        break
            except Exception as e2:
                print(f"   Failed {array_path}: {e2}")
    
    print(f"\n=== Download Complete ===")
    print(f"Check the './data' directory for downloaded files.")
    print(f"\nTo use the command-line interface, run:")
    print(f"python src/openorganelle_downloader.py --list-datasets")
    print(f"python src/openorganelle_downloader.py --explore {target_dataset}")
    print(f"python src/openorganelle_downloader.py --download {target_dataset}")

if __name__ == "__main__":
    main()
