#!/usr/bin/env python3
"""
OpenOrganelle Data Downloader

This script provides tools to download cellular imaging data from the OpenOrganelle platform.
It supports downloading FIB-SEM data, organelle segmentations, and related datasets.

Author: GitHub Copilot
License: MIT
"""

import os
import sys
import zarr
import fsspec
import numpy as np
import dask.array as da
from typing import List, Dict, Optional, Tuple
import requests
import json
from tqdm import tqdm
import argparse
import logging

# Import for zarr v3 compatibility
try:
    from zarr.storage import FSStore
    ZARR_V3 = True
except ImportError:
    try:
        from zarr import N5FSStore
        ZARR_V3 = False
    except ImportError:
        print("Warning: Neither zarr v3 FSStore nor N5FSStore available")
        ZARR_V3 = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenOrganelleDownloader:
    """
    A class to download data from the OpenOrganelle platform.
    """
    
    def __init__(self, output_dir: str = "./downloads"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = output_dir
        self.base_s3_url = "s3://janelia-cosem-datasets"
        self.api_base = "https://openorganelle.janelia.org"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"OpenOrganelle downloader initialized. Output directory: {output_dir}")
    
    def list_datasets(self) -> List[str]:
        """
        List available datasets on OpenOrganelle.
        
        Returns:
            List of dataset names
        """
        try:
            # Use fsspec to list datasets
            fs = fsspec.filesystem('s3', anon=True)
            datasets = fs.ls(self.base_s3_url.replace('s3://', ''))
            dataset_names = [os.path.basename(path) for path in datasets if fs.isdir(path)]
            
            logger.info(f"Found {len(dataset_names)} datasets")
            return sorted(dataset_names)
        
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    def _list_s3_groups(self, n5_path: str) -> List[str]:
        """
        List groups in the N5 dataset using direct filesystem access.
        """
        try:
            fs = fsspec.filesystem('s3', anon=True)
            # Remove s3:// prefix for filesystem operations
            clean_path = n5_path.replace('s3://', '')
            
            # List contents
            contents = fs.ls(clean_path)
            
            # Filter for directories (groups)
            groups = []
            for item in contents:
                # Get the base name
                base_name = item.split('/')[-1]
                # Skip attributes.json and other metadata files
                if not base_name.endswith('.json') and fs.isdir(item):
                    groups.append(base_name)
            
            return groups
        except Exception as e:
            logger.error(f"Error listing S3 groups: {e}")
            return []
    
    def _list_s3_arrays_in_group(self, n5_path: str, group_name: str) -> List[str]:
        """
        List arrays in a specific group using direct filesystem access.
        """
        try:
            fs = fsspec.filesystem('s3', anon=True)
            clean_path = n5_path.replace('s3://', '')
            group_path = f"{clean_path}/{group_name}"
            
            contents = fs.ls(group_path)
            arrays = []
            
            for item in contents:
                base_name = item.split('/')[-1]
                if fs.isdir(item):
                    arrays.append(base_name)
            
            return arrays
        except Exception as e:
            logger.error(f"Error listing arrays in group {group_name}: {e}")
            return []
        """
        Open a zarr group with compatibility for both zarr v2 and v3.
        """
        try:
            # Try direct fsspec approach first - most compatible
            logger.info(f"Attempting to open {n5_path}")
            store = fsspec.get_mapper(n5_path, anon=True)
            group = zarr.open(store, mode='r')
            logger.info(f"Successfully opened zarr group with fsspec")
            return group
        except Exception as e:
            logger.error(f"Failed to open zarr group: {e}")
            raise e
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            # Use filesystem-based approach for N5 compatibility
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            
            info = {
                'name': dataset_name,
                'n5_path': n5_path,
                'groups': [],
                'arrays': [],
                'metadata': {}
            }
            
            # List groups using filesystem approach
            info['groups'] = self._list_s3_groups(n5_path)
            
            # Get detailed information about each group
            for group_name in info['groups']:
                try:
                    arrays = self._list_s3_arrays_in_group(n5_path, group_name)
                    info[f'{group_name}_arrays'] = arrays
                    
                    # For some groups, there might be sub-groups
                    subgroups = []
                    for array_name in arrays:
                        subarray_path = f"{n5_path.replace('s3://', '')}/{group_name}/{array_name}"
                        fs = fsspec.filesystem('s3', anon=True)
                        try:
                            sub_contents = fs.ls(subarray_path)
                            # Check if there are further subdivisions
                            sub_dirs = [item for item in sub_contents if fs.isdir(item) and not item.endswith('.json')]
                            if sub_dirs:
                                subgroups.append(array_name)
                        except:
                            pass
                    
                    if subgroups:
                        info[f'{group_name}_groups'] = subgroups
                        
                except Exception as e:
                    logger.warning(f"Could not explore subgroup {group_name}: {e}")
            
            logger.info(f"Retrieved information for dataset: {dataset_name}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_name}: {e}")
            return {'name': dataset_name, 'error': str(e)}
    
    def list_data_types(self, dataset_name: str) -> List[str]:
        """
        List available data types for a dataset (e.g., 'em', 'labels', 'analysis').
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of available data types
        """
        try:
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            group = self._open_zarr_group(n5_path)
            
            data_types = []
            if hasattr(group, 'group_keys'):
                data_types = list(group.group_keys())
            elif hasattr(group, 'keys'):
                # For zarr v3, filter for groups
                all_keys = list(group.keys())
                data_types = [k for k in all_keys if hasattr(group.get(k, None), 'keys')]
                
            logger.info(f"Found data types for {dataset_name}: {data_types}")
            return data_types
            
        except Exception as e:
            logger.error(f"Error listing data types for {dataset_name}: {e}")
            return []
    
    def get_array_info(self, dataset_name: str, data_path: str) -> Dict:
        """
        Get information about a specific array in the dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to the array (e.g., 'em/fibsem-uint16/s0')
            
        Returns:
            Dictionary containing array information
        """
        try:
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            group = self._open_zarr_group(n5_path)
            
            array = group[data_path]
            
            info = {
                'shape': array.shape,
                'dtype': str(array.dtype),
                'chunks': array.chunks if hasattr(array, 'chunks') else None,
                'size_mb': np.prod(array.shape) * array.dtype.itemsize / (1024 * 1024),
                'path': data_path,
                'metadata': dict(array.attrs) if hasattr(array, 'attrs') else {}
            }
            
            logger.info(f"Array info for {dataset_name}/{data_path}: {info['shape']} {info['dtype']}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting array info for {dataset_name}/{data_path}: {e}")
            return {'error': str(e)}
    
    def download_array_slice(self, dataset_name: str, data_path: str, 
                           slice_spec: Tuple = None, output_filename: str = None) -> str:
        """
        Download a slice or subset of an array from the dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to the array (e.g., 'em/fibsem-uint16/s0')
            slice_spec: Tuple specifying the slice (e.g., (slice(0, 100), slice(0, 100), slice(0, 100)))
            output_filename: Name for the output file
            
        Returns:
            Path to the downloaded file
        """
        try:
            # Open the array
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            group = self._open_zarr_group(n5_path)
            zarray = group[data_path]
            
            # Create dask array for efficient processing
            chunks = zarray.chunks if hasattr(zarray, 'chunks') else None
            if chunks is None:
                # If no chunks, use a reasonable default
                chunks = tuple(min(64, s) for s in zarray.shape)
            
            darray = da.from_array(zarray, chunks=chunks)
            
            # Apply slice if specified
            if slice_spec:
                data = darray[slice_spec]
            else:
                # Default to a small sample (first 64x64x64 voxels)
                sample_size = min(64, min(darray.shape))
                data = darray[:sample_size, :sample_size, :sample_size]
            
            # Generate output filename if not provided
            if not output_filename:
                slice_str = "sample" if not slice_spec else "slice"
                output_filename = f"{dataset_name}_{data_path.replace('/', '_')}_{slice_str}.npy"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Compute and save the data
            logger.info(f"Downloading data slice: {data.shape}")
            result = data.compute()
            np.save(output_path, result)
            
            logger.info(f"Data saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading array slice: {e}")
            return None
    
    def download_metadata(self, dataset_name: str) -> str:
        """
        Download metadata for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the metadata file
        """
        try:
            info = self.get_dataset_info(dataset_name)
            
            output_filename = f"{dataset_name}_metadata.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            logger.info(f"Metadata saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading metadata: {e}")
            return None
    
    def explore_dataset(self, dataset_name: str):
        """
        Explore a dataset and print available data.
        
        Args:
            dataset_name: Name of the dataset to explore
        """
        print(f"\n=== Exploring Dataset: {dataset_name} ===")
        
        # Get basic info
        info = self.get_dataset_info(dataset_name)
        if 'error' in info:
            print(f"Error: {info['error']}")
            return
        
        print(f"Groups: {info.get('groups', [])}")
        print(f"Arrays: {info.get('arrays', [])}")
        
        # Explore each group
        for group_name in info.get('groups', []):
            print(f"\n--- Group: {group_name} ---")
            
            # List arrays in this group
            group_arrays = info.get(f'{group_name}_arrays', [])
            group_groups = info.get(f'{group_name}_groups', [])
            
            if group_arrays:
                print(f"Arrays: {group_arrays}")
            if group_groups:
                print(f"Subgroups: {group_groups}")
            
            # Get info for each array
            for array_name in group_arrays[:3]:  # Limit to first 3 arrays
                try:
                    array_path = f"{group_name}/{array_name}"
                    array_info = self.get_array_info(dataset_name, array_path)
                    if 'error' not in array_info:
                        print(f"  {array_name}: {array_info['shape']} {array_info['dtype']} "
                              f"({array_info['size_mb']:.1f} MB)")
                except:
                    pass


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Download data from OpenOrganelle')
    parser.add_argument('--list-datasets', action='store_true', 
                       help='List available datasets')
    parser.add_argument('--explore', type=str, metavar='DATASET',
                       help='Explore a specific dataset')
    parser.add_argument('--download', type=str, metavar='DATASET',
                       help='Download data from a dataset')
    parser.add_argument('--data-path', type=str, default='em/fibsem-uint16/s0',
                       help='Path to data within dataset (default: em/fibsem-uint16/s0)')
    parser.add_argument('--output-dir', type=str, default='./downloads',
                       help='Output directory for downloads')
    parser.add_argument('--sample-size', type=int, default=64,
                       help='Size of sample cube to download (default: 64)')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = OpenOrganelleDownloader(output_dir=args.output_dir)
    
    if args.list_datasets:
        print("Available datasets:")
        datasets = downloader.list_datasets()
        for i, dataset in enumerate(datasets, 1):
            print(f"{i:2d}. {dataset}")
    
    elif args.explore:
        downloader.explore_dataset(args.explore)
    
    elif args.download:
        print(f"Downloading sample from dataset: {args.download}")
        print(f"Data path: {args.data_path}")
        
        # Download metadata
        downloader.download_metadata(args.download)
        
        # Download a sample
        sample_slice = (slice(0, args.sample_size), 
                       slice(0, args.sample_size), 
                       slice(0, args.sample_size))
        
        output_file = downloader.download_array_slice(
            args.download, args.data_path, sample_slice
        )
        
        if output_file:
            print(f"Sample data downloaded to: {output_file}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
