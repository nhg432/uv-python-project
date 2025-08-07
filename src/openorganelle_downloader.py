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
import gzip
import struct

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

    def _read_n5_chunk(self, fs, chunk_path: str, dtype: str, shape: tuple) -> np.ndarray:
        """
        Read an individual N5 chunk file directly.
        
        Args:
            fs: Filesystem object
            chunk_path: Path to the chunk file
            dtype: Data type of the chunk
            shape: Expected shape of the chunk
            
        Returns:
            Numpy array containing the chunk data
        """
        try:
            # Convert dtype string to numpy dtype first
            if dtype == 'uint8':
                np_dtype = np.uint8
            elif dtype == 'uint16':
                np_dtype = np.uint16
            elif dtype == 'uint32':
                np_dtype = np.uint32
            elif dtype == 'uint64':
                np_dtype = np.uint64
            elif dtype == 'int8':
                np_dtype = np.int8
            elif dtype == 'int16':
                np_dtype = np.int16
            elif dtype == 'int32':
                np_dtype = np.int32
            elif dtype == 'int64':
                np_dtype = np.int64
            elif dtype == 'float32':
                np_dtype = np.float32
            elif dtype == 'float64':
                np_dtype = np.float64
            else:
                np_dtype = np.uint16  # Default fallback
            
            # Read the chunk file
            with fs.open(chunk_path, 'rb') as f:
                data = f.read()
            
            # N5 chunks are typically gzip compressed
            if data.startswith(b'\x1f\x8b'):  # gzip magic number
                try:
                    decompressed = gzip.decompress(data)
                except Exception as gzip_error:
                    logger.warning(f"Gzip decompression failed: {gzip_error}")
                    return np.zeros(shape, dtype=np_dtype)
            else:
                decompressed = data
            
            # N5 format has a header before the actual data
            # Try to skip potential header bytes and parse as array
            expected_size = np.prod(shape) * np_dtype().itemsize
            
            # Try different header skip amounts
            for header_skip in [0, 4, 8, 12, 16, 20, 24]:
                if len(decompressed) >= header_skip + expected_size:
                    try:
                        array_data = decompressed[header_skip:header_skip + expected_size]
                        array = np.frombuffer(array_data, dtype=np_dtype)
                        
                        if len(array) == np.prod(shape):
                            array = array.reshape(shape)
                            logger.debug(f"Successfully parsed chunk with {header_skip} byte header skip")
                            return array
                            
                    except Exception as parse_error:
                        continue
            
            # If standard parsing fails, try reading as much data as possible
            try:
                # Skip any header and read available data
                if len(decompressed) > 20:
                    # Try skipping first 20 bytes (common N5 header size)
                    array_data = decompressed[20:]
                    # Calculate how many elements we can read
                    available_elements = len(array_data) // np_dtype().itemsize
                    
                    if available_elements > 0:
                        array = np.frombuffer(array_data[:available_elements * np_dtype().itemsize], dtype=np_dtype)
                        
                        # Pad or truncate to expected shape
                        expected_elements = np.prod(shape)
                        if len(array) >= expected_elements:
                            array = array[:expected_elements].reshape(shape)
                        else:
                            # Pad with zeros
                            padded = np.zeros(expected_elements, dtype=np_dtype)
                            padded[:len(array)] = array
                            array = padded.reshape(shape)
                        
                        return array
                        
            except Exception as fallback_error:
                logger.warning(f"Fallback parsing failed: {fallback_error}")
            
            # Return empty chunk if all parsing fails
            logger.warning(f"Could not parse chunk data, returning zeros")
            return np.zeros(shape, dtype=np_dtype)
            
        except Exception as e:
            logger.warning(f"Failed to read N5 chunk {chunk_path}: {e}")
            # Return empty chunk with correct shape and dtype
            return np.zeros(shape, dtype=np_dtype)

    def _download_n5_slice_direct(self, dataset_name: str, data_path: str, 
                                slice_spec: Tuple = None) -> Optional[np.ndarray]:
        """
        Download N5 data by reading chunks directly from S3.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to the array (e.g., 'labels/mito_seg/s0')
            slice_spec: Tuple specifying the slice
            
        Returns:
            Numpy array with the requested data slice
        """
        try:
            # Get array metadata
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            array_path = f"{n5_path}/{data_path}"
            
            fs = fsspec.filesystem('s3', anon=True)
            clean_path = array_path.replace('s3://', '')
            
            # Read attributes
            attrs_path = f"{clean_path}/attributes.json"
            if not fs.exists(attrs_path):
                logger.error(f"No attributes.json found at {attrs_path}")
                return None
                
            with fs.open(attrs_path, 'r') as f:
                attrs = json.load(f)
            
            # Extract array information
            dimensions = attrs['dimensions']  # [z, y, x]
            block_size = attrs['blockSize']   # [z_block, y_block, x_block]
            dtype = attrs['dataType']
            
            logger.info(f"N5 array: {dimensions}, blocks: {block_size}, dtype: {dtype}")
            
            # Calculate slice bounds
            if slice_spec is None:
                # Default small sample
                slice_spec = (slice(0, min(32, dimensions[0])), 
                             slice(0, min(32, dimensions[1])), 
                             slice(0, min(32, dimensions[2])))
            
            z_slice, y_slice, x_slice = slice_spec
            z_start, z_stop = z_slice.start or 0, z_slice.stop or dimensions[0]
            y_start, y_stop = y_slice.start or 0, y_slice.stop or dimensions[1]
            x_start, x_stop = x_slice.start or 0, x_slice.stop or dimensions[2]
            
            # Calculate which chunks we need
            z_block_start = z_start // block_size[0]
            z_block_stop = (z_stop - 1) // block_size[0] + 1
            y_block_start = y_start // block_size[1]
            y_block_stop = (y_stop - 1) // block_size[1] + 1
            x_block_start = x_start // block_size[2]
            x_block_stop = (x_stop - 1) // block_size[2] + 1
            
            logger.info(f"Reading chunks: z={z_block_start}-{z_block_stop}, "
                       f"y={y_block_start}-{y_block_stop}, x={x_block_start}-{x_block_stop}")
            
            # Initialize output array
            output_shape = (z_stop - z_start, y_stop - y_start, x_stop - x_start)
            
            # Convert dtype for numpy
            if dtype == 'uint16':
                np_dtype = np.uint16
            elif dtype == 'uint8':
                np_dtype = np.uint8
            else:
                np_dtype = np.uint16  # Default
            
            result = np.zeros(output_shape, dtype=np_dtype)
            
            # Read required chunks
            chunks_read = 0
            for z_block in range(z_block_start, z_block_stop):
                for y_block in range(y_block_start, y_block_stop):
                    for x_block in range(x_block_start, x_block_stop):
                        # N5 chunk naming convention
                        chunk_name = f"{z_block}/{y_block}/{x_block}"
                        chunk_path = f"{clean_path}/{chunk_name}"
                        
                        if fs.exists(chunk_path):
                            try:
                                chunk_data = self._read_n5_chunk(
                                    fs, chunk_path, dtype, tuple(block_size)
                                )
                                
                                # Calculate where this chunk goes in the result
                                z_chunk_start = max(0, z_start - z_block * block_size[0])
                                z_chunk_stop = min(block_size[0], z_stop - z_block * block_size[0])
                                y_chunk_start = max(0, y_start - y_block * block_size[1])
                                y_chunk_stop = min(block_size[1], y_stop - y_block * block_size[1])
                                x_chunk_start = max(0, x_start - x_block * block_size[2])
                                x_chunk_stop = min(block_size[2], x_stop - x_block * block_size[2])
                                
                                if (z_chunk_start < z_chunk_stop and 
                                    y_chunk_start < y_chunk_stop and 
                                    x_chunk_start < x_chunk_stop):
                                    
                                    # Extract the relevant portion from the chunk
                                    chunk_slice = chunk_data[z_chunk_start:z_chunk_stop,
                                                           y_chunk_start:y_chunk_stop,
                                                           x_chunk_start:x_chunk_stop]
                                    
                                    # Calculate position in result array
                                    result_z_start = z_block * block_size[0] + z_chunk_start - z_start
                                    result_y_start = y_block * block_size[1] + y_chunk_start - y_start
                                    result_x_start = x_block * block_size[2] + x_chunk_start - x_start
                                    
                                    result_z_end = result_z_start + chunk_slice.shape[0]
                                    result_y_end = result_y_start + chunk_slice.shape[1]
                                    result_x_end = result_x_start + chunk_slice.shape[2]
                                    
                                    # Place chunk data in result
                                    if (result_z_start >= 0 and result_y_start >= 0 and result_x_start >= 0 and
                                        result_z_end <= result.shape[0] and result_y_end <= result.shape[1] and 
                                        result_x_end <= result.shape[2]):
                                        
                                        result[result_z_start:result_z_end,
                                               result_y_start:result_y_end,
                                               result_x_start:result_x_end] = chunk_slice
                                        
                                        chunks_read += 1
                                        
                            except Exception as chunk_error:
                                logger.warning(f"Failed to read chunk {chunk_name}: {chunk_error}")
                                continue
            
            logger.info(f"Successfully read {chunks_read} chunks, result shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to read N5 data directly: {e}")
            return None

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
            
            # Use direct filesystem access instead of zarr group
            fs = fsspec.filesystem('s3', anon=True)
            clean_path = n5_path.replace('s3://', '')
            
            contents = fs.ls(clean_path)
            data_types = []
            
            for item in contents:
                base_name = item.split('/')[-1]
                if fs.isdir(item) and not base_name.startswith('.'):
                    data_types.append(base_name)
                
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
            
            # Try to open the array directly using fsspec
            array_path = f"{n5_path}/{data_path}"
            store = fsspec.get_mapper(array_path, anon=True)
            
            try:
                array = zarr.open(store, mode='r')
                
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
                
            except Exception as zarr_error:
                # If zarr fails, try to get basic info from filesystem
                fs = fsspec.filesystem('s3', anon=True)
                clean_path = array_path.replace('s3://', '')
                
                try:
                    # Look for attributes.json file
                    attrs_path = f"{clean_path}/attributes.json"
                    if fs.exists(attrs_path):
                        import json
                        with fs.open(attrs_path, 'r') as f:
                            attrs = json.load(f)
                        
                        info = {
                            'shape': attrs.get('dimensions', 'Unknown'),
                            'dtype': attrs.get('dataType', 'Unknown'),
                            'chunks': attrs.get('blockSize', 'Unknown'),
                            'size_mb': 'Unknown',
                            'path': data_path,
                            'metadata': attrs
                        }
                        
                        logger.info(f"Array info (from attributes) for {dataset_name}/{data_path}")
                        return info
                    else:
                        return {'error': f'Could not access array data: {zarr_error}'}
                        
                except Exception as fs_error:
                    return {'error': f'Array access failed: {zarr_error}, filesystem: {fs_error}'}
            
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
            # Open the array directly using fsspec
            n5_path = f"{self.base_s3_url}/{dataset_name}/{dataset_name}.n5"
            array_path = f"{n5_path}/{data_path}"
            
            # First check if the path exists
            fs = fsspec.filesystem('s3', anon=True)
            clean_array_path = array_path.replace('s3://', '')
            
            if not fs.exists(clean_array_path):
                return None  # Path doesn't exist, return None instead of error
            
            # Try different approaches for N5 compatibility
            try:
                # Method 1: Try zarr with fsspec (standard approach)
                store = fsspec.get_mapper(array_path, anon=True)
                zarray = zarr.open(store, mode='r')
                
            except Exception as zarr_error:
                logger.warning(f"Standard zarr.open failed: {zarr_error}")
                
                # Method 2: Try direct N5 chunk reading
                logger.info("Attempting direct N5 chunk reading...")
                try:
                    # Use our custom N5 reader
                    array_data = self._download_n5_slice_direct(dataset_name, data_path, slice_spec)
                    
                    if array_data is not None:
                        # Generate output filename if not provided
                        if not output_filename:
                            slice_str = "sample" if not slice_spec else "slice"
                            output_filename = f"{dataset_name}_{data_path.replace('/', '_')}_{slice_str}.npy"
                        
                        output_path = os.path.join(self.output_dir, output_filename)
                        
                        # Save the data
                        np.save(output_path, array_data)
                        logger.info(f"N5 data saved to: {output_path}")
                        return output_path
                    else:
                        logger.error("Direct N5 reading failed")
                        return None
                        
                except Exception as n5_error:
                    logger.error(f"Direct N5 reading failed: {n5_error}")
                    return None
            
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
