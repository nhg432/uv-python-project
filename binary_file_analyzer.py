#!/usr/bin/env python3
"""
🔍 Binary File Image Analyzer
Analyzes binary files to determine if they contain image data and attempts to display them.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gzip

def analyze_binary_file(file_path):
    """
    Comprehensive analysis of a binary file to determine if it contains image data.
    
    Args:
        file_path (str): Path to the binary file to analyze
    """
    print("🔍 Binary File Image Analyzer")
    print("=" * 60)
    print(f"📁 Analyzing file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return False
    
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    success = False
    
    # Method 1: Try to open as a standard image format
    print("\n🖼️ Method 1: Attempting to open as standard image...")
    try:
        img = Image.open(file_path)
        print(f"✅ Successfully opened as image!")
        print(f"   Format: {img.format}")
        print(f"   Mode: {img.mode}")
        print(f"   Size: {img.size}")
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Image from file: {os.path.basename(file_path)}\nFormat: {img.format}, Size: {img.size}")
        plt.axis('off')
        plt.show()
        
        success = True
        
    except Exception as e:
        print(f"❌ Failed to open as standard image: {str(e)[:100]}")
        
        # Method 2: Try to interpret as raw binary data
        print("\n🔢 Method 2: Interpreting as raw binary data...")
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Convert to numpy array
            data_array = np.frombuffer(data, dtype=np.uint8)
            print(f"   📊 Data array shape: {data_array.shape}")
            print(f"   📈 Value range: {data_array.min()} - {data_array.max()}")
            
            # Try different image dimensions
            total_pixels = len(data_array)
            possible_dims = []
            
            # Look for common image dimensions
            for width in [32, 64, 128, 256, 512, 1024, 2048]:
                if total_pixels % width == 0:
                    height = total_pixels // width
                    if height > 0 and height < 10000:  # Reasonable height limit
                        possible_dims.append((width, height))
            
            print(f"   🎯 Possible dimensions: {possible_dims[:8]}")
            
            if possible_dims:
                # Try the most square-like dimension
                best_dim = min(possible_dims, key=lambda x: abs(x[0] - x[1]))
                width, height = best_dim
                
                if width * height <= total_pixels:
                    # Reshape and display
                    img_data = data_array[:width*height].reshape(height, width)
                    
                    plt.figure(figsize=(15, 6))
                    
                    # Original interpretation
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_data, cmap='gray')
                    plt.title(f'Raw Binary as Grayscale\n{width}x{height}')
                    plt.colorbar()
                    
                    # Enhanced contrast
                    plt.subplot(1, 3, 2)
                    plt.imshow(img_data, cmap='viridis')
                    plt.title(f'Enhanced Contrast\n{width}x{height}')
                    plt.colorbar()
                    
                    # Try different data type interpretation
                    if len(data_array) % 2 == 0:
                        data_uint16 = np.frombuffer(data, dtype=np.uint16)
                        if len(data_uint16) >= width * height // 2:
                            img_data_16 = data_uint16[:width*height//2].reshape(height//2, width)
                            plt.subplot(1, 3, 3)
                            plt.imshow(img_data_16, cmap='plasma')
                            plt.title(f'As 16-bit data\n{width}x{height//2}')
                            plt.colorbar()
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"✅ Displayed as {width}x{height} image")
                    success = True
                else:
                    print("❌ Calculated dimensions exceed available data")
            else:
                print("❌ Could not determine suitable image dimensions")
                
        except Exception as e:
            print(f"❌ Failed to interpret as binary data: {str(e)[:100]}")
    
    # Method 3: Check if it's compressed data
    print("\n📦 Method 3: Checking for compressed data...")
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to check for gzip magic number
            header = f.read(10)
            f.seek(0)  # Reset file pointer
            
        if header[:2] == b'\x1f\x8b':  # gzip magic number
            print("✅ Detected gzip compression!")
            try:
                with gzip.open(file_path, 'rb') as f:
                    decompressed = f.read()
                print(f"   📊 Decompressed size: {len(decompressed):,} bytes")
                
                # Try to interpret decompressed data as image
                decompressed_array = np.frombuffer(decompressed, dtype=np.uint8)
                print(f"   📈 Decompressed value range: {decompressed_array.min()} - {decompressed_array.max()}")
                
                # Save decompressed data to temporary file and analyze
                temp_file = file_path + "_decompressed"
                with open(temp_file, 'wb') as f:
                    f.write(decompressed)
                
                print(f"   💾 Saved decompressed data to: {temp_file}")
                print("   🔄 Analyzing decompressed data...")
                analyze_binary_file(temp_file)  # Recursive call
                
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                
            except Exception as e:
                print(f"❌ Failed to decompress: {str(e)[:50]}")
        else:
            print("❌ Not gzip compressed")
            
    except Exception as e:
        print(f"❌ Error checking compression: {str(e)[:50]}")
    
    # Method 4: Hexdump analysis
    print("\n🔍 Method 4: File header analysis...")
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)
        
        print(f"   📄 First 32 bytes (hex): {header.hex()}")
        print(f"   📄 First 32 bytes (repr): {repr(header)}")
        
        # Check for known file signatures
        signatures = {
            b'\xff\xd8\xff': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF8': 'GIF',
            b'BM': 'BMP',
            b'RIFF': 'RIFF (possibly WebP)',
            b'\x1f\x8b': 'GZIP',
            b'PK\x03\x04': 'ZIP',
            b'\x00\x00\x01\x00': 'ICO',
            b'II*\x00': 'TIFF (little endian)',
            b'MM\x00*': 'TIFF (big endian)',
        }
        
        detected = False
        for sig, format_name in signatures.items():
            if header.startswith(sig):
                print(f"✅ Detected file signature: {format_name}")
                detected = True
                break
        
        if not detected:
            print("❌ No known file signature detected")
            print("   💡 This might be raw image data or a proprietary format")
            
    except Exception as e:
        print(f"❌ Error reading file header: {str(e)[:50]}")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Analysis complete! Image data was successfully displayed above.")
    else:
        print("❌ Analysis complete! No displayable image data was found.")
        print("💡 The file may be in a proprietary format, corrupted, or not contain image data.")
    
    return success

def main():
    """Main function to run the binary file analyzer."""
    # You can change this path to analyze different files
    file_path = r"C:\Users\nhg43\Downloads\1 (1)"
    
    analyze_binary_file(file_path)

if __name__ == "__main__":
    main()
