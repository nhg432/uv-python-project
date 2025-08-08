#!/usr/bin/env python3
"""
🔍 Enhanced Binary File Image Analyzer
Advanced analysis of binary files with more dimension possibilities and data interpretations.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gzip
import math

def find_image_dimensions(total_bytes, channels=1):
    """Find possible image dimensions for given total bytes."""
    total_pixels = total_bytes // channels
    dimensions = []
    
    # Calculate possible square and rectangular dimensions
    sqrt_pixels = int(math.sqrt(total_pixels))
    
    # Try dimensions around the square root
    for width in range(max(1, sqrt_pixels - 200), sqrt_pixels + 200):
        if total_pixels % width == 0:
            height = total_pixels // width
            if 10 <= height <= 10000 and 10 <= width <= 10000:  # Reasonable bounds
                aspect_ratio = max(width, height) / min(width, height)
                dimensions.append((width, height, aspect_ratio))
    
    # Sort by aspect ratio (prefer more square images)
    dimensions.sort(key=lambda x: x[2])
    return dimensions[:15]  # Return top 15 candidates

def analyze_enhanced_binary_file(file_path):
    """
    Enhanced analysis of a binary file to determine if it contains image data.
    """
    print("🔍 Enhanced Binary File Image Analyzer")
    print("=" * 60)
    print(f"📁 Analyzing file: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return False
    
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Read the binary data
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Method 1: Standard image formats (already tried in basic analyzer)
    print("\n🖼️ Method 1: Standard image formats...")
    try:
        img = Image.open(file_path)
        print(f"✅ Successfully opened as standard image!")
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Standard Image: {os.path.basename(file_path)}")
        plt.axis('off')
        plt.show()
        return True
    except:
        print("❌ Not a standard image format")
    
    # Method 2: Enhanced raw binary interpretation
    print("\n🔢 Method 2: Enhanced raw binary analysis...")
    data_array = np.frombuffer(data, dtype=np.uint8)
    print(f"   📊 Total bytes: {len(data_array):,}")
    print(f"   📈 Value range: {data_array.min()} - {data_array.max()}")
    print(f"   📊 Unique values: {len(np.unique(data_array))}")
    
    # Try different data type interpretations
    interpretations = [
        ('uint8', np.uint8, 1),
        ('uint16_le', np.uint16, 2),
        ('uint32_le', np.uint32, 4),
        ('float32_le', np.float32, 4),
    ]
    
    best_results = []
    
    for dtype_name, dtype, bytes_per_pixel in interpretations:
        if len(data) % bytes_per_pixel != 0:
            continue
            
        try:
            typed_data = np.frombuffer(data, dtype=dtype)
            dimensions = find_image_dimensions(len(typed_data))
            
            if dimensions:
                print(f"\n   🔍 Trying {dtype_name} interpretation...")
                print(f"      Data points: {len(typed_data):,}")
                print(f"      Value range: {typed_data.min()} - {typed_data.max()}")
                print(f"      Top dimensions: {dimensions[:3]}")
                
                # Try the best dimension
                width, height, aspect_ratio = dimensions[0]
                if width * height <= len(typed_data):
                    img_data = typed_data[:width*height].reshape(height, width)
                    best_results.append((dtype_name, img_data, width, height, aspect_ratio))
        except Exception as e:
            print(f"      ❌ Failed {dtype_name}: {str(e)[:50]}")
    
    # Display best interpretations
    if best_results:
        print(f"\n🎨 Displaying {len(best_results)} interpretation(s)...")
        
        fig_width = min(5 * len(best_results), 20)
        plt.figure(figsize=(fig_width, 10))
        
        for i, (dtype_name, img_data, width, height, aspect_ratio) in enumerate(best_results):
            # Normalize data for display
            if img_data.dtype in [np.float32, np.float64]:
                # Handle float data
                if np.isfinite(img_data).all():
                    img_normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                else:
                    img_normalized = np.nan_to_num(img_data)
            else:
                # Handle integer data
                img_normalized = img_data.astype(np.float64)
                if img_normalized.max() > 1:
                    img_normalized = img_normalized / img_normalized.max()
            
            # Display with different colormaps
            plt.subplot(2, len(best_results), i+1)
            plt.imshow(img_normalized, cmap='gray')
            plt.title(f'{dtype_name}\n{width}x{height}\nAR: {aspect_ratio:.1f}')
            plt.axis('off')
            
            plt.subplot(2, len(best_results), i+1+len(best_results))
            plt.imshow(img_normalized, cmap='viridis')
            plt.title(f'{dtype_name} (enhanced)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    # Method 3: Check for specific patterns or headers
    print("\n🔍 Method 3: Pattern analysis...")
    
    # Look for repeating patterns that might indicate structure
    header = data[:64]
    print(f"   📄 First 64 bytes (hex): {header.hex()}")
    
    # Check for potential image headers at different offsets
    offsets_to_check = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    for offset in offsets_to_check:
        if offset < len(data):
            chunk = data[offset:offset+16]
            # Look for patterns that might indicate width/height
            if len(chunk) >= 8:
                # Try interpreting as little-endian 32-bit integers
                try:
                    vals = np.frombuffer(chunk[:8], dtype=np.uint32)
                    if len(vals) >= 2:
                        w, h = vals[0], vals[1]
                        if 10 <= w <= 5000 and 10 <= h <= 5000:
                            total_pixels = w * h
                            remaining_bytes = len(data) - offset - 8
                            bytes_per_pixel = remaining_bytes / total_pixels
                            if 0.5 <= bytes_per_pixel <= 4:
                                print(f"   🎯 Potential header at offset {offset}: {w}x{h} ({bytes_per_pixel:.1f} bpp)")
                                
                                # Try to extract and display this potential image
                                try:
                                    img_start = offset + 8
                                    img_bytes = int(w * h * bytes_per_pixel)
                                    if img_start + img_bytes <= len(data):
                                        img_data_bytes = data[img_start:img_start + img_bytes]
                                        
                                        if bytes_per_pixel == 1:
                                            img_array = np.frombuffer(img_data_bytes[:w*h], dtype=np.uint8)
                                        elif bytes_per_pixel == 2:
                                            img_array = np.frombuffer(img_data_bytes[:w*h*2], dtype=np.uint16)
                                        else:
                                            continue
                                        
                                        img_2d = img_array.reshape(h, w)
                                        
                                        plt.figure(figsize=(12, 6))
                                        plt.subplot(1, 2, 1)
                                        plt.imshow(img_2d, cmap='gray')
                                        plt.title(f'Potential Image at offset {offset}\n{w}x{h}, {bytes_per_pixel:.1f} bpp')
                                        plt.colorbar()
                                        
                                        plt.subplot(1, 2, 2)
                                        plt.imshow(img_2d, cmap='viridis')
                                        plt.title('Enhanced contrast')
                                        plt.colorbar()
                                        
                                        plt.tight_layout()
                                        plt.show()
                                        
                                        print(f"   ✅ Successfully displayed potential image!")
                                        return True
                                        
                                except Exception as e:
                                    print(f"      ❌ Failed to extract image: {str(e)[:50]}")
                except:
                    pass
    
    print("\n❌ No clear image patterns found")
    return False

def main():
    """Main function to run the enhanced binary file analyzer."""
    file_path = r"C:\Users\nhg43\Downloads\1 (1)"
    
    success = analyze_enhanced_binary_file(file_path)
    
    if not success:
        print("\n💡 Suggestions:")
        print("   • The file might be compressed or encrypted")
        print("   • It could be a proprietary format requiring specific software")
        print("   • Try opening with hex editor to examine structure manually")
        print("   • Check if it's part of a larger dataset with documentation")

if __name__ == "__main__":
    main()
