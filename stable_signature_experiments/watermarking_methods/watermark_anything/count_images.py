#!/usr/bin/env python3
"""
Enhanced script to count the number of image files in a folder
Supports common image formats and provides detailed statistics
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter

def count_images(folder_path, recursive=False, show_breakdown=True):
    """Count image files in the specified folder"""
    
    # Common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico'}
    
    try:
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: Folder '{folder_path}' does not exist.")
            return None
            
        if not folder.is_dir():
            print(f"Error: '{folder_path}' is not a directory.")
            return None
        
        # Count image files
        image_files = []
        extension_count = Counter()
        
        # Use rglob for recursive search, glob for non-recursive
        search_pattern = "**/*" if recursive else "*"
        
        for file_path in folder.glob(search_pattern):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                if file_extension in image_extensions:
                    image_files.append(file_path)
                    extension_count[file_extension] += 1
        
        return len(image_files), image_files, extension_count
        
    except PermissionError:
        print(f"Error: Permission denied accessing '{folder_path}'.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def main():
    parser = argparse.ArgumentParser(description="Count image files in a folder")
    parser.add_argument("folder", nargs="?", default=".", 
                       help="Folder path to count images in (default: current directory)")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="Search recursively in subdirectories")
    parser.add_argument("-l", "--list", action="store_true",
                       help="List all image files found")
    parser.add_argument("-s", "--size", action="store_true",
                       help="Show file sizes")
    parser.add_argument("--no-breakdown", action="store_true",
                       help="Don't show breakdown by file type")
    
    args = parser.parse_args()
    
    folder_path = args.folder
    
    print(f"Counting images in: {os.path.abspath(folder_path)}")
    if args.recursive:
        print("(Including subdirectories)")
    print("-" * 60)
    
    result = count_images(folder_path, args.recursive, not args.no_breakdown)
    
    if result is not None:
        image_count, image_files, extension_count = result
        print(f"Total images found: {image_count}")
        
        # Show breakdown by file type
        if not args.no_breakdown and extension_count:
            print("\nBreakdown by file type:")
            for ext, count in sorted(extension_count.items()):
                print(f"  {ext.upper():5s}: {count:4d}")
        
        # Calculate total size if requested
        if args.size and image_files:
            total_size = sum(file_path.stat().st_size for file_path in image_files)
            print(f"\nTotal size: {format_file_size(total_size)}")
        
        # List files if requested
        if args.list and image_files:
            print(f"\nImage files:")
            for i, file_path in enumerate(sorted(image_files), 1):
                if args.size:
                    file_size = format_file_size(file_path.stat().st_size)
                    relative_path = file_path.relative_to(folder_path)
                    print(f"{i:3d}. {relative_path} ({file_size})")
                else:
                    relative_path = file_path.relative_to(folder_path)
                    print(f"{i:3d}. {relative_path}")
    else:
        print("Failed to count images.")

if __name__ == "__main__":
    main()