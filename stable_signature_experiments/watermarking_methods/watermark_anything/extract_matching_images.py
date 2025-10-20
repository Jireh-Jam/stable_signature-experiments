#!/usr/bin/env python3
"""
Script to extract images from a source folder that match the base part of 
filenames in a reference folder (e.g., _original matches _watermarked)
"""

import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

def get_base_filename(filename):
    """Extract the base part of a filename (before the last underscore + suffix)"""
    # Common suffixes to remove
    suffixes_to_remove = ['_watermarked', '_original', '_edited', '_processed', '_copy']
    
    name_without_ext = filename
    
    # Try to remove known suffixes first
    for suffix in suffixes_to_remove:
        if name_without_ext.endswith(suffix):
            return name_without_ext[:-len(suffix)]
    
    # If no known suffix found, split on last underscore
    if '_' in name_without_ext:
        parts = name_without_ext.rsplit('_', 1)  # Split from right, only once
        return parts[0]
    
    # If no underscore, return the whole name
    return name_without_ext

def get_image_files(folder_path, return_ordered=False):
    """Get all image files from a folder with base filename mapping"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico'}
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return {} if not return_ordered else ([], {})
    
    image_files = {}
    ordered_files = []
    
    # Get files in sorted order to maintain consistent ordering
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Extract base filename (without suffix like _watermarked, _original)
            base_name = get_base_filename(file_path.stem)
            image_files[base_name] = file_path
            ordered_files.append(base_name)
    
    if return_ordered:
        return ordered_files, image_files
    return image_files

def extract_matching_images(source_folder, reference_folder, output_folder, dry_run=False):
    """
    Extract images from source folder that match the base part of filenames in reference folder
    Processing follows the sorted order of the reference folder
    
    Args:
        source_folder: Folder containing images to extract from
        reference_folder: Folder with existing images (used for filename matching)
        output_folder: Destination folder for extracted images
        dry_run: If True, only shows what would be copied without actually copying
    """
    
    print(f"Source folder: {os.path.abspath(source_folder)}")
    print(f"Reference folder: {os.path.abspath(reference_folder)}")
    print(f"Output folder: {os.path.abspath(output_folder)}")
    print("-" * 70)
    
    # Get image files - reference folder maintains sorted order
    source_images = get_image_files(source_folder)
    reference_order, reference_images = get_image_files(reference_folder, return_ordered=True)
    
    if not source_images:
        print(f"No image files found in source folder: {source_folder}")
        return
    
    if not reference_images:
        print(f"No image files found in reference folder: {reference_folder}")
        return
    
    print(f"Found {len(source_images)} images in source folder")
    print(f"Found {len(reference_images)} images in reference folder")
    print(f"Processing in reference folder order (sorted)")
    print("Matching strategy: Base filename parts (e.g., '0_123abc_original' matches '0_123abc_watermarked')")
    print()
    
    # Show some examples of base filename extraction
    print("Base filename extraction examples:")
    sample_ref = list(reference_images.keys())[:3]
    sample_src = list(source_images.keys())[:3]
    
    for i, ref_base in enumerate(sample_ref):
        ref_file = reference_images[ref_base]
        print(f"  Reference: {ref_file.name} -> base: '{ref_base}'")
        
    for i, src_base in enumerate(sample_src):
        src_file = source_images[src_base]
        print(f"  Source: {src_file.name} -> base: '{src_base}'")
    print()
    
    # Find matching files in the order of reference folder
    matches = []
    
    for ref_base in reference_order:
        if ref_base in source_images:
            source_file = source_images[ref_base]
            reference_file = reference_images[ref_base]
            matches.append((source_file, reference_file, ref_base))
    
    print(f"Found {len(matches)} matching files (in reference folder order):")
    print()
    
    if not matches:
        print("No matching files found.")
        print("Note: Script compares base filenames (before '_original', '_watermarked', etc.)")
        print("Example: '0_123abc_original.png' matches '0_123abc_watermarked.png'")
        return
    
    # Create output directory if it doesn't exist (unless dry run)
    output_path = Path(output_folder)
    if not dry_run and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy matching files in the order they appear in reference folder
    copied_count = 0
    skipped_count = 0
    
    for i, (source_file, reference_file, base_name) in enumerate(matches, 1):
        output_file = output_path / source_file.name
        
        print(f"[{i:3d}/{len(matches)}] Base: {base_name}")
        print(f"         Source: {source_file.name}")
        print(f"      Reference: {reference_file.name}")
        
        if dry_run:
            print(f"      [DRY RUN] Would copy: {source_file.name} -> {output_file.name}")
        else:
            try:
                # Check if file already exists
                if output_file.exists():
                    print(f"         [SKIP] File already exists: {output_file.name}")
                    skipped_count += 1
                else:
                    # Copy the file
                    shutil.copy2(source_file, output_file)
                    print(f"         [COPY] Copied to: {output_file.name}")
                    copied_count += 1
                
            except Exception as e:
                print(f"        [ERROR] Failed to copy {source_file.name}: {e}")
        
        print()
    
    # Summary
    print("-" * 70)
    if dry_run:
        print(f"DRY RUN COMPLETE: {len(matches)} files would be processed")
        print("Files would be processed in reference folder order")
    else:
        print(f"EXTRACTION COMPLETE (processed in reference folder order):")
        print(f"  Files copied: {copied_count}")
        print(f"  Files skipped: {skipped_count}")
        print(f"  Total processed: {copied_count + skipped_count}")
        print(f"  Reference order maintained: YES")

def find_similar_names(source_folder, reference_folder, threshold=0.7):
    """Find files with similar names using basic string similarity"""
    from difflib import SequenceMatcher
    
    source_images = get_image_files(source_folder)
    reference_order, reference_images = get_image_files(reference_folder, return_ordered=True)
    
    similar_pairs = []
    
    # Process in reference folder order
    for ref_name in reference_order:
        best_match = None
        best_ratio = 0
        
        for src_name in source_images:
            ratio = SequenceMatcher(None, ref_name.lower(), src_name.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = src_name
        
        if best_match:
            similar_pairs.append((best_match, ref_name, best_ratio))
    
    return similar_pairs

def main():
    parser = argparse.ArgumentParser(description="Extract images with matching filenames (first part matching)")
    parser.add_argument("source", help="Source folder containing images to extract")
    parser.add_argument("reference", help="Reference folder with existing images")
    parser.add_argument("output", help="Output folder for extracted images")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be copied without actually copying")
    parser.add_argument("--similar", action="store_true",
                       help="Find files with similar names (fuzzy matching)")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for fuzzy matching (0.0-1.0, default: 0.7)")
    
    args = parser.parse_args()
    
    # Validate input folders
    if not Path(args.source).exists():
        print(f"Error: Source folder does not exist: {args.source}")
        return
    
    if not Path(args.reference).exists():
        print(f"Error: Reference folder does not exist: {args.reference}")
        return
    
    if args.similar:
        print("Finding files with similar names...")
        similar_pairs = find_similar_names(args.source, args.reference, args.threshold)
        if similar_pairs:
            print(f"\nFound {len(similar_pairs)} similar filename pairs:")
            for src_name, ref_name, ratio in similar_pairs:
                print(f"  {src_name} <-> {ref_name} (similarity: {ratio:.2f})")
        else:
            print("No similar filenames found.")
        return
    
    extract_matching_images(
        args.source, 
        args.reference, 
        args.output,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()