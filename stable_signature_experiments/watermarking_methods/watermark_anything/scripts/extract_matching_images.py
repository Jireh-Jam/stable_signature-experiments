#!/usr/bin/env python3
"""
CLI to extract images from a source folder that match the base part of filenames
in a reference folder (e.g., _original matches _watermarked).

Moved under scripts/ for better project organization.
"""

from watermarking_methods.watermark_anything.extract_matching_images import main

if __name__ == "__main__":
    main()
