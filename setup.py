"""
Setup script for Watermark Robustness Testing Pipeline

This script helps you set up the pipeline quickly by:
1. Creating necessary directories
2. Verifying dependencies
3. Checking for required files
4. Optionally downloading sample data

Usage:
    python setup.py --method trustmark
    python setup.py --method stable_signature --download-models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def create_directory_structure():
    """Create the necessary directory structure for the pipeline."""
    print("Creating directory structure...")
    
    directories = [
        "watermark_models/stable-signature/models",
        "watermark_models/stable-signature/watermarked_images",
        "watermark_models/watermark-anything/models",
        "watermark_models/watermark-anything/watermarked_images",
        "watermark_models/trustmark/watermarked_images",
        "outputs/stable_signature/transformed_images",
        "outputs/stable_signature/results",
        "outputs/watermark_anything/transformed_images",
        "outputs/watermark_anything/results",
        "outputs/trustmark/transformed_images",
        "outputs/trustmark/results",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory}")
    
    print("✓ Directory structure created successfully\n")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}\n")
    return True


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("skimage", "scikit-image"),
        ("cv2", "opencv-python"),
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        return False, missing
    
    print("✓ All core dependencies installed\n")
    return True, []


def install_dependencies(missing_packages: List[str] = None):
    """Install missing dependencies."""
    print("Installing dependencies...")
    
    if missing_packages:
        print(f"Installing: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
        )
        print("✓ Dependencies installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}\n")
        return False


def install_method_dependencies(method: str):
    """Install method-specific dependencies."""
    print(f"Installing dependencies for {method}...")
    
    if method == "trustmark":
        packages = ["trustmark"]
    elif method == "stable_signature":
        packages = ["omegaconf", "einops"]
    elif method == "watermark_anything":
        packages = ["timm"]
    else:
        print(f"  Unknown method: {method}")
        return False
    
    try:
        for package in packages:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )
            print(f"  ✓ Installed {package}")
        print(f"✓ {method} dependencies installed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {method} dependencies: {e}\n")
        return False


def download_stable_signature_models():
    """Download Stable Signature pre-trained models."""
    print("Downloading Stable Signature models...")
    
    model_dir = Path("watermark_models/stable-signature/models")
    model_url = "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt"
    model_path = model_dir / "dec_48b_whit.torchscript.pt"
    
    if model_path.exists():
        print(f"  ✓ Model already exists: {model_path}")
        return True
    
    try:
        import urllib.request
        print(f"  Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"  ✓ Model downloaded to {model_path}\n")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download model: {e}")
        print(f"  Please download manually from: {model_url}\n")
        return False


def create_sample_trustmark_image():
    """Create a sample TrustMark watermarked image for testing."""
    print("Creating sample TrustMark image...")
    
    output_dir = Path("watermark_models/trustmark/watermarked_images")
    output_path = output_dir / "sample_001.png"
    
    if output_path.exists():
        print(f"  ✓ Sample image already exists: {output_path}")
        return True
    
    try:
        from trustmark import TrustMark
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Initialise TrustMark
        tm = TrustMark(verbose=False, model_type='Q')
        
        # Create watermark
        watermark_bits = [1, 0, 1, 1, 0, 1, 0, 0] * 4  # 32 bits
        
        # Encode and save
        watermarked = tm.encode(image, watermark_bits)
        watermarked.save(output_path)
        
        print(f"  ✓ Sample image created: {output_path}\n")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create sample image: {e}")
        print(f"  You'll need to create watermarked images manually\n")
        return False


def verify_setup(method: str):
    """Verify that the setup is correct for the specified method."""
    print(f"Verifying setup for {method}...")
    
    from config import create_config
    
    try:
        config = create_config(method=method.replace("_", " ").title().replace(" ", "_"))
        is_valid, issues = config.verify_setup()
        
        if is_valid:
            print("✓ Setup verification passed!\n")
            config.print_configuration()
            return True
        else:
            print("⚠️  Setup verification found issues:")
            for issue in issues:
                print(f"  - {issue}")
            print()
            return False
    except Exception as e:
        print(f"✗ Verification failed: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Set up the Watermark Robustness Testing Pipeline"
    )
    parser.add_argument(
        "--method",
        choices=["trustmark", "stable_signature", "watermark_anything"],
        default="trustmark",
        help="Watermarking method to set up (default: trustmark)"
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download pre-trained models (Stable Signature only)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample watermarked image (TrustMark only)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies automatically"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WATERMARK PIPELINE SETUP")
    print("=" * 70)
    print(f"Method: {args.method}\n")
    
    # Check Python version
    if not check_python_version():
        print("Please upgrade to Python 3.8 or higher")
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Check and install dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        if args.install_deps:
            install_dependencies(missing)
        else:
            print("To install missing dependencies, run:")
            print("  pip install -r requirements.txt")
            print("or rerun with --install-deps flag\n")
    
    # Install method-specific dependencies
    if args.install_deps or not deps_ok:
        install_method_dependencies(args.method)
    
    # Download models if requested
    if args.download_models and args.method == "stable_signature":
        download_stable_signature_models()
    
    # Create sample image if requested
    if args.create_sample and args.method == "trustmark":
        create_sample_trustmark_image()
    
    # Verify setup
    if not args.skip_verify:
        verify_setup(args.method)
    
    print("=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Open pipeline_mk4.ipynb in Jupyter")
    print("2. Update configuration in Section 1")
    print("3. Run the cells sequentially")
    print("\nFor more information, see QUICKSTART.md\n")


if __name__ == "__main__":
    main()
