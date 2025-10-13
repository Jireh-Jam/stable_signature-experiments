#!/usr/bin/env python3
"""
Setup script for the Watermark Testing Pipeline.

This script helps users set up the environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
import argparse
from pathlib import Path


def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🔐 WATERMARK TESTING PIPELINE SETUP")
    print("=" * 60)
    print("This script will help you set up the watermarking pipeline.")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing Python packages")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "experiments/data/raw",
        "experiments/data/watermarked", 
        "experiments/data/transformed",
        "experiments/results",
        "experiments/logs",
        "experiments/temp",
        "models/checkpoints",
        "models/configs",
        "docs/examples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("✅ Directory structure created")
    return True


def download_models(skip_large_models=False):
    """Download watermark models."""
    print("\n🔽 Downloading watermark models...")
    
    models = [
        {
            "name": "Stable Signature Decoder",
            "url": "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt",
            "path": "models/checkpoints/dec_48b_whit.torchscript.pt",
            "size": "~100MB"
        },
        {
            "name": "Alternative Stable Signature Decoder", 
            "url": "https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b_whit.torchscript.pt",
            "path": "models/checkpoints/other_dec_48b_whit.torchscript.pt",
            "size": "~100MB"
        }
    ]
    
    if skip_large_models:
        print("⏭️ Skipping model downloads (--skip-models flag used)")
        print("💡 You can download models later by running:")
        for model in models:
            print(f"   wget {model['url']} -P models/checkpoints/")
        return True
    
    success = True
    for model in models:
        print(f"📥 Downloading {model['name']} ({model['size']})...")
        
        try:
            # Create directory if it doesn't exist
            Path(model['path']).parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists
            if os.path.exists(model['path']):
                print(f"✅ {model['name']} already exists, skipping")
                continue
                
            # Download the file
            urllib.request.urlretrieve(model['url'], model['path'])
            print(f"✅ {model['name']} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Error downloading {model['name']}: {str(e)}")
            success = False
            
    return success


def create_sample_config():
    """Create a sample configuration file."""
    print("\n⚙️ Creating sample configuration...")
    
    config_path = "experiments/configs/user_config.yaml"
    
    if os.path.exists(config_path):
        print("✅ Configuration file already exists")
        return True
        
    sample_config = """# Your Personal Watermark Testing Configuration
# Copy this file and modify the settings below for your experiments

# User Information (PLEASE UPDATE)
user:
  name: "Your.Username"  # Change this to your actual username
  email: "your.email@example.com"
  azure_root_dir: "/home/azureuser/cloudfiles/code/Users/"

# Experiment Settings
experiment:
  name: "my_watermark_test"
  description: "Testing watermark robustness"
  
# Watermarking Method
watermarking:
  method: "stable_signature"  # stable_signature, trustmark, watermark_anything
  message: "my_test_watermark_message_48bits_long"

# Data Settings  
data:
  max_images_to_process: 5  # Start small for testing
  
# What to test
transformations:
  apply_standard: true      # Basic transformations
  apply_aggressive: false   # More challenging transformations
  
# Output settings
evaluation:
  generate_plots: true      # Create charts
  save_detailed_results: true
"""
    
    try:
        with open(config_path, 'w') as f:
            f.write(sample_config)
        print(f"✅ Sample configuration created: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating configuration: {str(e)}")
        return False


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("📋 NEXT STEPS:")
    print("1. 📝 Update your configuration:")
    print("   • Edit experiments/configs/user_config.yaml")
    print("   • Change 'Your.Username' to your actual username")
    print()
    print("2. 🖼️ Add test images:")
    print("   • Place images in experiments/data/raw/")
    print("   • Supported formats: .jpg, .png, .bmp, .tiff")
    print()
    print("3. 🚀 Run the pipeline:")
    print("   • Open: pipeline_mk4_user_friendly.ipynb")
    print("   • Follow the step-by-step instructions")
    print()
    print("4. 📊 View results:")
    print("   • Check experiments/results/ for charts and data")
    print()
    print("💡 HELPFUL COMMANDS:")
    print("   • Start Jupyter: jupyter notebook")
    print("   • View config: cat experiments/configs/user_config.yaml")
    print("   • Check models: ls -la models/checkpoints/")
    print()
    print("🆘 NEED HELP?")
    print("   • Read README.md for detailed instructions")
    print("   • Check docs/ folder for guides")
    print("   • Open an issue on GitHub")
    print()


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up the Watermark Testing Pipeline")
    parser.add_argument("--skip-models", action="store_true", 
                       help="Skip downloading large model files")
    parser.add_argument("--skip-requirements", action="store_true",
                       help="Skip installing Python requirements")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not args.skip_requirements:
        if not install_requirements():
            print("⚠️ Warning: Failed to install some packages")
    else:
        print("⏭️ Skipping requirements installation")
    
    # Create directories
    if not create_directories():
        print("❌ Error: Failed to create directories")
        sys.exit(1)
    
    # Download models
    if not download_models(skip_large_models=args.skip_models):
        print("⚠️ Warning: Some models failed to download")
        print("💡 You can download them manually later")
    
    # Create sample configuration
    if not create_sample_config():
        print("⚠️ Warning: Failed to create sample configuration")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()