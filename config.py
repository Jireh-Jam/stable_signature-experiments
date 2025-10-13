"""
Configuration Module for Watermark Robustness Pipeline

This module provides easy-to-use configuration for the watermark testing pipeline.
It handles paths, model settings, and transformation parameters for different
watermarking methods.

Usage:
    from config import WatermarkConfig
    
    config = WatermarkConfig(method="Watermark_Anything", user_name="YourName")
    print(config.raw_images_path)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os


class WatermarkConfig:
    """
    Configuration class for watermark robustness testing pipeline.
    
    This class automatically sets up all necessary paths and parameters
    based on the chosen watermarking method and environment.
    """
    
    # Supported watermarking methods
    SUPPORTED_METHODS = ["Stable_Signature", "Watermark_Anything", "TrustMark"]
    
    def __init__(
        self,
        method: str = "Watermark_Anything",
        user_name: str = "YourName",
        azure_root: str = "/home/azureuser/cloudfiles/code/Users/",
        max_images: Optional[int] = None
    ):
        """
        Initialise configuration for watermark pipeline.
        
        Args:
            method: Watermarking method to use (Stable_Signature, Watermark_Anything, or TrustMark)
            user_name: Azure AI username (only used if running on Azure)
            azure_root: Root directory for Azure AI (default: Azure AI standard path)
            max_images: Maximum number of images to process (None for all images)
        
        Raises:
            ValueError: If an unsupported method is specified
        """
        # Validate method
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {', '.join(self.SUPPORTED_METHODS)}"
            )
        
        self.method = method
        self.user_name = user_name
        self.max_images = max_images
        
        # Detect environment (Azure vs local)
        self.is_azure = os.path.exists(azure_root)
        
        # Set root directory
        if self.is_azure:
            self.root_dir = Path(azure_root) / user_name
        else:
            self.root_dir = Path.cwd()
        
        # Set up paths
        self.repo_dir = self.root_dir / "ost-embedding-research"
        self.watermark_models_dir = self.repo_dir / "watermark_models"
        
        # Method-specific paths
        self._setup_method_paths()
        
        # Create output directories
        self._create_output_directories()
    
    def _setup_method_paths(self):
        """Set up paths specific to the chosen watermarking method."""
        method_map = {
            "Watermark_Anything": "watermark-anything",
            "TrustMark": "trustmark",
            "Stable_Signature": "stable-signature"
        }
        
        method_dir_name = method_map[self.method]
        self.method_dir = self.watermark_models_dir / method_dir_name
        self.raw_images_path = self.method_dir / "watermarked_images"
        
        # Checkpoint paths
        if self.method == "Watermark_Anything":
            self.checkpoint_path = self.method_dir / "models" / "watermark_anything_model.pth"
        elif self.method == "TrustMark":
            self.checkpoint_path = None  # TrustMark uses built-in models
        elif self.method == "Stable_Signature":
            self.checkpoint_path = self.method_dir / "models" / "dec_48b_whit.torchscript.pt"
        
        # Output directory
        output_name_map = {
            "Watermark_Anything": "watermark_anything",
            "TrustMark": "trustmark",
            "Stable_Signature": "stable_signature"
        }
        output_dir_name = output_name_map[self.method]
        self.output_dir = self.repo_dir / "outputs" / output_dir_name
    
    def _create_output_directories(self):
        """Create necessary output directories if they don't exist."""
        self.transformed_images_dir = self.output_dir / "transformed_images"
        self.results_dir = self.output_dir / "results"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transformed_images_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def get_transformation_config(self) -> Dict:
        """
        Get default transformation configuration.
        
        Returns:
            Dictionary containing transformation settings
        """
        return {
            "resize": {
                "enabled": True,
                "dimensions": [(256, 256), (512, 512), (1024, 1024)]
            },
            "crop": {
                "enabled": True,
                "percentages": [0.99, 0.90, 0.80, 0.70, 0.60, 0.50]
            },
            "rotation": {
                "enabled": True,
                "angles": [5, 15, 30, 45, 90, 180]
            },
            "blur": {
                "enabled": True,
                "kernel_sizes": [3, 11, 21, 51]
            },
            "compression": {
                "enabled": True,
                "qualities": [95, 85, 75, 50, 25]
            },
            "noise": {
                "enabled": True,
                "levels": [0.01, 0.02, 0.05]
            },
            "colour_jitter": {
                "enabled": True,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2
            }
        }
    
    def verify_setup(self) -> Tuple[bool, List[str]]:
        """
        Verify that all required directories and files exist.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check root directory
        if not self.root_dir.exists():
            issues.append(f"Root directory does not exist: {self.root_dir}")
        
        # Check method directory
        if not self.method_dir.exists():
            issues.append(f"Method directory does not exist: {self.method_dir}")
        
        # Check raw images directory
        if not self.raw_images_path.exists():
            issues.append(f"Watermarked images directory does not exist: {self.raw_images_path}")
        elif not list(self.raw_images_path.glob("*.png")) and not list(self.raw_images_path.glob("*.jpg")):
            issues.append(f"No images found in: {self.raw_images_path}")
        
        # Check checkpoint (if applicable)
        if self.checkpoint_path and not self.checkpoint_path.exists():
            issues.append(f"Model checkpoint not found: {self.checkpoint_path}")
        
        return len(issues) == 0, issues
    
    def print_configuration(self):
        """Print the current configuration in a readable format."""
        print("=" * 70)
        print("WATERMARK PIPELINE CONFIGURATION")
        print("=" * 70)
        print(f"Method:                 {self.method}")
        print(f"Environment:            {'Azure AI' if self.is_azure else 'Local'}")
        print(f"User:                   {self.user_name if self.is_azure else 'N/A'}")
        print(f"Root Directory:         {self.root_dir}")
        print(f"Method Directory:       {self.method_dir}")
        print(f"Raw Images Path:        {self.raw_images_path}")
        print(f"Checkpoint Path:        {self.checkpoint_path if self.checkpoint_path else 'Built-in model'}")
        print(f"Output Directory:       {self.output_dir}")
        print(f"Max Images to Process:  {self.max_images if self.max_images else 'All'}")
        print("=" * 70)
        
        # Verify setup
        is_valid, issues = self.verify_setup()
        if is_valid:
            print("✓ Configuration is valid - all paths exist")
        else:
            print("⚠️  Configuration issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        print("=" * 70)


# Convenience function for quick configuration
def create_config(method: str = "Watermark_Anything", **kwargs) -> WatermarkConfig:
    """
    Create a configuration instance with sensible defaults.
    
    Args:
        method: Watermarking method to use
        **kwargs: Additional arguments passed to WatermarkConfig
    
    Returns:
        Configured WatermarkConfig instance
    
    Example:
        >>> config = create_config("TrustMark", user_name="Alice")
        >>> config.print_configuration()
    """
    return WatermarkConfig(method=method, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create configuration for Watermark Anything
    config = create_config("Watermark_Anything", user_name="TestUser", max_images=10)
    config.print_configuration()
