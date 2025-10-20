"""
Configuration utilities for watermarking methods.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: Union[str, Path]) -> Optional[DictConfig]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object or None if loading failed
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return None
        
        config = OmegaConf.load(config_path)
        print(f"✅ Loaded config from {config_path}")
        return config
        
    except Exception as e:
        print(f"❌ Error loading config {config_path}: {str(e)}")
        return None


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configuration objects.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge (takes priority)
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(config, output_path)
        print(f"✅ Saved config to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving config {output_path}: {str(e)}")
        return False


def get_default_config() -> DictConfig:
    """
    Get default configuration for watermarking methods.
    
    Returns:
        Default configuration object
    """
    default_config = {
        "watermarking": {
            "method": "stable_signature",
            "message_length": 48,
            "detection_threshold": 0.5,
        },
        "data": {
            "input_size": [256, 256],
            "batch_size": 1,
            "max_images": -1,  # -1 means no limit
        },
        "transformations": {
            "apply_standard": True,
            "apply_aggressive": False,
            "jpeg_quality_levels": [90, 70, 50, 30],
            "blur_kernel_sizes": [3, 5, 7],
            "crop_ratios": [0.9, 0.8, 0.7],
        },
        "evaluation": {
            "confidence_threshold": 0.5,
            "generate_plots": True,
            "save_detailed_results": True,
        },
        "paths": {
            "models_dir": "models",
            "data_dir": "data",
            "results_dir": "results",
        }
    }
    
    return OmegaConf.create(default_config)


def update_config_from_args(config: DictConfig, args: Dict[str, Any]) -> DictConfig:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Base configuration
        args: Dictionary of arguments to override
        
    Returns:
        Updated configuration
    """
    # Convert flat argument names to nested config structure
    for key, value in args.items():
        if value is not None:
            # Handle nested keys like "data.batch_size"
            if '.' in key:
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
    
    return config