"""
Configuration management utilities.

This module provides tools for loading and managing configuration files
for the watermarking pipeline.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Configuration manager for the watermarking pipeline.
    
    This class handles loading, validating, and providing access to
    configuration settings from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
            
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.error(f"Configuration file not found: {config_path}")
                return False
                
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.config_path = config_path
            self.logger.info(f"Configuration loaded from: {config_path}")
            
            # Validate configuration
            if self._validate_config():
                return True
            else:
                self.logger.error("Configuration validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return False
            
    def load_default_config(self) -> bool:
        """
        Load the default configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        # Try to find default config in common locations
        possible_paths = [
            "experiments/configs/default_config.yaml",
            "configs/default_config.yaml",
            "default_config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return self.load_config(path)
                
        # If no config file found, create minimal default
        self.logger.warning("No configuration file found, using minimal defaults")
        self.config = self._get_minimal_config()
        return True
        
    def _validate_config(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['user', 'watermarking', 'data', 'transformations', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
                
        # Validate user settings
        if 'name' not in self.config['user']:
            self.logger.error("User name not specified in configuration")
            return False
            
        # Validate watermarking method
        valid_methods = ['stable_signature', 'trustmark', 'watermark_anything']
        method = self.config['watermarking'].get('method', '').lower()
        if method not in valid_methods:
            self.logger.error(f"Invalid watermarking method: {method}. "
                            f"Valid options: {valid_methods}")
            return False
            
        return True
        
    def _get_minimal_config(self) -> Dict[str, Any]:
        """
        Get a minimal default configuration.
        
        Returns:
            Dictionary with minimal configuration settings
        """
        return {
            'experiment': {
                'name': 'watermark_test',
                'description': 'Basic watermark testing',
                'version': '1.0'
            },
            'user': {
                'name': 'user',
                'azure_root_dir': '/home/azureuser/cloudfiles/code/Users/'
            },
            'watermarking': {
                'method': 'stable_signature',
                'message': 'test_message_48bits'
            },
            'data': {
                'raw_images_dir': 'data/raw',
                'watermarked_images_dir': 'data/watermarked',
                'transformed_images_dir': 'data/transformed',
                'results_dir': 'results',
                'max_images_to_process': 5
            },
            'transformations': {
                'apply_standard': True,
                'apply_aggressive': False,
                'apply_custom': False
            },
            'evaluation': {
                'confidence_threshold': 0.5,
                'generate_plots': True,
                'save_detailed_results': True
            },
            'performance': {
                'use_gpu': True,
                'batch_size': 1
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': False
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'user.name' or 'watermarking.method')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'user.name')
            value: Value to set
        """
        keys = key.split('.')
        config_section = self.config
        
        # Navigate to the parent section
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
            
        # Set the value
        config_section[keys[-1]] = value
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name (e.g., 'watermarking', 'data')
            
        Returns:
            Dictionary with section contents
        """
        return self.config.get(section, {})
        
    def update_user_paths(self, username: str) -> None:
        """
        Update user-specific paths in the configuration.
        
        Args:
            username: Username to use for path construction
        """
        self.set('user.name', username)
        
        # Update Azure root directory path
        azure_root = self.get('user.azure_root_dir', '/home/azureuser/cloudfiles/code/Users/')
        home_dir = os.path.join(azure_root, username)
        
        # Update data paths to be relative to user home directory
        data_config = self.get_section('data')
        for key in ['raw_images_dir', 'watermarked_images_dir', 'transformed_images_dir', 'results_dir']:
            if key in data_config:
                # Make paths relative to user home directory if they're not absolute
                current_path = data_config[key]
                if not os.path.isabs(current_path):
                    self.set(f'data.{key}', os.path.join(home_dir, current_path))
                    
    def save_config(self, output_path: str) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path where to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
                
            self.logger.info(f"Configuration saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False
            
    def print_config(self) -> None:
        """Print the current configuration in a readable format."""
        print("=" * 50)
        print("🔧 CURRENT CONFIGURATION")
        print("=" * 50)
        
        def print_section(section_name: str, section_data: Dict[str, Any], indent: int = 0) -> None:
            """Recursively print configuration sections."""
            prefix = "  " * indent
            print(f"{prefix}📋 {section_name.upper()}:")
            
            for key, value in section_data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{prefix}  {key}: {value}")
            print()
            
        for section_name, section_data in self.config.items():
            if isinstance(section_data, dict):
                print_section(section_name, section_data)
            else:
                print(f"{section_name}: {section_data}")
                
    def get_summary(self) -> Dict[str, str]:
        """
        Get a summary of key configuration settings.
        
        Returns:
            Dictionary with key configuration values
        """
        return {
            "Experiment": self.get('experiment.name', 'Unknown'),
            "User": self.get('user.name', 'Unknown'),
            "Watermarking Method": self.get('watermarking.method', 'Unknown'),
            "Max Images": str(self.get('data.max_images_to_process', 'Unknown')),
            "Apply Standard Transforms": str(self.get('transformations.apply_standard', False)),
            "Apply Aggressive Transforms": str(self.get('transformations.apply_aggressive', False)),
            "Generate Plots": str(self.get('evaluation.generate_plots', False)),
            "Results Directory": self.get('data.results_dir', 'Unknown')
        }