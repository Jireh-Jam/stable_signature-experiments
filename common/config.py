"""
Configuration management for adversarial ML tooling.

This module provides centralized configuration management with type safety
and validation for model parameters, attack configurations, and system settings.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelParams:
    """Parameters for encoder/decoder models."""
    encoder_depth: int = 4
    encoder_channels: int = 64
    decoder_depth: int = 8
    decoder_channels: int = 64
    num_bits: int = 48
    attenuation: str = "jnd"
    scale_channels: bool = False
    scaling_i: float = 1.0
    scaling_w: float = 1.5
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1")
        if self.decoder_depth < 1:
            raise ValueError("decoder_depth must be >= 1")
        if self.num_bits < 1:
            raise ValueError("num_bits must be >= 1")
        if self.scaling_i <= 0:
            raise ValueError("scaling_i must be > 0")
        if self.scaling_w <= 0:
            raise ValueError("scaling_w must be > 0")


@dataclass
class AttackConfig:
    """Configuration for watermark attacks."""
    attack_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        """Validate attack configuration."""
        if not self.attack_type:
            raise ValueError("attack_type cannot be empty")


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 4
    seed: Optional[int] = None
    log_level: str = "INFO"
    output_dir: str = "output"
    
    def __post_init__(self):
        """Validate and process system configuration."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration container."""
    model: ModelParams = field(default_factory=ModelParams)
    system: SystemConfig = field(default_factory=SystemConfig)
    attacks: List[AttackConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse model parameters
            model_data = data.get('model', {})
            model = ModelParams(**model_data)
            
            # Parse system configuration
            system_data = data.get('system', {})
            system = SystemConfig(**system_data)
            
            # Parse attack configurations
            attacks_data = data.get('attacks', [])
            attacks = [AttackConfig(**attack_data) for attack_data in attacks_data]
            
            return cls(model=model, system=system, attacks=attacks)
            
        except Exception as e:
            raise ValueError(f"Failed to parse config file {config_path}: {str(e)}")
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model.__dict__,
            'system': self.system.__dict__,
            'attacks': [attack.__dict__ for attack in self.attacks]
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {output_path}: {str(e)}")
            raise
    
    def get_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.system.device)
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self.system.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.system.seed is not None:
            torch.manual_seed(self.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.system.seed)
            logger.info(f"Set random seed to {self.system.seed}")


# Default configuration
DEFAULT_CONFIG = Config()

# Attack parameter templates
ATTACK_TEMPLATES = {
    'high_frequency': {
        'threshold_percentile': [75, 90, 95, 98],
        'filter_strength': [0.5, 0.8, 0.95]
    },
    'gaussian_blur': {
        'kernel_size': [3, 5, 7],
        'sigma': [0.5, 1.0, 2.0]
    },
    'gaussian_noise': {
        'std': [0.01, 0.05, 0.1]
    },
    'jpeg_compression': {
        'quality': [10, 30, 50, 70, 90]
    },
    'brightness': {
        'factor': [0.2, 0.5, 0.8, 1.2, 1.5]
    },
    'contrast': {
        'factor': [0.2, 0.5, 0.8, 1.2, 1.5]
    },
    'rotation': {
        'degrees': [5, 15, 30, 45, 90]
    },
    'scale': {
        'factor': [0.1, 0.3, 0.5, 0.8]
    },
    'crop': {
        'ratio': [0.3, 0.5, 0.7, 0.9]
    },
    'diffusion_inpainting': {
        'mask_ratio': [0.2, 0.4, 0.6],
        'strength': [0.5, 0.75, 1.0]
    },
    'diffusion_regeneration': {
        'strength': [0.3, 0.5, 0.7]
    },
    'adversarial': {
        'attack_type': ['FGSM', 'PGD', 'DeepFool'],
        'epsilon': [0.01, 0.03, 0.05]
    }
}


def create_attack_configs(attack_types: List[str]) -> List[AttackConfig]:
    """
    Create attack configurations from templates.
    
    Args:
        attack_types: List of attack types to create configs for
        
    Returns:
        List of AttackConfig objects
    """
    configs = []
    
    for attack_type in attack_types:
        if attack_type in ATTACK_TEMPLATES:
            template = ATTACK_TEMPLATES[attack_type]
            config = AttackConfig(
                attack_type=attack_type,
                parameters=template,
                description=f"Auto-generated config for {attack_type}"
            )
            configs.append(config)
        else:
            logger.warning(f"No template found for attack type: {attack_type}")
    
    return configs


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or return default.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config object
    """
    if config_path is None:
        logger.info("Using default configuration")
        return DEFAULT_CONFIG
    
    try:
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG