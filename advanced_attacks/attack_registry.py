"""
Attack registry system for managing and discovering watermark attacks.

This module provides a centralized registry for all available attack methods,
allowing for dynamic discovery, configuration, and execution of attacks.
"""

import logging
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AttackInfo:
    """Information about a registered attack method."""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_models: List[str] = field(default_factory=list)
    computational_cost: str = "medium"  # low, medium, high
    effectiveness: str = "unknown"  # low, medium, high, unknown


class BaseAttack(ABC):
    """Base class for all attack implementations."""
    
    def __init__(self, name: str, description: str = "", category: str = "general"):
        self.name = name
        self.description = description
        self.category = category
    
    @abstractmethod
    def attack(self, image: Image.Image, **kwargs) -> Image.Image:
        """
        Apply the attack to an image.
        
        Args:
            image: Input image
            **kwargs: Attack-specific parameters
            
        Returns:
            Attacked image
        """
        pass
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this attack."""
        return {}
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate attack parameters."""
        return True


class AttackRegistry:
    """Registry for managing watermark attacks."""
    
    def __init__(self):
        self._attacks: Dict[str, BaseAttack] = {}
        self._attack_info: Dict[str, AttackInfo] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, 
                attack: BaseAttack,
                info: Optional[AttackInfo] = None) -> None:
        """
        Register a new attack method.
        
        Args:
            attack: Attack implementation
            info: Optional attack information
        """
        name = attack.name
        
        if name in self._attacks:
            logger.warning(f"Overwriting existing attack: {name}")
        
        self._attacks[name] = attack
        
        # Create default info if not provided
        if info is None:
            info = AttackInfo(
                name=name,
                description=attack.description,
                category=attack.category
            )
        
        self._attack_info[name] = info
        
        # Update category mapping
        category = info.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        logger.debug(f"Registered attack: {name} (category: {category})")
    
    def get_attack(self, name: str) -> Optional[BaseAttack]:
        """Get an attack by name."""
        return self._attacks.get(name)
    
    def get_attack_info(self, name: str) -> Optional[AttackInfo]:
        """Get attack information by name."""
        return self._attack_info.get(name)
    
    def list_attacks(self, category: Optional[str] = None) -> List[str]:
        """
        List available attacks, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of attack names
        """
        if category is None:
            return list(self._attacks.keys())
        else:
            return self._categories.get(category, [])
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(self._categories.keys())
    
    def apply_attack(self, 
                    image: Image.Image,
                    attack_name: str,
                    **kwargs) -> Image.Image:
        """
        Apply an attack by name.
        
        Args:
            image: Input image
            attack_name: Name of attack to apply
            **kwargs: Attack parameters
            
        Returns:
            Attacked image
            
        Raises:
            ValueError: If attack is not found
        """
        attack = self.get_attack(attack_name)
        if attack is None:
            raise ValueError(f"Attack '{attack_name}' not found")
        
        # Validate parameters
        if not attack.validate_parameters(**kwargs):
            logger.warning(f"Invalid parameters for attack '{attack_name}': {kwargs}")
        
        return attack.attack(image, **kwargs)
    
    def get_attack_parameters(self, attack_name: str) -> Dict[str, Any]:
        """Get default parameters for an attack."""
        attack = self.get_attack(attack_name)
        if attack is None:
            return {}
        return attack.get_default_parameters()
    
    def search_attacks(self, 
                      query: str,
                      search_in: List[str] = None) -> List[str]:
        """
        Search for attacks by name or description.
        
        Args:
            query: Search query
            search_in: Fields to search in ('name', 'description', 'category')
            
        Returns:
            List of matching attack names
        """
        if search_in is None:
            search_in = ['name', 'description', 'category']
        
        query_lower = query.lower()
        matches = []
        
        for name, info in self._attack_info.items():
            if 'name' in search_in and query_lower in name.lower():
                matches.append(name)
            elif 'description' in search_in and query_lower in info.description.lower():
                matches.append(name)
            elif 'category' in search_in and query_lower in info.category.lower():
                matches.append(name)
        
        return matches
    
    def get_attacks_by_cost(self, max_cost: str = "high") -> List[str]:
        """
        Get attacks filtered by computational cost.
        
        Args:
            max_cost: Maximum computational cost ('low', 'medium', 'high')
            
        Returns:
            List of attack names
        """
        cost_order = {'low': 0, 'medium': 1, 'high': 2}
        max_cost_level = cost_order.get(max_cost, 2)
        
        matching_attacks = []
        for name, info in self._attack_info.items():
            attack_cost_level = cost_order.get(info.computational_cost, 1)
            if attack_cost_level <= max_cost_level:
                matching_attacks.append(name)
        
        return matching_attacks
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        total_attacks = len(self._attacks)
        categories = self._categories
        
        cost_distribution = {}
        effectiveness_distribution = {}
        
        for info in self._attack_info.values():
            cost = info.computational_cost
            cost_distribution[cost] = cost_distribution.get(cost, 0) + 1
            
            effectiveness = info.effectiveness
            effectiveness_distribution[effectiveness] = effectiveness_distribution.get(effectiveness, 0) + 1
        
        return {
            'total_attacks': total_attacks,
            'categories': {cat: len(attacks) for cat, attacks in categories.items()},
            'cost_distribution': cost_distribution,
            'effectiveness_distribution': effectiveness_distribution
        }


# Global registry instance
attack_registry = AttackRegistry()


def register_attack(attack: BaseAttack, info: Optional[AttackInfo] = None):
    """Convenience function to register an attack."""
    attack_registry.register(attack, info)


def get_available_attacks(category: Optional[str] = None) -> List[str]:
    """Convenience function to list available attacks."""
    return attack_registry.list_attacks(category)


def apply_attack(image: Image.Image, attack_name: str, **kwargs) -> Image.Image:
    """Convenience function to apply an attack."""
    return attack_registry.apply_attack(image, attack_name, **kwargs)


# Wrapper classes for integrating existing attack methods
class FrequencyAttackWrapper(BaseAttack):
    """Wrapper for frequency domain attacks."""
    
    def __init__(self, name: str, attack_func: Callable, description: str = ""):
        super().__init__(name, description, "frequency")
        self.attack_func = attack_func
    
    def attack(self, image: Image.Image, **kwargs) -> Image.Image:
        return self.attack_func(image, **kwargs)
    
    def get_default_parameters(self) -> Dict[str, Any]:
        sig = inspect.signature(self.attack_func)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param_name != 'image' and param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
        return defaults


class DiffusionAttackWrapper(BaseAttack):
    """Wrapper for diffusion-based attacks."""
    
    def __init__(self, name: str, attack_obj: Any, method_name: str, description: str = ""):
        super().__init__(name, description, "diffusion")
        self.attack_obj = attack_obj
        self.method_name = method_name
    
    def attack(self, image: Image.Image, **kwargs) -> Image.Image:
        method = getattr(self.attack_obj, self.method_name)
        return method(image, **kwargs)
    
    def get_default_parameters(self) -> Dict[str, Any]:
        method = getattr(self.attack_obj, self.method_name)
        sig = inspect.signature(method)
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param_name not in ['self', 'image'] and param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
        return defaults


class TransformAttackWrapper(BaseAttack):
    """Wrapper for transform-based attacks."""
    
    def __init__(self, name: str, transform_name: str, transform_registry: Any, description: str = ""):
        super().__init__(name, description, "transform")
        self.transform_name = transform_name
        self.transform_registry = transform_registry
    
    def attack(self, image: Image.Image, **kwargs) -> Image.Image:
        result = self.transform_registry.apply_transform(image, self.transform_name, **kwargs)
        return result.image if result.success else image
    
    def get_default_parameters(self) -> Dict[str, Any]:
        # This would need to be implemented based on the transform registry
        return {}


def register_frequency_attacks():
    """Register frequency domain attacks."""
    from .frequency_attacks import FrequencyAttacks
    
    # High frequency filter
    register_attack(
        FrequencyAttackWrapper(
            "high_frequency_filter",
            FrequencyAttacks.high_frequency_filter,
            "Removes high frequency components that may contain watermarks"
        ),
        AttackInfo(
            name="high_frequency_filter",
            description="Filters high frequency components in the frequency domain",
            category="frequency",
            parameters={
                "threshold_percentile": [75, 90, 95, 98],
                "filter_strength": [0.5, 0.8, 0.95]
            },
            computational_cost="low",
            effectiveness="medium"
        )
    )
    
    # Low pass filter
    register_attack(
        FrequencyAttackWrapper(
            "low_pass_filter",
            FrequencyAttacks.low_pass_filter,
            "Applies low-pass filter to remove high frequencies"
        ),
        AttackInfo(
            name="low_pass_filter",
            description="Butterworth low-pass filter for frequency domain filtering",
            category="frequency",
            parameters={"cutoff_frequency": [0.1, 0.3, 0.5]},
            computational_cost="low",
            effectiveness="medium"
        )
    )
    
    # Notch filter
    register_attack(
        FrequencyAttackWrapper(
            "notch_filter",
            FrequencyAttacks.notch_filter,
            "Removes specific frequency components using notch filter"
        ),
        AttackInfo(
            name="notch_filter",
            description="Notch filter for removing specific frequency components",
            category="frequency",
            parameters={
                "center_freq": [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
                "radius": [0.05, 0.1, 0.15]
            },
            computational_cost="low",
            effectiveness="low"
        )
    )


def register_diffusion_attacks(diffusion_attacks_obj):
    """Register diffusion-based attacks."""
    
    # Inpainting attack
    register_attack(
        DiffusionAttackWrapper(
            "diffusion_inpainting",
            diffusion_attacks_obj,
            "inpainting_attack",
            "Uses diffusion inpainting to regenerate masked regions"
        ),
        AttackInfo(
            name="diffusion_inpainting",
            description="Stable Diffusion inpainting attack for watermark removal",
            category="diffusion",
            parameters={
                "mask_ratio": [0.2, 0.4, 0.6],
                "guidance_scale": [5.0, 7.5, 10.0],
                "num_inference_steps": [20, 30, 50]
            },
            requires_models=["stable-diffusion-inpainting"],
            computational_cost="high",
            effectiveness="high"
        )
    )
    
    # Img2img attack
    register_attack(
        DiffusionAttackWrapper(
            "diffusion_img2img",
            diffusion_attacks_obj,
            "img2img_attack",
            "Uses diffusion img2img to regenerate images"
        ),
        AttackInfo(
            name="diffusion_img2img",
            description="Stable Diffusion img2img attack for watermark removal",
            category="diffusion",
            parameters={
                "strength": [0.3, 0.5, 0.7],
                "guidance_scale": [5.0, 7.5, 10.0],
                "num_inference_steps": [20, 30, 50]
            },
            requires_models=["stable-diffusion-v1-5"],
            computational_cost="high",
            effectiveness="high"
        )
    )
    
    # ReSD attack
    register_attack(
        DiffusionAttackWrapper(
            "diffusion_resd",
            diffusion_attacks_obj,
            "resd_attack",
            "Uses ReSD approach for watermark removal"
        ),
        AttackInfo(
            name="diffusion_resd",
            description="ReSD (Regeneration Stable Diffusion) attack",
            category="diffusion",
            parameters={
                "noise_step": [10, 20, 30],
                "strength": [0.3, 0.5, 0.7],
                "guidance_scale": [5.0, 7.5, 10.0]
            },
            requires_models=["stable-diffusion-v1-5", "resd-pipeline"],
            computational_cost="high",
            effectiveness="high"
        )
    )


def register_transform_attacks():
    """Register transform-based attacks."""
    from ..common.transforms import transform_registry, get_standard_transform_configs
    
    configs = get_standard_transform_configs()
    
    for config_name, params in configs.items():
        # Extract base transform name
        transform_name = config_name.split('_')[0]
        
        # Map to actual transform names
        transform_mapping = {
            'crop': 'crop_center',
            'resize': 'resize',
            'rotate': 'rotate',
            'blur': 'gaussian_blur',
            'brighten': 'brightness',
            'darken': 'brightness',
            'high': 'contrast',
            'low': 'contrast',
            'saturate': 'saturation',
            'desaturate': 'saturation',
            'noise': 'gaussian_noise',
            'jpeg': 'jpeg_compression'
        }
        
        actual_transform = transform_mapping.get(transform_name, transform_name)
        
        if actual_transform in transform_registry.list_transforms():
            register_attack(
                TransformAttackWrapper(
                    config_name,
                    actual_transform,
                    transform_registry,
                    f"Transform-based attack: {config_name}"
                ),
                AttackInfo(
                    name=config_name,
                    description=f"Transform attack using {actual_transform}",
                    category="transform",
                    parameters=params,
                    computational_cost="low",
                    effectiveness="medium"
                )
            )


def initialize_registry():
    """Initialize the attack registry with all available attacks."""
    logger.info("Initializing attack registry...")
    
    # Register frequency attacks
    register_frequency_attacks()
    
    # Register transform attacks
    register_transform_attacks()
    
    # Diffusion attacks will be registered when DiffusionAttacks is instantiated
    
    logger.info(f"Registry initialized with {len(attack_registry.list_attacks())} attacks")


# Initialize registry on import
initialize_registry()