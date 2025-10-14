"""
Main watermark attacker class that orchestrates all attack methods.

This module provides a unified interface for applying various watermark attacks
and evaluating their effectiveness.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from ..common.config import Config, load_config
from ..common.image_utils import (
    load_image, save_image, calculate_image_metrics, 
    create_image_difference, get_device
)
from .attack_registry import attack_registry, register_diffusion_attacks
from .frequency_attacks import FrequencyAttacks
from .diffusion_attacks import DiffusionAttacks

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Result of applying a watermark attack."""
    attack_name: str
    success: bool
    attacked_image: Optional[Image.Image] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating multiple attacks."""
    original_image_path: str
    watermarked_image_path: str
    attack_results: List[AttackResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0


class WatermarkAttacker:
    """
    Main class for applying watermark attacks and evaluating their effectiveness.
    
    This class provides a unified interface for:
    - Applying individual attacks
    - Running comprehensive evaluations
    - Comparing attack effectiveness
    - Generating reports
    """
    
    def __init__(self, config: Optional[Config] = None, device: Optional[str] = None):
        """
        Initialize the watermark attacker.
        
        Args:
            config: Configuration object (uses default if None)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.config = config or Config()
        
        if device is None:
            self.device = self.config.get_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized WatermarkAttacker on device: {self.device}")
        
        # Initialize attack components
        self.frequency_attacks = FrequencyAttacks()
        self.diffusion_attacks = None  # Lazy initialization
        
        # Register diffusion attacks when available
        self._register_diffusion_attacks()
        
        # Setup logging
        self.config.setup_logging()
    
    def _register_diffusion_attacks(self):
        """Register diffusion attacks if available."""
        try:
            self.diffusion_attacks = DiffusionAttacks(device=str(self.device))
            register_diffusion_attacks(self.diffusion_attacks)
            logger.info("Diffusion attacks registered successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize diffusion attacks: {str(e)}")
    
    def apply_attack(self, 
                    image: Union[str, Path, Image.Image],
                    attack_name: str,
                    parameters: Optional[Dict[str, Any]] = None,
                    original_image: Optional[Union[str, Path, Image.Image]] = None) -> AttackResult:
        """
        Apply a single attack to an image.
        
        Args:
            image: Input image (path or PIL Image)
            attack_name: Name of the attack to apply
            parameters: Attack parameters (uses defaults if None)
            original_image: Original image for metrics calculation
            
        Returns:
            AttackResult containing the result and metrics
        """
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            try:
                image = load_image(image)
            except Exception as e:
                return AttackResult(
                    attack_name=attack_name,
                    success=False,
                    error_message=f"Failed to load image: {str(e)}"
                )
        
        # Use default parameters if none provided
        if parameters is None:
            parameters = attack_registry.get_attack_parameters(attack_name)
        
        try:
            # Apply the attack
            attacked_image = attack_registry.apply_attack(image, attack_name, **parameters)
            
            # Calculate metrics if original image provided
            metrics = {}
            if original_image is not None:
                if isinstance(original_image, (str, Path)):
                    original_image = load_image(original_image)
                
                metrics = calculate_image_metrics(original_image, attacked_image)
                
                # Add watermark vs attacked metrics
                watermark_metrics = calculate_image_metrics(image, attacked_image)
                metrics.update({
                    'watermark_psnr': watermark_metrics['psnr'],
                    'watermark_ssim': watermark_metrics['ssim']
                })
            
            execution_time = time.time() - start_time
            
            return AttackResult(
                attack_name=attack_name,
                success=True,
                attacked_image=attacked_image,
                parameters=parameters,
                metrics=metrics,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Attack '{attack_name}' failed: {str(e)}")
            
            return AttackResult(
                attack_name=attack_name,
                success=False,
                parameters=parameters,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def run_comprehensive_evaluation(self,
                                   original_image_path: Union[str, Path],
                                   watermarked_image_path: Union[str, Path],
                                   attack_names: Optional[List[str]] = None,
                                   output_dir: Optional[Union[str, Path]] = None) -> EvaluationResult:
        """
        Run a comprehensive evaluation with multiple attacks.
        
        Args:
            original_image_path: Path to original image
            watermarked_image_path: Path to watermarked image
            attack_names: List of attacks to run (uses all if None)
            output_dir: Directory to save results
            
        Returns:
            EvaluationResult with all attack results
        """
        start_time = time.time()
        
        # Load images
        try:
            original_image = load_image(original_image_path)
            watermarked_image = load_image(watermarked_image_path)
        except Exception as e:
            logger.error(f"Failed to load images: {str(e)}")
            return EvaluationResult(
                original_image_path=str(original_image_path),
                watermarked_image_path=str(watermarked_image_path),
                attack_results=[],
                total_time=time.time() - start_time
            )
        
        # Use all available attacks if none specified
        if attack_names is None:
            attack_names = attack_registry.list_attacks()
        
        logger.info(f"Running evaluation with {len(attack_names)} attacks")
        
        # Run attacks
        attack_results = []
        for attack_name in attack_names:
            logger.info(f"Running attack: {attack_name}")
            
            result = self.apply_attack(
                watermarked_image,
                attack_name,
                original_image=original_image
            )
            
            attack_results.append(result)
            
            # Save attacked image if output directory provided
            if output_dir and result.success and result.attacked_image:
                output_path = Path(output_dir) / f"{attack_name}_attacked.png"
                save_image(result.attacked_image, output_path)
        
        # Calculate summary statistics
        successful_attacks = [r for r in attack_results if r.success]
        
        summary_stats = {
            'total_attacks': len(attack_results),
            'successful_attacks': len(successful_attacks),
            'success_rate': len(successful_attacks) / len(attack_results) if attack_results else 0,
        }
        
        if successful_attacks:
            # Calculate average metrics
            avg_metrics = {}
            for metric_name in ['psnr', 'ssim', 'watermark_psnr', 'watermark_ssim']:
                values = [r.metrics.get(metric_name, 0) for r in successful_attacks if metric_name in r.metrics]
                if values:
                    avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                    avg_metrics[f'std_{metric_name}'] = np.std(values)
            
            summary_stats.update(avg_metrics)
            
            # Find best and worst attacks by SSIM
            ssim_results = [(r.attack_name, r.metrics.get('watermark_ssim', 1.0)) 
                           for r in successful_attacks if 'watermark_ssim' in r.metrics]
            
            if ssim_results:
                ssim_results.sort(key=lambda x: x[1])  # Lower SSIM = more effective
                summary_stats['most_effective_attack'] = ssim_results[0][0]
                summary_stats['least_effective_attack'] = ssim_results[-1][0]
        
        total_time = time.time() - start_time
        
        result = EvaluationResult(
            original_image_path=str(original_image_path),
            watermarked_image_path=str(watermarked_image_path),
            attack_results=attack_results,
            summary_stats=summary_stats,
            total_time=total_time
        )
        
        # Save evaluation report if output directory provided
        if output_dir:
            self._save_evaluation_report(result, output_dir)
        
        return result
    
    def compare_attacks(self,
                       images: List[Tuple[str, str]],  # (original, watermarked) pairs
                       attack_names: Optional[List[str]] = None,
                       output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Compare attack effectiveness across multiple images.
        
        Args:
            images: List of (original_path, watermarked_path) tuples
            attack_names: List of attacks to compare
            output_dir: Directory to save results
            
        Returns:
            Comparison results dictionary
        """
        if attack_names is None:
            attack_names = attack_registry.list_attacks()
        
        logger.info(f"Comparing {len(attack_names)} attacks on {len(images)} images")
        
        all_results = []
        for i, (original_path, watermarked_path) in enumerate(images):
            logger.info(f"Processing image pair {i+1}/{len(images)}")
            
            result = self.run_comprehensive_evaluation(
                original_path,
                watermarked_path,
                attack_names,
                output_dir / f"image_{i}" if output_dir else None
            )
            all_results.append(result)
        
        # Aggregate results across images
        attack_performance = {}
        for attack_name in attack_names:
            attack_results = []
            for eval_result in all_results:
                for attack_result in eval_result.attack_results:
                    if attack_result.attack_name == attack_name and attack_result.success:
                        attack_results.append(attack_result)
            
            if attack_results:
                # Calculate aggregate metrics
                metrics = {}
                for metric_name in ['psnr', 'ssim', 'watermark_psnr', 'watermark_ssim']:
                    values = [r.metrics.get(metric_name, 0) for r in attack_results if metric_name in r.metrics]
                    if values:
                        metrics[f'mean_{metric_name}'] = np.mean(values)
                        metrics[f'std_{metric_name}'] = np.std(values)
                        metrics[f'min_{metric_name}'] = np.min(values)
                        metrics[f'max_{metric_name}'] = np.max(values)
                
                attack_performance[attack_name] = {
                    'success_count': len(attack_results),
                    'total_attempts': len(all_results),
                    'success_rate': len(attack_results) / len(all_results),
                    'avg_execution_time': np.mean([r.execution_time for r in attack_results]),
                    'metrics': metrics
                }
        
        # Rank attacks by effectiveness (lower watermark SSIM = more effective)
        rankings = {}
        if attack_performance:
            ssim_scores = [(name, perf['metrics'].get('mean_watermark_ssim', 1.0)) 
                          for name, perf in attack_performance.items()]
            ssim_scores.sort(key=lambda x: x[1])
            
            rankings['by_effectiveness'] = [name for name, _ in ssim_scores]
            rankings['by_speed'] = sorted(attack_performance.keys(), 
                                        key=lambda x: attack_performance[x]['avg_execution_time'])
            rankings['by_success_rate'] = sorted(attack_performance.keys(),
                                               key=lambda x: attack_performance[x]['success_rate'],
                                               reverse=True)
        
        comparison_result = {
            'attack_performance': attack_performance,
            'rankings': rankings,
            'total_images': len(images),
            'total_attacks': len(attack_names)
        }
        
        # Save comparison report
        if output_dir:
            self._save_comparison_report(comparison_result, output_dir)
        
        return comparison_result
    
    def _save_evaluation_report(self, result: EvaluationResult, output_dir: Union[str, Path]):
        """Save evaluation report to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("WATERMARK ATTACK EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Original Image: {result.original_image_path}\n")
            f.write(f"Watermarked Image: {result.watermarked_image_path}\n")
            f.write(f"Total Execution Time: {result.total_time:.2f} seconds\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            for key, value in result.summary_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("INDIVIDUAL ATTACK RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for attack_result in result.attack_results:
                f.write(f"\nAttack: {attack_result.attack_name}\n")
                f.write(f"Success: {attack_result.success}\n")
                f.write(f"Execution Time: {attack_result.execution_time:.3f}s\n")
                
                if attack_result.success and attack_result.metrics:
                    f.write("Metrics:\n")
                    for metric, value in attack_result.metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                
                if attack_result.error_message:
                    f.write(f"Error: {attack_result.error_message}\n")
                
                f.write(f"Parameters: {attack_result.parameters}\n")
        
        logger.info(f"Saved evaluation report to {report_path}")
    
    def _save_comparison_report(self, result: Dict[str, Any], output_dir: Union[str, Path]):
        """Save comparison report to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "comparison_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("WATERMARK ATTACK COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Images: {result['total_images']}\n")
            f.write(f"Total Attacks: {result['total_attacks']}\n\n")
            
            # Rankings
            f.write("ATTACK RANKINGS\n")
            f.write("-" * 30 + "\n")
            
            rankings = result['rankings']
            if 'by_effectiveness' in rankings:
                f.write("By Effectiveness (most to least effective):\n")
                for i, attack in enumerate(rankings['by_effectiveness'], 1):
                    f.write(f"  {i}. {attack}\n")
                f.write("\n")
            
            if 'by_speed' in rankings:
                f.write("By Speed (fastest to slowest):\n")
                for i, attack in enumerate(rankings['by_speed'], 1):
                    f.write(f"  {i}. {attack}\n")
                f.write("\n")
            
            if 'by_success_rate' in rankings:
                f.write("By Success Rate (highest to lowest):\n")
                for i, attack in enumerate(rankings['by_success_rate'], 1):
                    f.write(f"  {i}. {attack}\n")
                f.write("\n")
            
            # Detailed performance
            f.write("DETAILED PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            
            for attack_name, performance in result['attack_performance'].items():
                f.write(f"\nAttack: {attack_name}\n")
                f.write(f"Success Rate: {performance['success_rate']:.2%}\n")
                f.write(f"Avg Execution Time: {performance['avg_execution_time']:.3f}s\n")
                
                metrics = performance['metrics']
                if metrics:
                    f.write("Metrics:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Saved comparison report to {report_path}")
    
    def get_available_attacks(self, category: Optional[str] = None) -> List[str]:
        """Get list of available attacks."""
        return attack_registry.list_attacks(category)
    
    def get_attack_info(self, attack_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific attack."""
        info = attack_registry.get_attack_info(attack_name)
        if info:
            return {
                'name': info.name,
                'description': info.description,
                'category': info.category,
                'parameters': info.parameters,
                'computational_cost': info.computational_cost,
                'effectiveness': info.effectiveness,
                'requires_models': info.requires_models
            }
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.diffusion_attacks:
            self.diffusion_attacks.cleanup()
        logger.info("Cleaned up WatermarkAttacker resources")