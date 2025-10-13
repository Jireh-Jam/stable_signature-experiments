"""
Evaluation utilities for watermark robustness testing.

This module provides tools for measuring watermark detection performance
and generating comprehensive reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class WatermarkEvaluator:
    """
    Comprehensive evaluation toolkit for watermark robustness testing.
    
    This class provides methods to analyse detection results, calculate
    performance metrics, and generate detailed reports.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = []
        self.summary_stats = {}
        
    def add_result(self, image_name: str, transformation: str, 
                   detected: bool, confidence: float, 
                   message: Optional[str] = None, **kwargs) -> None:
        """
        Add a detection result to the evaluation dataset.
        
        Args:
            image_name: Name of the test image
            transformation: Name of the applied transformation
            detected: Whether watermark was detected
            confidence: Detection confidence score (0.0 to 1.0)
            message: Extracted message (if any)
            **kwargs: Additional metadata
        """
        result = {
            'image_name': image_name,
            'transformation': transformation,
            'detected': detected,
            'confidence': confidence,
            'message': message,
            **kwargs
        }
        self.results.append(result)
        
    def add_results_batch(self, results: List[Dict[str, Any]]) -> None:
        """
        Add multiple results at once.
        
        Args:
            results: List of result dictionaries
        """
        self.results.extend(results)
        
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        return pd.DataFrame(self.results)
        
    def calculate_detection_rates(self) -> Dict[str, float]:
        """
        Calculate detection rates for each transformation.
        
        Returns:
            Dictionary mapping transformation names to detection rates
        """
        if not self.results:
            return {}
            
        df = self.get_dataframe()
        detection_rates = df.groupby('transformation')['detected'].mean()
        return detection_rates.to_dict()
        
    def calculate_confidence_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence score statistics for each transformation.
        
        Returns:
            Dictionary with confidence statistics by transformation
        """
        if not self.results:
            return {}
            
        df = self.get_dataframe()
        
        stats = {}
        for transformation in df['transformation'].unique():
            trans_data = df[df['transformation'] == transformation]
            
            # Separate detected and not detected
            detected_conf = trans_data[trans_data['detected']]['confidence']
            not_detected_conf = trans_data[~trans_data['detected']]['confidence']
            
            stats[transformation] = {
                'overall_mean': trans_data['confidence'].mean(),
                'overall_std': trans_data['confidence'].std(),
                'detected_mean': detected_conf.mean() if len(detected_conf) > 0 else 0.0,
                'detected_std': detected_conf.std() if len(detected_conf) > 0 else 0.0,
                'not_detected_mean': not_detected_conf.mean() if len(not_detected_conf) > 0 else 0.0,
                'not_detected_std': not_detected_conf.std() if len(not_detected_conf) > 0 else 0.0,
            }
            
        return stats
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Dictionary with summary statistics and analysis
        """
        if not self.results:
            return {"error": "No results available"}
            
        df = self.get_dataframe()
        
        # Overall statistics
        total_images = len(df)
        overall_detection_rate = df['detected'].mean()
        overall_confidence = df['confidence'].mean()
        
        # Detection rates by transformation
        detection_rates = self.calculate_detection_rates()
        
        # Confidence statistics
        confidence_stats = self.calculate_confidence_stats()
        
        # Best and worst performing transformations
        best_transformation = max(detection_rates.items(), key=lambda x: x[1])
        worst_transformation = min(detection_rates.items(), key=lambda x: x[1])
        
        # Robustness classification
        if overall_detection_rate >= 0.9:
            robustness_level = "Excellent"
        elif overall_detection_rate >= 0.8:
            robustness_level = "Good"
        elif overall_detection_rate >= 0.6:
            robustness_level = "Moderate"
        elif overall_detection_rate >= 0.4:
            robustness_level = "Poor"
        else:
            robustness_level = "Very Poor"
            
        # Transformation categories analysis
        transformation_categories = self._categorize_transformations(detection_rates)
        
        summary = {
            "overall_statistics": {
                "total_tests": total_images,
                "overall_detection_rate": round(overall_detection_rate, 4),
                "overall_detection_percentage": round(overall_detection_rate * 100, 2),
                "overall_confidence": round(overall_confidence, 4),
                "robustness_level": robustness_level
            },
            "transformation_analysis": {
                "detection_rates": {k: round(v, 4) for k, v in detection_rates.items()},
                "best_performance": {
                    "transformation": best_transformation[0],
                    "detection_rate": round(best_transformation[1], 4)
                },
                "worst_performance": {
                    "transformation": worst_transformation[0],
                    "detection_rate": round(worst_transformation[1], 4)
                }
            },
            "confidence_analysis": confidence_stats,
            "robustness_categories": transformation_categories,
            "recommendations": self._generate_recommendations(detection_rates, overall_detection_rate)
        }
        
        self.summary_stats = summary
        return summary
        
    def _categorize_transformations(self, detection_rates: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Categorize transformations by robustness level.
        
        Args:
            detection_rates: Dictionary of detection rates by transformation
            
        Returns:
            Dictionary categorizing transformations by performance
        """
        categories = {
            "robust": [],      # >= 90% detection
            "good": [],        # 70-89% detection
            "moderate": [],    # 50-69% detection
            "vulnerable": []   # < 50% detection
        }
        
        for transformation, rate in detection_rates.items():
            if rate >= 0.9:
                categories["robust"].append(transformation)
            elif rate >= 0.7:
                categories["good"].append(transformation)
            elif rate >= 0.5:
                categories["moderate"].append(transformation)
            else:
                categories["vulnerable"].append(transformation)
                
        return categories
        
    def _generate_recommendations(self, detection_rates: Dict[str, float], 
                                overall_rate: float) -> List[str]:
        """
        Generate recommendations based on the results.
        
        Args:
            detection_rates: Detection rates by transformation
            overall_rate: Overall detection rate
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Overall performance recommendations
        if overall_rate < 0.6:
            recommendations.append("Consider improving watermark strength or embedding method")
            recommendations.append("Current watermark shows poor robustness to common transformations")
        elif overall_rate < 0.8:
            recommendations.append("Watermark shows moderate robustness but could be improved")
            recommendations.append("Focus on vulnerable transformation types for enhancement")
        else:
            recommendations.append("Watermark shows good overall robustness")
            
        # Specific vulnerability recommendations
        vulnerable_transforms = [t for t, r in detection_rates.items() if r < 0.5]
        if vulnerable_transforms:
            recommendations.append(f"Particularly vulnerable to: {', '.join(vulnerable_transforms[:3])}")
            
        # Strong performance acknowledgment
        robust_transforms = [t for t, r in detection_rates.items() if r >= 0.9]
        if robust_transforms:
            recommendations.append(f"Shows excellent robustness against: {', '.join(robust_transforms[:3])}")
            
        return recommendations
        
    def save_results(self, output_dir: str, filename_prefix: str = "watermark_evaluation") -> Dict[str, str]:
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save detailed results as CSV
        df = self.get_dataframe()
        csv_path = output_path / f"{filename_prefix}_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        saved_files['detailed_csv'] = str(csv_path)
        
        # Save summary report as JSON
        summary = self.generate_summary_report()
        json_path = output_path / f"{filename_prefix}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary_json'] = str(json_path)
        
        # Save detection rates as CSV
        detection_rates = self.calculate_detection_rates()
        rates_df = pd.DataFrame(list(detection_rates.items()), 
                               columns=['transformation', 'detection_rate'])
        rates_csv_path = output_path / f"{filename_prefix}_detection_rates.csv"
        rates_df.to_csv(rates_csv_path, index=False)
        saved_files['rates_csv'] = str(rates_csv_path)
        
        return saved_files
        
    def create_visualizations(self, output_dir: str, 
                            filename_prefix: str = "watermark_evaluation") -> Dict[str, str]:
        """
        Create and save visualization plots.
        
        Args:
            output_dir: Directory to save plots
            filename_prefix: Prefix for plot filenames
            
        Returns:
            Dictionary with paths to saved plots
        """
        if not self.results:
            return {}
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = self.get_dataframe()
        saved_plots = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Detection rates bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        detection_rates = self.calculate_detection_rates()
        
        # Sort by detection rate
        sorted_rates = sorted(detection_rates.items(), key=lambda x: x[1])
        transformations, rates = zip(*sorted_rates)
        
        bars = ax.barh(range(len(transformations)), [r * 100 for r in rates])
        ax.set_yticks(range(len(transformations)))
        ax.set_yticklabels(transformations)
        ax.set_xlabel('Detection Rate (%)')
        ax.set_title('Watermark Detection Rates by Transformation')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plot_path = output_path / f"{filename_prefix}_detection_rates.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots['detection_rates'] = str(plot_path)
        
        # 2. Confidence distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        detected_conf = df[df['detected']]['confidence']
        not_detected_conf = df[~df['detected']]['confidence']
        
        ax.hist(detected_conf, alpha=0.7, label='Detected', bins=20, color='green')
        ax.hist(not_detected_conf, alpha=0.7, label='Not Detected', bins=20, color='red')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Confidence Scores')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / f"{filename_prefix}_confidence_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots['confidence_distribution'] = str(plot_path)
        
        # 3. Detection success/failure by transformation
        fig, ax = plt.subplots(figsize=(12, 8))
        transform_counts = df.groupby(['transformation', 'detected']).size().unstack(fill_value=0)
        transform_counts.plot(kind='barh', ax=ax, color=['red', 'green'], alpha=0.7)
        ax.set_title('Detection Success vs Failure by Transformation')
        ax.set_xlabel('Number of Images')
        ax.legend(['Not Detected', 'Detected'])
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / f"{filename_prefix}_success_failure.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots['success_failure'] = str(plot_path)
        
        return saved_plots
        
    def print_summary(self) -> None:
        """Print a formatted summary of the evaluation results."""
        summary = self.generate_summary_report()
        
        print("=" * 60)
        print("🔐 WATERMARK EVALUATION SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        overall = summary['overall_statistics']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total tests: {overall['total_tests']}")
        print(f"   Detection rate: {overall['overall_detection_percentage']:.1f}%")
        print(f"   Average confidence: {overall['overall_confidence']:.3f}")
        print(f"   Robustness level: {overall['robustness_level']}")
        
        # Best and worst performance
        trans_analysis = summary['transformation_analysis']
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   {trans_analysis['best_performance']['transformation']}: "
              f"{trans_analysis['best_performance']['detection_rate']*100:.1f}%")
        
        print(f"\n⚠️  WORST PERFORMANCE:")
        print(f"   {trans_analysis['worst_performance']['transformation']}: "
              f"{trans_analysis['worst_performance']['detection_rate']*100:.1f}%")
        
        # Robustness categories
        categories = summary['robustness_categories']
        print(f"\n📈 ROBUSTNESS CATEGORIES:")
        for category, transforms in categories.items():
            if transforms:
                print(f"   {category.title()}: {len(transforms)} transformations")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        
        print("\n" + "=" * 60)