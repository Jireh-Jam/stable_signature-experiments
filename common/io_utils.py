"""
Common I/O utilities for file operations and data management.

This module provides utilities for file operations, CSV handling, and directory management.
"""

import os
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def create_output_directories(base_dir: Union[str, Path] = "output") -> Dict[str, Path]:
    """
    Create standard output directory structure.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_path = Path(base_dir)
    
    directories = {
        'watermarked': base_path / 'watermarked',
        'original': base_path / 'original',
        'attacked': base_path / 'attacked',
        'difference': base_path / 'difference',
        'combined': base_path / 'combined',
        'metrics': base_path / 'metrics',
        'logs': base_path / 'logs'
    }
    
    for name, path in directories.items():
        ensure_directory(path)
    
    logger.info(f"Created output directory structure in {base_path}")
    return directories


def save_metrics_csv(
    metrics: Union[Dict[str, Any], List[Dict[str, Any]]],
    filepath: Union[str, Path],
    append: bool = False
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary or list of dictionaries containing metrics
        filepath: Output CSV file path
        append: Whether to append to existing file
    """
    filepath = Path(filepath)
    
    # Ensure metrics is a list
    if isinstance(metrics, dict):
        metrics = [metrics]
    
    if not metrics:
        logger.warning("No metrics to save")
        return
    
    # Get all unique keys
    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())
    fieldnames = sorted(all_keys)
    
    mode = 'a' if append and filepath.exists() else 'w'
    
    try:
        with open(filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only for new files
            if mode == 'w' or not filepath.exists():
                writer.writeheader()
            
            writer.writerows(metrics)
        
        logger.info(f"Saved {len(metrics)} metrics to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save metrics to {filepath}: {str(e)}")
        raise


def load_metrics_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load metrics from CSV file.
    
    Args:
        filepath: CSV file path
        
    Returns:
        List of dictionaries containing metrics
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Metrics file does not exist: {filepath}")
        return []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            metrics = list(reader)
        
        logger.info(f"Loaded {len(metrics)} metrics from {filepath}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to load metrics from {filepath}: {str(e)}")
        return []


def save_json(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output JSON file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Saved JSON to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {str(e)}")
        raise


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"JSON file does not exist: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {str(e)}")
        return None


def save_pickle(
    data: Any,
    filepath: Union[str, Path]
) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Output pickle file path
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved pickle to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save pickle to {filepath}: {str(e)}")
        raise


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Pickle file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Pickle file does not exist: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded pickle from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load pickle from {filepath}: {str(e)}")
        return None


def get_image_paths(
    directory: Union[str, Path],
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
    recursive: bool = True
) -> List[Path]:
    """
    Get all image paths from a directory.
    
    Args:
        directory: Directory to search
        extensions: List of valid image extensions
        recursive: Whether to search subdirectories
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    image_paths = []
    
    if recursive:
        for ext in extensions:
            image_paths.extend(directory.rglob(f"*{ext}"))
            image_paths.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))
    
    logger.info(f"Found {len(image_paths)} images in {directory}")
    return image_paths


def create_timestamp_filename(
    prefix: str = "result",
    extension: str = ".csv"
) -> str:
    """
    Create a filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"


def backup_file(
    filepath: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Create a backup of a file.
    
    Args:
        filepath: File to backup
        backup_dir: Directory for backup (defaults to same directory)
        
    Returns:
        Path to backup file, or None if backup failed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"File does not exist, cannot backup: {filepath}")
        return None
    
    # Determine backup directory
    if backup_dir is None:
        backup_dir = filepath.parent
    else:
        backup_dir = ensure_directory(backup_dir)
    
    # Create backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
    backup_path = backup_dir / backup_name
    
    try:
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return None