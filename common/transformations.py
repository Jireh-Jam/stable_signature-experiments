"""
Transformations pipeline and registry used across the project.

This module provides a set of pure PIL-based transformations and simple
registry helpers for composing transformation pipelines in a deterministic
and testable way.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random


TransformFn = Callable[[Image.Image], Image.Image]


class ImageTransformations:
    """Collection of image transformations for watermark robustness testing.

    All functions are pure (no I/O) and operate on PIL Images.
    """

    @staticmethod
    def crop_center(image: Image.Image, crop_percentage: float = 0.1) -> Image.Image:
        width, height = image.size
        crop_w = int(width * crop_percentage)
        crop_h = int(height * crop_percentage)
        left = crop_w
        top = crop_h
        right = width - crop_w
        bottom = height - crop_h
        return image.crop((left, top, right, bottom))

    @staticmethod
    def crop_random(image: Image.Image, crop_percentage: float = 0.1) -> Image.Image:
        width, height = image.size
        crop_w = int(width * crop_percentage)
        crop_h = int(height * crop_percentage)
        left = random.randint(0, crop_w)
        top = random.randint(0, crop_h)
        right = width - random.randint(0, crop_w)
        bottom = height - random.randint(0, crop_h)
        return image.crop((left, top, right, bottom))

    @staticmethod
    def blur_gaussian(image: Image.Image, radius: float = 1.0) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def blur_motion(image: Image.Image, size: int = 5) -> Image.Image:
        kernel = ImageFilter.Kernel((size, size), [1 / size] * size + [0] * (size * size - size))
        return image.filter(kernel)

    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float = 1.2) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(factor)

    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(factor)

    @staticmethod
    def adjust_saturation(image: Image.Image, factor: float = 1.2) -> Image.Image:
        return ImageEnhance.Color(image).enhance(factor)

    @staticmethod
    def resize_image(image: Image.Image, scale: float = 0.8) -> Image.Image:
        width, height = image.size
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def rotate_image(image: Image.Image, angle: float = 5.0) -> Image.Image:
        return image.rotate(angle, expand=True, fillcolor="white")

    @staticmethod
    def add_noise(image: Image.Image, noise_level: float = 0.1) -> Image.Image:
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    @staticmethod
    def jpeg_compression(image: Image.Image, quality: int = 85) -> Image.Image:
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    @classmethod
    def get_standard_transformations(cls) -> Dict[str, TransformFn]:
        return {
            "crop_10_percent": lambda img: cls.crop_center(img, 0.1),
            "crop_20_percent": lambda img: cls.crop_center(img, 0.2),
            "crop_random_15": lambda img: cls.crop_random(img, 0.15),
            "blur_light": lambda img: cls.blur_gaussian(img, 1.0),
            "blur_medium": lambda img: cls.blur_gaussian(img, 2.0),
            "blur_heavy": lambda img: cls.blur_gaussian(img, 3.0),
            "motion_blur": lambda img: cls.blur_motion(img, 5),
            "brighten_20": lambda img: cls.adjust_brightness(img, 1.2),
            "darken_20": lambda img: cls.adjust_brightness(img, 0.8),
            "brighten_50": lambda img: cls.adjust_brightness(img, 1.5),
            "darken_50": lambda img: cls.adjust_brightness(img, 0.5),
            "high_contrast": lambda img: cls.adjust_contrast(img, 1.5),
            "low_contrast": lambda img: cls.adjust_contrast(img, 0.7),
            "saturate": lambda img: cls.adjust_saturation(img, 1.3),
            "desaturate": lambda img: cls.adjust_saturation(img, 0.7),
            "grayscale": lambda img: cls.adjust_saturation(img, 0.0),
            "resize_80": lambda img: cls.resize_image(img, 0.8),
            "resize_60": lambda img: cls.resize_image(img, 0.6),
            "resize_120": lambda img: cls.resize_image(img, 1.2),
            "rotate_5": lambda img: cls.rotate_image(img, 5.0),
            "rotate_neg5": lambda img: cls.rotate_image(img, -5.0),
            "rotate_10": lambda img: cls.rotate_image(img, 10.0),
            "noise_light": lambda img: cls.add_noise(img, 0.05),
            "noise_medium": lambda img: cls.add_noise(img, 0.1),
            "jpeg_90": lambda img: cls.jpeg_compression(img, 90),
            "jpeg_70": lambda img: cls.jpeg_compression(img, 70),
            "jpeg_50": lambda img: cls.jpeg_compression(img, 50),
        }

    @classmethod
    def get_aggressive_transformations(cls) -> Dict[str, TransformFn]:
        return {
            "crop_30_percent": lambda img: cls.crop_center(img, 0.3),
            "blur_extreme": lambda img: cls.blur_gaussian(img, 5.0),
            "brighten_extreme": lambda img: cls.adjust_brightness(img, 2.0),
            "darken_extreme": lambda img: cls.adjust_brightness(img, 0.3),
            "resize_40": lambda img: cls.resize_image(img, 0.4),
            "rotate_30": lambda img: cls.rotate_image(img, 30.0),
            "noise_heavy": lambda img: cls.add_noise(img, 0.2),
            "jpeg_30": lambda img: cls.jpeg_compression(img, 30),
            "jpeg_10": lambda img: cls.jpeg_compression(img, 10),
        }

    @classmethod
    def apply_transformation_chain(
        cls, image: Image.Image, transformations: List[Tuple[str, TransformFn]]
    ) -> Image.Image:
        result = image.copy()
        for name, transform_func in transformations:
            try:
                result = transform_func(result)
            except Exception:
                # keep pipeline robust; skip failing transformations
                continue
        return result
