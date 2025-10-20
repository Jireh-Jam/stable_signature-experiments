"""
Stable Signature Watermarking Method

Implementation of the Stable Signature watermarking technique for latent diffusion models.
Based on "The Stable Signature: Rooting Watermarks in Latent Diffusion Models" (ICCV 2023).
"""

from .method import StableSignatureMethod

__all__ = ["StableSignatureMethod"]