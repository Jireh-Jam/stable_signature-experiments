"""
Model management and loading utilities for watermark detection.

This module provides utilities for loading and managing watermark detection models,
including the HiDDeN architecture and related components.
"""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from ..common.config import ModelParams

logger = logging.getLogger(__name__)


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. 
    Sequence of Convolution, Batch Normalization, and GELU activation.
    """
    def __init__(self, channels_in: int, channels_out: int):
        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HiddenEncoder(nn.Module):
    """
    Encoder that inserts a watermark into an image.
    """
    def __init__(self, num_blocks: int, num_bits: int, channels: int, last_tanh: bool = True):
        super(HiddenEncoder, self).__init__()
        
        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)
        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            imgs: Input images (B, 3, H, W)
            msgs: Messages to embed (B, num_bits)
            
        Returns:
            Watermarked images (B, 3, H, W)
        """
        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # (B, num_bits, 1, 1)
        msgs = msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1))  # (B, num_bits, H, W)

        encoded_image = self.conv_bns(imgs)  # (B, channels, H, W)

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)  # (B, num_bits + channels + 3, H, W)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w


class HiddenDecoder(nn.Module):
    """
    Decoder that extracts watermarks from images.
    
    The input image may have various kinds of noise applied to it,
    such as crop, JPEG compression, etc.
    """
    def __init__(self, num_blocks: int, num_bits: int, channels: int):
        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            img_w: Watermarked images (B, 3, H, W)
            
        Returns:
            Decoded message features (B, num_bits)
        """
        x = self.layers(img_w)  # (B, num_bits, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, num_bits)
        x = self.linear(x)  # (B, num_bits)
        return x


class JND(nn.Module):
    """
    Just Noticeable Difference (JND) module for perceptual masking.
    Based on luminance and contrast masking.
    """
    
    def __init__(self, preprocess=lambda x: x):
        super(JND, self).__init__()
        
        # Sobel kernels for edge detection
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        
        # Luminance kernel
        kernel_lum = [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.FloatTensor(kernel_lum).unsqueeze(0).unsqueeze(0)

        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.register_buffer('weight_lum', kernel_lum)

        self.preprocess = preprocess
    
    def jnd_la(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Luminance masking component."""
        la = torch.nn.functional.conv2d(x, self.weight_lum, padding=2) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum] / 127)) + 3
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x: torch.Tensor, beta: float = 0.117) -> torch.Tensor:
        """Contrast masking component."""
        grad_x = torch.nn.functional.conv2d(x, self.weight_x, padding=1)
        grad_y = torch.nn.functional.conv2d(x, self.weight_y, padding=1)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    def heatmaps(self, x: torch.Tensor, clc: float = 0.3) -> torch.Tensor:
        """
        Generate JND heatmaps for perceptual masking.
        
        Args:
            x: Input images in [0,1] range (B, 3, H, W)
            clc: Cross-luminance-contrast parameter
            
        Returns:
            JND heatmaps (B, 1, H, W)
        """
        x = 255 * self.preprocess(x)
        # Convert to luminance
        x = 0.299 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :] + 0.114 * x[..., 2:3, :, :]
        
        la = self.jnd_la(x)
        cm = self.jnd_cm(x)
        
        return (la + cm - clc * torch.minimum(la, cm)) / 255


class EncoderWithJND(nn.Module):
    """
    Encoder combined with JND-based perceptual masking.
    """
    def __init__(self, 
                 encoder: HiddenEncoder, 
                 attenuation: Optional[JND], 
                 scale_channels: bool,
                 scaling_i: float,
                 scaling_w: float):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with JND-based masking.
        
        Args:
            imgs: Input images (B, 3, H, W)
            msgs: Messages to embed (B, num_bits)
            
        Returns:
            Watermarked images (B, 3, H, W)
        """
        # Encode watermark
        deltas_w = self.encoder(imgs, msgs)  # (B, 3, H, W)

        # Scale channels (give more weight to blue channel)
        if self.scale_channels:
            aa = 1/4.6  # Normalization factor
            aas = torch.tensor([aa*(1/0.299), aa*(1/0.587), aa*(1/0.114)]).to(imgs.device)
            deltas_w = deltas_w * aas[None, :, None, None]

        # Apply JND-based attenuation
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs)  # (B, 1, H, W)
            deltas_w = deltas_w * heatmaps  # (B, 3, H, W) * (B, 1, H, W) -> (B, 3, H, W)
        
        # Combine original image with watermark
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w
        
        return imgs_w


class ModelManager:
    """
    Manager for loading and handling watermark detection models.
    """
    
    def __init__(self, model_params: ModelParams, device: torch.device):
        """
        Initialize model manager.
        
        Args:
            model_params: Model configuration parameters
            device: Device to load models on
        """
        self.model_params = model_params
        self.device = device
        
        # Standard image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    
    def create_encoder(self) -> HiddenEncoder:
        """Create a new encoder model."""
        return HiddenEncoder(
            num_blocks=self.model_params.encoder_depth,
            num_bits=self.model_params.num_bits,
            channels=self.model_params.encoder_channels
        )
    
    def create_decoder(self) -> HiddenDecoder:
        """Create a new decoder model."""
        return HiddenDecoder(
            num_blocks=self.model_params.decoder_depth,
            num_bits=self.model_params.num_bits,
            channels=self.model_params.decoder_channels
        )
    
    def create_encoder_with_jnd(self) -> EncoderWithJND:
        """Create encoder with JND-based attenuation."""
        encoder = self.create_encoder()
        
        attenuation = None
        if self.model_params.attenuation == "jnd":
            attenuation = JND(preprocess=self.unnormalize)
        
        return EncoderWithJND(
            encoder=encoder,
            attenuation=attenuation,
            scale_channels=self.model_params.scale_channels,
            scaling_i=self.model_params.scaling_i,
            scaling_w=self.model_params.scaling_w
        )
    
    def load_decoder(self, checkpoint_path: Union[str, Path]) -> HiddenDecoder:
        """
        Load a decoder model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded decoder model
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Create decoder model
            decoder = self.create_decoder()
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract decoder state dict
            if 'encoder_decoder' in checkpoint:
                state_dict = checkpoint['encoder_decoder']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Extract decoder-specific parameters
            decoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decoder.'):
                    decoder_state_dict[k.replace('decoder.', '')] = v
                elif 'decoder' not in k and k in [p[0] for p in decoder.named_parameters()]:
                    # Handle case where decoder params don't have 'decoder.' prefix
                    decoder_state_dict[k] = v
            
            # Load state dict
            decoder.load_state_dict(decoder_state_dict, strict=False)
            
            # Move to device and set to eval mode
            decoder = decoder.to(self.device).eval()
            
            logger.info(f"Successfully loaded decoder from {checkpoint_path}")
            return decoder
            
        except Exception as e:
            raise ValueError(f"Failed to load decoder from {checkpoint_path}: {str(e)}")
    
    def load_encoder(self, checkpoint_path: Union[str, Path]) -> HiddenEncoder:
        """
        Load an encoder model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded encoder model
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Create encoder model
            encoder = self.create_encoder()
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract encoder state dict
            if 'encoder_decoder' in checkpoint:
                state_dict = checkpoint['encoder_decoder']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Extract encoder-specific parameters
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '')] = v
            
            # Load state dict
            encoder.load_state_dict(encoder_state_dict, strict=False)
            
            # Move to device and set to eval mode
            encoder = encoder.to(self.device).eval()
            
            logger.info(f"Successfully loaded encoder from {checkpoint_path}")
            return encoder
            
        except Exception as e:
            raise ValueError(f"Failed to load encoder from {checkpoint_path}: {str(e)}")
    
    def save_model(self, 
                   model: nn.Module, 
                   save_path: Union[str, Path],
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a model to checkpoint.
        
        Args:
            model: Model to save
            save_path: Path to save checkpoint
            metadata: Optional metadata to include
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_params': self.model_params.__dict__,
            'model_type': type(model).__name__
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model to {save_path}")


def load_detection_model(checkpoint_path: Union[str, Path],
                        model_params: Optional[ModelParams] = None,
                        device: Optional[torch.device] = None) -> HiddenDecoder:
    """
    Convenience function to load a detection model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_params: Model parameters (uses default if None)
        device: Device to load on (auto-detect if None)
        
    Returns:
        Loaded decoder model
    """
    if model_params is None:
        model_params = ModelParams()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    manager = ModelManager(model_params, device)
    return manager.load_decoder(checkpoint_path)