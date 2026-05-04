"""Image processing utilities for adversarial dataset generation."""

import io
import math
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import gc
from typing import Tuple, Optional


def round_by_factor(number: int, factor: int) -> int:
    """Round a number to the nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Ceiling of number to the nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Floor of number to the nearest multiple of factor."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
) -> Tuple[int, int]:
    """
    Resize dimensions intelligently to meet pixel constraints.
    
    Args:
        height: Original height
        width: Original width
        factor: Alignment factor (both dimensions must be multiples)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels
        
    Returns:
        Tuple of (new_height, new_width)
        
    Raises:
        ValueError: If aspect ratio is too extreme
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme: {height}/{width}")

    # First align to factor
    h = max(factor, round_by_factor(height, factor))
    w = max(factor, round_by_factor(width, factor))

    # If too many pixels, downscale
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = floor_by_factor(height / beta, factor)
        w = floor_by_factor(width / beta, factor)
    # If too few pixels, upscale
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = ceil_by_factor(height * beta, factor)
        w = ceil_by_factor(width * beta, factor)

    return h, w


def resize_and_save(img: Image.Image, output_path: str, max_pixels: int = 1_000_000) -> None:
    """
    Resize an image to fit within max_pixels and save it.
    
    Args:
        img: PIL Image to resize and save
        output_path: Path to save the image
        max_pixels: Maximum allowed pixels (0 or negative to skip resize)
    """
    if max_pixels > 0 and img.width * img.height > max_pixels:
        h, w = smart_resize(img.height, img.width, factor=14, max_pixels=max_pixels)
        img = img.resize((w, h), resample=Image.BICUBIC)
    
    img.save(output_path)


def encode_image_to_base64(
    img: Image.Image,
    target_size: int = -1,
    fmt: str = 'JPEG',
    max_pixels: int = 2_000_000
) -> str:
    """
    Encode a PIL Image to base64 string.
    
    Args:
        img: PIL Image to encode
        target_size: Max thumbnail dimension (-1 to skip)
        fmt: Image format (JPEG, PNG, etc.)
        max_pixels: Maximum allowed pixels
        
    Returns:
        Base64 encoded string
    """
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    
    if max_pixels > 0 and img.width * img.height > max_pixels:
        h, w = smart_resize(img.height, img.width, factor=14, max_pixels=max_pixels)
        img = img.resize((w, h), resample=Image.BICUBIC)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')


def encode_image_file_to_base64(image_path: str, target_size: int = -1) -> str:
    """Encode an image file to base64 string."""
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)


def add_caption_to_image(image_path: str, caption: str, output_path: str) -> None:
    """
    Add a caption at the bottom of an image.
    
    Args:
        image_path: Path to the source image
        caption: Text to add as caption
        output_path: Path to save the captioned image
    """
    img = Image.open(image_path)
    
    # Calculate caption height - scale based on image width
    caption_height = max(80, int(img.width * 0.1))
    
    # Create new image with space for caption
    new_img = Image.new('RGB', (img.width, img.height + caption_height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    
    # Add caption text
    draw = ImageDraw.Draw(new_img)
    
    # Try to use a standard font
    font_size = max(8, int(img.width * 0.02))
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # Wrap text to fit image width
    avg_char_width = font_size * 0.8
    chars_per_line = max(40, int(img.width * 0.9 / avg_char_width))
    wrapped_text = textwrap.fill(caption, width=chars_per_line)
    
    # Calculate text dimensions for centering
    if hasattr(draw, 'textbbox'):
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        try:
            text_width, text_height = draw.textsize(wrapped_text, font=font)
        except:
            text_width = img.width * 0.8
            text_height = caption_height * 0.6
    
    # Calculate centered position
    text_x = (img.width - text_width) // 2
    text_y = img.height + (caption_height - text_height) // 2
    
    # Draw the text
    draw.text((text_x, text_y), wrapped_text, font=font, fill=(0, 0, 0))

    new_img.save(output_path)