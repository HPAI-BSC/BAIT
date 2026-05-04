"""
Refactored Adversarial Datasets Framework

A clean, modular, and optimized framework for generating adversarial
evaluation datasets for vision-language models.

Key Features:
- Clean architecture with base classes
- Minimal code duplication
- Easy to extend with new approaches
- Full type annotations
- Comprehensive documentation

Quick Start:
    from src_refactored.generator import DatasetGenerator
    from omegaconf import DictConfig
    
    config = DictConfig({...})
    generator = DatasetGenerator(config)
    generator.run()

For more information, see README.md
"""

__version__ = "2.0.0"
__author__ = "Refactored Framework Team"

from . import core
from . import approaches

__all__ = [
    'core',
    'approaches',
]
