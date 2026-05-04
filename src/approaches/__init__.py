"""Adversarial approach implementations."""

from .base import BaseApproach, ClassificationApproach, BBoxApproach, ApproachConfig
from .baseline_classification import BaselineClassificationApproach
from .baseline_detection import BaselineDetectionApproach
from .captions import CaptionsApproach
from .captions_text import CaptionsTextApproach
from .legends import LegendsApproach
from .sycophancy import SycophancyApproach

__all__ = [
    'BaseApproach',
    'ClassificationApproach',
    'BBoxApproach',
    'ApproachConfig',
    'BaselineClassificationApproach',
    'BaselineDetectionApproach',
    'CaptionsApproach',
    'CaptionsTextApproach',
    'LegendsApproach',
    'SycophancyApproach',
]
