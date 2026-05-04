"""Baseline approach: standard classification without modifications."""
import random
import os
from typing import List, Dict, Any, Tuple
from PIL import Image
import cv2

from .base import ClassificationApproach
from ..core import resize_and_save


class BaselineClassificationApproach(ClassificationApproach):
    """
    Baseline approach for classification tasks.
    
    Generates standard multiple-choice classification questions
    from base images without any adversarial modifications.
    """
    
    def get_output_subdir(self) -> str:
        """Output directory for baseline images."""
        return 'base'
    
    def process_sample(self, sample_data: Tuple, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a single image for baseline classification.
        
        Args:
            sample_data: (img_id, row)
            **kwargs: Contains use_image_paths flag
            
        Returns:
            List with single classification question
        """

        img_id, row = sample_data
        use_image_paths = kwargs.get('use_image_paths', False)
        
        # Prepare and save base image
        img_path = os.path.join(
            self.config.image_dir,
            self.get_output_subdir(),
            f"{img_id}.jpg"
        )
        
        # Create classification question
        question_sample = self.create_classification_question(
            img_id=img_id,
            image_path=img_path,
            true_class=row['class'],
            use_image_paths=use_image_paths,
            additional_fields={'sample_type': 'baseline'}
        )
        
        return [question_sample]
