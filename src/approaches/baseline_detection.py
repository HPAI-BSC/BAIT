"""Baseline BBox approach: standard bounding box identification."""

import os
from typing import List, Dict, Any, Tuple

from .base import BBoxApproach
from ..core import format_bounding_box, log_message

class BaselineDetectionApproach(BBoxApproach):
    """
    Baseline bounding box approach.
    
    Generates questions asking to identify which bounding box
    contains an object of a specific class, without adversarial
    modifications to the legend or visual presentation.
    """
    
    def get_output_subdir(self) -> str:
        """Output directory for painted bbox images."""
        return 'painted'
    
    def prepare_samples(self, data, **kwargs):
        """Prepare samples with bbox annotations and mappings."""
        normalized_annotations = kwargs.get('normalized_annotations', {})
        box_mapping = kwargs.get('box_mapping', {})
        
        valid_samples = []
        for img_id, row in data.iterrows():
            if (img_id in normalized_annotations and
                normalized_annotations[img_id] and                  # Has at least one bbox
                normalized_annotations[img_id][0][0] != -1 and      # At least one real bbox
                img_id in box_mapping and                           # Has box mapping (painted image has been created)
                box_mapping[img_id] is not None):                   # Box mapping is valid
                valid_samples.append((img_id, normalized_annotations[img_id], box_mapping[img_id]))
        
        return valid_samples
    
    def process_sample(self, sample_data: Tuple, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a single image for detection baseline questions.
        
        Args:
            sample_data: (img_id, bounding_boxes, box_mapping_for_img)
            **kwargs: Contains use_image_paths flag
            
        Returns:
            List of detection questions (one per class in image)
        """
        img_id, bounding_boxes, box_mapping_for_img = sample_data
        use_image_paths = kwargs.get('use_image_paths', False)
        
        sample_results = []
        classes_set = set(box_mapping_for_img.keys())
        
        for bbox in bounding_boxes:
            bbox_class = bbox[0]
            
            if bbox_class not in classes_set:
                continue
            
            # Get painted image path
            painted_img_path = os.path.join(
                self.config.image_dir,
                self.get_output_subdir(),
                f"{img_id}_class_{bbox_class}.jpg"
            )
            
            # Get box mapping for this class
            box_mapping = box_mapping_for_img[bbox_class]
            other_bboxes = [b for b in box_mapping.keys() if b != bbox]
            
            # Create box options
            n_options = getattr(self.config, 'n_options_detection', 4)
            if n_options is None or n_options < 2:
                n_options = 2
            box_options = [f"Box {i+1}" for i in range(n_options)]
            correct_option = box_mapping[bbox]
            
            
            # Count real boxes (excluding fake ones)
            real_boxes = [b for b in other_bboxes if b[0] >= 0 and b[0] != 99]
            n_classes = len(set(b[0] for b in real_boxes)) + 1 if real_boxes else 1
            
            # Create question
            additional_fields = {
                'sample_type': 'baseline',
                'fake_bboxes': [format_bounding_box(b) for b in other_bboxes],
                'fake_bboxes_classes': [b[0] for b in other_bboxes],
            }
            
            question_sample = self.create_bbox_question(
                img_id=img_id,
                image_path=painted_img_path,
                bbox=bbox,
                box_options=box_options,
                correct_option=correct_option,
                bbox_class=bbox_class,
                use_image_paths=use_image_paths,
                additional_fields=additional_fields
            )
            
            sample_results.append(question_sample)
            
            # Process each class only once
            classes_set.remove(bbox_class)
        
        return sample_results
