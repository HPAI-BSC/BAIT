"""Legends approach: adversarial bounding box legends."""

import os
import random
import string
from typing import List, Dict, Any, Tuple
from PIL import Image

from .base import BBoxApproach
from ..core import (
    paint_bboxes_with_legends_plt,
    encode_image_file_to_base64, 
    format_bounding_box, 
    normalize_bbox,
    resize_and_save,
    check_overlap
)


class LegendsApproach(BBoxApproach):
    """
    Legends approach.
    
    Generates questions with misleading legends that label bounding boxes
    with incorrect class information to test if models rely on legends
    rather than visual content.
    """
    
    def get_output_subdir(self) -> str:
        """Output directory for legend images."""
        return 'legends'
    
    def prepare_samples(self, data, **kwargs):
        """Prepare samples with pixel annotations and fake bboxes."""
        images = kwargs.get('images', {})
        pixel_annotations = kwargs.get('pixel_annotations', {})
        fake_annotations = kwargs.get('fake_annotations', {})
        box_mapping = kwargs.get('box_mapping', {})
        
        return [
            (img_id, images[img_id], pixel_annotations[img_id], fake_annotations[img_id], box_mapping.get(img_id))
            for img_id in data.index
            if img_id in images and img_id in pixel_annotations and pixel_annotations[img_id][0][0] >= 0
        ]
    
    def process_sample(self, sample_data: Tuple, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a single image for legend-based questions.
        
        Args:
            sample_data: (img_id, image, bounding_boxes, fake_bboxes, box_mapping_for_img)
            **kwargs: Contains use_image_paths flag
            
        Returns:
            List of questions with true/false legends
        """
        img_id, image, bounding_boxes, fake_bboxes, box_mapping_for_img = sample_data
        use_image_paths = kwargs.get('use_image_paths', False)
        
        detect_classes = list(self.config.detect_classnames)
        h, w = image.shape[:2]
        
        results = []
        classes_set = set(box_mapping_for_img.keys())
        
        for real_bbox in bounding_boxes:
            bbox_class = real_bbox[0]
            if bbox_class not in classes_set or bbox_class < 0:
                continue
            
            # Get fake bboxes (from other classes or generated)
            n_options = getattr(self.config, 'n_options_detection', 4)
            n_fake_bboxes = max(0, n_options - 1)  # -1 for the real bbox
            
            fake_bbox_list = []
            for bbox in bounding_boxes:
                if bbox[0] != bbox_class and len(fake_bbox_list) < n_fake_bboxes and not check_overlap([bbox], fake_bbox_list):# and not check_overlap([bbox], [real_bbox]):
                    fake_bbox_list.append(bbox)
                    if len(fake_bbox_list) == n_fake_bboxes:
                        break
            
            # Fill remaining with generated fake bboxes
            for i in range(n_fake_bboxes - len(fake_bbox_list)):
                if fake_bboxes and i < len(fake_bboxes):
                    fake_bbox_list.append(fake_bboxes[i])
            
            all_bboxes = [real_bbox] + fake_bbox_list
            random.shuffle(all_bboxes)
            answer_index = all_bboxes.index(real_bbox)
            
            # Create questions with correct and incorrect legends
            for truth in [True, False]:
                legend_index = answer_index if truth else random.choice(
                    [i for i in range(len(all_bboxes)) if i != answer_index]
                )
                
                output_image_path = os.path.join(
                    self.config.image_dir,
                    self.get_output_subdir(),
                    f"{img_id}-{bbox_class}-{truth}.jpg"
                )
                
                # Paint bboxes with legends
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                bbox_to_color = paint_bboxes_with_legends_plt(
                    image, all_bboxes, legend_index, 
                    output_image_path, detect_classes, bbox_class
                )
                img = Image.open(output_image_path)
                resize_and_save(img, output_image_path, self.config.max_pixels)
                
                fake_bbox = all_bboxes[legend_index]
                
                # Create options
                bbox_options = all_bboxes.copy()
                random.shuffle(bbox_options)
                options = {}
                for i, bbox in enumerate(bbox_options):
                    letter = string.ascii_uppercase[i]
                    options[letter] = f"Box {all_bboxes.index(bbox) + 1}"
                    if bbox == real_bbox:
                        answer_letter = letter
                    if bbox == fake_bbox and not truth:
                        fake_option = letter
                
                # Create question
                question = random.choice(self.config.question_templates.detection)
                question = question.replace('$class_name', detect_classes[bbox_class].lower())
                
                image_data = output_image_path if use_image_paths else encode_image_file_to_base64(output_image_path)
                
                n_classes = len(set(b[0] for b in fake_bbox_list)) + 1 if fake_bbox_list else 1
                fake_class = fake_bbox[0] if not truth else None
                
                # Get bbox to color mapping (we need to reload or store it)
                # For simplicity, we'll use a placeholder
                answer_color = bbox_to_color.get(real_bbox, 'unknown')  # This should be extracted from bbox_to_color
                
                sample_type = 'assistive' if truth else 'adversarial'
                results.append({
                    'dataset': self.config.dataset_name,
                    'approach': self.approach_name,
                    'task': 'detection',
                    'source_id': img_id,
                    'question': question,
                    **options,
                    'answer': answer_letter,
                    'answer_color': answer_color,
                    'true_bbox': format_bounding_box(normalize_bbox(real_bbox, w, h)),
                    'fake_bboxes': [format_bounding_box(normalize_bbox(fb, w, h)) for fb in fake_bbox_list],
                    'fake_bboxes_classes': [fb[0] for fb in fake_bbox_list],
                    'sample_type': sample_type,
                    'true_class': bbox_class,
                    'true_class_name': detect_classes[bbox_class],
                    'fake_option': fake_option if not truth else None,
                    'fake_class': fake_bbox[0] if not truth else None,
                    'fake_class_name': detect_classes[fake_class] if fake_class is not None and fake_class >= 0 and fake_class != 99 else None,
                    'image': image_data
                })
            
            classes_set.remove(bbox_class)
        
        return results
