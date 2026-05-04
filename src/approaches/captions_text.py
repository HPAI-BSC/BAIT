"""Captions Text approach: captions as text in questions."""

import os
import random
import string
from typing import List, Dict, Any, Tuple

from .base import BaseApproach
from ..core import create_options, encode_image_file_to_base64, format_bounding_box


class CaptionsTextApproach(BaseApproach):
    """
    Captions Text approach.
    
    Generates questions with caption text prepended to the question itself,
    rather than added as visual elements in the image.
    Tests if models handle textual vs visual caption information differently.
    Supports both classification and bounding box tasks.
    """
    
    def __init__(self, config):
        """Initialize with caption templates."""
        super().__init__(config)
        self.caption_templates = config.caption_templates if hasattr(config, 'caption_templates') else {}
    
    def get_output_subdir(self) -> str:
        """Uses base images (captions added as text)."""
        return 'base'
    
    def prepare_samples(self, data, **kwargs):
        """Prepare samples with annotations and mappings."""
        normalized_annotations = kwargs.get('normalized_annotations', {})
        box_mapping = kwargs.get('box_mapping', {})
        
        return [
            (img_id, row, normalized_annotations.get(img_id, []), box_mapping.get(img_id))
            for img_id, row in data.iterrows()
        ]
    
    def process_sample(self, sample_data: Tuple, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a single sample for text caption questions.
        
        Args:
            sample_data: (img_id, row, bounding_boxes, box_mapping_for_img)
            **kwargs: Contains use_image_paths, question_templates
            
        Returns:
            List of questions with caption text prepended
        """
        img_id, row, bounding_boxes, box_mapping_for_img = sample_data
        use_image_paths = kwargs.get('use_image_paths', False)
        
        sample_results = []
        fig_number = str(random.randint(1, 10))
        base_image_path = os.path.join(self.config.image_dir, 'base', f"{img_id}.jpg")
        
        # Process classification questions
        if (self.config.classif_classnames is not None and 
            row.get('classification', False)):
            sample_results.extend(
                self._process_classification_text_captions(
                    img_id, row, base_image_path, fig_number, use_image_paths
                )
            )
        
        # Process bbox questions
        if (self.config.detect_classnames is not None and 
            bounding_boxes and 
            bounding_boxes[0][0] != -1 and 
            row.get('detection', False) and 
            box_mapping_for_img is not None):
            sample_results.extend(
                self._process_bbox_text_captions(
                    img_id, row, bounding_boxes, box_mapping_for_img, fig_number, use_image_paths
                )
            )
        
        return sample_results
    
    def _process_classification_text_captions(
        self, 
        img_id: str, 
        row, 
        base_image_path: str, 
        fig_number: str, 
        use_image_paths: bool
    ) -> List[Dict[str, Any]]:
        """Process classification questions with text captions."""
        class_names = list(self.config.classif_classnames)
        question_templates = self.config.question_templates.get('classification', [])
        caption_templates = self.caption_templates.get('classification', [])
        
        orig_question = random.choice(question_templates)
        caption_template = random.choice(caption_templates).replace('$number', fig_number)
        
        true_class = int(row['class'])
        n_options = getattr(self.config, 'n_options_classification', None)
        if n_options is not None and n_options < 2:
            n_options = 2
        
        true_caption = caption_template.replace('$class_name', class_names[true_class].lower())
        
        options, answer_letter = create_options(class_names, true_class, n_options=n_options)
        distractor_class_names = [name for name in options.values() if name != class_names[true_class]]
        fake_class_name = random.choice(distractor_class_names)
        fake_class = class_names.index(fake_class_name)
        fake_option = next(letter for letter, value in options.items() if value == fake_class_name)
        false_caption = caption_template.replace('$class_name', fake_class_name.lower())
        
        image = base_image_path if use_image_paths else encode_image_file_to_base64(base_image_path)
        
        results = []
        for caption_type, caption in [(True, true_caption), (False, false_caption)]:
            # Prepend caption to question as text
            question = f"Caption: {caption}\n\n{orig_question}"
            
            sample_type = 'assistive' if caption_type else 'adversarial'
            results.append({
				'dataset': self.config.dataset_name,
				'approach': self.approach_name,
				'task': 'classification',
                'source_id': img_id,
                'question': question,
                **options,
                'answer': answer_letter,
                'sample_type': sample_type,
                'true_class': true_class,
                'true_class_name': class_names[true_class],
                'fake_option': fake_option if not caption_type else None,
                'fake_class': fake_class if not caption_type else None,
                'fake_class_name': class_names[fake_class] if not caption_type else None,
                'caption': caption,
                'image': image
            })
        
        return results
    
    def _process_bbox_text_captions(
        self, 
        img_id: str, 
        row, 
        bounding_boxes: List[Tuple], 
        box_mapping_for_img: Dict, 
        fig_number: str, 
        use_image_paths: bool
    ) -> List[Dict[str, Any]]:
        """Process bbox questions with text captions."""
        detect_classes = list(self.config.detect_classnames)
        question_templates = self.config.question_templates.get('detection', [])
        caption_templates = self.caption_templates.get('detection', [])
        
        results = []
        classes_set = set(box_mapping_for_img.keys())
        
        for bbox in bounding_boxes:
            bbox_class = bbox[0]
            if bbox_class not in classes_set or bbox_class < 0:
                continue
            
            painted_img_path = os.path.join(
                self.config.image_dir, 'painted', f"{img_id}_class_{bbox_class}.jpg"
            )
            true_class = detect_classes[bbox_class]
            
            class_mapping = box_mapping_for_img[bbox_class]
            other_bboxes = [b for b in class_mapping.keys() if b != bbox]
            
            orig_question = random.choice(question_templates).replace('$class_name', true_class.lower())
            orig_caption = random.choice(caption_templates).replace('$number', fig_number).replace('$class_name', true_class.lower())
            
            for truth in [True, False]:
                # Select which bbox to mention in caption
                other_real = [b for b in other_bboxes if b[0] >= 0 and b[0] != 99]
                if other_real:
                    question_bbox = bbox if truth else random.choice([b for b in other_real if b != bbox])
                else:
                    question_bbox = bbox if truth else random.choice([b for b in other_bboxes if b != bbox])
                
                caption = orig_caption.replace('$bbox', class_mapping[question_bbox])
                
                # Create options
                n_options = getattr(self.config, 'n_options_detection', 4)
                box_options = [f"Box {i+1}" for i in range(n_options)]
                random.shuffle(box_options)
                correct_option = class_mapping[bbox]
                fake_option = class_mapping[question_bbox]
                correct_index = box_options.index(correct_option)
                fake_index = box_options.index(fake_option)
                options = {letter: opt for letter, opt in zip(string.ascii_uppercase, box_options)}
                answer_letter = string.ascii_uppercase[correct_index]
                fake_option_letter = string.ascii_uppercase[fake_index]
                
                # Prepend caption to question as text
                question = f"Caption: {caption}\n{orig_question}"
                
                image = painted_img_path if use_image_paths else encode_image_file_to_base64(painted_img_path)
                
                n_classes = len(set(b[0] for b in other_real)) + 1 if other_real else 1
                fake_class = question_bbox[0] if not truth else None
                
                sample_type = 'assistive' if truth else 'adversarial'
                results.append({
                    'dataset': self.config.dataset_name,
                    'approach': self.approach_name,
                    'task': 'detection',
                    'source_id': img_id,
                    'question': question,
                    **options,
                    'answer': answer_letter,
                    'true_bbox': format_bounding_box(bbox),
                    'fake_bboxes': [format_bounding_box(b) for b in other_bboxes if b != bbox],
                    'fake_bboxes_classes': [fb[0] for fb in other_bboxes if fb != bbox],
                    'sample_type': sample_type,
                    'true_class': bbox_class,
                    'true_class_name': true_class,
                    'fake_option': fake_option_letter if not truth else None,
                    'fake_class': fake_class if not truth else None,
                    'fake_class_name': detect_classes[fake_class] if fake_class is not None and fake_class >= 0 and fake_class != 99 else None,
                    'caption': caption,
                    'image': image,
                })
            
            classes_set.remove(bbox_class)
        
        return results
