"""Sycophancy approach: questions with misleading context."""

import os
import random
import string
from typing import List, Dict, Any, Tuple

from .base import BaseApproach
from ..core import create_options, encode_image_file_to_base64, format_bounding_box


class SycophancyApproach(BaseApproach):
	"""
	Sycophancy approach.
	
	Generates questions with misleading context that suggests a specific answer.
	Tests if models are influenced by question wording rather than image content.
	Supports both classification and detection tasks.
	"""
	
	def __init__(self, config):
		"""Initialize with trigger templates."""
		super().__init__(config)
		self.sycophancy_triggers = config.sycophancy_triggers if hasattr(config, 'sycophancy_triggers') else {}
	
	def get_output_subdir(self) -> str:
		"""Uses base images."""
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
		Process a single sample for sycophancy questions.
		
		Args:
			sample_data: (img_id, row, bounding_boxes, box_mapping_for_img)
			**kwargs: Contains use_image_paths, question_templates
			
		Returns:
			List of questions with correct/misleading context
		"""
		img_id, row, bounding_boxes, box_mapping_for_img = sample_data
		use_image_paths = kwargs.get('use_image_paths', False)
		
		sample_results = []
		base_img_path = os.path.join(self.config.image_dir, 'base', f"{img_id}.jpg")
		
		# Process classification questions
		if (self.config.classif_classnames is not None and 
			row.get('classification', False)):
			sample_results.extend(
				self._process_classification_sycophancy(
					img_id, row, base_img_path, use_image_paths
				)
			)
		
		# Process detection questions
		if (self.config.detect_classnames is not None and 
			bounding_boxes and 
			bounding_boxes[0][0] != -1 and 
			row.get('detection', False) and 
			box_mapping_for_img is not None):
			sample_results.extend(
				self._process_detection_sycophancy(
					img_id, row, bounding_boxes, box_mapping_for_img, use_image_paths
				)
			)
		
		return sample_results
	
	def _process_classification_sycophancy(
		self, 
		img_id: str, 
		row, 
		base_img_path: str, 
		use_image_paths: bool
	) -> List[Dict[str, Any]]:
		"""Process classification questions with sycophancy."""
		class_names = list(self.config.classif_classnames)
		question_templates = self.config.question_templates.get('classification', [])
		sycophancy_triggers = self.sycophancy_triggers.get('classification', [])
		
		true_class = int(row['class'])
		n_options = getattr(self.config, 'n_options_classification', None)
		if n_options is not None and n_options < 2:
			n_options = 2
		options, answer_letter = create_options(class_names, true_class, n_options=n_options)
		image = base_img_path if use_image_paths else encode_image_file_to_base64(base_img_path)
		
		results = []

		original_trigger = random.choice(sycophancy_triggers)
		original_question = random.choice(question_templates)
		
		# Create questions with correct and misleading context
		for matches_class in [True, False]:
			distractor_class_names = [name for name in options.values() if name != class_names[true_class]]
			question_class = true_class if matches_class else class_names.index(random.choice(distractor_class_names))
			
			# Get trigger and question
			trigger = original_trigger.replace('$class_name', class_names[question_class].lower())
			question = f"{trigger} {original_question}"
			
			fake_option = next(
				letter for letter, value in options.items()
				if value == class_names[question_class]
			)
			
			sample_type = 'assistive' if matches_class else 'adversarial'
			results.append({
				'dataset': self.config.dataset_name,
				'approach': self.approach_name,
				'task': 'classification',
				'source_id': img_id,
				'question': question,
				**options,
				'answer': answer_letter,
				'true_bbox': None,
				'sample_type': sample_type,
				'true_class': true_class,
				'true_class_name': class_names[true_class],
				'fake_option': fake_option if not matches_class else None,
				'fake_class': question_class if not matches_class else None,
				'fake_class_name': class_names[question_class] if not matches_class else None,
				'syco_trigger': trigger,
				'image': image,
			})
		
		return results
	
	def _process_detection_sycophancy(
		self, 
		img_id: str, 
		row, 
		bounding_boxes: List[Tuple], 
		box_mapping_for_img: Dict, 
		use_image_paths: bool
	) -> List[Dict[str, Any]]:
		"""Process detection questions with sycophancy."""
		detection_classes = list(self.config.detect_classnames)
		question_templates = self.config.question_templates.get('detection', [])
		sycophancy_triggers = self.sycophancy_triggers.get('detection', [])
		
		results = []
		classes_set = set(box_mapping_for_img.keys())
		
		for bbox in bounding_boxes:
			bbox_class = bbox[0]
			if bbox_class not in classes_set or bbox_class < 0:
				continue
			
			painted_img_path = os.path.join(
				self.config.image_dir, 'painted', f"{img_id}_class_{bbox_class}.jpg"
			)
			true_class = detection_classes[bbox_class]
			
			class_mapping = box_mapping_for_img[bbox_class]
			other_bboxes = [b for b in class_mapping.keys() if b != bbox]
			
			original_trigger = random.choice(sycophancy_triggers)
			original_question = random.choice(question_templates).replace('$class_name', true_class.lower())
			
			for truth in [True, False]:
				# Select which bbox to mention in trigger
				other_real = [b for b in other_bboxes if b[0] >= 0 and b[0] != 99]
				if other_real:
					question_bbox = bbox if truth else random.choice([b for b in other_real if b != bbox])
				else:
					question_bbox = bbox if truth else random.choice([b for b in other_bboxes if b != bbox])
				
				image = painted_img_path if use_image_paths else encode_image_file_to_base64(painted_img_path)
				
				# Get trigger and replace placeholders
				trigger = original_trigger.replace('$class_name', true_class.lower())
				trigger = trigger.replace('$bbox', class_mapping[question_bbox])
				
				# Combine trigger with question
				question = f"{trigger} {original_question}"
				
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
					'fake_bboxes_classes': [b[0] for b in other_bboxes if b != bbox],
					'sample_type': sample_type,
					'true_class': bbox_class,
					'true_class_name': true_class,
					'fake_option': fake_option_letter if not truth else None,
					'fake_class': fake_class if not truth else None,
					'fake_class_name': detection_classes[fake_class] if fake_class is not None and fake_class >= 0 and fake_class != 99 else None,
					'syco_trigger': trigger,
					'image': image,
				})
			
			classes_set.remove(bbox_class)
		
		return results
