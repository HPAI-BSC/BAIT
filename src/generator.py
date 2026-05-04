"""Main dataset generator orchestrator."""

import os
import cv2
import pickle
import pandas as pd
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import random
import numpy as np

from .core import (
	log_message, 
	get_file_paths, 
	validate_paths,
	load_annotations,
	generate_fake_bboxes,
	paint_bboxes_with_labels,
	resize_and_save,
	check_overlap
)
from .approaches import (
	ApproachConfig,
	BaselineClassificationApproach,
	BaselineDetectionApproach,
	CaptionsApproach,
	CaptionsTextApproach,
	LegendsApproach,
	SycophancyApproach,
)
from PIL import Image

random.seed(42)
np.random.seed(42)
class DatasetGenerator:
	"""
	Main orchestrator for adversarial dataset generation.
	
	This class manages:
	- Data loading and validation
	- Image preprocessing
	- Fake bbox generation
	- Approach instantiation and execution
	"""
	
	# Map approach names to classes
	APPROACH_MAP = {
		'baseline_classification': BaselineClassificationApproach,
		'baseline_detection': BaselineDetectionApproach,
		'captions': CaptionsApproach,
		'captions_text': CaptionsTextApproach,
		'legends': LegendsApproach,
		'sycophancy': SycophancyApproach,
	}
	
	def __init__(self, config: DictConfig, dfs: Optional[List[pd.DataFrame]] = None):
		"""
		Initialize the dataset generator.
		
		Args:
			config: Hydra configuration object
			dfs: Optional list of pre-loaded dataframes
		"""
		self.config = config
		self.data_dir = config.get("data_dir")
		self.dataset_path = config.get("dataset_path")
		self.use_image_paths = config.get("use_image_paths", True)
		self.input_image_ext = config.get("input_image_ext", ".jpg")
		
		# Ensure output directories exist
		if self.config.get('output_dir'):
			os.makedirs(config.get('output_dir'), exist_ok=True)
		if self.config.get('image_dir'):
			os.makedirs(config.get('image_dir'), exist_ok=True)
		
		# Initialize data containers
		self.data: Optional[pd.DataFrame] = None
		self.images: Dict[str, Any] = {}
		self.paths: Dict[str, str] = {}
		self.pixel_annotations: Dict[str, List] = {}
		self.normalized_annotations: Dict[str, List] = {}
		self.fake_annotations: Dict[str, List] = {}
		self.box_mappings: Dict[str, Dict] = {}
		self.generated_files: List[str] = []
		self.generated_dfs: List[pd.DataFrame] = dfs if dfs is not None else []
	
	def run(self):
		"""Execute the complete dataset generation pipeline."""
		log_message(f"Starting dataset generation for {self.config['dataset_name']}")
		
		# Load and validate data
		self._load_data()
		self._load_images()
		self._load_annotations()
		
		# Apply subsampling if configured
		self._apply_subsampling()
		
		# Generate fake bboxes if needed
		needs_fake_bboxes = self._check_needs_fake_bboxes()
		if needs_fake_bboxes:
			self._generate_fake_bboxes()
		
		# Create base and painted images
		self._create_base_images()
		
		# Run each requested approach
		self._run_approaches()
		self._merge_outputs()
		
		log_message("Dataset generation complete!")
	
	def _load_data(self):
		"""Load the dataset CSV."""
		log_message("Loading data...")
		if not self.dataset_path:
			raise ValueError("Dataset path not found in the config file.")
		
		self.data = pd.read_csv(
			self.dataset_path, 
			index_col='image_id', 
			dtype={'image_id': str}
		)
		log_message(f"Loaded {len(self.data)} samples")
	
	def _load_images(self):
		"""Load all images in parallel."""
		log_message("Loading images...")
		
		# Get image paths
		self.paths = get_file_paths(self.data, self.data_dir, self.input_image_ext)
		
		# Validate all paths exist
		missing_paths = validate_paths(self.paths)
		if missing_paths:
			log_message(f"Error: {len(missing_paths)} image paths do not exist:")
			for path in missing_paths[:10]:
				log_message(f"  - {path}")
			if len(missing_paths) > 10:
				log_message(f"  ... and {len(missing_paths) - 10} more")
			raise FileNotFoundError(f"{len(missing_paths)} image files not found")
		
		log_message(f"All {len(self.paths)} image paths verified")
		
		# Load images in parallel
		def load_image(img_id: str, path: str):
			return img_id, cv2.imread(path)
		
		with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
			self.images = dict(
				executor.map(lambda item: load_image(*item), self.paths.items())
			)
		
		# Validate all images loaded
		failed_loads = [img_id for img_id, img in self.images.items() if img is None]
		if failed_loads:
			log_message(f"Error: {len(failed_loads)} images failed to load:")
			for img_id in failed_loads[:10]:
				log_message(f"  - {img_id}")
			raise RuntimeError(f"{len(failed_loads)} images failed to load")
		
		log_message(f"All {len(self.images)} images loaded successfully")
	
	def _load_annotations(self):
		"""Load bounding box annotations."""
		log_message("Loading annotations...")
		
		for img_id in self.data.index:
			annot_path = os.path.join(self.data_dir, "annotations", f"{img_id}.txt")
			h, w = self.images[img_id].shape[:2]            
			self.pixel_annotations[img_id], self.normalized_annotations[img_id] = load_annotations(annot_path, w, h)
		
		log_message("Annotations loaded")
	
	def _check_needs_fake_bboxes(self) -> bool:
		"""Check if any approach requires fake bboxes."""
		detection_approaches = ['baseline_detection', 'sycophancy', 'captions', 'legends']
		return (any(approach in self.config['to_create'] for approach in detection_approaches) and 
				self.config.get('detect_classes') is not None)
	
	def _apply_subsampling(self):
		"""Apply subsampling if configured."""
		if 'subsample_path' not in self.config:
			self.data_detect = self.data.copy()
			self.data_classif = self.data.copy()
			self.detect_pairs = None
			return
		
		log_message("Applying subsampling...")
		
		subsample_path = self.config['subsample_path']
		dataset_name = self.config['dataset_name']
		
		# Load detection subsample
		detection_file = os.path.join(subsample_path, f"{dataset_name}_detection.txt")
		if self.config.get('detect_classes') is None:
			ids_detection = []
			self.detect_pairs = None
		elif os.path.exists(detection_file):
			pairs_detection = pd.read_csv(detection_file, header=None)[0].tolist()
			ids_detection = ["-".join(id_class.split("-")[:-1]) for id_class in pairs_detection]
			
			self.detect_pairs = {img_id: [] for img_id in ids_detection}
			for img_id in ids_detection:
				for bbox in self.pixel_annotations[img_id]:
					if f"{img_id}-{bbox[0]}" in pairs_detection:
						self.detect_pairs[img_id].append(bbox[0])
		else:
			ids_detection = self.data.index.tolist()
			self.detect_pairs = {}
		
		# Load classification subsample
		classif_file = os.path.join(subsample_path, f"{dataset_name}_classification.txt")
		if self.config.get('classif_classes') is None:
			ids_classif = []
		elif os.path.exists(classif_file):
			pairs_classif = pd.read_csv(classif_file, header=None)[0].tolist()
			ids_classif = ["-".join(id_class.split("-")[:-1]) for id_class in pairs_classif]
		else:
			ids_classif = self.data.index.tolist()

		self.data_detect = self.data[self.data.index.isin(ids_detection)]
		self.data_classif = self.data[self.data.index.isin(ids_classif)]
		self.data = self.data[self.data.index.isin(ids_detection) | self.data.index.isin(ids_classif)]
		
		log_message(f"Classification samples: {len(self.data_classif)}")
		log_message(f"Detection samples: {len(self.data_detect)}")
	
	def _generate_fake_bboxes(self):
		"""Generate or load fake bounding boxes."""
		fake_bbox_path = self.config.get('fake_bbox_path')
		fake_bbox_arguments = self.config.get("fake_bbox_arguments", {})
		
		if fake_bbox_path and os.path.exists(fake_bbox_path):
			log_message(f"Loading fake bbox annotations from {fake_bbox_path}...")
			self.fake_annotations = pickle.load(open(fake_bbox_path, 'rb'))
		else:
			log_message("Generating fake bbox annotations...")
			
			for img_id in self.data_detect.index:
				if self.pixel_annotations[img_id][0][0] != -1:
					self.fake_annotations[img_id] = generate_fake_bboxes(
						self.images[img_id],
						self.pixel_annotations[img_id],
						n=3,
						**fake_bbox_arguments
					)
				else:
					self.fake_annotations[img_id] = None
			
			if fake_bbox_path:
				log_message(f"Saving fake bbox annotations to {fake_bbox_path}...")
				os.makedirs(os.path.dirname(fake_bbox_path), exist_ok=True)
				with open(fake_bbox_path, 'wb') as f:
					pickle.dump(self.fake_annotations, f)
		
		# Mark invalid samples
		for img_id in self.data_detect.index:
			if self.fake_annotations.get(img_id) is None:
				self.pixel_annotations[img_id] = [(-1, -1, -1, -1, -1)]
				self.normalized_annotations[img_id] = [(-1, -1, -1, -1, -1)]
	
	def _create_base_images(self):
		"""Create base and images with painted bounding boxes."""
		log_message("Creating base images...")
		
		# Create directories
		dirs = ['base', 'painted', 'captions/classification', 'captions/detection', 'legends']
		for d in dirs:
			os.makedirs(os.path.join(self.config['image_dir'], d), exist_ok=True)
		
		img_dir = self.config['image_dir']
		
		for img_id, image in self.images.items():
			# Save base image
			if self.config.get('overwrite_images', False) or not os.path.exists(os.path.join(img_dir, 'base', f"{img_id}.jpg")):
				if img_id in self.data_classif.index:
					base_path = os.path.join(img_dir, 'base', f"{img_id}.jpg")
					base_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
					resize_and_save(base_image, base_path, self.config['max_pixels'])
			
			# Create painted images for detection tasks
			if (img_id in self.data_detect.index and 
				self.pixel_annotations[img_id][0][0] != -1):
				self._create_painted_images(img_id, image)
	
	def _create_painted_images(self, img_id: str, image):
		"""Create painted images with labeled bounding boxes."""
		n_options = self.config.get('n_options_detection', 4)
		if n_options is None or n_options < 2:
			n_options = 2
		unique_classes = set([bbox[0] for bbox in self.pixel_annotations[img_id]])
		self.box_mappings[img_id] = {}
		
		for target_class in unique_classes:
			if self.detect_pairs is not None and self.detect_pairs and target_class not in self.detect_pairs.get(img_id, []):
				continue

			# Get bboxes for this class
			target_bbox_indices = [
				i for i, bbox in enumerate(self.pixel_annotations[img_id]) 
				if bbox[0] == target_class
			]
			
			if not target_bbox_indices:
				continue
			
			target_idx = target_bbox_indices[0]
			true_bbox = self.pixel_annotations[img_id][target_idx]
			bboxes = [true_bbox]
			
			# Add bboxes from other classes
			for bbox in self.pixel_annotations[img_id]:
				if bbox[0] != target_class and not check_overlap([bbox], bboxes):
					bboxes.append(bbox)
					if len(bboxes) == n_options:
						break
			
			# Fill with fake bboxes if needed
			if self.fake_annotations.get(img_id) is not None and len(bboxes) < n_options:
				bboxes.extend(self.fake_annotations[img_id][:n_options - len(bboxes)])
			
			# Paint bboxes
			paint_image =  self.config.get('overwrite_images', False) or not os.path.exists(os.path.join(self.config['image_dir'], 'painted', f"{img_id}_class_{target_class}.jpg"))
			painted_path = os.path.join(
				self.config['image_dir'], 
				'painted', 
				f"{img_id}_class_{target_class}.jpg"
			)

			mapping = paint_bboxes_with_labels(image, bboxes, painted_path, paint_image)
			if paint_image:
				painted = Image.open(painted_path)
				resize_and_save(painted, painted_path, self.config['max_pixels'])
			self.box_mappings[img_id][target_class] = mapping
	
	def _run_approaches(self):
		"""Run all requested approaches."""
		# Mark which samples are for bbox vs classification
		# Initialize with False defaults
		self.data['detection'] = False
		self.data['classification'] = False
		
		# Mark detection samples
		for idx in self.data_detect.index:
			if idx in self.data.index and self.pixel_annotations[idx][0][0] != -1:
				self.data.loc[idx, 'detection'] = True
		
		# Mark classification samples
		for idx in self.data_classif.index:
			if idx in self.data.index and not np.isnan(self.data.loc[idx, 'class']):
				self.data.loc[idx, 'classification'] = True
		
		# Run each approach
		for approach_name in self.config['to_create']:
			if approach_name in self.APPROACH_MAP:
				self._run_approach(approach_name)
			else:
				log_message(f"Warning: Approach '{approach_name}' not implemented")
	
	def _run_approach(self, approach_name: str):
		"""Run a specific approach."""
		print("\n")
		log_message(f"Running approach: {approach_name}")
		
		# Create approach config
		approach_config = ApproachConfig(
			dataset_name=self.config['dataset_name'],
			output_dir=self.config['output_dir'],
			image_dir=self.config['image_dir'],
			max_pixels=self.config.get('max_pixels', 1_000_000),
			n_options_classification=self.config.get('n_options_classification'),
			n_options_detection=self.config.get('n_options_detection'),
			classif_classnames=self.config.get('classif_classes'),
			detect_classnames=self.config.get('detect_classes'),
		)
		
		# Add approach-specific config
		# Question templates are at root level (shared across approaches)
		if 'question_templates' in self.config:
			approach_config.question_templates = self.config['question_templates']
		
		# Caption templates (for captions and captions_text)
		if approach_name in ['captions', 'captions_text']:
			try:
				approach_config.caption_templates = self.config['captions']['caption_templates']
			except KeyError:
				log_message("Warning: 'Caption Templates' not found in config for captions approach/es")
		
		# Sycophancy triggers (for sycophancy)
		if approach_name == 'sycophancy':
			try:
				approach_config.sycophancy_triggers = self.config['sycophancy']['sycophancy_triggers']
			except KeyError:
				log_message("Warning: 'Sycophancy Triggers' not found in config for sycophancy approach")
		
		# Instantiate approach
		approach_class = self.APPROACH_MAP[approach_name]
		approach = approach_class(approach_config)
		
		# Prepare data and kwargs based on approach type
		if approach_name == 'baseline_classification':
			data_to_use = self.data[self.data['classification']]
			kwargs = {'paths': self.paths}
		elif approach_name == 'baseline_detection':
			data_to_use = self.data[self.data['detection']]
			kwargs = {
				'normalized_annotations': self.normalized_annotations,
				'box_mapping': self.box_mappings
			}
		elif approach_name in ['captions', 'captions_text', 'sycophancy']:
			data_to_use = self.data
			kwargs = {
				'normalized_annotations': self.normalized_annotations,
				'box_mapping': self.box_mappings,
				'question_templates': approach_config.question_templates,
			}
		elif approach_name == 'legends':
			data_to_use = self.data[self.data['detection']]
			kwargs = {
				'images': self.images,
				'pixel_annotations': self.pixel_annotations,
				'fake_annotations': self.fake_annotations,
				'box_mapping': self.box_mappings,
			}
		else:
			data_to_use = self.data
			kwargs = {}
		
		# Run approach
		df = approach.create_dataset(data_to_use, use_image_paths=self.use_image_paths, **kwargs)
		if df is not None and not df.empty:
			output_path = os.path.join(
				self.config['output_dir'],
				f"{self.config['dataset_name']}_{approach.approach_name}.tsv"
			)
			self.generated_files.append(output_path)
			self.generated_dfs.append(df)

	def _merge_outputs(self):
		"""Merge all generated outputs into a single TSV with preferred column order."""
		if not self.generated_dfs:
			log_message("No generated datasets to merge")
			return

		log_message("Merging generated datasets...")
		merged = pd.concat(self.generated_dfs, ignore_index=True, sort=False)

		preferred_order = [
			'index',
			'dataset',
			'task',
			'approach',
			'sample_type',
			'source_id',
			'question',
			'A', 'B', 'C', 'D',
			'answer',
			'fake_option',

			'true_bbox',
			'true_class',
			'true_class_name',

			'fake_bboxes',
			'fake_bboxes_classes',
			'fake_class',
			'fake_class_name',

			'caption',
			'syco_trigger',
			'answer_color',

			'image',
		]

		all_columns = list(dict.fromkeys(list(merged.columns)))
		ordered_columns = [c for c in preferred_order if c in all_columns]
		ordered_columns += [c for c in all_columns if c not in ordered_columns]
		merged = merged.reindex(columns=ordered_columns)
		merged['index'] = merged.index

		merged_path = os.path.join(
			self.config['output_dir'],
			f"{self.config['output_name']}.tsv"
		)
		merged.to_csv(merged_path, index=False, sep='\t')
		log_message(f"Merged dataset saved to {merged_path}")
	
	def get_generated_dfs(self) -> List[pd.DataFrame]:
		"""Get list of generated dataset DataFrames."""
		return self.generated_dfs


@hydra.main(config_path="../configs", config_name="default_general", version_base="1.3")
def main(config: DictConfig):
	"""
	Main entry point for dataset generation.
	
	Supports both single config and batch processing:
	- Single config: runs the generator with the provided config
	- Batch config: if 'batch_configs' field exists, iterates through the list
	  of config paths/names and runs the generator for each
	"""
	# Check if this is a batch config
	if 'batch_configs' in config and config.batch_configs:
		log_message("=" * 80)
		log_message(f"BATCH MODE: Processing {len(config.batch_configs)} configs")
		log_message("=" * 80)
		
		# Get the absolute path to the configs directory
		config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))
		generated_dfs = []

		for idx, config_name in enumerate(config.batch_configs, 1):
			print("\n")
			log_message("=" * 80)
			log_message(f"Processing config {idx}/{len(config.batch_configs)}: {config_name}")
			log_message("=" * 80)
			
			try:
				# Clear GlobalHydra before re-initializing
				GlobalHydra.instance().clear()
				
				# Initialize Hydra with the config directory
				with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
					# Compose the config (this respects defaults and overrides)
					sub_config = compose(config_name=config_name)
					
					# Run the generator with this config
					generator = DatasetGenerator(sub_config)
					generator.run()
					generated_dfs.extend(generator.get_generated_dfs())
					log_message("=" * 80)
					log_message(f"Added {len(generator.get_generated_dfs())} dfs from {config_name}")
					log_message("=" * 80)
					
				log_message(f"✓ Successfully completed config: {config_name}")
				
			except Exception as e:
				log_message(f"✗ Error processing config '{config_name}': {str(e)}")
				# Continue with next config instead of crashing
				import traceback
				log_message(traceback.format_exc())

		# Set output_dir to the last sub_config's output_dir
		if 'output_dir' in sub_config:
			config['output_dir'] = os.sep.join(sub_config.output_dir.split(os.sep)[:-1])
		
		merger = DatasetGenerator(config, dfs=generated_dfs)
		log_message(f"Merging outputs ({len(generated_dfs)}) from all configs into a single dataset...")
		merger._merge_outputs()

		print("\n")
		log_message("=" * 80)
		log_message("BATCH PROCESSING COMPLETE")
		log_message("=" * 80)
	else:
		# Single config mode
		generator = DatasetGenerator(config)
		generator.run()


if __name__ == "__main__":
	main()
