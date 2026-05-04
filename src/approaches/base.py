"""Base classes for approach implementations."""

import os
import numpy as np
import random
import string
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..core import log_message, create_options, encode_image_file_to_base64, format_bounding_box


@dataclass
class ApproachConfig:
    """Configuration for an approach."""
    dataset_name: str
    output_dir: str
    image_dir: str
    max_pixels: int = 1_000_000
    n_options_classification: Optional[int] = None
    n_options_detection: Optional[int] = None
    question_templates: List[str] = field(default_factory=list)
    classif_classnames: Optional[List[str]] = None
    detect_classnames: Optional[List[str]] = None

    def __post_init__(self):
        """Ensure directories exist and validate option counts."""
        # Validate option counts
        if self.n_options_classification is not None:
            if self.n_options_classification < 2:
                raise ValueError(f"n_options_classification must be >= 2, got {self.n_options_classification}")
        
        if self.n_options_detection is not None:
            if self.n_options_detection < 2:
                raise ValueError(f"n_options_detection must be >= 2, got {self.n_options_detection}")
            if self.n_options_detection > 4:
                raise ValueError(f"n_options_detection must be <= 4 (framework generates max 4 bboxes per image), got {self.n_options_detection}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)


class BaseApproach(ABC):
    """
    Base class for all adversarial dataset generation approaches.
    
    This class provides common functionality for:
    - Parallel processing
    - Sample generation
    - DataFrame creation and saving
    - Image encoding
    """
    
    def __init__(self, config: ApproachConfig):
        """
        Initialize the approach.
        
        Args:
            config: Configuration object for this approach
        """
        self.config = config
        class_name = self.__class__.__name__.replace('Approach', '')
        self.approach_name = ''.join('_' + c if c.isupper() else c for c in class_name).lower().strip('_')
    
    @abstractmethod
    def process_sample(self, sample_data: Tuple, **kwargs) -> List[Dict[str, Any]]:
        """
        Process a single sample and return generated questions.
        
        Args:
            sample_data: Data for a single sample (format depends on approach)
            **kwargs: Additional approach-specific arguments
            
        Returns:
            List of sample dictionaries (one sample can generate multiple questions)
        """
        pass
    
    @abstractmethod
    def get_output_subdir(self) -> str:
        """
        Get the subdirectory name for storing generated images.
        
        Returns:
            Subdirectory name (e.g., 'base', 'painted', 'captions')
        """
        pass
    
    def create_dataset(
        self,
        data: pd.DataFrame,
        use_image_paths: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate the complete adversarial dataset.
        
        Args:
            data: Input DataFrame with sample information
            use_image_paths: If True, store paths; if False, store base64
            **kwargs: Additional approach-specific arguments
            
        Returns:
            DataFrame with generated adversarial samples
        """
        log_message(f"Creating {self.approach_name} dataset")

		# Reset random seeds for reproducibility in each approach
        random.seed(42)
        np.random.seed(42)
        
        # Prepare output directory
        output_subdir = self.get_output_subdir()
        if output_subdir:
            os.makedirs(os.path.join(self.config.image_dir, output_subdir), exist_ok=True)
        
        # Prepare samples for processing
        samples = self.prepare_samples(data, **kwargs)
        
        if not samples:
            log_message(f"\t- No valid samples to process")
            return pd.DataFrame()
        
        log_message(f"\t- Processing {len(samples)} samples...")
        
        process_kwargs = self.get_process_kwargs(use_image_paths, **kwargs)
        dataset = []
        import time

        start = time.time()
        for sample in samples:
            result = self.process_sample(sample, **process_kwargs)
            dataset.extend(result)
        print("-" * 50)
        print(f"{self.approach_name} Processing time: {time.time() - start:.2f} seconds")
        print("-" * 50)
        
        if not dataset:
            log_message(f"\t- No samples generated")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        df['index'] = df.index
        df.insert(0, 'index', df.pop('index'))
        if 'image' in df.columns:
            df['image'] = df.pop('image')
        
        # Save to file
        output_path = os.path.join(
            self.config.output_dir,
            f'{self.config.dataset_name}_{self.approach_name}.tsv'
        )
        log_message(f"\t- Saving {len(df)} questions to {output_path}")
        df.to_csv(output_path, index=False, sep='\t')
        
        return df
    
    def prepare_samples(self, data: pd.DataFrame, **kwargs) -> List[Tuple]:
        """
        Prepare samples for processing.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional arguments
            
        Returns:
            List of sample data tuples
        """
        # Default: just return rows as tuples
        return list(data.iterrows())
    
    def get_process_kwargs(self, use_image_paths: bool, **kwargs) -> Dict[str, Any]:
        """
        Get keyword arguments to pass to process_sample.
        
        Args:
            use_image_paths: Whether to use image paths or base64
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of keyword arguments
        """
        return {'use_image_paths': use_image_paths, **kwargs}
    
    def create_question_sample(
        self,
        img_id: str,
        question: str,
        options: Dict[str, str],
        answer: str,
        image: str,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized question sample dictionary.
        
        Args:
            img_id: Image identifier
            question: Question text
            options: Dictionary mapping letters to option texts
            answer: Correct answer letter
            image: Image data (path or base64)
            additional_fields: Optional additional fields to include
            
        Returns:
            Sample dictionary
        """
        sample = {
            'dataset': self.config.dataset_name,
            'approach': self.approach_name,
            'source_id': img_id,
            'question': question,
            **options,
            'answer': answer,
        }
        
        if additional_fields:
            sample.update(additional_fields)
        
        sample['image'] = image
        
        return sample


class ClassificationApproach(BaseApproach):
    """Base class for classification-based approaches."""
    
    def create_classification_question(
        self,
        img_id: str,
        image_path: str,
        true_class: int,
        use_image_paths: bool = False,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standard classification question.
        
        Args:
            img_id: Image identifier
            image_path: Path to image file
            true_class: True class index
            use_image_paths: Whether to use paths or base64
            additional_fields: Optional additional fields
            
        Returns:
            Question sample dictionary
        """
        class_names = list(self.config.classif_classnames)
        question = random.choice(self.config.question_templates.classification)
        n_options = getattr(self.config, 'n_options_classification', None)
        options, answer_letter = create_options(class_names, true_class, n_options=n_options)
        image = image_path if use_image_paths else encode_image_file_to_base64(image_path)
        
        base_fields = {
            'task': 'classification',
            'true_class': true_class,
            'true_class_name': class_names[true_class],
        }
        
        if additional_fields:
            base_fields.update(additional_fields)
        
        return self.create_question_sample(
            img_id, question, options, answer_letter, image, base_fields
        )


class BBoxApproach(BaseApproach):
    """Base class for bounding box-based approaches."""
    
    def create_bbox_question(
        self,
        img_id: str,
        image_path: str,
        bbox: Tuple,
        box_options: List[str],
        correct_option: str,
        bbox_class: int,
        use_image_paths: bool = False,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standard bounding box question.
        
        Args:
            img_id: Image identifier
            image_path: Path to painted image file
            bbox: True bounding box tuple
            box_options: List of box option strings (e.g., ["Box 1", "Box 2", ...])
            correct_option: Correct box string
            bbox_class: Class index of the bbox
            use_image_paths: Whether to use paths or base64
            additional_fields: Optional additional fields
            
        Returns:
            Question sample dictionary
        """
        detect_classes = list(self.config.detect_classnames)
        true_class_name = detect_classes[bbox_class]
        
        # Create question with class name
        question_template = random.choice(self.config.question_templates.detection)
        question = question_template.replace('$class_name', true_class_name.lower())
        
        # Create options - limit to n_options_detection if specified
        n_options = getattr(self.config, 'n_options_detection', len(box_options))
        n_options = min(n_options, len(box_options))
        selected_options = [correct_option] + random.sample(
            [opt for opt in box_options if opt != correct_option], 
            n_options - 1
        )
        random.shuffle(selected_options)
        correct_index = selected_options.index(correct_option)
        options = {letter: opt for letter, opt in zip(string.ascii_uppercase, selected_options)}
        answer_letter = string.ascii_uppercase[correct_index]
        
        # Encode image
        image = image_path if use_image_paths else encode_image_file_to_base64(image_path)
        
        base_fields = {
            'task': 'detection',
            'true_bbox': format_bounding_box(bbox),
            'true_class': bbox_class,
            'true_class_name': true_class_name,
        }
        
        if additional_fields:
            base_fields.update(additional_fields)
        
        return self.create_question_sample(
            img_id, question, options, answer_letter, image, base_fields
        )
