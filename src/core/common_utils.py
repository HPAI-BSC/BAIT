"""Common utilities for the framework."""

import random
import string
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np


def log_message(message: str, file: str = None) -> None:
    """
    Log a message with timestamp.
    
    Args:
        message: Message to log
        file: Optional file path to append log to
    """
    timestamp = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
    formatted_message = f"[{timestamp}] {message}"
    
    if file is not None:
        with open(file, 'a') as f:
            f.write(formatted_message + "\n")
    else:
        print(formatted_message)


def create_options(class_names: List[str], true_class: int, n_options: int = None) -> Tuple[Dict[str, str], str]:
    """
    Create shuffled multiple choice options.
    
    Args:
        class_names: List of all class names
        true_class: Index of the correct class
        n_options: Number of options to create. If None, uses all class names.
                   If less than len(class_names), randomly selects n_options classes.
        
    Returns:
        Tuple of (options_dict, correct_answer_letter)
        where options_dict maps letters (A, B, C...) to class names
    """
    if n_options is None:
        n_options = len(class_names)
    
    correct_class_name = class_names[true_class] if isinstance(true_class, (int, np.integer)) else true_class
    
    # If we need all classes, use all; otherwise select n_options randomly including the true class
    if n_options >= len(class_names):
        selected_classes = class_names.copy()
    else:
        # Ensure the correct class is always included
        other_classes = [c for c in class_names if c != correct_class_name]
        selected_classes = [correct_class_name] + random.sample(other_classes, n_options - 1)
    
    shuffled_class_names = selected_classes.copy()
    random.shuffle(shuffled_class_names)
    options = {letter: name for letter, name in zip(string.ascii_uppercase, shuffled_class_names)}
    
    # Find the correct answer letter
    answer_letter = next(letter for letter, name in options.items() if name == correct_class_name)
    
    return options, answer_letter


def get_file_paths(
    data: pd.DataFrame,
    data_dir: str,
    extension: str = ".jpg"
) -> Dict[str, str]:
    """
    Generate file paths for all images in the dataset.
    
    Args:
        data: DataFrame with image_id index
        data_dir: Base data directory
        extension: File extension for images
        
    Returns:
        Dictionary mapping image_id to file path
    """
    return {
        img_id: str(Path(data_dir) / "images" / f"{img_id}{extension}")
        for img_id in data.index
    }


def validate_paths(paths: Dict[str, str]) -> List[str]:
    """
    Validate that all file paths exist.
    
    Args:
        paths: Dictionary of paths to validate
        
    Returns:
        List of missing paths (empty if all exist)
    """
    missing = []
    for path in paths.values():
        if not Path(path).exists():
            missing.append(path)
    return missing


def balanced_sample_ids(
    df: pd.DataFrame,
    target_ids: int = 250,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample IDs with balanced class distribution.
    
    Args:
        df: DataFrame with image_id and true_class columns
        target_ids: Target number of samples
        random_state: Random seed
        
    Returns:
        Balanced subset of the DataFrame
    """
    # Create a combined key
    df['id_class'] = df['image_id'].astype(str) + "-" + df['true_class'].astype(str)
    
    # Map IDs to their class
    id_class_map = df.set_index('id_class')['true_class']
    selected_ids = []
    remaining_ids = id_class_map.copy()
    
    # Try different subsets in order of preference
    for subset in ["test", "val", "train", ""]:
        remaining_ids_subset = remaining_ids[remaining_ids.index.str.contains(subset)]
        
        while len(selected_ids) < target_ids and not remaining_ids_subset.empty:
            classes_remaining = remaining_ids_subset.value_counts()
            n_classes = len(classes_remaining)
            
            ids_per_class = max(1, (target_ids - len(selected_ids)) // n_classes)
            
            round_ids = set()
            for cls in classes_remaining.index:
                cls_ids = remaining_ids_subset[remaining_ids_subset == cls].index
                n_pick = min(ids_per_class, len(cls_ids))
                if n_pick > 0:
                    sampled = pd.Series(cls_ids).sample(
                        n=n_pick, random_state=random_state
                    ).tolist()
                    round_ids.update(sampled)
            
            selected_ids.extend(round_ids)
            remaining_ids_subset = remaining_ids_subset.drop(round_ids, errors="ignore")
    
    # Trim to exact target if we oversampled
    if len(selected_ids) > target_ids:
        selected_ids = pd.Series(selected_ids).sample(
            n=target_ids, random_state=random_state
        ).tolist()
    
    balanced_df = df[df['id_class'].isin(selected_ids)].copy()
    balanced_df.drop(columns=['id_class'], inplace=True)
    
    return balanced_df
