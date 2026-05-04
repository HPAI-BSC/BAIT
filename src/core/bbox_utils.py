"""Bounding box utilities for adversarial dataset generation."""

import time
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from scipy.ndimage import uniform_filter
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

COLOR_MAP = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 128, 0),
    'yellow': (0, 255, 255)
}

@dataclass
class BBox:
    """Bounding box representation."""
    cls_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        """Convert to tuple format."""
        return (self.cls_id, self.x1, self.y1, self.x2, self.y2)
    
    @classmethod
    def from_tuple(cls, bbox_tuple: Tuple[int, int, int, int, int]) -> 'BBox':
        """Create from tuple format."""
        return cls(*bbox_tuple)
    
    def area(self) -> int:
        """Calculate bounding box area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def overlaps_with(self, other: 'BBox') -> bool:
        """Check if this bbox overlaps with another."""
        return (self.x1 < other.x2 and self.x2 > other.x1 and
                self.y1 < other.y2 and self.y2 > other.y1)


def normalize_bbox(bbox: Tuple[int, int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int, int]:
    """
    Normalize bounding box coordinates to 0-1000 scale.
    
    Args:
        bbox: (cls_id, x1, y1, x2, y2) in pixel coordinates
        w: Image width
        h: Image height
        
    Returns:
        Normalized bbox tuple
    """
    cls_id, x1, y1, x2, y2 = bbox
    x1 = int(x1 * 1000 / w)
    y1 = int(y1 * 1000 / h)
    x2 = int(x2 * 1000 / w)
    y2 = int(y2 * 1000 / h)
    return (cls_id, x1, y1, x2, y2)


def format_bounding_box(bbox: Tuple[int, int, int, int, int]) -> str:
    """
    Convert bounding box to string representation.
    
    Args:
        bbox: (cls_id, x1, y1, x2, y2)
        human: If True, use human-readable format
        
    Returns:
        String representation of bbox
    """
    bbox = [int(x) for x in bbox]
    cls_id, x1, y1, x2, y2 = bbox
    return ((x1,y1), (x2,y2))


def check_overlap(bbox_list1: List[Tuple], bbox_list2: List[Tuple]) -> bool:
    """
    Check if any bboxes from two lists overlap.
    
    Args:
        bbox_list1: First list of bboxes
        bbox_list2: Second list of bboxes
        
    Returns:
        True if any overlap detected
    """
    for bbox1 in bbox_list1:
        for bbox2 in bbox_list2:
            if (bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1] and
                bbox1[2] < bbox2[4] and bbox1[4] > bbox2[2]):
                return True
    return False


def load_annotations(annot_path: str, w: int, h: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Load bounding box annotations from YOLO format file.
    
    Args:
        annot_path: Path to annotation file
        w: Image width
        h: Image height
        normalize: If True, normalize coordinates to 0-1000
        
    Returns:
        List of bounding boxes as (cls_id, x1, y1, x2, y2) tuples
    """
    with open(annot_path, 'r') as f:
        annotations = f.readlines()
    
    bboxes = []
    if annotations:
        for annotation in annotations:
            parts = annotation.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                center_x = float(parts[1]) * w
                center_y = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)
                
                bboxes.append((cls_id, x1, y1, x2, y2))
    
    # Calculate coverage of bounding boxes
    if bboxes:
        mask = np.zeros((h, w), dtype=bool)
        largest_box_area = 0
        
        for _, x1, y1, x2, y2 in bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
                area = (x2 - x1) * (y2 - y1)
                largest_box_area = max(largest_box_area, area)
        
        coverage = np.sum(mask) / (w * h)
        largest_box_coverage = largest_box_area / (w * h)
        
        # If coverage is too high, return invalid bbox
        if coverage > 0.75 or largest_box_coverage > 0.3:
            bbox = (-1, np.random.randint(0, w), np.random.randint(0, h),
                    np.random.randint(0, w), np.random.randint(0, h))
            bboxes = [bbox]
    else:
        # No annotations found
        bbox = (-1, np.random.randint(0, w), np.random.randint(0, h),
                np.random.randint(0, w), np.random.randint(0, h))
        bboxes = [bbox]
    
    norm_bboxes = [normalize_bbox(bbox, w, h) for bbox in bboxes]
    
    return bboxes, norm_bboxes


def check_non_overlapping_boxes(bboxes: List[Tuple]) -> int:
    """
    Count non-overlapping bounding boxes (one per class).
    
    Args:
        bboxes: List of bounding boxes
        
    Returns:
        Count of non-overlapping boxes
    """
    non_overlapping_count = 0
    counted_classes = set()
    
    for i in range(len(bboxes)):
        if bboxes[i][0] in counted_classes:
            continue
            
        overlap_found = False
        for j in range(len(bboxes)):
            if bboxes[i][0] == bboxes[j][0] or i == j:
                continue
            
            if (bboxes[i][1] < bboxes[j][3] and bboxes[i][3] > bboxes[j][1] and
                bboxes[i][2] < bboxes[j][4] and bboxes[i][4] > bboxes[j][2]):
                overlap_found = True
                break
        
        if not overlap_found:
            non_overlapping_count += 1
            counted_classes.add(bboxes[i][0])
    
    return non_overlapping_count


def generate_fake_bboxes(
    image: np.ndarray,
    real_bboxes: List[Tuple],
    n: int = 3,
    size_variation: float = 0.35,
    mean_diff_threshold: float = 0.25,
    std_diff_threshold: float = 0.5,
    exclude_black: Optional[float] = None,
    max_attempts: int = 10000,
) -> Optional[List[Tuple]]:
    """
    Generate fake bounding boxes that are visually similar to real ones.
    
    Args:
        image: Input image as numpy array
        real_bboxes: List of real bounding boxes
        n: Number of fake boxes to generate
        size_variation: Allowed size variation (0-1)
        mean_diff_threshold: Maximum allowed mean difference ratio
        std_diff_threshold: Maximum allowed std difference ratio
        exclude_black: Exclude boxes with black pixel percentage above this threshold
        max_attempts: Maximum generation attempts
        
    Returns:
        List of fake bounding boxes, or None if generation failed
    """
    h, w = image.shape[:2]
    
    # Adjust n based on non-overlapping real boxes
    n = n - check_non_overlapping_boxes(real_bboxes) + 1
    
    # Calculate statistics from real bounding boxes
    real_bbox_means = []
    real_bbox_stds = []
    
    for _, rx1, ry1, rx2, ry2 in real_bboxes:
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        if rx2 > rx1 and ry2 > ry1:
            roi = image[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                real_bbox_means.append(np.mean(roi))
                real_bbox_stds.append(np.std(roi))
    
    target_mean = np.mean(real_bbox_means)
    target_std = np.mean(real_bbox_stds)
    
    # Calculate average bbox dimensions
    real_widths = [rx2 - rx1 for _, rx1, ry1, rx2, ry2 in real_bboxes]
    real_heights = [ry2 - ry1 for _, rx1, ry1, rx2, ry2 in real_bboxes]
    avg_width = int(np.mean(real_widths))
    avg_height = int(np.mean(real_heights))
    
    # Create a similarity heatmap
    kernel_size = (avg_height // 2, avg_width // 2)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        img_gray = np.mean(image, axis=2).astype(float)
    else:
        img_gray = image.astype(float)
    
    # Compute local statistics
    mean_img = uniform_filter(img_gray, size=kernel_size)
    mean_sq_img = uniform_filter(img_gray ** 2, size=kernel_size)
    local_std = np.sqrt(np.maximum(0, mean_sq_img - mean_img ** 2))
    
    # Create similarity map
    mean_similarity = 1 - np.abs(mean_img - target_mean) / (target_mean + 1e-10)
    std_similarity = 1 - np.abs(local_std - target_std) / (target_std + 1e-10)
    similarity_map = mean_similarity * std_similarity
    
    # Normalize to probability distribution
    similarity_map = np.clip(similarity_map, 0, None)
    if similarity_map.sum() > 0:
        similarity_map = similarity_map / similarity_map.sum()
    
    # Flatten for sampling
    flat_map = similarity_map.flatten()
    coords = np.arange(flat_map.size)
    
    fake_bboxes = []
    attempts = 0
    
    # Dynamic threshold adjustment
    current_mean_threshold = mean_diff_threshold
    current_std_threshold = std_diff_threshold
    
    while len(fake_bboxes) < n:
        attempts += 1
        
        if attempts > max_attempts:
            return None
        
        # Relax thresholds progressively
        if attempts % 1000 == 0:
            fake_bboxes = []  # Reset
            
        if attempts % 500 == 0 and attempts > 0:
            current_mean_threshold = min(1.01, current_mean_threshold * 1.1)
            current_std_threshold = min(1.01, current_std_threshold * 1.1)
        
        # Sample from probability distribution
        if flat_map.sum() > 0:
            idx = np.random.choice(coords, p=flat_map)
            y, x = np.unravel_index(idx, similarity_map.shape)
        else:
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
        
        # Calculate bbox with random variation
        rand_width = int(np.random.uniform(1 - size_variation, 1 + size_variation) * avg_width)
        rand_height = int(np.random.uniform(1 - size_variation, 1 + size_variation) * avg_height)
        
        # Center bbox around sampled point
        rand_x1 = max(0, x - rand_width // 2)
        rand_y1 = max(0, y - rand_height // 2)
        rand_x2 = min(w, rand_x1 + rand_width)
        rand_y2 = min(h, rand_y1 + rand_height)
        
        # Check pixel similarity
        roi = img_gray[rand_y1:rand_y2, rand_x1:rand_x2]
        if roi.size > 0:
            fake_mean = np.mean(roi)
            fake_std = np.std(roi)
            
            mean_diff_ratio = abs(fake_mean - target_mean) / (target_mean + 1e-10)
            std_diff_ratio = abs(fake_std - target_std) / (target_std + 1e-10)
            
            if mean_diff_ratio >= current_mean_threshold or std_diff_ratio >= current_std_threshold:
                continue
            
            if exclude_black:
                black_pixel_percentage = np.sum(roi == 0) / roi.size
                if black_pixel_percentage > exclude_black:
                    continue
            
            # Check for overlap
            bbox1 = [(99, rand_x1, rand_y1, rand_x2, rand_y2)]
            all_bboxes = real_bboxes + fake_bboxes
            
            if not check_overlap(bbox1, all_bboxes):
                fake_bboxes.extend(bbox1)
                
                # Update probability map to reduce chances near chosen bbox
                y_indices, x_indices = np.mgrid[
                    max(0, rand_y1 - rand_height):min(h, rand_y2 + rand_height),
                    max(0, rand_x1 - rand_width):min(w, rand_x2 + rand_width)
                ]
                flat_indices = np.ravel_multi_index(
                    (y_indices.flatten(), x_indices.flatten()),
                    similarity_map.shape
                )
                flat_map[flat_indices] *= 0.1
                if flat_map.sum() > 0:
                    flat_map = flat_map / flat_map.sum()
    
    return fake_bboxes

def paint_bboxes_with_labels(
    image: np.ndarray,
    bboxes: List[Tuple],
    output_path: str,
    paint_image: bool = True
) -> Dict[Tuple, str]:
    """
    Paint bounding boxes with numbered labels.
    
    Args:
        image: Input image (BGR format from cv2)
        bboxes: List of bounding boxes
        output_path: Path to save painted image
        paint_image: Whether to paint the image or not
        
    Returns:
        Dictionary mapping normalized bboxes to their label strings
    """
    if paint_image:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        
        if len(image_rgb.shape) == 3:
            plt.imshow(image_rgb)
        else:
            plt.imshow(image_rgb, cmap='gray')
    
    mapping = {}
    random.shuffle(bboxes)
    
    for i, bbox in enumerate(bboxes):
        _, x1, y1, x2, y2 = bbox
        
        if paint_image:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='red', facecolor='none',
                linewidth=3, linestyle='-'
            )
            plt.gca().add_patch(rect)
            
            plt.text(
                x1, y1 - 10, f"Box {i+1}",
                color='white', fontsize=11, fontweight='bold',
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal()
                ]
            )
        
        norm_bbox = normalize_bbox(bbox, image.shape[1], image.shape[0])
        mapping[norm_bbox] = f"Box {i+1}"
    
    if paint_image:
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
    
    return mapping

def paint_bboxes_with_legends_plt(
    image: np.ndarray,
    bboxes: List[Tuple],
    correct_index: int,
    output_path: str,
    classes: List[str],
    true_class: int
) -> Dict[Tuple, str]:
    """
    Paint bounding boxes with class-based legends.
    
    Args:
        image: Input image (BGR format from cv2)
        bboxes: List of bounding boxes
        correct_index: Index of the correct bbox
        output_path: Path to save painted image
        classes: List of class names
        true_class: Index of the true class
        
    Returns:
        Dictionary mapping bboxes to their assigned colors
    """
    
    colors = ['red', 'blue', 'green', 'yellow']
    random.shuffle(colors)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_rgb, cmap=None if image_rgb.ndim == 3 else 'gray')
    
    legend_labels = {}
    class_to_color = {}
    bbox_to_color = {}
    color_index = 0
    
    wrong_classes = [c for c in range(len(classes)) if c != true_class]
    
    # Assign classes and colors
    for i, bbox in enumerate(bboxes):
        cls_id, x1, y1, x2, y2 = bbox
        
        # Assign true class to impostor bbox at correct_index
        if i == correct_index and cls_id != true_class:
            cls_id = true_class
        # Assign wrong class to correct bbox if not at correct_index
        elif i != correct_index and cls_id == true_class:
            cls_id = bboxes[correct_index][0]
        # For generated bboxes, assign random wrong class
        if cls_id == 99:
            cls_id = random.choice(wrong_classes)
        
        if cls_id not in class_to_color:
            class_to_color[cls_id] = colors[color_index]
            color_index += 1
        
        color = class_to_color[cls_id]
        bbox_to_color[(cls_id, x1, y1, x2, y2)] = color
        
        if i == correct_index:
            label = f'Contains {classes[true_class].lower()}'
        else:
            label = f'Contains {classes[cls_id].lower()}'

        rect_kwargs = dict(
            edgecolor=color,
            facecolor='none',
            linewidth=3
        )

        if label not in legend_labels:
            rect_kwargs['label'] = label
            legend_labels[label] = color
        
        ax.add_patch(
            mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, **rect_kwargs)
        )
        
        ax.text(
            x1, y1 - 10, f"Box {i + 1}",
            color='white', fontsize=12, fontweight='bold',
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()
            ]
        )
    
    ax.axis('off')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)    
    plt.close()
    
    return bbox_to_color