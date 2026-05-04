"""Core utilities package."""

from .image_utils import (
    smart_resize,
    resize_and_save,
    encode_image_to_base64,
    encode_image_file_to_base64,
    add_caption_to_image,
)

from .bbox_utils import (
    BBox,
    normalize_bbox,
    format_bounding_box,
    check_overlap,
    load_annotations,
    check_non_overlapping_boxes,
    generate_fake_bboxes,
    paint_bboxes_with_labels,
    paint_bboxes_with_legends_plt,
)

from .common_utils import (
    log_message,
    create_options,
    get_file_paths,
    validate_paths,
    balanced_sample_ids,
)

__all__ = [
    # Image utils
    'smart_resize',
    'resize_and_save',
    'encode_image_to_base64',
    'encode_image_file_to_base64',
    'add_caption_to_image',
    'calculate_histogram',
    'histogram_similarity',
    'regional_similarity',
    'compute_similarity',
    
    # BBox utils
    'BBox',
    'normalize_bbox',
    'format_bounding_box',
    'check_overlap',
    'load_annotations',
    'check_non_overlapping_boxes',
    'generate_fake_bboxes',
    'paint_bboxes_with_labels',
    'paint_bboxes_with_legends_plt',
    
    # Common utils
    'log_message',
    'create_options',
    'get_file_paths',
    'validate_paths',
    'balanced_sample_ids',
]
