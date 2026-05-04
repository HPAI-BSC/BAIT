# BAIT (Blind-trust Adversarial Injection Toolkit)

Generate adversarial multimodal benchmark datasets from image classification and detection annotations.

This project loads a source dataset made of images, class labels, and bounding-box annotations, then produces multiple question-answer TSV datasets using a set of generation approaches:

- `baseline_classification`
- `baseline_detection`
- `sycophancy`
- `captions`
- `captions_text`
- `legends`

The generator also creates derived images such as painted bounding-box views and caption/legend variants, then merges the resulting samples into a single output TSV.

## What The Pipeline Expects

The generator expects each dataset to live under the path defined in the config with a structure similar to:

```text
data/sources/Fruits/
  classification.csv
  images/
    <image_id>.jpg
  annotations/
    <image_id>.txt
```

The `classification.csv` file is read with `image_id` as the index. The annotation files are expected to contain bounding-box information in the format used by the utilities in `src/core`.

## Outputs

Generated data is written under `data/output/<dataset_name>/` by default.

Typical artifacts include:

- One TSV per approach, for example `COCO_baseline_classification.tsv`
- A merged TSV named after `output_name`, for example `COCO.tsv`
- Generated images under `images/`, including:
  - `base/`
  - `painted/`
  - `captions/classification/`
  - `captions/detection/`
  - `legends/`

The TSV rows include the question, answer options, correct answer, approach name, source image id, and either image paths or base64-encoded images depending on configuration.

## Installation

The project is a Python application. Create a virtual environment and install the runtime dependencies:

```bash
pip install hydra-core omegaconf pandas numpy opencv-python pillow
```

If you already have a project environment, activate it first and then install the packages above.

## Configuration

Configuration is driven by Hydra files in `configs/`.

- `configs/default.yaml` defines shared defaults such as input/output paths, template strings, and enabled approaches.
- `configs/main_config.yaml` provides the Fruits example dataset configuration.
- `configs/example_dataset.yaml` is another example dataset definition.

The same files are mirrored in `examples/` for convenience.

Key fields you will usually customize:

- `dataset_name`
- `output_name`
- `data_dir`
- `dataset_path`
- `classif_classes`
- `detect_classes`
- `to_create`
- `fake_bbox_arguments`

## Usage

Run the generator through Hydra and point it at the config you want to execute:

```bash
python -m src.generator --config-name main_config
```

For the example dataset shipped with the repository, the command above uses `configs/main_config.yaml`.

You can also run other configs in the same way, for example:

```bash
python -m src.generator --config-name example_dataset
```

## Batch Generation

Hydra batch mode is supported through the `batch_configs` field. When present, the generator iterates through each listed config and merges the generated outputs into a single TSV.

Example:

```yaml
batch_configs:
  - example_dataset
```

## Repository Layout

```text
configs/   Hydra configuration files
scripts/   Example launch scripts
src/      Generator and approach implementations
```

## Implementation Notes

The main orchestration lives in `src/generator.py`. Approach-specific behavior is implemented in `src/approaches/`, and reusable image / annotation helpers live in `src/core/`.

Random seeds can be fixed to make generation reproducible across runs.