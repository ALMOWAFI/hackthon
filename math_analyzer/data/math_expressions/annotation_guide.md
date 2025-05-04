# Math Expression Detection Dataset Guide

## Dataset Format

YOLOv8 requires a specific folder structure and annotation format:

```
math_expressions/
├── images/
│   ├── train/     # Training images (.jpg, .png, etc.)
│   └── val/       # Validation images (.jpg, .png, etc.)
├── labels/
│   ├── train/     # Training annotations (.txt)
│   └── val/       # Validation annotations (.txt)
└── dataset.yaml   # Dataset configuration
```

## Annotation Format

Each image requires a corresponding text file with the same name in the labels directory.
For example, if you have `images/train/math1.jpg`, you need `labels/train/math1.txt`.

### Annotation Structure

Each line in the .txt file represents ONE bounding box in this format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: Always 0 for math_expression
- `x_center`, `y_center`: Center coordinates of the box (normalized 0-1)
- `width`, `height`: Width and height of the box (normalized 0-1)

Example: `0 0.5 0.5 0.3 0.2` means:
- Class 0 (math_expression)
- Center at 50% across, 50% down the image
- Box is 30% of image width, 20% of image height

## Annotation Tools

We recommend using:
1. [LabelImg](https://github.com/tzutalin/labelImg)
2. [CVAT](https://www.cvat.ai/)
3. [Roboflow](https://roboflow.com/)

## Annotation Guidelines

1. **What to label:**
   - Any mathematical expression (equations, formulas, numbers with operations)
   - Include complete expressions (don't split an equation)
   - Include all symbols that belong to the expression

2. **Tight bounding boxes:**
   - Draw boxes as tight as possible around the expressions
   - Include all parts of the expression (superscripts, fractions, etc.)
   - For expressions written across multiple lines, use separate boxes for each line

3. **Common errors to avoid:**
   - Boxes too large/small
   - Missing parts of expressions
   - Including non-mathematical elements
   - Overlapping annotations

## Example Dataset Creation

We've included a script `create_sample_annotations.py` to help you create some initial annotations:

1. Copy homework images to `images/train/`
2. Run `python math_analyzer/create_sample_annotations.py`
3. Review and edit the generated annotations
4. Manually create validation set annotations

## Training

Once your dataset is prepared:

```
pip install ultralytics
python math_analyzer/train_yolo_detector.py --epochs 50 --img 640 --batch 16
```

This will train the model and save the best weights to `math_analyzer/models/best.pt`.
