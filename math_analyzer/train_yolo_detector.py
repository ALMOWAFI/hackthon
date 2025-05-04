"""
Train a YOLOv8 model to detect math expressions in images.

This script trains a YOLO model based on the dataset configuration in 
math_analyzer/data/math_expressions/dataset.yaml.

Usage:
    python math_analyzer/train_yolo_detector.py --epochs 50 --img 640 --batch 16
"""
import argparse
from pathlib import Path
import shutil
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with 'pip install ultralytics'")
    sys.exit(1)

def train_yolo_detector(epochs=50, img_size=640, batch_size=16, pretrained=True):
    """Train a YOLOv8 model for math expression detection."""
    print(f"Training YOLOv8 for {epochs} epochs with image size {img_size}px...")
    
    # Data path
    data_yaml = Path(__file__).parent / "data" / "math_expressions" / "dataset.yaml"
    if not data_yaml.exists():
        print(f"Error: Dataset config not found at {data_yaml}")
        return
    
    # Start with a pre-trained model
    model_size = "n"  # nano - smallest, fastest model (can be 'n', 's', 'm', 'l', 'x')
    model = YOLO(f"yolov8{model_size}.pt" if pretrained else None)
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=15,  # Early stopping patience
        device="0" if model.device.type != "cpu" else "cpu",  # GPU if available
        project=str(Path(__file__).parent / "runs"),
        name="train",
        save=True,
        exist_ok=True,
        verbose=True
    )
    
    # Save model to our models directory
    model_path = Path(__file__).parent / "models" / "best.pt"
    try:
        best_model = Path(__file__).parent / "runs" / "train" / "weights" / "best.pt"
        if best_model.exists():
            shutil.copy(best_model, model_path)
            print(f"Best model saved to {model_path}")
        else:
            print("Warning: Best model not found after training")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for math expression detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights")
    args = parser.parse_args()
    
    train_yolo_detector(
        epochs=args.epochs,
        img_size=args.img,
        batch_size=args.batch,
        pretrained=not args.no_pretrained
    )
