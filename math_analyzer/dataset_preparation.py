"""
Dataset preparation utilities for training the handwritten math OCR system
This module helps download and preprocess Kaggle datasets
"""

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import subprocess
import requests
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define recommended Kaggle datasets for handwritten math recognition
KAGGLE_DATASETS = {
    "handwritten_math_symbols": {
        "url": "xainano/handwrittenmathsymbols",
        "description": "100,000+ images of handwritten math symbols (75 classes)",
        "suitable_for": "Symbol recognition",
        "size": "~1GB"
    },
    "handwritten_math_expressions": {
        "url": "rtatman/handwritten-mathematical-expressions",
        "description": "Handwritten math expressions with LaTeX annotations",
        "suitable_for": "Expression recognition",
        "size": "~400MB"
    },
    "mnist_digits": {
        "url": "oddrationale/mnist-in-csv",
        "description": "Classic MNIST digit dataset (0-9)",
        "suitable_for": "Digit recognition base training",
        "size": "~120MB"
    },
    "handwritten_operators": {
        "url": "clarencezhao/handwritten-math-symbol-dataset",
        "description": "Handwritten mathematical operators dataset",
        "suitable_for": "Operator recognition",
        "size": "~50MB"
    },
    "crohme": {
        "url": "ranzeet013/handwritten-mathematical-expression-database",
        "description": "Competition on Recognition of Handwritten Mathematical Expressions (CROHME)",
        "suitable_for": "Complex expression recognition",
        "size": "~200MB"
    }
}


def list_recommended_datasets():
    """
    Print information about recommended Kaggle datasets
    """
    print("\n=== Recommended Kaggle Datasets for Handwritten Math OCR ===\n")
    
    for name, info in KAGGLE_DATASETS.items():
        print(f"Dataset: {name}")
        print(f"URL: kaggle.com/datasets/{info['url']}")
        print(f"Description: {info['description']}")
        print(f"Best for: {info['suitable_for']}")
        print(f"Size: {info['size']}")
        print("-" * 60)


def setup_kaggle_api():
    """
    Set up the Kaggle API credentials
    
    Returns:
        bool: True if setup was successful
    """
    # Check if Kaggle is installed
    try:
        import kaggle
        print("Kaggle API already installed.")
    except ImportError:
        print("Installing Kaggle API...")
        try:
            subprocess.check_call(["pip", "install", "kaggle"])
            print("Kaggle API installed successfully.")
        except Exception as e:
            print(f"Failed to install Kaggle API: {str(e)}")
            return False
    
    # Check for Kaggle API credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("Kaggle API credentials found.")
        return True
    
    # Guide user to set up credentials
    print("\n=== Kaggle API Credentials Setup ===")
    print("To download datasets directly from Kaggle, you need to set up your API credentials:")
    print("1. Log in to your Kaggle account")
    print("2. Go to 'Account' tab")
    print("3. Scroll down to 'API' section and click 'Create New API Token'")
    print("4. This will download a kaggle.json file")
    print("5. Place the kaggle.json file in the ~/.kaggle/ directory")
    print(f"   (On Windows, this would be: {os.path.expanduser('~/.kaggle/')})")
    
    # Ask for manual input if credentials file doesn't exist
    key = input("\nEnter your Kaggle username: ")
    token = input("Enter your Kaggle API key: ")
    
    # Create the directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Save the credentials
    with open(kaggle_json, 'w') as f:
        f.write(f'{{"username":"{key}","key":"{token}"}}')
    
    # Set correct permissions
    os.chmod(kaggle_json, 0o600)
    
    print(f"Credentials saved to {kaggle_json}")
    return True


def download_kaggle_dataset(dataset_url, target_dir="datasets"):
    """
    Download a dataset from Kaggle
    
    Args:
        dataset_url: Kaggle dataset URL (e.g., 'xainano/handwrittenmathsymbols')
        target_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    # Ensure Kaggle API is set up
    if not setup_kaggle_api():
        print("Kaggle API setup failed. Cannot download dataset.")
        return None
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Download the dataset
        print(f"Downloading Kaggle dataset: {dataset_url}")
        
        # Use Kaggle API
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_url, path=target_dir, unzip=True)
        
        print(f"Dataset downloaded successfully to {target_dir}")
        return target_dir
    
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Alternative: Download manually from kaggle.com/datasets/" + dataset_url)
        return None


def prepare_handwritten_math_symbols(dataset_path, target_size=(45, 45), max_samples_per_class=1000):
    """
    Prepare the Handwritten Math Symbols dataset for training
    
    Args:
        dataset_path: Path to the extracted dataset
        target_size: Target image size for training
        max_samples_per_class: Maximum number of samples per class
        
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, label_mapping
    """
    # Path to the dataset
    data_path = Path(dataset_path) / "handwritten_math_symbols"
    
    if not data_path.exists():
        possible_data_paths = list(Path(dataset_path).glob("**/data"))
        if possible_data_paths:
            data_path = possible_data_paths[0]
        else:
            print(f"Could not find dataset directory in {dataset_path}")
            return None
    
    print(f"Loading dataset from {data_path}")
    
    # Get class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    # Create label mapping
    label_mapping = {d.name: i for i, d in enumerate(sorted(class_dirs))}
    
    # Initialize data arrays
    images = []
    labels = []
    
    # Load images and labels
    print("Loading images...")
    for class_dir in tqdm(class_dirs):
        class_label = label_mapping[class_dir.name]
        
        # Get image files
        image_files = list(class_dir.glob("*.jpg"))
        image_files.extend(class_dir.glob("*.png"))
        
        # Limit samples per class
        if max_samples_per_class > 0:
            image_files = image_files[:max_samples_per_class]
        
        for img_file in image_files:
            # Read image
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                
                # Add to dataset
                images.append(img)
                labels.append(class_label)
    
    # Convert to numpy arrays
    X = np.array(images).reshape(-1, target_size[0], target_size[1], 1)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(label_mapping))
    
    # Split into train, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Dataset prepared: {x_train.shape[0]} training, {x_val.shape[0]} validation, {x_test.shape[0]} test samples")
    
    return x_train, y_train, x_val, y_val, x_test, y_test, label_mapping


def prepare_handwritten_expressions(dataset_path, target_size=(640, 480), max_samples=2000):
    """
    Prepare the Handwritten Mathematical Expressions dataset for training
    
    Args:
        dataset_path: Path to the extracted dataset
        target_size: Target image size for training
        max_samples: Maximum number of samples to use
        
    Returns:
        images, expressions (formatted as a dict with image and LaTeX pairs)
    """
    # This is a placeholder for processing a math expressions dataset
    # The actual implementation would depend on the specific dataset structure
    pass


def visualize_dataset_samples(X, y, label_mapping, num_samples=10):
    """
    Visualize random samples from the dataset
    
    Args:
        X: Image data
        y: Labels (one-hot encoded)
        label_mapping: Mapping from indices to class names
        num_samples: Number of samples to visualize
    """
    # Create a reverse mapping from indices to class names
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Convert one-hot encoded labels back to indices
    if len(y.shape) > 1:
        y_indices = np.argmax(y, axis=1)
    else:
        y_indices = y
    
    # Get random indices
    indices = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
    
    # Plot the samples
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx].reshape(X[idx].shape[0], X[idx].shape[1]), cmap='gray')
        plt.title(reverse_mapping[y_indices[idx]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def prepare_dataset_for_training(dataset_name="handwritten_math_symbols", max_samples=10000):
    """
    Download and prepare a Kaggle dataset for training
    
    Args:
        dataset_name: Name of the dataset (must be one of KAGGLE_DATASETS keys)
        max_samples: Maximum number of samples to use (per class for class-based datasets)
        
    Returns:
        Training data ready for model training
    """
    if dataset_name not in KAGGLE_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        list_recommended_datasets()
        return None
    
    # Dataset info
    dataset_info = KAGGLE_DATASETS[dataset_name]
    dataset_url = dataset_info["url"]
    
    # Create dataset directory
    dataset_dir = Path("datasets") / dataset_name
    
    # Check if dataset already exists
    if not dataset_dir.exists():
        # Download the dataset
        download_path = download_kaggle_dataset(dataset_url, str(dataset_dir))
        
        if download_path is None:
            return None
    else:
        print(f"Dataset directory already exists: {dataset_dir}")
    
    # Prepare the dataset based on its type
    if dataset_name == "handwritten_math_symbols":
        return prepare_handwritten_math_symbols(dataset_dir, max_samples_per_class=max_samples//100)
    elif dataset_name == "handwritten_math_expressions":
        return prepare_handwritten_expressions(dataset_dir, max_samples=max_samples)
    elif dataset_name == "mnist_digits":
        # Specific handling for MNIST dataset
        pass
    else:
        print(f"No specific preparation function for {dataset_name}. Using default preparation.")
        return prepare_handwritten_math_symbols(dataset_dir, max_samples_per_class=max_samples//100)


if __name__ == "__main__":
    # List recommended datasets
    list_recommended_datasets()
    
    # Example: Prepare a dataset for training
    # dataset = input("\nEnter the name of the dataset to download and prepare: ")
    # if dataset in KAGGLE_DATASETS:
    #     data = prepare_dataset_for_training(dataset)
    # else:
    #     print("Invalid dataset name.")
