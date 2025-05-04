"""
Train the handwritten math OCR model using Kaggle datasets
This script demonstrates the full pipeline from dataset download to model training
"""

import os
import argparse
import matplotlib.pyplot as plt
from math_analyzer.dataset_preparation import (
    list_recommended_datasets,
    setup_kaggle_api,
    prepare_dataset_for_training
)
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train handwritten math OCR model")
    
    parser.add_argument("--dataset", type=str, default="handwritten_math_symbols",
                        choices=["handwritten_math_symbols", "mnist_digits", 
                                 "handwritten_operators", "crohme"],
                        help="Which dataset to use for training")
    
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use from the dataset")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save the trained model")
    
    parser.add_argument("--list_datasets", action="store_true",
                        help="List recommended datasets and exit")
    
    return parser.parse_args()

def main():
    """Main function to train the OCR model"""
    args = parse_args()
    
    # Just list datasets if requested
    if args.list_datasets:
        list_recommended_datasets()
        return
    
    print(f"\n=== Training handwritten math OCR model with {args.dataset} ===\n")
    
    # Step 1: Set up Kaggle API (if needed)
    if not setup_kaggle_api():
        print("Failed to set up Kaggle API. Exiting.")
        return
    
    # Step 2: Download and prepare dataset
    print(f"\nPreparing dataset: {args.dataset}")
    print(f"This may take a while for large datasets...\n")
    
    data = prepare_dataset_for_training(args.dataset, max_samples=args.max_samples)
    
    if data is None:
        print(f"Failed to prepare dataset: {args.dataset}")
        return
    
    x_train, y_train, x_val, y_val, x_test, y_test, label_mapping = data
    
    print(f"\nDataset prepared successfully:")
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Validation samples: {x_val.shape[0]}")
    print(f"  Test samples: {x_test.shape[0]}")
    print(f"  Number of classes: {len(label_mapping)}")
    
    # Step 3: Initialize OCR model
    print("\nInitializing handwritten math OCR model...")
    math_ocr = HandwrittenMathOCR()
    
    # Step 4: Visualize some samples from the dataset
    print("\nVisualizing sample images from the dataset...")
    plt.figure(figsize=(15, 8))
    
    # Get class names for visualization
    class_names = {v: k for k, v in label_mapping.items()}
    
    # Display random samples
    for i in range(min(10, x_train.shape[0])):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i].reshape(x_train[i].shape[0], x_train[i].shape[1]), cmap='gray')
        plt.title(class_names[y_train[i].argmax()])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_samples.png")
    print(f"Sample visualization saved to dataset_samples.png")
    
    # Step 5: Train the model
    print(f"\nTraining model for {args.epochs} epochs with batch size {args.batch_size}...")
    history = math_ocr.train(
        (x_train, y_train),
        (x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 6: Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc = math_ocr.evaluate((x_test, y_test))
    
    # Step 7: Plot training history
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print(f"Training history saved to training_history.png")
    
    # Step 8: Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"handwritten_math_{args.dataset}.h5")
    math_ocr.save_model(model_path)
    
    print(f"\nTraining complete! Model saved to {model_path}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Step 9: Save class mapping for inference
    mapping_path = os.path.join(args.output_dir, f"handwritten_math_{args.dataset}_mapping.txt")
    with open(mapping_path, "w") as f:
        for symbol, idx in label_mapping.items():
            f.write(f"{symbol},{idx}\n")
    
    print(f"Class mapping saved to {mapping_path}")
    
    print("\n=== Training pipeline complete! ===")
    print("You can now use this model for handwritten math recognition.")
    print(f"To use this model: system = MathRecognitionSystem(ocr_model_path='{model_path}')")

if __name__ == "__main__":
    main()
