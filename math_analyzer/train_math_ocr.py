"""
Train OCR model for math notation recognition using TrOCR and MathPix
"""
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from tqdm import tqdm

class MathOCRTrainer:
    def __init__(self):
        # Load environment variables
        load_dotenv("math_analyzer/api.env")
        
        # Initialize TrOCR model and processor
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Configure model
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Set up MathPix credentials (trial account)
        self.mathpix_app_id = os.getenv("MATHPIX_APP_ID")
        self.mathpix_app_key = os.getenv("MATHPIX_APP_KEY")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def prepare_dataset(self):
        """Load and prepare MNIST dataset as initial training data"""
        print("Loading MNIST dataset...")
        dataset = load_dataset("mnist", split="train[:1000]")
        
        # Convert MNIST images to RGB and resize
        processed_images = []
        labels = []
        
        for item in tqdm(dataset, desc="Processing images"):
            # Convert numpy array to PIL Image
            image = Image.fromarray(np.uint8(item["image"]))
            # Convert to RGB
            image = image.convert("RGB")
            # Resize to TrOCR expected size
            image = image.resize((384, 384))
            
            # Process image for model
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            processed_images.append(pixel_values[0])
            
            # Convert label to string
            labels.append(str(item["label"]))
        
        return processed_images, labels
    
    def train(self, num_epochs=3):
        """Train the model on processed dataset"""
        print("Preparing dataset...")
        processed_images, labels = self.prepare_dataset()
        
        # Set up training parameters
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            
            for idx in tqdm(range(len(processed_images)), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Prepare batch
                pixel_values = processed_images[idx].unsqueeze(0).to(self.device)
                labels_encoded = self.processor.tokenizer(
                    labels[idx], 
                    return_tensors="pt",
                    padding="max_length",
                    max_length=24
                ).input_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels_encoded)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(processed_images)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_dir = f"math_ocr_model/checkpoint-epoch-{epoch + 1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_model(self, output_dir="math_ocr_model"):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def test_prediction(self, image_path):
        """Test the model on a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate prediction
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

def main():
    trainer = MathOCRTrainer()
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
