"""
Advanced OCR Training Script for Mathematical Notation
Using Hugging Face, Azure Custom Vision, and MathPix
"""

import os
import json
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch
import requests
from PIL import Image
import numpy as np
from dotenv import load_dotenv

class MathOCRTrainer:
    def __init__(self):
        load_dotenv()
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.azure_key = os.getenv("AZURE_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.mathpix_app_id = os.getenv("MATHPIX_APP_ID")
        self.mathpix_app_key = os.getenv("MATHPIX_APP_KEY")
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/trocr-base-handwritten")
        
        # Initialize Azure Custom Vision client
        self.custom_vision_client = CustomVisionTrainingClient(
            self.azure_key,
            endpoint=self.azure_endpoint
        )
        
    def prepare_datasets(self):
        """Load and prepare training datasets"""
        # Load IM2LATEX dataset from Hugging Face
        im2latex = load_dataset("im2latex-100k")
        
        # Load handwritten math dataset
        crohme = load_dataset("crohme")
        
        # Combine datasets
        combined_dataset = {
            'train': self._combine_datasets([im2latex['train'], crohme['train']]),
            'validation': self._combine_datasets([im2latex['validation'], crohme['validation']]),
            'test': self._combine_datasets([im2latex['test'], crohme['test']])
        }
        
        return combined_dataset
        
    def _combine_datasets(self, datasets):
        """Combine multiple datasets into one"""
        combined = []
        for dataset in datasets:
            combined.extend(dataset)
        return combined
        
    def augment_with_mathpix(self, image_path):
        """Use MathPix API to get additional training data"""
        headers = {
            'app_id': self.mathpix_app_id,
            'app_key': self.mathpix_app_key,
            'Content-type': 'application/json'
        }
        
        # Read image and convert to base64
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
            
        data = {
            'src': f'data:image/jpeg;base64,{image_data}',
            'formats': ['latex', 'text'],
            'data_options': {
                'include_asciimath': True,
                'include_latex': True
            }
        }
        
        response = requests.post(
            'https://api.mathpix.com/v3/text',
            headers=headers,
            json=data
        )
        
        return response.json()
        
    def train_custom_vision(self, dataset_path):
        """Train Azure Custom Vision model"""
        # Create a new project
        project = self.custom_vision_client.create_project(
            "MathOCR",
            domain_id="ee85a72c-c7cc-4944-8a90-67d66b579c9d"  # Optical Character Recognition domain
        )
        
        # Upload and tag images
        image_list = []
        for image_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, image_file), "rb") as image_contents:
                image_list.append(
                    ImageFileCreateEntry(
                        name=image_file,
                        contents=image_contents.read(),
                        tag_ids=[tag.id for tag in tags]
                    )
                )
                
        # Upload images in batches
        batch_size = 64
        for i in range(0, len(image_list), batch_size):
            batch = ImageFileCreateBatch(images=image_list[i:i + batch_size])
            self.custom_vision_client.create_images_from_files(project.id, batch)
            
        # Train the model
        iteration = self.custom_vision_client.train_project(project.id)
        
        return project, iteration
        
    def fine_tune_transformer(self, dataset):
        """Fine-tune the Transformer model"""
        training_args = TrainingArguments(
            output_dir="./math_ocr_model",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        return eval_results
        
    def train(self, custom_dataset_path=None):
        """Main training pipeline"""
        print("Starting training pipeline...")
        
        # Prepare datasets
        print("Preparing datasets...")
        datasets = self.prepare_datasets()
        
        # Train Custom Vision model if dataset provided
        if custom_dataset_path:
            print("Training Azure Custom Vision model...")
            project, iteration = self.train_custom_vision(custom_dataset_path)
            print(f"Custom Vision model trained: {iteration.status}")
            
        # Fine-tune transformer model
        print("Fine-tuning transformer model...")
        eval_results = self.fine_tune_transformer(datasets)
        print(f"Training completed. Evaluation results: {eval_results}")
        
        # Save the final model
        self.model.save_pretrained("./final_math_ocr_model")
        self.tokenizer.save_pretrained("./final_math_ocr_model")
        
        return {
            'eval_results': eval_results,
            'model_path': "./final_math_ocr_model"
        }

if __name__ == "__main__":
    trainer = MathOCRTrainer()
    results = trainer.train()
