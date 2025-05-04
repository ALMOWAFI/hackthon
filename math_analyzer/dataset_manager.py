"""
Dataset Manager for Math OCR Training
Handles dataset collection, preprocessing, and augmentation
"""

import os
import json
import requests
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub
import base64
import shutil
from pathlib import Path
import csv

class MathDatasetManager:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        self.datasets = {}
        
        # APIs setup
        self.mathpix_headers = {
            'app_id': os.getenv('MATHPIX_APP_ID'),
            'app_key': os.getenv('MATHPIX_APP_KEY'),
            'Content-type': 'application/json'
        }
        
        # Dataset sources
        self.dataset_sources = {
            'mathinstruct': 'thedevastator/mathinstruct-dataset-hybrid-math-instruction-tun',
            'student_perf': 'whenamancodes/student-performance',
            'student_perf2': 'abdelrahmanmohamed26/students-performance',
            'math_score': 'soumyadiptadas/students-math-score-for-different-teaching-style'
        }
        
    def _load_config(self, config_path):
        """Load configuration file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'data_dir': './data',
            'cache_dir': './cache',
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'image_size': (384, 384)  # TrOCR expected size
        }
        
    def download_datasets(self):
        """Download datasets from various sources"""
        print("Downloading Kaggle datasets...")
        for name, dataset_id in self.dataset_sources.items():
            try:
                path = kagglehub.dataset_download(dataset_id)
                self.datasets[name] = path
                print(f"Successfully downloaded {name} dataset to {path}")
            except Exception as e:
                print(f"Error downloading {name} dataset: {str(e)}")
        
        return self.datasets
        
    def process_datasets(self):
        """Process downloaded datasets and organize them"""
        if not self.datasets:
            print("No datasets available. Please download datasets first.")
            return
            
        data_dir = Path(self.config['data_dir'])
        processed_dir = data_dir / 'processed'
        
        # Create directories if they don't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nProcessing datasets...")
        # Process each dataset
        for name, path in self.datasets.items():
            try:
                dataset_path = Path(path)
                print(f"\nProcessing {name} dataset from {dataset_path}...")
                
                # Handle different dataset types
                if name == 'mathinstruct':
                    self._process_math_instruct(dataset_path, processed_dir)
                elif name == 'student_perf':
                    self._process_student_performance(dataset_path, processed_dir)
                elif name == 'student_perf2':
                    self._process_student_performance2(dataset_path, processed_dir)
                elif name == 'math_score':
                    self._process_math_score(dataset_path, processed_dir)
                    
            except Exception as e:
                print(f"Error processing {name} dataset: {str(e)}")
                
    def _process_math_instruct(self, dataset_path, output_dir):
        """Process math instruction dataset"""
        try:
            train_file = dataset_path / 'train.csv'
            if train_file.exists():
                print(f"Processing math instruction data from {train_file}")
                df = pd.read_csv(train_file)
                
                # Extract relevant columns and filter math problems
                data = []
                for _, row in df.iterrows():
                    instruction = str(row.get('instruction', ''))
                    output = str(row.get('output', ''))
                    
                    # Only include math-related problems
                    if any(keyword in instruction.lower() for keyword in ['math', 'equation', 'solve', 'calculate', 'number']):
                        item = {
                            'instruction': instruction,
                            'solution': output,
                            'type': 'math_problem'
                        }
                        data.append(item)
                    
                print(f"Found {len(data)} math-related problems")
                
                # Save processed data
                output_file = output_dir / 'mathinstruct.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved processed data to {output_file}")
                    
        except Exception as e:
            print(f"Error processing math instruction dataset: {str(e)}")
            
    def _process_student_performance(self, dataset_path, output_dir):
        """Process student performance dataset"""
        try:
            math_file = dataset_path / 'Maths.csv'
            if math_file.exists():
                print(f"Processing math performance data from {math_file}")
                df = pd.read_csv(math_file, encoding='latin1')  # Try different encoding
                
                # Extract relevant columns
                data = {
                    'math_scores': df['G3'].tolist(),  # Final math grade
                    'study_time': df['studytime'].tolist(),
                    'failures': df['failures'].tolist(),
                    'absences': df['absences'].tolist(),
                    'type': 'student_performance'
                }
                
                # Save processed data
                output_file = output_dir / 'student_performance.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved processed data to {output_file}")
                    
        except Exception as e:
            print(f"Error processing student performance dataset: {str(e)}")
            
    def _process_student_performance2(self, dataset_path, output_dir):
        """Process second student performance dataset"""
        try:
            data_file = dataset_path / 'student-por.csv'
            if data_file.exists():
                print(f"Processing student performance data from {data_file}")
                df = pd.read_csv(data_file, sep=';')
                
                # Extract relevant columns
                data = {
                    'math_scores': df['G3'].tolist(),  # Final grade
                    'study_time': df['studytime'].tolist(),
                    'failures': df['failures'].tolist(),
                    'absences': df['absences'].tolist(),
                    'type': 'student_performance'
                }
                
                # Save processed data
                output_file = output_dir / 'student_performance2.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved processed data to {output_file}")
                    
        except Exception as e:
            print(f"Error processing student performance dataset: {str(e)}")
            
    def _process_math_score(self, dataset_path, output_dir):
        """Process math score dataset"""
        try:
            # Try both possible locations
            data_files = [
                dataset_path / 'data.csv',
                dataset_path / 'versions/1/data.csv'
            ]
            
            data_file = None
            for file in data_files:
                if file.exists():
                    data_file = file
                    break
                    
            if data_file:
                print(f"Processing math score data from {data_file}")
                df = pd.read_csv(data_file)
                
                # Extract relevant columns
                data = {
                    'math_scores': df['math_score'].tolist(),
                    'teaching_style': df['teaching_style'].tolist(),
                    'student_level': df['student_level'].tolist(),
                    'type': 'math_score'
                }
                
                # Save processed data
                output_file = output_dir / 'math_score.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved processed data to {output_file}")
            else:
                print("Math score data file not found")
                    
        except Exception as e:
            print(f"Error processing math score dataset: {str(e)}")
            
    class MathDataset(Dataset):
        """Custom dataset for math data"""
        def __init__(self, data_dir, transform=None):
            self.data_dir = Path(data_dir)
            self.transform = transform
            self.samples = self._load_samples()
            print(f"Loaded {len(self.samples)} samples from {data_dir}")
            
        def _load_samples(self):
            samples = []
            processed_dir = self.data_dir / 'processed'
            
            if not processed_dir.exists():
                print(f"Processed directory not found: {processed_dir}")
                return samples
                
            # Load all JSON files in the processed directory
            for json_file in processed_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # For mathinstruct dataset
                            samples.extend(data)
                        else:
                            # For performance datasets
                            samples.append(data)
                    print(f"Loaded {json_file.name}")
                except Exception as e:
                    print(f"Error loading {json_file}: {str(e)}")
                    
            if not samples:
                print("No valid samples found!")
                print(f"Processed directory contents: {list(processed_dir.glob('*'))}")
                
            return samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            return self.samples[idx]
            
    def prepare_dataset(self, data_dir):
        """Prepare dataset for training"""
        print(f"\nPreparing dataset from {data_dir}")
        
        # Create dataset instance
        dataset = self.MathDataset(
            data_dir,
            transform=None  # No image transform needed for this data
        )
        
        if len(dataset) == 0:
            print("No samples found in the dataset!")
            return None
            
        # Split dataset
        train_size = int(len(dataset) * self.config['train_split'])
        val_size = int(len(dataset) * self.config['val_split'])
        test_size = len(dataset) - train_size - val_size
        
        print(f"Splitting dataset into: train={train_size}, val={val_size}, test={test_size}")
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }
        
    def create_dataloaders(self, datasets, batch_size=32):
        """Create DataLoader instances"""
        if not datasets:
            print("No datasets provided to create dataloaders!")
            return None
            
        dataloaders = {}
        for split, dataset in datasets.items():
            if len(dataset) > 0:
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=4
                )
                print(f"Created dataloader for {split} split with {len(dataset)} samples")
            else:
                print(f"Warning: {split} split has no samples!")
                
        return dataloaders
        
    def export_dataset(self, output_dir):
        """Export processed dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            'num_samples': len(self.samples) if hasattr(self, 'samples') else 0,
            'splits': {
                'train': self.config['train_split'],
                'validation': self.config['val_split'],
                'test': self.config['test_split']
            },
            'preprocessing': {
                'normalization': 'per_channel'
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
if __name__ == "__main__":
    # Example usage
    manager = MathDatasetManager()
    
    # Download datasets
    datasets = manager.download_datasets()
    
    # Process datasets
    manager.process_datasets()
    
    # Prepare dataset
    processed_datasets = manager.prepare_dataset('./data')
    
    # Create dataloaders if we have processed datasets
    if processed_datasets:
        dataloaders = manager.create_dataloaders(processed_datasets)
        
        # Export dataset
        if dataloaders:
            manager.export_dataset('./processed_data')
