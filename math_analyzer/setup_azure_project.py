"""
Set up Azure Custom Vision project for math notation recognition
"""
import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import time

def create_math_vision_project():
    # Load credentials
    load_dotenv("math_analyzer/api.env")
    endpoint = "https://kellychopsvision.cognitiveservices.azure.com/"
    api_key = os.getenv("AZURE_KEY")
    
    # Set up credentials
    credentials = ApiKeyCredentials(in_headers={"Training-key": api_key})
    trainer = CustomVisionTrainingClient(endpoint, credentials)
    
    # Create new project
    print("Creating new Custom Vision project for math notation...")
    project = trainer.create_project(
        "MathNotationOCR",
        domain_id="ee85a72c-c7cc-4944-8a90-67d66b579c9d",  # OCR domain
        classification_type="Multilabel"
    )
    
    # Create tags for different math symbols
    print("Creating tags for math symbols...")
    tags = {}
    symbol_types = [
        "digit", "operator", "parenthesis", "equals",
        "variable", "exponent", "fraction", "root"
    ]
    
    for symbol_type in symbol_types:
        tag = trainer.create_tag(project.id, symbol_type)
        tags[symbol_type] = tag
        print(f"Created tag: {symbol_type}")
    
    print(f"\nProject created successfully!")
    print(f"Project ID: {project.id}")
    print(f"Project Name: {project.name}")
    
    return project, tags

if __name__ == "__main__":
    project, tags = create_math_vision_project()
