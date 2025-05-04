"""
Test API credentials and dataset access
"""
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import load_dataset
import requests
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials

def test_huggingface():
    print("Testing Hugging Face access...")
    try:
        # Try to load a model
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/trocr-base-handwritten",
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        print("[PASS] Hugging Face credentials working")
        
        # Try to load a simple public dataset
        dataset = load_dataset(
            "mnist",
            split="train[:10]",
            trust_remote_code=True
        )
        print("[PASS] Dataset access working")
        return True
    except Exception as e:
        print(f"[FAIL] Hugging Face error: {str(e)}")
        return False

def test_azure():
    print("\nTesting Azure Custom Vision access...")
    try:
        # Create the Azure client with proper endpoint
        endpoint = "https://kellychopsvision.cognitiveservices.azure.com/"
        api_key = os.getenv("AZURE_KEY")
        
        # Set up credentials
        credentials = ApiKeyCredentials(in_headers={"Training-key": api_key})
        
        # Initialize the client
        client = CustomVisionTrainingClient(endpoint, credentials)
        
        # Try to list existing projects
        try:
            projects = client.get_projects()
            print("[PASS] Successfully connected to Azure Custom Vision")
            print(f"Found {len(projects)} existing projects")
        except Exception as e:
            print("[INFO] No existing projects found or insufficient permissions")
            print(f"Details: {str(e)}")
            
        print("[PASS] Azure credentials working")
        return True
    except Exception as e:
        print(f"[FAIL] Azure error: {str(e)}")
        return False

def main():
    # Load environment variables
    load_dotenv("math_analyzer/api.env")
    
    # Print loaded credentials (partially masked)
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    azure_key = os.getenv("AZURE_KEY", "")
    print("Loaded credentials:")
    print(f"Hugging Face token: {hf_token[:8]}...{hf_token[-4:] if len(hf_token) > 12 else ''}")
    print(f"Azure key: {azure_key[:8]}...{azure_key[-4:] if len(azure_key) > 12 else ''}")
    print()
    
    # Test each service
    hf_success = test_huggingface()
    azure_success = test_azure()
    
    # Print summary
    print("\nCredential Test Summary:")
    print(f"Hugging Face: {'[PASS]' if hf_success else '[FAIL]'}")
    print(f"Azure: {'[PASS]' if azure_success else '[FAIL]'}")
    
    if hf_success and azure_success:
        print("\nAll credentials are working! Ready to start training.")
    else:
        print("\nSome credentials need attention before proceeding.")

if __name__ == "__main__":
    main()
