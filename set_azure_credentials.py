"""
Set Azure credentials as environment variables securely.
This is a helper script to set your credentials before running the server.

This script sets multiple environment variable formats to ensure compatibility
with different parts of your Azure integration codebase.
"""

import os
import sys
import subprocess
import json

def set_azure_credentials():
    # Azure endpoint and API key
    AZURE_ENDPOINT = "https://kellychopsvision.cognitiveservices.azure.com/"
    AZURE_API_KEY = "Bz8oqt8Etc10RgPR6nNchr4oCkbVRSCpIDKfRdSiLfoFHIQ5x4B9JQQJ99BEACYeBjFXJ3w3AAAFACOG4BGA"
    
    # Set for Form Recognizer
    os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_FORM_RECOGNIZER_KEY"] = AZURE_API_KEY
    
    # Set for Computer Vision
    os.environ["AZURE_VISION_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_VISION_KEY"] = AZURE_API_KEY
    
    # Set generic Azure credentials
    os.environ["AZURE_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_KEY"] = AZURE_API_KEY
    
    # Set as OpenAI key for compatibility
    os.environ["AZURE_OPENAI_KEY"] = AZURE_API_KEY
    os.environ["OPENAI_API_KEY"] = AZURE_API_KEY
    
    # Also set cognitive services credentials generic format
    os.environ["AZURE_COGNITIVE_SERVICE_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_COGNITIVE_SERVICE_KEY"] = AZURE_API_KEY
    
    # Write to .env file for persistence
    try:
        with open('.env', 'w') as env_file:
            env_file.write(f"AZURE_FORM_RECOGNIZER_ENDPOINT={AZURE_ENDPOINT}\n")
            env_file.write(f"AZURE_FORM_RECOGNIZER_KEY={AZURE_API_KEY}\n")
            env_file.write(f"AZURE_VISION_ENDPOINT={AZURE_ENDPOINT}\n")
            env_file.write(f"AZURE_VISION_KEY={AZURE_API_KEY}\n")
            env_file.write(f"AZURE_ENDPOINT={AZURE_ENDPOINT}\n")
            env_file.write(f"AZURE_KEY={AZURE_API_KEY}\n")
            env_file.write(f"AZURE_OPENAI_KEY={AZURE_API_KEY}\n")
            env_file.write(f"OPENAI_API_KEY={AZURE_API_KEY}\n")
        print("Azure credentials written to .env file")
    except Exception as e:
        print(f"Warning: Could not write to .env file: {e}")
    
    print("Azure credentials set successfully in environment variables!")
    
    # Create a configuration file as well
    try:
        config = {
            "azure": {
                "form_recognizer": {
                    "endpoint": AZURE_ENDPOINT,
                    "key": AZURE_API_KEY
                },
                "vision": {
                    "endpoint": AZURE_ENDPOINT,
                    "key": AZURE_API_KEY
                },
                "openai": {
                    "endpoint": AZURE_ENDPOINT,
                    "key": AZURE_API_KEY
                }
            }
        }
        with open('azure_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Configuration written to azure_config.json")
    except Exception as e:
        print(f"Warning: Could not write configuration file: {e}")
    
    # Launch the server with these credentials
    if len(sys.argv) > 1 and sys.argv[1] == "--run-server":
        print("Starting server with Azure credentials...")
        subprocess.run(["python", "mvp_server.py"])

if __name__ == "__main__":
    set_azure_credentials()
