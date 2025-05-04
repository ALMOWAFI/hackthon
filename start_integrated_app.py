"""
Start both the React web app and the Enhanced Math System servers
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser

# Define the paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REACT_APP_PATH = os.path.join(PROJECT_ROOT, 'paper-pal-prodigy-prime')
ENHANCED_SERVER_PATH = os.path.join(PROJECT_ROOT, 'enhanced_web_server.py')

def run_react_app():
    """Start the React application"""
    print("Starting React application...")
    
    # Make sure we're in the correct directory
    os.chdir(REACT_APP_PATH)
    print(f"Running React app from: {os.getcwd()}")
    
    # Use npm run dev to start the React app
    if os.name == 'nt':  # Windows
        process = subprocess.Popen("npm run dev", cwd=REACT_APP_PATH, shell=True)
    else:  # Unix/Linux/MacOS
        process = subprocess.Popen(["npm", "run", "dev"], cwd=REACT_APP_PATH)
        
    print("React application started. It will be available at http://localhost:8080")

def run_enhanced_server():
    """Start the Enhanced Math System backend"""
    print("Starting Enhanced Math System backend...")
    python_executable = sys.executable
    
    # Run the enhanced_web_server.py script
    os.chdir(PROJECT_ROOT)
    if os.name == 'nt':  # Windows
        subprocess.Popen([python_executable, ENHANCED_SERVER_PATH], shell=True)
    else:  # Unix/Linux/MacOS
        subprocess.Popen([python_executable, ENHANCED_SERVER_PATH], shell=True)
        
    print("Enhanced Math System backend started. API is available at http://localhost:5000")

def open_browser():
    """Open the browser to the React app URL after a delay"""
    time.sleep(5)  # Wait for servers to start
    print("Opening browser...")
    webbrowser.open("http://localhost:8080")

if __name__ == "__main__":
    # Start both servers in separate threads
    react_thread = threading.Thread(target=run_react_app)
    enhanced_thread = threading.Thread(target=run_enhanced_server)
    browser_thread = threading.Thread(target=open_browser)
    
    react_thread.start()
    enhanced_thread.start()
    browser_thread.start()
    
    # Print instructions
    print("""
=====================================================================
   Integrated Math Feedback System with vLLM
=====================================================================

React Frontend UI: http://localhost:8080
Enhanced Math API: http://localhost:5000

- Upload math homework images through the React UI
- The UI will communicate with the backend API
- The backend uses your fine-tuned models from the hackthon folder
- You'll get detailed feedback and practice worksheets

Press Ctrl+C in this terminal to stop both servers.
    """)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # The servers will be killed when the script exits
        sys.exit(0)
