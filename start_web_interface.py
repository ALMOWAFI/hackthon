#!/usr/bin/env python3
"""
Academic Document Analyzer - Web Interface Launcher
This script starts the web interface for the document analyzer.
"""

import argparse
from math_analyzer import MathHomeworkAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Start the Academic Document Analyzer web interface')
    parser.add_argument('--host', default='127.0.0.1', help='Host IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Academic Document Analyzer web interface at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")
    
    analyzer = MathHomeworkAnalyzer()
    analyzer.start_web_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
