#!/usr/bin/env python3
"""
Docker helper script for the Math Feedback System
This script helps manage the Docker containers for the math feedback system.
"""

import os
import argparse
import subprocess
import time

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def start_containers():
    """Start all Docker containers"""
    print("Starting Docker containers...")
    run_command("docker-compose up -d")
    print("\nContainers started successfully!")
    print("Services available at:")
    print("- Flask API:       http://localhost:5000")
    print("- vLLM API:        http://localhost:8000")
    print("- Web Interface:   http://localhost:3000")

def stop_containers():
    """Stop all Docker containers"""
    print("Stopping Docker containers...")
    run_command("docker-compose down")
    print("Containers stopped successfully!")

def check_status():
    """Check status of containers"""
    print("Checking container status...")
    run_command("docker-compose ps")

def view_logs(service=None):
    """View logs for a specific service or all services"""
    if service:
        print(f"Viewing logs for {service}...")
        run_command(f"docker-compose logs {service}")
    else:
        print("Viewing logs for all services...")
        run_command("docker-compose logs")

def rebuild_containers():
    """Rebuild and restart containers"""
    print("Rebuilding containers...")
    run_command("docker-compose down")
    run_command("docker-compose build")
    run_command("docker-compose up -d")
    print("Containers rebuilt and restarted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker helper for Math Feedback System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start all containers")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop all containers")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check container status")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View container logs")
    logs_parser.add_argument("--service", "-s", help="Service name (flask-app, vllm-server, web-interface)")
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild and restart containers")
    
    args = parser.parse_args()
    
    # Execute the requested command
    if args.command == "start":
        start_containers()
    elif args.command == "stop":
        stop_containers()
    elif args.command == "status":
        check_status()
    elif args.command == "logs":
        view_logs(args.service)
    elif args.command == "rebuild":
        rebuild_containers()
    else:
        parser.print_help()
