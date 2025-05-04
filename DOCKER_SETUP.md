# Docker Setup for Math Feedback System

This document explains how to run the Math Feedback System using Docker.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (comes with Docker Desktop for Windows/Mac)

## System Architecture

The containerized system consists of three main services:

1. **vLLM Server**: Runs the AI models for math analysis
2. **Flask App**: Handles the backend logic, including OCR, math analysis, and feedback generation
3. **Web Interface**: Provides a user-friendly interface for interacting with the system

## Quick Start

We've created a helper script to make working with Docker easier. You can use it with the following commands:

```bash
# Start all services
python docker_run.py start

# Check status of running containers
python docker_run.py status

# View logs from all services
python docker_run.py logs

# View logs from a specific service
python docker_run.py logs --service flask-app

# Rebuild containers (after code changes)
python docker_run.py rebuild

# Stop all services
python docker_run.py stop
```

## Service URLs

When the containers are running, you can access:

- Flask API: http://localhost:5000
- vLLM API: http://localhost:8000
- Web Interface: http://localhost:3000

## Manual Docker Commands

If you prefer to use Docker commands directly:

```bash
# Start services in the background
docker-compose up -d

# View container status
docker-compose ps

# View logs
docker-compose logs

# View logs for a specific service
docker-compose logs flask-app

# Rebuild containers after changes
docker-compose down
docker-compose build
docker-compose up -d

# Stop services
docker-compose down
```

## Development Workflow

When making changes to the codebase:

1. Make your code changes
2. Run `python docker_run.py rebuild` to rebuild and restart the containers
3. Test your changes

Most code changes will be immediately available in the containers due to volume mounting, but changes to dependencies or Dockerfiles require a rebuild.

## Troubleshooting

- **Error: port already in use**: Make sure no other applications are using ports 3000, 5000, or 8000
- **Container fails to start**: Check the logs with `python docker_run.py logs`
- **Changes not showing up**: For some files, you may need to rebuild the containers with `python docker_run.py rebuild`
