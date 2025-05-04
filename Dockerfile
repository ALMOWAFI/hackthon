FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir vllm fastapi uvicorn

# Copy model server and necessary files
COPY run_vllm_server.py .

# Copy the hackthon folder (contains first vLLM model)
COPY hackthon/ /app/hackthon/

# Copy any other model directories from the main folder
# This will include the second vLLM model outside the hackthon folder
COPY models/ /app/models/

# Expose port for vLLM server
EXPOSE 8000

# Start the vLLM server
CMD ["python3", "run_vllm_server.py", "--port", "8000"]
