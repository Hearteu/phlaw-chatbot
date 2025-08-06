# Use the official NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.11, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        build-essential libpq-dev git \
        curl \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA build
ENV CMAKE_ARGS="-DGGML_CUDA=on"

# Copy the backend requirements file and install Python dependencies
COPY requirements.txt ./requirements.txt
RUN python3.11 -m pip install --default-timeout=300 --retries=10 -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/
WORKDIR /app/backend

# Expose the Django server port
EXPOSE 8000

# Start the Django server
CMD ["python3.11", "manage.py", "runserver", "0.0.0.0:8000"]
