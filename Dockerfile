# Use the official NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.11, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        build-essential libpq-dev git \
        ninja-build \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA build
ENV CMAKE_ARGS="-DGGML_CUDA=on"

# Install node.js and npm for Next.js dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy the backend requirements and install Python dependencies
COPY requirements.txt ./requirements.txt
RUN python3.11 -m pip install --default-timeout=300 --retries=10 -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/
WORKDIR /app/backend

# Expose the Django server port
EXPOSE 8000

# Install Next.js frontend dependencies and build
COPY frontend/ ./frontend/
WORKDIR /app/frontend

# Install npm dependencies for Next.js
RUN npm install

# Expose the Next.js server port
EXPOSE 3000

# Run both backend and frontend in parallel using `supervisor` or a similar tool

# Supervisor approach (you can install it in the base container)
RUN apt-get install -y supervisor

COPY supervisor.conf /etc/supervisor/conf.d/supervisord.conf

# Start both the Django backend and Next.js frontend
CMD ["/usr/bin/supervisord"]
