FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.11, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        build-essential libpq-dev git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for CUDA build
ENV CMAKE_ARGS="-DGGML_CUDA=on"

COPY requirements.txt ./requirements.txt

# Use Python 3.11 for pip install
RUN python3.11 -m pip install --default-timeout=300 --retries=10 -r requirements.txt

COPY backend/ ./backend/
WORKDIR /app/backend

EXPOSE 8000

CMD ["python3.11", "manage.py", "runserver", "0.0.0.0:8000"]
