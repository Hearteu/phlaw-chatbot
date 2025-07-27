# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only requirements.txt to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend code only (not entire root folder)
COPY backend/ ./backend/
WORKDIR /app/backend

# Expose port for Django
EXPOSE 8000

# Run Django development server (for production, use gunicorn/uvicorn instead)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
