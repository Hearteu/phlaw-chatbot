# Dockerfile in phlaw-chatbot/
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the backend code
COPY backend /app/backend

WORKDIR /app/backend

# Expose the port if needed (for Django)
EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
