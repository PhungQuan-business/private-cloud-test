# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY main.py .

# Default command
CMD ["python", "main.py"]
