# TIA-SMART-CONNECT-SERVICE/app/Dockerfile (or equivalent location)

# Use a base image with NVIDIA CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set up the environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and pip
RUN apt-get update && \
  apt-get install -y python3.11 python3-pip && \
  apt-get clean

# Set up a working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies for GPU
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY ./src/ .

# Expose the port the API will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
