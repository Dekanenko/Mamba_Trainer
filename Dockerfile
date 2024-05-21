# Use the official PyTorch image as a base
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    transformers \
    pip install torch \
    pip install causal-conv1d>=1.2.0 \
    pip install mamba-ssm \
    accelerate==0.27.2 \
    bitsandbytes==0.41.3 \
    scipy==1.11.4

# Copy application code to the container
COPY . /app

# Command to run training script
CMD ["python", "train.py"]