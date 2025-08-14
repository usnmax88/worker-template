# Use RunPod's CUDA 12.9 base image for better OmniAvatar compatibility
FROM runpod/base:0.7.0-ubuntu2404-cuda1290

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget curl ca-certificates \
    libgl1 libglib2.0-0 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Set working directory to /workspace (OmniAvatar expects this)
WORKDIR /workspace

# Install Python dependencies using pip instead of uv
COPY requirements.txt /requirements.txt
RUN pip install --upgrade -r /requirements.txt --no-cache-dir

# Add the handler to /workspace (OmniAvatar path)
COPY handler.py /workspace/handler.py

# Run the handler from /workspace
CMD ["python", "-u", "/workspace/handler.py"]
