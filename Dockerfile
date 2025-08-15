# CUDA 12.9 base
FROM runpod/base:0.7.0-ubuntu2404-cuda1290

# System deps (no uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget curl ca-certificates \
    libgl1 libglib2.0-0 python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Make sure "python" exists and points to python3
RUN ln -sf $(which python3) /usr/local/bin/python

# Workdir
WORKDIR /workspace

# Copy first, then install — and always bind pip to the same interpreter
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# App
COPY handler.py /workspace/handler.py

# Run
CMD ["python", "-u", "/workspace/handler.py"]
