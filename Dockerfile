# CUDA 12.1 base (works on more driver versions than 12.9)
FROM runpod/base:0.6.2-cuda12.1.0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ffmpeg wget curl ca-certificates \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Make sure "python" exists and is python3
RUN ln -sf $(which python3) /usr/local/bin/python

# Workdir expected by most Runpod workers
WORKDIR /workspace

# Install Python deps (bind pip to interpreter explicitly)
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# A tiny probe to print versions inside the image (shows in build logs)
RUN python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch, torchvision, torchaudio
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Torchaudio:", torchaudio.__version__)
    print("Torch CUDA reported by torch:", torch.version.cuda)
except Exception as e:
    print("Torch import failed:", e)
PY

# (Optional) Your handler; comment this line out if you don't have handler.py yet
# COPY handler.py /workspace/handler.py

# Default command (you can override to run the test)
CMD ["python", "-u", "/workspace/handler.py"]
