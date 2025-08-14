#!/bin/bash
# Custom bootstrap script for OmniAvatar RunPod worker

echo "Starting OmniAvatar worker..."

# Set environment variables
export OMNI_MODEL_SIZE=${OMNI_MODEL_SIZE:-14B}
export OMNI_TOKENS=${OMNI_TOKENS:-30000}
export OMNI_OVERLAP=${OMNI_OVERLAP:-1}
export OMNI_STEPS=${OMNI_STEPS:-25}

# Set HuggingFace cache location if network volume is attached
if [ -d "/runpod-volume" ]; then
    export HF_HOME=/runpod-volume/.cache/huggingface
    echo "Using network volume for HF cache: $HF_HOME"
fi

# Start the handler
python -u /workspace/handler.py
