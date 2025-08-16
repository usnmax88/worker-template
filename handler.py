# /workspace/handler.py
import os, base64, subprocess, pathlib, traceback
from urllib.request import urlretrieve
import runpod

ROOT = pathlib.Path("/workspace/OmniAvatar")  # worker-template uses /workspace
CONFIG_14B = ROOT / "configs" / "inference.yaml"      # 14B
CONFIG_13B = ROOT / "configs" / "inference_1.3B.yaml" # fallback if ever needed

MODEL_SIZE = os.getenv("OMNI_MODEL_SIZE", "14B")  # default to 14B
TOKENS     = os.getenv("OMNI_TOKENS", "30000")    # authors trained 14B at ~30k tokens
OVERLAP    = os.getenv("OMNI_OVERLAP", "1")       # try 1 or 13 per README
STEPS      = os.getenv("OMNI_STEPS", "25")

def ensure_repo():
    """Ensure OmniAvatar repository is cloned and available."""
    try:
        if not ROOT.exists():
            print(f"Cloning OmniAvatar to {ROOT}")
            subprocess.run(["bash","-lc",
                "git clone https://github.com/Omni-Avatar/OmniAvatar.git /workspace/OmniAvatar"
            ], check=True, capture_output=True, text=True)
            print("OmniAvatar repository cloned successfully")
        else:
            print(f"OmniAvatar repository already exists at {ROOT}")
            
        # Check if key directories exist
        if not (ROOT / "scripts").exists():
            return False, "OmniAvatar scripts directory not found after clone"
        if not (ROOT / "configs").exists():
            return False, "OmniAvatar configs directory not found after clone"
            
        return True, "Repository ready"
    except Exception as e:
        return False, f"Failed to ensure repository: {str(e)}"

def ensure_models():
    """Ensure OmniAvatar models are downloaded and available."""
    try:
        # Define model paths
        model_dir = ROOT / "checkpoints" / "pretrained_models" / "OmniAvatar-14B"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if models already exist
        pytorch_model = model_dir / "pytorch_model.pt"
        if pytorch_model.exists():
            print(f"Models already exist at {model_dir}")
            return True, "Models ready"
        
        print(f"Models not found at {model_dir}, attempting to download...")
        
        # Try to download models using the OmniAvatar download script
        download_script = ROOT / "scripts" / "download_models.py"
        if download_script.exists():
            print("Found download script, running it...")
            result = subprocess.run([
                "python", str(download_script)
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            print("Download script completed successfully")
        else:
            print("Download script not found, checking for alternative methods...")
            
            # Try to find models in common locations or download from HuggingFace
            # This is a fallback approach
            print("Attempting to download models from HuggingFace...")
            
            # Use huggingface-hub to download models
            try:
                import huggingface_hub
                print("Downloading OmniAvatar-14B models from HuggingFace...")
                
                # Download the model files
                huggingface_hub.snapshot_download(
                    repo_id="Omni-Avatar/OmniAvatar-14B",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )
                print("Models downloaded successfully from HuggingFace")
                
            except ImportError:
                print("huggingface-hub not available, trying alternative approach...")
                # Fallback: try to download specific files
                return False, "Models not available and no download method found"
        
        # Verify models were downloaded
        if pytorch_model.exists():
            print(f"Models successfully downloaded to {model_dir}")
            return True, "Models ready"
        else:
            return False, "Models still not found after download attempt"
            
    except Exception as e:
        return False, f"Failed to ensure models: {str(e)}"

def check_environment():
    """Check if the environment is properly set up."""
    checks = []
    
    # Check CUDA
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append("✅ CUDA/GPU available")
        else:
            checks.append("❌ CUDA/GPU not available")
    except:
        checks.append("❌ nvidia-smi not found")
    
    # Check Python packages
    try:
        import torch
        checks.append(f"✅ PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            checks.append(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            checks.append("❌ PyTorch CUDA not available")
    except ImportError:
        checks.append("❌ PyTorch not available")
    
    # Check directories
    if ROOT.exists():
        checks.append(f"✅ OmniAvatar root: {ROOT}")
        if (ROOT / "scripts").exists():
            checks.append("✅ OmniAvatar scripts directory exists")
        else:
            checks.append("❌ OmniAvatar scripts directory missing")
    else:
        checks.append(f"❌ OmniAvatar root missing: {ROOT}")
    
    # Check models (simple check - no downloading yet)
    model_dir = ROOT / "checkpoints" / "pretrained_models" / "OmniAvatar-14B"
    if model_dir.exists():
        pytorch_model = model_dir / "pytorch_model.pt"
        if pytorch_model.exists():
            checks.append(f"✅ OmniAvatar models available: {pytorch_model}")
        else:
            checks.append(f"❌ OmniAvatar models missing: {pytorch_model}")
    else:
        checks.append(f"❌ OmniAvatar models directory missing: {model_dir}")
    
    return checks

def run_inference(prompt, img_path, wav_path):
    """Run OmniAvatar inference with detailed error handling."""
    try:
        # First check environment
        env_checks = check_environment()
        print("Environment checks:", env_checks)
        
        # Ensure repository
        repo_ok, repo_msg = ensure_repo()
        if not repo_ok:
            raise Exception(f"Repository setup failed: {repo_msg}")
        
        # Ensure models are available
        models_ok, models_msg = ensure_models()
        if not models_ok:
            raise Exception(f"Models setup failed: {models_msg}")
        
        cfg = CONFIG_14B if MODEL_SIZE == "14B" else CONFIG_13B
        if not cfg.exists():
            raise Exception(f"Config file not found: {cfg}")
        
        # Create examples directory and input file
        (ROOT / "examples").mkdir(parents=True, exist_ok=True)
        input_file = ROOT / "examples" / "infer_samples.txt"
        input_file.write_text(f"{prompt}@@{img_path}@@{wav_path}\n")
        print(f"Created input file: {input_file} with content: {input_file.read_text()}")
        
        # Build command
        cmd = [
            "bash","-lc",
            f"cd {ROOT} && "
            "torchrun --standalone --nproc_per_node=1 "
            f"scripts/inference.py --config {cfg} "
            f"--input_file examples/infer_samples.txt "
            f"--hp=num_steps={STEPS} --hp=num_tokens={TOKENS} --hp=overlap_frame={OVERLAP}"
        ]
        
        print(f"Executing command: {' '.join(cmd)}")
        print(f"Working directory: {ROOT}")
        print(f"Input file exists: {input_file.exists()}")
        print(f"Config file exists: {cfg.exists()}")
        
        # Run inference
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
        print(f"Inference stdout: {out.stdout}")
        print(f"Inference stderr: {out.stderr}")
        
        # Find generated videos
        vids = sorted(ROOT.rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not vids:
            raise Exception("No MP4 files generated")
        
        print(f"Generated videos: {[str(v) for v in vids]}")
        return vids[0], out.stdout
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"Inference setup failed: {str(e)}\nTraceback: {traceback.format_exc()}")

def handler(event):
    """Main handler function with comprehensive error handling."""
    try:
        inp = event.get("input", {})
        print(f"Received input: {inp}")
        
        # Handle test input (for RunPod testing)
        if "name" in inp and "image_url" not in inp and "audio_url" not in inp:
            env_checks = check_environment()
            return {
                "ok": True,
                "message": f"Hello {inp['name']}! This is a test response from OmniAvatar handler.",
                "model": MODEL_SIZE,
                "status": "test_mode",
                "description": "Handler is working correctly and ready for OmniAvatar inference",
                "environment": env_checks
            }
        
        # Handle OmniAvatar input
        prompt = inp.get("prompt", "Frontal talking head, neutral lighting.")
        image_url = inp.get("image_url")
        audio_url = inp.get("audio_url")
        
        print(f"Processing: prompt='{prompt}', image='{image_url}', audio='{audio_url}'")
        
        # Validate required inputs
        if not image_url or not audio_url:
            return {
                "ok": False,
                "error": "Missing required inputs: image_url and audio_url are required",
                "received_input": inp
            }
        
        img_path, wav_path = "/tmp/ref.png", "/tmp/line.wav"
        
        try:
            print(f"Downloading image from {image_url} to {img_path}")
            urlretrieve(image_url, img_path)
            print(f"Downloading audio from {audio_url} to {wav_path}")
            urlretrieve(audio_url, wav_path)
            
            # Verify files were downloaded
            if not os.path.exists(img_path):
                raise Exception(f"Image file not found after download: {img_path}")
            if not os.path.exists(wav_path):
                raise Exception(f"Audio file not found after download: {wav_path}")
                
            print(f"Files downloaded successfully: image={os.path.getsize(img_path)} bytes, audio={os.path.getsize(wav_path)} bytes")
            
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to download files: {str(e)}",
                "image_url": image_url,
                "audio_url": audio_url,
                "traceback": traceback.format_exc()
            }
        
        try:
            mp4, logs = run_inference(prompt, img_path, wav_path)
            print(f"Inference completed successfully, video: {mp4}")
            
            # Encode video to base64
            with open(mp4, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            
            return {
                "ok": True, 
                "model": MODEL_SIZE, 
                "mp4_base64": b64, 
                "logs_tail": logs[-1000:] if logs else "No logs available",
                "video_path": str(mp4),
                "video_size": os.path.getsize(mp4)
            }
            
        except Exception as e:
            return {
                "ok": False,
                "error": f"Inference failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "environment": check_environment()
            }
            
    except Exception as e:
        return {
            "ok": False,
            "error": f"Handler error: {str(e)}",
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Test locally
    test_event = {"input": {"name": "test"}}
    result = handler(test_event)
    print("Test result:", result)

runpod.serverless.start({"handler": handler})

