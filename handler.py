# /workspace/handler.py
import os, base64, subprocess, pathlib, traceback, time, json
from pathlib import Path
import runpod
import requests

# ---------- Constants / paths ----------
ROOT = Path("/workspace/OmniAvatar")
PERSIST_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODEL_STORE = PERSIST_ROOT / "pretrained_models"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # faster downloads

CONFIG_14B = ROOT / "configs" / "inference.yaml"
CONFIG_13B = ROOT / "configs" / "inference_1.3B.yaml"

MODEL_SIZE = os.getenv("OMNI_MODEL_SIZE", "14B")
TOKENS     = os.getenv("OMNI_TOKENS", "30000")
OVERLAP    = os.getenv("OMNI_OVERLAP", "1")
STEPS      = os.getenv("OMNI_STEPS", "25")

WAN_DIR   = MODEL_STORE / "Wan2.1-T2V-14B"
W2V_DIR   = MODEL_STORE / "wav2vec2-base-960h"
OMNI_DIR  = MODEL_STORE / "OmniAvatar-14B"

# Both of these must point to MODEL_STORE so either path in the repo works.
WORK_PRETRAINED = ROOT / "pretrained_models"
WORK_CKPT_PRETRAINED = ROOT / "checkpoints" / "pretrained_models"

# ---------- Small utils ----------
def _log(msg): print(msg, flush=True)

def _symlink(target: Path, link_path: Path):
    """Create symlink from link_path to target, handling existing files."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() and not link_path.is_symlink():
        # remove plain folder created by previous runs
        import shutil
        shutil.rmtree(link_path)
    if not link_path.exists():
        link_path.symlink_to(target, target_is_directory=True)

def _with_lock(lock_path: Path, fn):
    """Simple cross-process lock with retry; avoids multi-GB duplicate downloads."""
    for _ in range(300):  # ~5 min
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                return fn()
            finally:
                os.close(fd)
                os.unlink(lock_path)
        except FileExistsError:
            time.sleep(1)
    # last resort: run anyway
    return fn()

# ---------- Repo setup ----------
def ensure_repo():
    """Ensure OmniAvatar repository is cloned and available."""
    try:
        if not ROOT.exists():
            _log(f"Cloning OmniAvatar to {ROOT}")
            subprocess.run(
                ["bash", "-lc",
                 "git clone --depth 1 https://github.com/Omni-Avatar/OmniAvatar.git /workspace/OmniAvatar"],
                check=True, capture_output=True, text=True
            )
        if not (ROOT / "scripts").exists() or not (ROOT / "configs").exists():
            return False, "OmniAvatar repo missing required folders after clone"
        return True, "Repository ready"
    except Exception as e:
        return False, f"Repo setup failed: {e}"

# ---------- Model setup ----------
def _download(repo_id: str, local_dir: Path):
    """Download a HuggingFace repository to local directory with compatibility handling."""
    try:
        from huggingface_hub import snapshot_download
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN
        )
    except ImportError as e:
        if "HfHubHTTPError" in str(e):
            _log(f"HuggingFace Hub version compatibility issue: {e}")
            _log("Attempting to use alternative download method...")
            # Fallback to git clone if huggingface_hub is not compatible
            try:
                import subprocess
                if local_dir.exists():
                    import shutil
                    shutil.rmtree(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)
                
                # Use git clone as fallback
                clone_url = f"https://huggingface.co/{repo_id}"
                if HF_TOKEN:
                    clone_url = f"https://{HF_TOKEN}@huggingface.co/{repo_id}"
                
                subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, str(local_dir)],
                    check=True, capture_output=True, text=True
                )
                _log(f"Successfully cloned {repo_id} using git fallback")
            except Exception as git_error:
                raise Exception(f"Both huggingface_hub and git fallback failed: {git_error}")
        else:
            raise e
    except Exception as e:
        _log(f"Download failed for {repo_id}: {e}")
        raise e

def _verify_wan():
    """Verify that all required Wan2.1-T2V-14B model files are present."""
    shards = [WAN_DIR / f"diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)]
    shards.append(WAN_DIR / "Wan2.1_VAE.pth")
    missing = [str(p) for p in shards if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Wan2.1 files:\n" + "\n".join(missing))

def ensure_models():
    """Idempotent setup to persistent volume + workspace symlinks."""
    # symlink both locations used by the repo to the single persistent store
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    _symlink(MODEL_STORE, WORK_PRETRAINED)
    _symlink(MODEL_STORE, WORK_CKPT_PRETRAINED)

    def _do():
        # download any missing repo snapshots
        if not (WAN_DIR.exists() and any(WAN_DIR.iterdir())):
            _log("Downloading Wan2.1-T2V-14B ...")
            _download("Wan-AI/Wan2.1-T2V-14B", WAN_DIR)
        if not (W2V_DIR.exists() and any(W2V_DIR.iterdir())):
            _log("Downloading wav2vec2-base-960h ...")
            _download("facebook/wav2vec2-base-960h", W2V_DIR)
        if not (OMNI_DIR.exists() and any(OMNI_DIR.iterdir())):
            _log("Downloading OmniAvatar-14B ...")
            _download("OmniAvatar/OmniAvatar-14B", OMNI_DIR)
        _verify_wan()
        return True, "Models ready"

    lock = MODEL_STORE / ".setup.lock"
    try:
        return _with_lock(lock, _do)
    except Exception as e:
        return False, f"Model setup failed: {e}"

# ---------- Downloads (image/audio) ----------
def _download_to(path: Path, url: str, kind: str):
    """Download a file from URL to local path."""
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk: f.write(chunk)
    _log(f"{kind} downloaded: {path} ({path.stat().st_size} bytes)")

# ---------- Env check ----------
def check_environment():
    """Check if the environment is properly set up."""
    checks = []
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        checks.append("✅ CUDA/GPU available" if r.returncode == 0 else "❌ CUDA/GPU not available")
    except Exception:
        checks.append("❌ nvidia-smi not found")

    try:
        import torch
        checks.append(f"✅ PyTorch {torch.__version__} available")
        checks.append(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "❌ PyTorch CUDA not available")
    except Exception:
        checks.append("❌ PyTorch not available")

    checks.append(f"✅ OmniAvatar root: {ROOT}" if ROOT.exists() else f"❌ OmniAvatar root missing: {ROOT}")
    checks.append(f"✅ OmniAvatar scripts directory exists" if (ROOT / "scripts").exists() else "❌ OmniAvatar scripts directory missing")

    checks.append(f"✅ Models dir (persistent): {MODEL_STORE} exists" if MODEL_STORE.exists() else f"❌ Models dir missing: {MODEL_STORE}")
    
    # Check model availability
    if WAN_DIR.exists():
        base_model_check = WAN_DIR / "diffusion_pytorch_model-00001-of-00006.safetensors"
        checks.append(f"✅ Base Wan2.1-T2V-14B models available: {base_model_check}" if base_model_check.exists() else f"❌ Base Wan2.1-T2V-14B models missing: {base_model_check}")
    else:
        checks.append(f"❌ Base models directory missing: {WAN_DIR}")
    
    if OMNI_DIR.exists():
        omni_model_check = OMNI_DIR / "pytorch_model.pt"
        checks.append(f"✅ OmniAvatar weights available: {omni_model_check}" if omni_model_check.exists() else f"❌ OmniAvatar weights missing: {omni_model_check}")
    else:
        checks.append(f"❌ OmniAvatar weights directory missing: {OMNI_DIR}")

    return checks

# ---------- Inference ----------
def run_inference(prompt, img_path, wav_path):
    """Run OmniAvatar inference with detailed error handling."""
    env_checks = check_environment()
    _log("Environment checks: " + json.dumps(env_checks, ensure_ascii=False))

    ok, msg = ensure_repo()
    if not ok:
        raise Exception(f"Repository setup failed: {msg}")

    ok, msg = ensure_models()
    if not ok:
        raise Exception(f"Models setup failed: {msg}")

    cfg = CONFIG_14B if MODEL_SIZE == "14B" else CONFIG_13B
    if not cfg.exists():
        raise Exception(f"Config file not found: {cfg}")

    (ROOT / "examples").mkdir(parents=True, exist_ok=True)
    input_file = ROOT / "examples" / "infer_samples.txt"
    input_file.write_text(f"{prompt}@@{img_path}@@{wav_path}\n")

    cmd = [
        "bash","-lc",
        f"cd {ROOT} && "
        "torchrun --standalone --nproc_per_node=1 "
        f"scripts/inference.py --config {cfg} "
        f"--input_file examples/infer_samples.txt "
        f"--hp=num_steps={STEPS} --hp=num_tokens={TOKENS} --hp=overlap_frame={OVERLAP}"
    ]
    _log("Executing command: " + " ".join(cmd))

    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    _log("Inference stdout:\n" + out.stdout)
    if out.stderr: _log("Inference stderr:\n" + out.stderr)

    vids = sorted(ROOT.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vids:
        raise Exception("No MP4 files generated")
    return vids[0], out.stdout

# ---------- Handler ----------
def handler(event):
    """Main handler function for RunPod serverless."""
    try:
        inp = event.get("input", {}) or {}
        _log("Received input: " + json.dumps(inp, ensure_ascii=False))

        # quick test mode
        if "name" in inp and "image_url" not in inp and "audio_url" not in inp:
            return {"ok": True, "message": f"Hello {inp['name']}!", "status": "test_mode", "environment": check_environment()}

        prompt = inp.get("prompt", "Frontal talking head, neutral lighting.")
        image_url = inp.get("image_url")
        audio_url = inp.get("audio_url")
        return_b64 = bool(inp.get("return_base64", True))

        if not image_url or not audio_url:
            return {"ok": False, "error": "Missing required inputs: image_url and audio_url", "received_input": inp}

        img_path = Path("/tmp/ref.png")
        wav_path = Path("/tmp/line.wav")
        try:
            _download_to(img_path, image_url, "Image")
            _download_to(wav_path, audio_url, "Audio")
        except Exception as e:
            return {"ok": False, "error": f"Failed to download inputs: {e}", "image_url": image_url, "audio_url": audio_url, "traceback": traceback.format_exc()}

        try:
            mp4, logs = run_inference(prompt, str(img_path), str(wav_path))
            resp = {"ok": True, "model": MODEL_SIZE, "video_path": str(mp4), "video_size": os.path.getsize(mp4), "logs_tail": logs[-1000:] if logs else ""}
            if return_b64:
                with open(mp4, "rb") as f:
                    resp["mp4_base64"] = base64.b64encode(f.read()).decode("utf-8")
            return resp
        except subprocess.CalledProcessError as e:
            msg = f"Command failed with return code {e.returncode}"
            if e.stdout: msg += f"\nSTDOUT: {e.stdout}"
            if e.stderr: msg += f"\nSTDERR: {e.stderr}"
            return {"ok": False, "error": msg, "environment": check_environment()}
        except Exception as e:
            return {"ok": False, "error": f"Inference failed: {e}", "traceback": traceback.format_exc(), "environment": check_environment()}

    except Exception as e:
        return {"ok": False, "error": f"Handler error: {e}", "traceback": traceback.format_exc()}

# Optional: init hook runs once per container warmup
def init():
    """Initialize models and environment once per container warmup."""
    ensure_repo()
    ok, msg = ensure_models()
    return {"models": msg, "env": check_environment()}

runpod.serverless.start({"handler": handler, "init": init})
