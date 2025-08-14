# /workspace/handler.py
import os, base64, subprocess, pathlib
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
    if not ROOT.exists():
        subprocess.run(["bash","-lc",
            "git clone https://github.com/Omni-Avatar/OmniAvatar.git /workspace/OmniAvatar"
        ], check=True)

def run_inference(prompt, img_path, wav_path):
    cfg = CONFIG_14B if MODEL_SIZE == "14B" else CONFIG_13B
    (ROOT / "examples").mkdir(parents=True, exist_ok=True)
    (ROOT / "examples" / "infer_samples.txt").write_text(
        f"{prompt}@@{img_path}@@{wav_path}\n"
    )
    cmd = [
        "bash","-lc",
        f"cd {ROOT} && "
        "torchrun --standalone --nproc_per_node=1 "
        f"scripts/inference.py --config {cfg} "
        f"--input_file examples/infer_samples.txt "
        f"--hp=num_steps={STEPS} --hp=num_tokens={TOKENS} --hp=overlap_frame={OVERLAP}"
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    vids = sorted(ROOT.rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    return vids[0], out.stdout

def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt", "Frontal talking head, neutral lighting.")
    image_url = inp["image_url"]
    audio_url = inp["audio_url"]
    img_path, wav_path = "/tmp/ref.png", "/tmp/line.wav"
    urlretrieve(image_url, img_path)
    urlretrieve(audio_url, wav_path)
    ensure_repo()
    mp4, logs = run_inference(prompt, img_path, wav_path)
    b64 = base64.b64encode(open(mp4,'rb').read()).decode("utf-8")
    return {"ok": True, "model": MODEL_SIZE, "mp4_base64": b64, "logs_tail": logs[-1000:]}

runpod.serverless.start({"handler": handler})
