import argparse
import os
import sys
from huggingface_hub import snapshot_download

from mage_flow.pipeline_xpu import MageFlowPipeline

# ==============================================================================
# LOCAL MODEL RESOLVER
# Checks /models/ folder first. If missing, downloads it cleanly.
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def resolve_model_path(repo_or_path):
    if os.path.isdir(repo_or_path): return os.path.abspath(repo_or_path)
    model_name = repo_or_path.split("/")[-1]
    local_model_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "model_index.json")):
        print(f">> Found local model at: {local_model_dir}")
        return local_model_dir
    print(f"\n>> Downloading '{repo_or_path}' from Hugging Face...")
    try:
        os.makedirs(local_model_dir, exist_ok=True)
        snapshot_download(repo_id=repo_or_path, local_dir=local_model_dir, local_dir_use_symlinks=False)
        return local_model_dir
    except Exception as e:
        print(f"\n[ERROR] Failed to download: {e}")
        sys.exit(1)

# ==============================================================================
# CLI LOGIC
# ==============================================================================
def _add_common_args(p):
    p.add_argument("--model_path", required=True, help="HF Repo ID or local directory name inside /models")
    p.add_argument("--neg_prompt", default=None, help="negative prompt applied to every sample (default: a single space)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42, help="base seed; sample i uses seed + i")
    p.add_argument("--static_shift", type=float, default=None)
    p.add_argument("--device", default="xpu")
    p.add_argument("--out", default="./outputs")

def _neg_list(neg_prompt, n):
    return [neg_prompt] * n if neg_prompt is not None else None

def _size_list(vals, n, name, parser):
    if len(vals) == 1: return vals * n
    if len(vals) == n: return vals
    parser.error(f"--{name} expects 1 value or {n} values")

def main():
    """``mage-flow-xpu`` — batched text-to-image generation."""
    p = argparse.ArgumentParser(prog="mage-flow-xpu", description="Sequential Low-RAM Mage-Flow T2I Generation for Intel ARC")
    p.add_argument("--prompt", nargs="+", required=True)
    p.add_argument("--height", type=int, nargs="+", default=[1024])
    p.add_argument("--width", type=int, nargs="+", default=[1024])
    p.add_argument("--prompt_template", default="mage-flow")
    _add_common_args(p)
    args = p.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    local_path = resolve_model_path(args.model_path)
    
    pipe = MageFlowPipeline.from_pretrained(local_path, device=args.device)
    n = len(args.prompt)
    
    imgs = pipe.generate(
        args.prompt,
        neg_prompts=_neg_list(args.neg_prompt, n),
        seeds=[args.seed + i for i in range(n)],
        heights=_size_list(args.height, n, "height", p),
        widths=_size_list(args.width, n, "width", p),
        steps=args.steps, cfg=args.cfg, static_shift=args.static_shift,
        prompt_template=args.prompt_template
    )
    
    for i, im in enumerate(imgs):
        path = os.path.join(args.out, f"gen_{i:03d}.png")
        im.save(path)
        print(f">> Saved {path}")

def main_edit():
    """``mage-flow-edit-xpu`` — batched instruction-based image editing."""
    p = argparse.ArgumentParser(prog="mage-flow-edit-xpu", description="Sequential Low-RAM Mage-Flow Edit Generation for Intel ARC")
    p.add_argument("--prompt", nargs="+", required=True)
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--max_size", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--vl_cond_long_edge", type=int, default=384)
    p.add_argument("--prompt_template", default="mage-flow-edit")
    _add_common_args(p)
    args = p.parse_args()
    
    if len(args.ref) != len(args.prompt):
        p.error(f"--ref count ({len(args.ref)}) must match --prompt count ({len(args.prompt)})")
    os.makedirs(args.out, exist_ok=True)
    
    local_path = resolve_model_path(args.model_path)
    pipe = MageFlowPipeline.from_pretrained(local_path, device=args.device)
    n = len(args.prompt)
    
    ref_images = [[s.strip() for s in r.split(",") if s.strip()] for r in args.ref]
    size_kw = {"heights": [args.height] * n, "widths": [args.width] * n} if args.height and args.width else {}
        
    outs = pipe.edit(
        args.prompt, ref_images,
        neg_prompts=_neg_list(args.neg_prompt, n),
        seeds=[args.seed + i for i in range(n)],
        max_size=args.max_size, steps=args.steps, cfg=args.cfg,
        static_shift=args.static_shift, prompt_template=args.prompt_template,
        vl_cond_long_edge=args.vl_cond_long_edge,
        **size_kw,
    )
    
    for i, im in enumerate(outs):
        path = os.path.join(args.out, f"edit_{i:03d}.png")
        im.save(path)
        print(f">> Saved {path}")

if __name__ == "__main__":
    main()