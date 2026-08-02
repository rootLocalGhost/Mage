import os
import gc
import json
import time
import torch
import torch.nn.functional as F
import sys
import numpy as np
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import transformers

# ==============================================================================
# 1. HF TRANSFORMERS SDPA PATCH
# Forces the base text encoder to use native PyTorch SDPA.
# ==============================================================================
transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = (
    lambda *args, **kwargs: "sdpa"
)

# ==============================================================================
# 2. INTERNAL MAGE FLOW IMPORTS
# ==============================================================================
from mage_flow.models.mage_flow import MageFlowModel, ModelConfig
from mage_flow.models.utils import get_noise, unpack
from mage_flow.models.modules.mage_latent import encode_noise, resolve_gs_key
from mage_flow.pipeline import (
    _build_pack_ctx,
    _velocity,
    _lens_to_cu,
    _make_divisible_by_16,
    _encode_texts_packed,
    _slice_packed,
    _template_info,
    build_scheduler,
)
import mage_flow.models.modules._attn_backend as attn_backend


# ==============================================================================
# 3. DIRECT CACHE INJECTION (THE FIX)
# Instead of faking the flash_attn library, we inject our native XPU math
# directly into the backend's resolved function cache.
# ==============================================================================
def _mock_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    out = torch.empty_like(q)
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    num_groups = q.shape[1] // k.shape[1]

    for i in range(len(cu_q) - 1):
        sq, eq, sk, ek = cu_q[i], cu_q[i + 1], cu_k[i], cu_k[i + 1]
        qi, ki, vi = (
            q[sq:eq].transpose(0, 1).unsqueeze(0),
            k[sk:ek].transpose(0, 1).unsqueeze(0),
            v[sk:ek].transpose(0, 1).unsqueeze(0),
        )

        if num_groups > 1:
            ki = torch.repeat_interleave(ki, repeats=num_groups, dim=1)
            vi = torch.repeat_interleave(vi, repeats=num_groups, dim=1)

        oi = F.scaled_dot_product_attention(
            qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        out[sq:eq] = oi.squeeze(0).transpose(0, 1)
    return out


# Hijack the resolved backend function so the Transformer never looks for NVIDIA kernels
attn_backend._RESOLVED_FN = _mock_flash_attn_varlen_func

# ==============================================================================
# 4. MEMORY MANAGEMENT
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def empty_cache_xpu():
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def resolve_model_path(repo_or_path):
    if os.path.isdir(repo_or_path):
        return os.path.abspath(repo_or_path)
    model_name = repo_or_path.split("/")[-1]
    local_model_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(local_model_dir) and os.path.exists(
        os.path.join(local_model_dir, "model_index.json")
    ):
        return local_model_dir
    print(f"\n>> Downloading '{repo_or_path}' from Hugging Face...")
    snapshot_download(
        repo_id=repo_or_path, local_dir=local_model_dir, local_dir_use_symlinks=False
    )
    return local_model_dir


# ==============================================================================
# 5. SEQUENTIAL GENERATION PIPELINE
# ==============================================================================
def generate_image(
    prompt: str,
    repo_path: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    num_steps: int = 4,
):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    print(f"\n--- Starting Generation: '{prompt}' (seed={seed}, steps={num_steps}) ---")
    repo_dir = resolve_model_path(repo_path)

    mi = json.load(open(os.path.join(repo_dir, "model_index.json")))
    tcfg = json.load(open(os.path.join(repo_dir, "transformer", "config.json")))
    _meta = {
        "_class_name",
        "txt_max_length",
        "max_sequence_length",
        "param_dtype",
        "packing",
        "schedule_mode",
        "static_shift",
        "use_time_shift",
        "rope_type",
        "apply_text_rotary_emb",
        "mlp_ratio",
        "depth_single_blocks",
        "theta",
        "qkv_bias",
        "guidance_embed",
        "vec_in_dim",
        "vec_type",
        "time_type",
        "double_block_type",
    }
    structure = {k: v for k, v in tcfg.items() if k not in _meta}

    vae_src = mi.get("_vae_source")
    txt_src = mi.get("_text_encoder_path")

    cfg = ModelConfig(
        vae_path=vae_src if os.path.isabs(vae_src) else os.path.join(repo_dir, vae_src),
        txt_enc_path=(
            txt_src if os.path.isabs(txt_src) else os.path.join(repo_dir, txt_src)
        ),
        model_structure=structure,
        txt_max_length=tcfg.get("txt_max_length", 2048),
        packing=tcfg.get("packing", True),
        static_shift=tcfg.get("static_shift", 6.0),
    )

    model_shell = MageFlowModel.__new__(MageFlowModel)
    torch.nn.Module.__init__(model_shell)
    model_shell.config = cfg

    # --- PHASE A: Text Encoder ---
    print("\n>> [1/3] Loading Text Encoder (Qwen3-VL)...")
    model_shell.txt_enc = model_shell.load_text_enc()
    model_shell.txt_enc.to(device)
    model_shell.txt_enc.to(dtype)

    print(">> Generating text embeddings...")
    info = _template_info("mage-flow")
    template, drop_idx = info.get("template", "{}"), int(info.get("start_idx", 0))

    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        txt_flat, vec_all, lens_t = _encode_texts_packed(
            model_shell, [prompt], template, drop_idx, device
        )
        txt, txt_cu, txt_mask, vec = _slice_packed(
            txt_flat, vec_all, lens_t, 0, 1, device
        )

    print(">> Unloading Text Encoder to free system RAM...")
    del model_shell.txt_enc
    empty_cache_xpu()

    # --- PREPARE LATENTS ---
    print("\n>> Preparing initial noise latents...")
    ch = 128
    h_, w_ = _make_divisible_by_16(height), _make_divisible_by_16(width)
    torch.manual_seed(seed)

    x = get_noise(
        num_samples=1,
        channel=ch,
        height=h_,
        width=w_,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    x = encode_noise(
        tuple(x.shape[1:]),
        key=resolve_gs_key(None),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    _, _, gh, gw = x.shape
    img = rearrange(x, "b c h w -> b (h w) c")

    ids = torch.zeros(gh, gw, 3, device=device)
    ids[..., 1], ids[..., 2] = (
        ids[..., 1] + torch.arange(gh, device=device)[:, None],
        ids[..., 2] + torch.arange(gw, device=device)[None, :],
    )
    img_ids = rearrange(ids, "h w c -> (h w) c").unsqueeze(0)

    lens = [gh * gw]
    img_cu = _lens_to_cu(lens, device)
    ctx = _build_pack_ctx(
        img_ids,
        img_cu,
        [[(1, gh, gw)]],
        lens,
        txt,
        txt_cu,
        txt_mask,
        vec,
        None,
        None,
        None,
        None,
        1.0,
        False,
        False,
        device,
    )

    # --- PHASE B: Transformer (DiT) ---
    print("\n>> [2/3] Loading MageFlow Transformer directly to XPU...")
    trans_path = os.path.join(
        repo_dir, "transformer", "diffusion_pytorch_model.safetensors"
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model_shell.transformer = model_shell.load_transformer()
    finally:
        torch.set_default_dtype(original_dtype)

    sd = load_file(trans_path, device="xpu")
    model_shell.transformer.load_state_dict(sd, strict=False, assign=True)
    del sd
    empty_cache_xpu()

    print(f">> Denoising for {num_steps} steps...")
    scheduler = build_scheduler(num_steps, device=device, shift=cfg.static_shift)

    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        for si, t in enumerate(scheduler.timesteps):
            pred = _velocity(
                model_shell.transformer, img, ctx, scheduler.sigmas[si].item()
            )
            img = scheduler.step(pred, t, img, return_dict=False)[0]

    print(">> Unloading Transformer to free VRAM...")
    del model_shell.transformer
    empty_cache_xpu()

    # --- PHASE C: Autoencoder (VAE) ---
    print("\n>> [3/3] Loading Autoencoder...")
    model_shell.vae = model_shell.load_vae()
    model_shell.vae.to(device)
    model_shell.vae.to(dtype)

    print(">> Decoding latent to image...")
    with torch.no_grad(), torch.autocast(device_type="xpu", dtype=dtype):
        out = model_shell.vae.decode(unpack(img.float(), h_, w_))

    out = rearrange(out.clamp(-1, 1), "b c h w -> b h w c")
    out = (127.5 * (out + 1.0)).cpu().byte().numpy()
    image = Image.fromarray(out[0])

    print(">> Unloading Autoencoder...")
    del model_shell.vae
    empty_cache_xpu()

    # --- SAVE ---
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"gen_{timestamp}_{seed}.png")
    image.save(output_path)

    metadata = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "device": "xpu",
        "num_steps": num_steps,
    }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n>> Success! Image saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        prog="run_mage_xpu",
        description="Sequential Low-RAM Mage-Flow Generation for Intel ARC",
    )
    p.add_argument(
        "--model",
        default="microsoft/Mage-Flow-Turbo",
        help="HF Repo ID or local folder name inside /models",
    )
    p.add_argument("--prompt", required=True, help="Your text prompt")
    p.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of denoising steps (4 for Turbo models)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (leave blank for random)"
    )
    args = p.parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(0, 2**31 - 1)
    generate_image(
        prompt=args.prompt, repo_path=args.model, seed=seed, num_steps=args.steps
    )
