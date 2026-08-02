from __future__ import annotations
import json
import os
import random
import gc
import torch
import torch.nn.functional as F
import sys
import numpy as np
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
import transformers

# ==============================================================================
# 1. HF TRANSFORMERS SDPA PATCH
# Forces the base text encoder to use native PyTorch SDPA.
# ==============================================================================
transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = lambda *args, **kwargs: "sdpa"

# ==============================================================================
# 2. BULLETPROOF ATTENTION INJECTION
# We must apply this patch BEFORE importing MageFlowModel so that all internal
# layers inherit our safe XPU math instead of the broken NVIDIA kernel.
# ==============================================================================
def _mock_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    out = torch.empty_like(q)
    cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    num_groups = q.shape[1] // k.shape[1]  
    
    for i in range(len(cu_q) - 1):
        sq, eq, sk, ek = cu_q[i], cu_q[i+1], cu_k[i], cu_k[i+1]
        qi, ki, vi = q[sq:eq].transpose(0, 1).unsqueeze(0), k[sk:ek].transpose(0, 1).unsqueeze(0), v[sk:ek].transpose(0, 1).unsqueeze(0)
        if num_groups > 1:
            ki = torch.repeat_interleave(ki, repeats=num_groups, dim=1)
            vi = torch.repeat_interleave(vi, repeats=num_groups, dim=1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        out[sq:eq] = oi.squeeze(0).transpose(0, 1)
    return out

# Import ONLY the backend layers and hard-patch them everywhere
import mage_flow.models.modules._attn_backend as attn_backend
import mage_flow.models.modules.mage_layers as mage_layers

attn_backend._RESOLVED_FN = _mock_flash_attn_varlen_func
attn_backend.flash_attn_varlen_func = _mock_flash_attn_varlen_func
mage_layers.flash_attn_varlen_func = _mock_flash_attn_varlen_func

# ==============================================================================
# 3. INTERNAL MAGE FLOW IMPORTS
# Now that the backend is safe, we can import the model classes.
# ==============================================================================
from .models.mage_flow import MageFlowModel, ModelConfig
from .models.utils import PROMPT_TEMPLATE, get_noise, unpack
from .models.modules.mage_text import make_refusal_image
from .models.modules.mage_latent import encode_noise, resolve_gs_key

def _clean_vram(device):
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

def build_scheduler(num_steps: int, device=None, shift: float = 6.0):
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    scheduler.set_timesteps(sigmas=torch.linspace(1.0, 1.0 / num_steps, num_steps).tolist(), device=device)
    return scheduler

def _as_list(val, default, n):
    if val is None: return [default] * n
    if isinstance(val, (list, tuple)): return list(val)
    return [val] * n

def _lens_to_cu(lens, device):
    t = torch.tensor(lens, device=device, dtype=torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(t, dim=0, dtype=torch.int32)])

def _make_divisible_by_16(size: int) -> int:
    return max(16, 16 * (size // 16))

def _compute_aspect_ratio_size(pil_img: Image.Image, max_size: int):
    w, h = pil_img.size
    if h >= w: new_h, new_w = max_size, int(round(w * max_size / h))
    else: new_w, new_h = max_size, int(round(h * max_size / w))
    return _make_divisible_by_16(new_h), _make_divisible_by_16(new_w)

def _edit_target_size(pil_img: Image.Image, max_size, height, width):
    if height and width: return _make_divisible_by_16(height), _make_divisible_by_16(width)
    if max_size: return _compute_aspect_ratio_size(pil_img, max_size)
    return _compute_aspect_ratio_size(pil_img, max(pil_img.size))

def _decode_one(model, tokens, height, width, dev):
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model.vae.decode(unpack(tokens.float(), height, width))
    out = rearrange(out.clamp(-1, 1), "b c h w -> b h w c")
    return Image.fromarray((127.5 * (out + 1.0)).cpu().byte().numpy()[0])

def _build_pack_ctx(img_ids, img_cu, img_shapes, img_lens, txt, txt_cu, txt_mask, vec, neg_txt, neg_cu, neg_mask, neg_vec, cfg, renormalization, batch_cfg, device):
    na = len(img_lens)
    ctx = {
        "na": na, "cfg": cfg, "renorm": renormalization, "batch_cfg": batch_cfg, "has_neg": neg_txt is not None,
        "img_ids": img_ids, "img_cu": img_cu, "img_shapes": img_shapes, "img_max": int(max(img_lens)),
        "txt": txt, "txt_ids": torch.zeros(1, txt.shape[1], 3, device=device), "txt_cu": txt_cu, "txt_mask": txt_mask, "vec": vec,
        "txt_max": int((txt_cu[1:] - txt_cu[:-1]).max().item()),
    }
    if neg_txt is not None:
        ctx.update({"neg_txt": neg_txt, "neg_ids": torch.zeros(1, neg_txt.shape[1], 3, device=device), "neg_cu": neg_cu, "neg_mask": neg_mask, "neg_vec": neg_vec, "neg_max": int((neg_cu[1:] - neg_cu[:-1]).max().item())})
    if batch_cfg and neg_txt is not None:
        pos_lens, neg_lens = (txt_cu[1:] - txt_cu[:-1]).tolist(), (neg_cu[1:] - neg_cu[:-1]).tolist()
        d_txt = torch.cat([txt, neg_txt], dim=1)
        ctx.update({
            "d_img_ids": torch.cat([img_ids, img_ids], dim=1), "d_img_cu": _lens_to_cu(list(img_lens) + list(img_lens), device), "d_img_shapes": [img_shapes[0] + img_shapes[0]],
            "d_txt": d_txt, "d_txt_ids": torch.zeros(1, d_txt.shape[1], 3, device=device), "d_txt_cu": _lens_to_cu(pos_lens + neg_lens, device),
            "d_txt_mask": torch.ones(1, d_txt.shape[1], device=device), "d_vec": torch.cat([vec, neg_vec], dim=0), "d_txt_max": int(max(pos_lens + neg_lens)),
        })
    return ctx

def _velocity(transformer, img, ctx, sigma):
    dev, na = img.device, ctx["na"]
    def _fwd(x, n, img_ids, img_cu, img_max, img_shapes, txt, txt_ids, txt_cu, txt_mask, txt_max, vec):
        return transformer(img=x, txt=txt, timesteps=torch.full((n,), sigma, dtype=x.dtype, device=dev), img_shapes=img_shapes, img_cu_seqlens=img_cu, txt_cu_seqlens=txt_cu)
    if not ctx["has_neg"]: return _fwd(img, na, ctx["img_ids"], ctx["img_cu"], ctx["img_max"], ctx["img_shapes"], ctx["txt"], ctx["txt_ids"], ctx["txt_cu"], ctx["txt_mask"], ctx["txt_max"], ctx["vec"])
    if ctx["batch_cfg"]:
        out = _fwd(torch.cat([img, img], dim=1), 2 * na, ctx["d_img_ids"], ctx["d_img_cu"], ctx["img_max"], ctx["d_img_shapes"], ctx["d_txt"], ctx["d_txt_ids"], ctx["d_txt_cu"], ctx["d_txt_mask"], ctx["d_txt_max"], ctx["d_vec"])
        cond, unc = out[:, :img.shape[1], :], out[:, img.shape[1]:, :]
    else:
        cond = _fwd(img, na, ctx["img_ids"], ctx["img_cu"], ctx["img_max"], ctx["img_shapes"], ctx["txt"], ctx["txt_ids"], ctx["txt_cu"], ctx["txt_mask"], ctx["txt_max"], ctx["vec"])
        unc = _fwd(img, na, ctx["img_ids"], ctx["img_cu"], ctx["img_max"], ctx["img_shapes"], ctx["neg_txt"], ctx["neg_ids"], ctx["neg_cu"], ctx["neg_mask"], ctx["neg_max"], ctx["neg_vec"])
    comb = unc + ctx["cfg"] * (cond - unc)
    return comb * (torch.norm(cond, dim=-1, keepdim=True) / (torch.norm(comb, dim=-1, keepdim=True) + 1e-6)) if ctx["renorm"] else comb

def _encode_texts_packed(model, prompts, template, drop_idx, device):
    tok = model.txt_enc.tokenizer
    ids_list = [tok(template.format(p), max_length=tok.model_max_length + drop_idx, truncation=True, return_tensors="pt").input_ids.squeeze(0) for p in prompts]
    res = model.txt_enc(torch.cat(ids_list).to(device), _lens_to_cu([int(t.numel()) for t in ids_list], device), drop_idx_override=drop_idx)
    return res["txt"], res["vec"], res["txt_seq_lens"].tolist()

def _slice_packed(txt_flat, vec, lens, start, count, device):
    seg_lens = lens[start:start + count]
    txt = txt_flat[sum(lens[:start]):sum(lens[:start]) + sum(seg_lens)].reshape(1, -1, txt_flat.shape[-1]).to(device)
    return txt, _lens_to_cu(seg_lens, device), torch.ones(1, txt.shape[1], device=device), vec[start:start + count].to(device)

def _encode_edits_packed(model, ref_pils_per_sample, instructions, template, drop_idx, device):
    proc, ids_list, pv_list, thw_list = model.txt_enc.processor, [], [], []
    for ref_pils, instr in zip(ref_pils_per_sample, instructions):
        formatted = template.format("".join(f"Image {j}: <|vision_start|><|image_pad|><|vision_end|>" for j in range(1, len(ref_pils) + 1)) + instr)
        vl = proc(text=[formatted], images=list(ref_pils), padding=True, return_tensors="pt")
        ids_list.append(vl["input_ids"].to(device).squeeze(0))
        if vl.get("pixel_values") is not None:
            pv_list.append(vl["pixel_values"].to(device)); thw_list.append(vl["image_grid_thw"].to(device))
    cu = _lens_to_cu([int(t.numel()) for t in ids_list], device)
    inputs = {"input_ids": torch.cat(ids_list), "cu_seqlens": cu}
    if pv_list: inputs.update({"pixel_values": torch.cat(pv_list, dim=0), "image_grid_thw": torch.cat(thw_list, dim=0)})
    res = model.txt_enc(inputs["input_ids"], cu, inputs=inputs, drop_idx_override=drop_idx)
    return res["txt"], res["vec"], res["txt_seq_lens"].tolist()

def _preprocess_ref_image(pil_img, height, width, device):
    from torchvision.transforms import functional as TF
    t = TF.normalize(TF.to_tensor(TF.resize(pil_img.convert("RGB"), [height, width], interpolation=TF.InterpolationMode.BICUBIC)), [0.5]*3, [0.5]*3)
    return t.to(device)

class MageFlowPipeline:
    def __init__(self, repo_dir: str, config: ModelConfig, device: str = "xpu"):
        self.repo_dir = repo_dir
        self.cfg = config
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        
        # Shell initialization to prevent 16GB RAM spike
        self.model_shell = MageFlowModel.__new__(MageFlowModel)
        torch.nn.Module.__init__(self.model_shell)
        self.model_shell.config = self.cfg

    @classmethod
    def from_pretrained(cls, repo_dir: str, device: str = "xpu"):
        if not os.path.isdir(repo_dir):
            try: repo_dir = snapshot_download(repo_id=repo_dir)
            except Exception as e: raise RuntimeError(f"Could not load {repo_dir}: {e}")
            
        mi = json.load(open(os.path.join(repo_dir, "model_index.json")))
        tcfg = json.load(open(os.path.join(repo_dir, "transformer", "config.json")))
        structure = {k: v for k, v in tcfg.items() if k not in {"_class_name", "txt_max_length", "max_sequence_length", "param_dtype", "packing", "schedule_mode", "static_shift", "use_time_shift", "rope_type", "apply_text_rotary_emb", "mlp_ratio", "depth_single_blocks", "theta", "qkv_bias", "guidance_embed", "vec_in_dim", "vec_type", "time_type", "double_block_type"}}
        
        v_src, t_src = mi.get("_vae_source"), mi.get("_text_encoder_path")
        cfg = ModelConfig(
            vae_path=v_src if os.path.isabs(v_src) else os.path.join(repo_dir, v_src),
            txt_enc_path=t_src if os.path.isabs(t_src) else os.path.join(repo_dir, t_src),
            model_structure=structure, txt_max_length=tcfg.get("txt_max_length", 2048),
            packing=tcfg.get("packing", True), static_shift=tcfg.get("static_shift", 6.0)
        )
        return cls(repo_dir, cfg, device)

    def generate(self, prompts, neg_prompts=None, seeds=None, steps=30, cfg=5.0, heights=None, widths=None, prompt_template="mage-flow", static_shift=None, gs_key=None, renormalization=False, batch_cfg=True):
        if isinstance(prompts, str): prompts = [prompts]
        n = len(prompts)
        neg_prompts, seeds, heights, widths = _as_list(neg_prompts, " ", n), _as_list(seeds, 42, n), _as_list(heights, 1024, n), _as_list(widths, 1024, n)
        tmpl, drop_idx = (info := PROMPT_TEMPLATE.get(prompt_template, {})).get("template", "{}"), int(info.get("start_idx", 0))
        results, active = [None] * n, []

        print(">> [1/3] Loading Text Encoder...")
        self.model_shell.txt_enc = self.model_shell.load_text_enc().to(self.device, self.dtype)
        for i in range(n):
            if seeds[i] == -1: seeds[i] = random.randint(0, 2**32 - 1)
            v = self.model_shell.txt_enc.screen_text(prompts[i])
            if v.violates: results[i] = make_refusal_image(v, height=_make_divisible_by_16(heights[i]), width=_make_divisible_by_16(widths[i]))
            else: active.append(i)
            
        if active:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
                txt_flat, vec_all, lens_t = _encode_texts_packed(self.model_shell, [prompts[i] for i in active] + ([neg_prompts[i] or " " for i in active] if cfg > 1.0 else []), tmpl, drop_idx, self.device)
                txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, len(active), self.device)
                neg_txt, neg_cu, neg_mask, neg_vec = _slice_packed(txt_flat, vec_all, lens_t, len(active), len(active), self.device) if cfg > 1.0 else (None, None, None, None)
        del self.model_shell.txt_enc
        _clean_vram(self.device)
        if not active: return results

        print(">> Preparing latents...")
        img_list, ids_list, lens, shapes, hw = [], [], [], [], []
        for i in active:
            h_, w_ = _make_divisible_by_16(heights[i]), _make_divisible_by_16(widths[i])
            torch.manual_seed(seeds[i])
            x = encode_noise(tuple(get_noise(1, 128, h_, w_, self.device, self.dtype, seeds[i]).shape[1:]), key=resolve_gs_key(gs_key), seed=seeds[i], device=self.device, dtype=self.dtype)
            _, _, gh, gw = x.shape
            img_list.append(rearrange(x, "b c h w -> b (h w) c")[0])
            ids = torch.zeros(gh, gw, 3, device=self.device)
            ids[..., 1], ids[..., 2] = ids[..., 1] + torch.arange(gh, device=self.device)[:, None], ids[..., 2] + torch.arange(gw, device=self.device)[None, :]
            ids_list.append(rearrange(ids, "h w c -> (h w) c"))
            lens.append(gh * gw); shapes.append((1, gh, gw)); hw.append((h_, w_))
            
        img, img_ids, img_cu = torch.cat(img_list, 0).unsqueeze(0), torch.cat(ids_list, 0).unsqueeze(0), _lens_to_cu(lens, self.device)
        ctx = _build_pack_ctx(img_ids, img_cu, [shapes], lens, txt, txt_cu, txt_mask, vec, neg_txt, neg_cu, neg_mask, neg_vec, cfg, renormalization, batch_cfg, self.device)

        print(">> [2/3] Loading Transformer...")
        orig_dt = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try: self.model_shell.transformer = self.model_shell.load_transformer()
        finally: torch.set_default_dtype(orig_dt)
        self.model_shell.transformer.load_state_dict(load_file(os.path.join(self.repo_dir, "transformer", "diffusion_pytorch_model.safetensors"), device=str(self.device)), strict=False, assign=True)

        scheduler = build_scheduler(steps, self.device, static_shift if static_shift else self.cfg.static_shift)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for si, t in enumerate(scheduler.timesteps):
                img = scheduler.step(_velocity(self.model_shell.transformer, img, ctx, scheduler.sigmas[si].item()), t, img, return_dict=False)[0]
        del self.model_shell.transformer
        _clean_vram(self.device)

        print(">> [3/3] Loading Autoencoder & Decoding...")
        self.model_shell.vae = self.model_shell.load_vae().to(self.device, self.dtype)
        off = 0
        for k, i in enumerate(active):
            L, (h_, w_) = lens[k], hw[k]
            results[i] = _decode_one(self.model_shell, img[:, off:off + L, :], h_, w_, self.device)
            off += L
        del self.model_shell.vae
        _clean_vram(self.device)
        return results

    def edit(self, prompts, ref_images, neg_prompts=None, seeds=None, steps=30, cfg=5.0, max_size=None, heights=None, widths=None, prompt_template="mage-flow-edit", static_shift=None, gs_key=None, vl_cond_long_edge=384, renormalization=False, batch_cfg=True):
        if isinstance(prompts, str): prompts, ref_images = [prompts], [ref_images]
        n = len(prompts)
        neg_prompts, seeds, heights, widths = _as_list(neg_prompts, " ", n), _as_list(seeds, 42, n), _as_list(heights, None, n), _as_list(widths, None, n)
        tmpl, drop_idx = (info := PROMPT_TEMPLATE.get(prompt_template, {})).get("template", "{}"), int(info.get("start_idx", 0))
        pils_per_sample = [[(Image.open(x) if isinstance(x, str) else x).convert("RGB") for x in (r if isinstance(r, (list, tuple)) else [r])] for r in ref_images]
        results, res_hw, active = [None] * n, [None] * n, []

        print(">> [1/4] Loading Text Encoder (Screening & Prompting)...")
        self.model_shell.txt_enc = self.model_shell.load_text_enc().to(self.device, self.dtype)
        for i in range(n):
            res_hw[i] = _edit_target_size(pils_per_sample[i][0], max_size, heights[i], widths[i])
            if seeds[i] == -1: seeds[i] = random.randint(0, 2**32 - 1)
            v = self.model_shell.txt_enc.screen_edit(prompts[i], pils_per_sample[i])
            if v.violates: results[i] = make_refusal_image(v, height=res_hw[i][0], width=res_hw[i][1])
            else: active.append(i)

        if active:
            edit_refs = [[(p.resize((max(1, int(round(w * vl_cond_long_edge / max(w, h)))), max(1, int(round(h * vl_cond_long_edge / max(w, h))))), Image.BICUBIC) if vl_cond_long_edge and max(p.size) > vl_cond_long_edge else p) for p in pils_per_sample[i]] for i, p in enumerate(pils_per_sample) if i in active]
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
                txt_flat, vec_all, lens_t = _encode_edits_packed(self.model_shell, edit_refs * 2 if cfg > 1.0 else edit_refs, [prompts[i] for i in active] + ([neg_prompts[i] or " " for i in active] if cfg > 1.0 else []), tmpl, drop_idx, self.device)
                txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, len(active), self.device)
                neg_txt, neg_cu, neg_mask, neg_vec = _slice_packed(txt_flat, vec_all, lens_t, len(active), len(active), self.device) if cfg > 1.0 else (None, None, None, None)
        del self.model_shell.txt_enc
        _clean_vram(self.device)
        if not active: return results

        print(">> [2/4] Loading Autoencoder (Encoding References)...")
        self.model_shell.vae = self.model_shell.load_vae().to(self.device, self.dtype)
        targets, refs, ids_list, shape_seq, samp_lens, tgt_lens, hw, target_idx_parts, off = [], [], [], [], [], [], [], [], 0
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for i in active:
                h_, w_ = res_hw[i]
                torch.manual_seed(seeds[i])
                ref_tok, ref_shapes, ref_ids = self.model_shell.compute_vae_encodings([_preprocess_ref_image(p, h_, w_, self.device) for p in pils_per_sample[i]], with_ids=True)
                x = encode_noise(tuple(get_noise(1, 128, h_, w_, self.device, self.dtype, seeds[i]).shape[1:]), key=resolve_gs_key(gs_key), seed=seeds[i], device=self.device, dtype=self.dtype)
                _, _, gh, gw = x.shape
                tgt_ids = rearrange(torch.zeros(gh, gw, 3, device=self.device).index_put_((..., 1), torch.arange(gh, device=self.device)[:, None]).index_put_((..., 2), torch.arange(gw, device=self.device)[None, :]), "h w c -> (h w) c").unsqueeze(0)
                tgt, lt, lr = rearrange(x, "b c h w -> b (h w) c"), x.shape[2]*x.shape[3], ref_tok.shape[1]
                targets.append(tgt); refs.append(ref_tok.to(self.dtype)); ids_list.append(torch.cat([tgt_ids, ref_ids.to(self.device)], dim=1)[0])
                shape_seq.append((1, gh, gw)); shape_seq.extend(s[0] for s in ref_shapes)
                samp_lens.append(lt + lr); tgt_lens.append(lt); hw.append((h_, w_))
                target_idx_parts.append(torch.arange(off, off + lt, device=self.device))
                off += lt + lr
        del self.model_shell.vae
        _clean_vram(self.device)

        ctx = _build_pack_ctx(torch.cat(ids_list, 0).unsqueeze(0), _lens_to_cu(samp_lens, self.device), [shape_seq], samp_lens, txt, txt_cu, txt_mask, vec, neg_txt, neg_cu, neg_mask, neg_vec, cfg, renormalization, batch_cfg, self.device)
        target_idx = torch.cat(target_idx_parts)

        print(">> [3/4] Loading Transformer...")
        orig_dt = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try: self.model_shell.transformer = self.model_shell.load_transformer()
        finally: torch.set_default_dtype(orig_dt)
        self.model_shell.transformer.load_state_dict(load_file(os.path.join(self.repo_dir, "transformer", "diffusion_pytorch_model.safetensors"), device=str(self.device)), strict=False, assign=True)

        scheduler = build_scheduler(steps, self.device, static_shift if static_shift else self.cfg.static_shift)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for si, t in enumerate(scheduler.timesteps):
                stepped = scheduler.step(_velocity(self.model_shell.transformer, torch.cat([x for p in zip(targets, refs) for x in p], dim=1), ctx, scheduler.sigmas[si].item())[:, target_idx, :], t, torch.cat(targets, dim=1), return_dict=False)[0]
                targets, o = [], 0
                for lt in tgt_lens: targets.append(stepped[:, o:o + lt, :]); o += lt
        del self.model_shell.transformer
        _clean_vram(self.device)

        print(">> [4/4] Loading Autoencoder (Decoding)...")
        self.model_shell.vae = self.model_shell.load_vae().to(self.device, self.dtype)
        for k, i in enumerate(active):
            results[i] = _decode_one(self.model_shell, targets[k], hw[k][0], hw[k][1], self.device)
        del self.model_shell.vae
        _clean_vram(self.device)
        return results