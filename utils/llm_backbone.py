import os
import json
from dataclasses import dataclass
from typing import Optional, List

import torch


KNOWN_MODELS = {
    "gpt2":              {"hidden_size": 768,  "num_layers": 12, "best_layer": 11},
    "gpt2-medium":       {"hidden_size": 1024, "num_layers": 24, "best_layer": 20},
    "gpt2-large":        {"hidden_size": 1280, "num_layers": 36, "best_layer": 30},
    "gpt2-xl":           {"hidden_size": 1600, "num_layers": 48, "best_layer": 40},

    "Qwen/Qwen2-0.5B":  {"hidden_size": 896,  "num_layers": 24, "best_layer": 20},
    "Qwen/Qwen2-1.5B":  {"hidden_size": 1536, "num_layers": 28, "best_layer": 24},
    "Qwen/Qwen2-7B":    {"hidden_size": 3584, "num_layers": 28, "best_layer": 24},
    "Qwen/Qwen2.5-0.5B": {"hidden_size": 896,  "num_layers": 24, "best_layer": 20},
    "Qwen/Qwen2.5-1.5B": {"hidden_size": 1536, "num_layers": 28, "best_layer": 24},
    "Qwen/Qwen2.5-3B":  {"hidden_size": 2048, "num_layers": 36, "best_layer": 30},
    "Qwen/Qwen2.5-7B":  {"hidden_size": 3584, "num_layers": 28, "best_layer": 24},
    "Qwen/Qwen2.5-7B-Instruct": {"hidden_size": 3584, "num_layers": 28, "best_layer": 24},

    "meta-llama/Llama-2-7b-hf":  {"hidden_size": 4096, "num_layers": 32, "best_layer": 28},
    "meta-llama/Llama-3.2-1B":   {"hidden_size": 2048, "num_layers": 16, "best_layer": 13},
    "meta-llama/Llama-3.2-3B":   {"hidden_size": 3072, "num_layers": 28, "best_layer": 24},

    "bert-base-uncased":    {"hidden_size": 768,  "num_layers": 12, "best_layer": 11},
    "bert-large-uncased":   {"hidden_size": 1024, "num_layers": 24, "best_layer": 20},

    "facebook/opt-125m":    {"hidden_size": 768,  "num_layers": 12, "best_layer": 10},
    "facebook/opt-350m":    {"hidden_size": 512,  "num_layers": 24, "best_layer": 20},
    "facebook/opt-1.3b":    {"hidden_size": 2048, "num_layers": 24, "best_layer": 20},
}


@dataclass
class LLMBackbone:
    model: torch.nn.Module
    tokenizer: object
    hidden_size: int
    num_layers: int
    best_layer: int
    model_name: str

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model = self.model.to(device)
        return self


def _normalize_model_name(name: str) -> str:
    return name.rstrip("/").strip()


def _detect_from_config_json(model_path: str) -> Optional[dict]:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    hidden_size = (
        config.get("hidden_size") or config.get("d_model") or
        config.get("n_embd") or config.get("dim") or config.get("model_dim")
    )
    num_layers = (
        config.get("num_hidden_layers") or config.get("n_layer") or
        config.get("num_layers") or config.get("n_layers")
    )
    if hidden_size is None:
        return None
    if num_layers is None:
        num_layers = 12
    best_layer = max(0, num_layers - 2)
    return {"hidden_size": int(hidden_size), "num_layers": int(num_layers), "best_layer": int(best_layer)}


def _detect_from_auto_config(model_name: str) -> dict:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'n_embd', None) or getattr(config, 'd_model', None)
    if hidden_size is None:
        raise ValueError(f"Cannot detect hidden_size for '{model_name}'. Add to KNOWN_MODELS.")
    num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', None) or 12
    best_layer = max(0, num_layers - 2)
    return {"hidden_size": int(hidden_size), "num_layers": int(num_layers), "best_layer": int(best_layer)}


def get_model_info(model_name_or_path: str) -> dict:
    name = _normalize_model_name(model_name_or_path)


    if name in KNOWN_MODELS:
        info = KNOWN_MODELS[name]
        print(f"[LLM] Quick lookup: {name} -> hidden={info['hidden_size']}")
        return info


    basename = os.path.basename(name)
    for known_name, info in KNOWN_MODELS.items():
        if basename == known_name.split("/")[-1]:
            print(f"[LLM] Matched local path '{name}' to known model '{known_name}' -> hidden={info['hidden_size']}")
            return info


    if os.path.isdir(name):
        info = _detect_from_config_json(name)
        if info is not None:
            print(f"[LLM] Local config: {name} -> hidden={info['hidden_size']}")
            return info


    print(f"[LLM] Auto-detecting: {name}")
    info = _detect_from_auto_config(name)
    print(f"[LLM] Detected: hidden={info['hidden_size']}, layers={info['num_layers']}")
    return info


def get_llm_hidden_size(model_name_or_path: str) -> int:
    return get_model_info(model_name_or_path)["hidden_size"]


def load_llm_backbone(
    model_name_or_path: str,
    device: str = "cpu",
    layer: Optional[int] = None,
    freeze: bool = True,
) -> LLMBackbone:
    from transformers import AutoTokenizer, AutoModel

    name = _normalize_model_name(model_name_or_path)
    info = get_model_info(name)

    print(f"[LLM] Loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"[LLM] pad_token = eos_token ('{tokenizer.eos_token}')")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("[LLM] Added [PAD] token")

    print(f"[LLM] Loading model: {name}")


    _dt = os.environ.get("FUCAL_LLM_DTYPE", "").strip().lower()
    _map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}
    if _dt in _map:
        print(f"[LLM] compute dtype: {_dt}")
        model = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=_map[_dt])
    else:
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    model = model.to(device)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("[LLM] Model frozen")

    best_layer = layer if layer is not None else info["best_layer"]

    backbone = LLMBackbone(
        model=model, tokenizer=tokenizer,
        hidden_size=info["hidden_size"], num_layers=info["num_layers"],
        best_layer=best_layer, model_name=name,
    )
    print(f"[LLM] Ready: hidden={info['hidden_size']}, extract_layer={best_layer}")
    return backbone


def encode_text(
    backbone: LLMBackbone,
    text: str,
    max_length: int = 128,
    pooling: str = "mean",
) -> torch.Tensor:
    tokens = backbone.tokenizer(
        text, return_tensors='pt', padding=False,
        truncation=True, max_length=max_length,
    )
    device = backbone.device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = backbone.model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[backbone.best_layer]

    if pooling == "mean":
        emb = hidden[0].mean(dim=0)
    elif pooling == "last":
        emb = hidden[0, -1, :]
    elif pooling == "first":
        emb = hidden[0, 0, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    return emb


def encode_text_batch(
    backbone: LLMBackbone,
    texts: List[str],
    max_length: int = 128,
    pooling: str = "mean",
    batch_size: int = 32,
    show_progress: bool = False,
) -> torch.Tensor:
    if len(texts) == 0:
        return torch.zeros(0, backbone.hidden_size)

    device = backbone.device
    tokenizer = backbone.tokenizer
    all_embeddings = []


    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    try:
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]


            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = backbone.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[backbone.best_layer]


                if pooling == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    summed = (hidden.float() * mask_expanded).sum(dim=1)
                    counts = mask_expanded.sum(dim=1).clamp(min=1e-8)
                    batch_embs = summed / counts
                elif pooling == "last":

                    seq_lens = attention_mask.sum(dim=1) - 1
                    batch_embs = hidden[torch.arange(len(batch_texts), device=device), seq_lens].float()
                elif pooling == "first":

                    first_real = attention_mask.argmax(dim=1)
                    batch_embs = hidden[torch.arange(len(batch_texts), device=device), first_real].float()
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")


                batch_embs = batch_embs / (batch_embs.norm(dim=-1, keepdim=True) + 1e-8)

            all_embeddings.append(batch_embs.cpu())

            if show_progress and (start // batch_size) % 10 == 0:
                print(f"  [Batch encode] {end}/{len(texts)} done")

    finally:

        tokenizer.padding_side = original_padding_side

    return torch.cat(all_embeddings, dim=0)


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python llm_backbone.py <model_name_or_path> [--bench]")
        sys.exit(0)

    model_name = sys.argv[1]
    do_bench = "--bench" in sys.argv

    print(f"\n{'='*60}")
    print(f" LLM Backbone Self-Check: {model_name}")
    print(f"{'='*60}\n")

    info = get_model_info(model_name)
    print(f"  hidden_size: {info['hidden_size']}")
    print(f"  num_layers:  {info['num_layers']}")
    print(f"  best_layer:  {info['best_layer']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_llm_backbone(model_name, device=device)


    test_text = "HUFL (high useful load). Trend: rising. Level: High. Stats: mean_z=0.5."
    emb = encode_text(backbone, test_text)
    print(f"\n  Single encode: shape={emb.shape}, norm={emb.norm():.4f}")


    test_texts = [
        f"V{i} (variable {i}). Level: {'High' if i%2==0 else 'Low'}. "
        f"Stats: mean_z={i*0.1:.1f}, std_z={0.5:.1f}."
        for i in range(100)
    ]


    if do_bench:
        print(f"\n  Benchmarking 100 prompts (one by one vs batched)...")

        t0 = time.time()
        for t in test_texts:
            encode_text(backbone, t)
        t_seq = time.time() - t0

        for bs in [8, 16, 32, 64]:
            t0 = time.time()
            embs = encode_text_batch(backbone, test_texts, batch_size=bs)
            t_batch = time.time() - t0
            speedup = t_seq / t_batch
            print(f"    Sequential: {t_seq:.2f}s | Batch(bs={bs}): {t_batch:.2f}s | "
                  f"Speedup: {speedup:.1f}x | Output: {embs.shape}")

    else:
        embs = encode_text_batch(backbone, test_texts[:10], batch_size=8)
        print(f"  Batch encode (10 texts): shape={embs.shape}")

    print(f"\n   d_llm = {backbone.hidden_size}")
    print(f"{'='*60}")