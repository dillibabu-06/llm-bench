"""Load a model at a given precision and measure memory, latency and throughput."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .metrics import env_info, mb, summarize, timed

PRECISIONS = ("fp32", "fp16", "int8", "nf4")


@dataclass
class CellConfig:
    """One measured point: a (precision, batch, prompt length) combination."""

    precision: str
    batch_size: int
    prompt_len: int
    new_tokens: int


def pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str, precision: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {PRECISIONS}")
    kwargs = {}
    if precision == "fp32":
        kwargs["dtype"] = torch.float32
    elif precision == "fp16":
        kwargs["dtype"] = torch.float16
    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if precision in ("int8", "nf4") and device != "cuda":
        raise RuntimeError(f"{precision} needs a CUDA GPU (bitsandbytes)")
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return model


def kv_cache_bytes(config, batch_size: int, seq_len: int, bytes_per_elem: int = 2) -> int:
    """Theoretical KV-cache size: 2 (K and V) * layers * kv_heads * head_dim * tokens * batch."""
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return 2 * config.num_hidden_layers * kv_heads * head_dim * seq_len * batch_size * bytes_per_elem


def _free(*objs):
    import torch

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure_cell(model, cell: CellConfig, device: str, repeats: int, warmup: int, seed: int):
    """Measure one cell. Returns a result dict (status 'ok' or 'oom')."""
    import torch

    cuda = device == "cuda"
    cfg = model.config
    g = torch.Generator().manual_seed(seed)
    low = 1000 if cfg.vocab_size > 2000 else 0  # stay clear of special tokens
    ids = torch.randint(low, cfg.vocab_size, (cell.batch_size, cell.prompt_len), generator=g).to(device)
    mask = torch.ones_like(ids)
    pad = getattr(cfg, "pad_token_id", None) or getattr(cfg, "eos_token_id", None) or 0
    if isinstance(pad, list):
        pad = pad[0]

    def generate(n):
        t = {}
        with timed(t):
            with torch.inference_mode():
                model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=n,
                    min_new_tokens=n,  # force full length so EOS can't shorten the run
                    do_sample=False,
                    pad_token_id=pad,
                )
        return t["elapsed"]

    result = {**asdict(cell), "status": "ok"}
    try:
        for _ in range(warmup):
            generate(min(8, cell.new_tokens))
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        ttft, total, decode_tps, total_tps = [], [], [], []
        for _ in range(repeats):
            t1 = generate(1)
            tn = generate(cell.new_tokens)
            decode_time = tn - t1
            ttft.append(t1)
            total.append(tn)
            total_tps.append(cell.batch_size * cell.new_tokens / tn)
            if decode_time > 0 and cell.new_tokens > 1:
                decode_tps.append(cell.batch_size * (cell.new_tokens - 1) / decode_time)
        result["ttft_s"] = summarize(ttft)
        result["e2e_s"] = summarize(total)
        result["decode_tok_per_s"] = summarize(decode_tps)
        result["total_tok_per_s"] = summarize(total_tps)
        if cuda:
            result["peak_alloc_mb"] = mb(torch.cuda.max_memory_allocated())
            result["peak_reserved_mb"] = mb(torch.cuda.max_memory_reserved())
        seq = cell.prompt_len + cell.new_tokens
        result["kv_cache_theoretical_mb"] = mb(kv_cache_bytes(cfg, cell.batch_size, seq))
    except torch.cuda.OutOfMemoryError if cuda else () as exc:  # noqa: PERF203
        result["status"] = "oom"
        result["error"] = str(exc).splitlines()[0]
        torch.cuda.empty_cache()
    return result


def run_benchmark(
    model_name: str,
    precisions,
    batch_sizes,
    prompt_lens,
    new_tokens: int,
    repeats: int,
    warmup: int,
    seed: int,
    out_dir: str,
    log=print,
):
    """Run the full grid. The result file is rewritten after every cell so a Colab
    disconnect never loses finished measurements."""
    import torch

    device = pick_device()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    slug = (Path(model_name).name if Path(model_name).exists() else model_name).replace("/", "_")
    path = out_path / f"{slug}_{time.strftime('%Y%m%d-%H%M%S')}.json"

    doc = {
        "schema_version": SCHEMA_VERSION,
        "llmbench_version": __version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_name,
        "device": device,
        "env": env_info(),
        "settings": {"repeats": repeats, "warmup": warmup, "seed": seed, "new_tokens": new_tokens},
        "runs": [],
    }

    def save():
        path.write_text(json.dumps(doc, indent=2))

    for precision in precisions:
        log(f"\n== loading {model_name} @ {precision} on {device}")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            model = load_model(model_name, precision, device)
        except Exception as exc:  # keep going so one bad precision doesn't kill the sweep
            log(f"   skipped: {exc}")
            doc["runs"].append({"precision": precision, "status": "load_failed", "error": str(exc)})
            save()
            continue
        weights_mb = mb(torch.cuda.memory_allocated()) if device == "cuda" else None
        log(f"   weights: {weights_mb:.0f} MB" if weights_mb is not None else "   (cpu: no GPU memory stats)")
        for bs in batch_sizes:
            for pl in prompt_lens:
                cell = CellConfig(precision, bs, pl, new_tokens)
                res = measure_cell(model, cell, device, repeats, warmup, seed)
                res["weights_mb"] = weights_mb
                if weights_mb is not None and res.get("peak_alloc_mb") is not None:
                    # Peak minus weights = KV cache + activations + temporary buffers.
                    res["runtime_overhead_mb"] = res["peak_alloc_mb"] - weights_mb
                doc["runs"].append(res)
                save()
                log("   " + format_run(res))
        _free(model)
    log(f"\nsaved: {path}")
    return path, doc


def format_run(r) -> str:
    head = f"{r['precision']:>4} bs={r['batch_size']:<3} prompt={r['prompt_len']:<5}"
    if r.get("status") != "ok":
        return f"{head} {r['status'].upper()}"
    peak = f"{r['peak_alloc_mb']:.0f}MB" if r.get("peak_alloc_mb") is not None else "n/a"
    return (
        f"{head} TTFT {r['ttft_s']['median'] * 1000:.0f}ms  "
        f"decode {r['decode_tok_per_s']['median']:.1f} tok/s  peak {peak}"
    )
