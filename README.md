# llmbench

Benchmark and profile quantized LLM inference: **per-component GPU memory, latency and throughput** across precisions (fp16, int8, nf4), batch sizes and prompt lengths.

## What it measures

| Metric | Meaning |
|---|---|
| `ttft_s` | Time to first token (a 1-token generation, prefill dominated) |
| `decode_tok_per_s` | Steady-state generation speed after the first token, summed over the batch |
| `e2e_s` / `total_tok_per_s` | Whole request latency and overall throughput |
| `weights_mb` | GPU memory right after loading (quantized weights + non-quantized layers) |
| `peak_alloc_mb` | Peak allocated GPU memory during generation |
| `runtime_overhead_mb` | `peak - weights`: KV cache + activations + temporary buffers |
| `kv_cache_theoretical_mb` | KV cache size computed from the model config, to compare with the measured overhead |
| `status` | `ok`, `oom` (out of memory at this point) or `load_failed` |

Every measurement is repeated (`--repeats`), preceded by warmup, and reported as median / mean / stdev / p95 / p99. Decoding is greedy with forced length (`min_new_tokens`) so runs are comparable. Prompts are fixed-seed synthetic token ids, so results do not depend on a dataset.

## Quick start (Colab, T4)

```python
!git clone https://github.com/<you>/llmbench && cd llmbench && pip install -q -e ".[quant]"
!llmbench run --model Qwen/Qwen2.5-1.5B-Instruct \
    --precision fp16 int8 nf4 --batch 1 4 --prompt-len 128 1024 --new-tokens 128
!llmbench show results/<file>.json
```

Results are written to `results/` after every cell, so a disconnected session keeps finished data. Mount Google Drive and pass `--out /content/drive/MyDrive/llmbench_results` to keep them across sessions.

## Local smoke test (CPU, no downloads)

See `tests/` and run `pytest`. CPU runs report latency only; GPU memory metrics need CUDA.

## Result file

JSON with `schema_version`, GPU and library versions, settings, and one entry per measured cell. Always compare runs with the same `env` block.

## Roadmap

- [ ] Quality check next to speed (perplexity, lm-evaluation-harness)
- [ ] Plots and comparison tables
- [ ] OpenAI-compatible server mode, so vLLM / llama.cpp / TGI can be benchmarked with the same tool
- [ ] Open-loop load generator (Poisson arrivals) with p50/p95/p99 vs request rate and goodput
- [ ] Validate numbers against vLLM's benchmark scripts
- [ ] Dockerfile, CI smoke test, PyPI release

## Known limits

- TTFT is measured as a separate 1-token run, not from a streamer, so it includes generate() setup overhead.
- Peak memory comes from the PyTorch allocator; it does not include CUDA context or non-PyTorch allocations.
- bitsandbytes int8/nf4 kernels are not the fastest quantized kernels; compare with GPTQ/AWQ later.
