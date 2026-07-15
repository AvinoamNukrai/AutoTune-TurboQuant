"""Measurement utilities: device-level VRAM sampling (NVML), per-request
latency extraction from vLLM outputs, and KV-cache-sensitive perplexity.

VRAM — vLLM v1 runs the model in a separate engine-core process, so
`torch.cuda.max_memory_allocated()` in the client process reads 0 (smoke-test
finding #9). We therefore sample *device-level* used memory via NVML from a
background thread, which sees all processes on the GPU.

Perplexity — subtle but critical (SPEC "simulated quantization matches the
mechanism" concern applies to measurement too): during a single prefill pass,
attention runs over the *unquantized* K/V and only then stores them compressed,
so prompt logprobs from one monolithic prefill would NOT reflect cache
quantization at all. We force small chunked-prefill chunks, so every chunk
after the first attends to the quantized cache of earlier chunks, and we score
only tokens past the first chunk. This makes PPL sensitive to the KV-cache
quantization that is actually under test.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# NVML device-level VRAM sampler
# --------------------------------------------------------------------------

class VramSampler:
    """Background thread sampling device used-memory via NVML.

    Usage:
        with VramSampler(device_index=0) as s:
            ... run engine ...
        print(s.peak_gb, s.baseline_gb)
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples_gb: list[float] = []
        self.baseline_gb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None

    def _read_used_gb(self) -> float:
        handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 2**30

    def __enter__(self) -> "VramSampler":
        import pynvml  # noqa: PLC0415 — ships inside the nvidia-ml-py dep of vllm
        self._nvml = pynvml
        pynvml.nvmlInit()
        self.baseline_gb = self._read_used_gb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples_gb.append(self._read_used_gb())
            except Exception:  # noqa: BLE001 — NVML hiccups shouldn't kill the run
                pass
            time.sleep(self.interval_s)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    @property
    def peak_gb(self) -> float:
        return max(self.samples_gb, default=0.0)


# --------------------------------------------------------------------------
# Per-request latency extraction
# --------------------------------------------------------------------------

@dataclass
class RequestLatency:
    ttft_s: float | None      # time to first token (None if metrics unavailable)
    e2e_s: float | None       # end-to-end request latency
    n_output_tokens: int
    tpot_s: float | None = None  # (e2e - ttft) / (n_out - 1)

    def __post_init__(self) -> None:
        if (self.tpot_s is None and self.ttft_s is not None
                and self.e2e_s is not None and self.n_output_tokens > 1):
            self.tpot_s = (self.e2e_s - self.ttft_s) / (self.n_output_tokens - 1)


def extract_latencies(outputs) -> list[RequestLatency]:
    """Pull per-request timing out of vLLM RequestOutput.metrics.

    vLLM populates `RequestOutput.metrics` (arrival_time, first_token_time,
    finished_time) when available; field availability varies by version, so
    every access is defensive — missing fields degrade to None rather than
    crashing the cell.
    """
    lats = []
    for out in outputs:
        n_out = sum(len(c.token_ids) for c in out.outputs)
        m = getattr(out, "metrics", None)
        ttft = e2e = None
        if m is not None:
            arrival = getattr(m, "arrival_time", None)
            first_tok = getattr(m, "first_token_time", None)
            finished = getattr(m, "finished_time", None)
            if arrival is not None and first_tok is not None:
                ttft = first_tok - arrival
            if arrival is not None and finished is not None:
                e2e = finished - arrival
        lats.append(RequestLatency(ttft_s=ttft, e2e_s=e2e, n_output_tokens=n_out))
    return lats


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile; None on empty input."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = (len(vals) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def summarize_latencies(lats: list[RequestLatency]) -> dict:
    def stats(vals: list[float | None]) -> dict:
        clean = [v for v in vals if v is not None]
        if not clean:
            return {"mean": None, "p50": None, "p95": None, "p99": None, "n": 0}
        return {
            "mean": sum(clean) / len(clean),
            "p50": percentile(clean, 0.50),
            "p95": percentile(clean, 0.95),
            "p99": percentile(clean, 0.99),
            "n": len(clean),
        }

    return {
        "ttft_s": stats([l.ttft_s for l in lats]),
        "tpot_s": stats([l.tpot_s for l in lats]),
        "e2e_s": stats([l.e2e_s for l in lats]),
        "total_output_tokens": sum(l.n_output_tokens for l in lats),
    }


# --------------------------------------------------------------------------
# KV-cache-sensitive perplexity
# --------------------------------------------------------------------------

@dataclass
class PPLConfig:
    n_docs: int = 4
    doc_tokens: int = 6144       # long docs so most tokens attend to quantized cache
    score_skip_tokens: int = 512  # skip tokens covered by the first (unquantized) chunk
    chunk_size: int = 512         # must match engine's max_num_batched_tokens


def load_wikitext_docs(cfg: PPLConfig, tokenizer) -> list[str]:
    """Concatenate WikiText-2 test split into n_docs documents of ~doc_tokens."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    ids = tokenizer.encode(text)
    docs = []
    for i in range(cfg.n_docs):
        chunk = ids[i * cfg.doc_tokens:(i + 1) * cfg.doc_tokens]
        if len(chunk) < cfg.doc_tokens // 2:
            break
        docs.append(tokenizer.decode(chunk))
    return docs


def compute_ppl(llm, cfg: PPLConfig) -> dict:
    """Perplexity over WikiText docs, scored only past the first prefill chunk.

    REQUIRES the engine to be launched with chunked prefill and
    max_num_batched_tokens == cfg.chunk_size, otherwise the quantized-cache
    path is never exercised and PPL is blind to the config under test.
    """
    from vllm import SamplingParams  # noqa: PLC0415

    tokenizer = llm.get_tokenizer()
    docs = load_wikitext_docs(cfg, tokenizer)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outputs = llm.generate(docs, sp)

    total_nll, total_tokens = 0.0, 0
    for out in outputs:
        plps = out.prompt_logprobs or []
        for tok_logprobs in plps[cfg.score_skip_tokens:]:
            if not tok_logprobs:
                continue
            # prompt_logprobs entry: {token_id: Logprob}; the actual token's
            # logprob is the entry whose rank is not None / matches the token.
            lp = next(iter(tok_logprobs.values()))
            total_nll -= lp.logprob
            total_tokens += 1

    ppl = math.exp(total_nll / total_tokens) if total_tokens else None
    return {"ppl": ppl, "scored_tokens": total_tokens, "n_docs": len(docs)}
