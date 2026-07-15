"""Workload trace generation for the three profiles (Chat / RAG / Batch).

Traces are generated deterministically from a seed, then frozen to JSON under
data/traces/. Every benchmark cell replays the identical frozen trace, so all
comparisons are paired per-request (SPEC §5.1).

Prompt content is synthetic but shaped to match each profile's token profile
(SPEC §4). Content variety comes from a word bank + templates so that requests
differ (no accidental prefix-cache hits between requests) while remaining
fully reproducible.

CLI:
    python -m src.workloads --out data/traces --seed 20260714
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

TOPICS = [
    "photosynthesis", "the French Revolution", "binary search trees",
    "plate tectonics", "the immune system", "supply and demand",
    "neural networks", "the water cycle", "Roman aqueducts", "black holes",
    "fermentation", "the printing press", "ocean currents", "game theory",
    "the Silk Road", "semiconductors", "coral reefs", "the Marshall Plan",
    "public key cryptography", "photosynthetic bacteria", "jet engines",
    "the Great Depression", "DNA replication", "sorting algorithms",
    "monetary policy", "volcanic eruptions", "medieval guilds",
    "reinforcement learning", "antibiotic resistance", "the telegraph",
]

CHAT_TEMPLATES = [
    "Explain {topic} to a high-school student in a few paragraphs.",
    "What are the three most important things to understand about {topic}? Elaborate on each.",
    "Write a short story that illustrates the concept of {topic}.",
    "Compare {topic} with {topic2}, covering similarities and differences.",
    "I'm preparing a lesson about {topic}. Draft an outline with talking points.",
    "Describe how {topic} affects everyday life, with concrete examples.",
]

# Word bank for building varied filler sentences (RAG contexts, batch docs).
_SUBJECTS = ["The committee", "A research team", "The engineer", "Local officials",
             "The laboratory", "An early prototype", "The survey", "Field measurements",
             "The archive", "A recent audit"]
_VERBS = ["documented", "reported", "analyzed", "confirmed", "estimated",
          "revised", "compared", "recorded", "projected", "disputed"]
_OBJECTS = ["a significant increase in throughput", "irregularities in the data",
            "seasonal variation across regions", "a decline in error rates",
            "unexpected correlations between variables", "gaps in the methodology",
            "improvements after recalibration", "differences between cohorts",
            "long-term structural trends", "inconsistencies in earlier records"]


@dataclass
class Request:
    prompt: str
    max_tokens: int


def _sentence(rng: random.Random) -> str:
    return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}."


def _passage(rng: random.Random, n_sentences: int, fact: str | None = None,
             fact_pos: float = 0.5) -> str:
    """Filler passage; optionally embeds a retrievable fact at a relative position."""
    sents = [_sentence(rng) for _ in range(n_sentences)]
    if fact is not None:
        sents.insert(int(len(sents) * fact_pos), fact)
    return " ".join(sents)


def gen_chat(rng: random.Random, n_requests: int) -> list[Request]:
    """Profile A: short input (<100 tok), long output (200-500 tok)."""
    reqs = []
    for _ in range(n_requests):
        t1, t2 = rng.sample(TOPICS, 2)
        prompt = rng.choice(CHAT_TEMPLATES).format(topic=t1, topic2=t2)
        reqs.append(Request(prompt=prompt, max_tokens=rng.randint(200, 500)))
    return reqs


def gen_rag(rng: random.Random, n_requests: int,
            min_sentences: int = 350, max_sentences: int = 700) -> list[Request]:
    """Profile B: long input (~4k-8k tok), short output, with a needle question.

    ~12 tokens/sentence -> 350-700 sentences ~ 4.2k-8.4k tokens.
    The needle makes answers checkable (retrieval accuracy signal for free).
    """
    reqs = []
    for i in range(n_requests):
        code = f"{rng.randint(100, 999)}-{rng.choice('ABCDEF')}{rng.randint(10, 99)}"
        fact = f"The internal reference code for this document is {code}."
        passage = _passage(rng, rng.randint(min_sentences, max_sentences),
                           fact=fact, fact_pos=rng.uniform(0.15, 0.85))
        prompt = (f"Document:\n{passage}\n\nBased only on the document above, "
                  f"what is the internal reference code? Answer with the code only.")
        reqs.append(Request(prompt=prompt, max_tokens=32))
    return reqs


def gen_batch(rng: random.Random, n_requests: int) -> list[Request]:
    """Profile C: medium docs (~1k tok) submitted together; summarization."""
    reqs = []
    for _ in range(n_requests):
        passage = _passage(rng, rng.randint(70, 120))
        prompt = (f"Summarize the following report in 3 bullet points:\n\n{passage}")
        reqs.append(Request(prompt=prompt, max_tokens=128))
    return reqs


PROFILES = {
    "chat": (gen_chat, 40),
    "rag": (gen_rag, 20),
    "batch": (gen_batch, 30),
}


def generate_trace(profile: str, seed: int, n_requests: int | None = None,
                   scale: float = 1.0) -> dict:
    """Build one trace dict. `scale` shrinks request counts (screening traces)."""
    gen_fn, default_n = PROFILES[profile]
    n = n_requests if n_requests is not None else max(4, int(default_n * scale))
    rng = random.Random(f"{profile}-{seed}")
    requests = gen_fn(rng, n)
    return {
        "profile": profile,
        "seed": seed,
        "n_requests": n,
        "requests": [asdict(r) for r in requests],
    }


def load_trace(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/traces")
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--scale", type=float, default=1.0,
                   help="request-count multiplier (screening uses 1.0; "
                        "validation traces use a different --seed and >1.0)")
    p.add_argument("--tag", default="screen", help="filename tag (screen/heldout)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        trace = generate_trace(profile, args.seed, scale=args.scale)
        path = out / f"{profile}_{args.tag}_seed{args.seed}.json"
        path.write_text(json.dumps(trace, indent=1))
        print(f"wrote {path} ({trace['n_requests']} requests)")


if __name__ == "__main__":
    main()
