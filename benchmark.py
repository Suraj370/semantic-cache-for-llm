#!/usr/bin/env python3
"""
Semantic Cache Benchmark
─────────────────────────────────────────────────────────────────────
Drives the cache with 2,000+ requests using realistic traffic patterns
(exact repeats, semantic paraphrases, novel queries) and reports:

  • Hit-rate convergence over time
  • Latency percentiles — HIT vs MISS vs overall
  • Total tokens and cost saved (from server-side monitoring API)

Prerequisites
─────────────
1. Stack must be running:
       docker compose up -d                          # real OpenAI key in .env
   OR with the bundled mock LLM (no API costs):
       docker compose --profile benchmark up -d
       # and add to .env: OPENAI_BASE_URL=http://localhost:9999/v1

2. httpx is installed (comes with fastapi[standard]):
       pip install httpx

Usage
─────
  python benchmark.py                               # 2 000 reqs, concurrency 20
  python benchmark.py --requests 5000 --concurrency 40
  python benchmark.py --host http://localhost:8000 --model gpt-4o-mini
  python benchmark.py --output results/run1.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Query corpus
# 60 concept groups × (1 seed + 3 paraphrases) = 240 unique phrasings.
# Paraphrases are close semantic equivalents — cosine similarity typically
# 0.87–0.96 for text-embedding-3-small, so they should cross the 0.85 threshold.
# ─────────────────────────────────────────────────────────────────────────────

QUERY_GROUPS: List[Dict[str, Any]] = [
    # ── Factual (12 groups) ──────────────────────────────────────────────────
    {"cat": "factual",
     "seed": "What is the capital of France?",
     "para": ["What city is the capital of France?", "Name the capital city of France.", "Which city serves as France's capital?"]},
    {"cat": "factual",
     "seed": "Who invented the telephone?",
     "para": ["Who was the inventor of the telephone?", "Which scientist created the telephone?", "Name the person who invented the telephone."]},
    {"cat": "factual",
     "seed": "What year did World War 2 end?",
     "para": ["When did World War Two end?", "In what year did WWII conclude?", "What year was the end of the Second World War?"]},
    {"cat": "factual",
     "seed": "What is the speed of light in a vacuum?",
     "para": ["How fast does light travel in a vacuum?", "What is the speed of light?", "Tell me how fast light travels."]},
    {"cat": "factual",
     "seed": "Who wrote Romeo and Juliet?",
     "para": ["Who authored Romeo and Juliet?", "Which playwright wrote Romeo and Juliet?", "Who is the author of the play Romeo and Juliet?"]},
    {"cat": "factual",
     "seed": "What is the largest planet in our solar system?",
     "para": ["Which planet is the biggest in our solar system?", "Name the largest planet in the solar system.", "What is the biggest planet orbiting our sun?"]},
    {"cat": "factual",
     "seed": "What is the chemical formula for water?",
     "para": ["What is the molecular formula of water?", "How do you write the formula for water?", "What chemical formula represents water?"]},
    {"cat": "factual",
     "seed": "Who painted the Mona Lisa?",
     "para": ["Which artist created the Mona Lisa?", "Who is the painter of the Mona Lisa?", "Name the artist who painted the Mona Lisa."]},
    {"cat": "factual",
     "seed": "What is the tallest mountain in the world?",
     "para": ["Which mountain is the tallest on Earth?", "Name the highest mountain in the world.", "What is the world's tallest peak?"]},
    {"cat": "factual",
     "seed": "What is photosynthesis?",
     "para": ["Can you explain photosynthesis?", "How does photosynthesis work?", "Describe the process of photosynthesis."]},
    {"cat": "factual",
     "seed": "What is the boiling point of water?",
     "para": ["At what temperature does water boil?", "What temperature causes water to boil?", "Tell me the boiling point of water."]},
    {"cat": "factual",
     "seed": "Who discovered gravity?",
     "para": ["Who is credited with discovering gravity?", "Which scientist discovered the law of gravity?", "Name the scientist who discovered gravity."]},

    # ── Coding (12 groups) ───────────────────────────────────────────────────
    {"cat": "coding",
     "seed": "How do I reverse a list in Python?",
     "para": ["What is the Python way to reverse a list?", "How to reverse a Python list?", "Show me how to reverse a list in Python."]},
    {"cat": "coding",
     "seed": "What is a dictionary in Python?",
     "para": ["Explain Python dictionaries.", "How do Python dicts work?", "What is a dict in Python?"]},
    {"cat": "coding",
     "seed": "How do I read a file in Python?",
     "para": ["What is the Python way to open and read a file?", "How to open a file in Python?", "Show me file reading in Python."]},
    {"cat": "coding",
     "seed": "What is the difference between a list and a tuple in Python?",
     "para": ["How are Python lists different from tuples?", "Compare Python lists and tuples.", "List versus tuple in Python — what is the difference?"]},
    {"cat": "coding",
     "seed": "How do I sort a list in Python?",
     "para": ["What is the Python way to sort a list?", "How to sort elements in a Python list?", "Show me list sorting in Python."]},
    {"cat": "coding",
     "seed": "What is a lambda function in Python?",
     "para": ["Explain Python lambda functions.", "How do lambda functions work in Python?", "What is a lambda in Python used for?"]},
    {"cat": "coding",
     "seed": "How do I handle exceptions in Python?",
     "para": ["What is the Python way to handle errors?", "How to use try and except in Python?", "Show me exception handling in Python."]},
    {"cat": "coding",
     "seed": "What is a REST API?",
     "para": ["Can you explain what REST APIs are?", "How does a REST API work?", "What are REST APIs used for?"]},
    {"cat": "coding",
     "seed": "What is a Python decorator?",
     "para": ["Explain Python decorators.", "How do decorators work in Python?", "What is a decorator used for in Python?"]},
    {"cat": "coding",
     "seed": "How do I create a virtual environment in Python?",
     "para": ["Show me how to use Python virtual environments.", "What command creates a Python venv?", "How to set up a Python virtual environment?"]},
    {"cat": "coding",
     "seed": "What is the difference between == and is in Python?",
     "para": ["Explain == versus is in Python.", "How does == differ from is in Python?", "When should I use == instead of is in Python?"]},
    {"cat": "coding",
     "seed": "How do I use list comprehensions in Python?",
     "para": ["Explain Python list comprehensions.", "Show me list comprehensions in Python.", "How do list comprehensions work in Python?"]},

    # ── Math (12 groups) ─────────────────────────────────────────────────────
    {"cat": "math",
     "seed": "What is 15 percent of 240?",
     "para": ["Calculate 15% of 240.", "Find fifteen percent of 240.", "What is 15% of two hundred forty?"]},
    {"cat": "math",
     "seed": "What is the square root of 144?",
     "para": ["Calculate the square root of 144.", "Find sqrt of 144.", "What number squared equals 144?"]},
    {"cat": "math",
     "seed": "What is the Pythagorean theorem?",
     "para": ["Explain the Pythagorean theorem.", "How does the Pythagorean theorem work?", "State the Pythagorean theorem formula."]},
    {"cat": "math",
     "seed": "What is 2 raised to the power of 10?",
     "para": ["Calculate 2 to the power of 10.", "What is 2 to the 10th?", "Compute two raised to the tenth power."]},
    {"cat": "math",
     "seed": "What is the area of a circle with radius 5?",
     "para": ["Calculate the area of a circle with radius 5.", "Find the area when the circle radius is 5.", "What is the area of a circle of radius five?"]},
    {"cat": "math",
     "seed": "What is the derivative of x squared?",
     "para": ["Find the derivative of x squared.", "Differentiate x to the power of two.", "What is d over dx of x squared?"]},
    {"cat": "math",
     "seed": "What is the sum of the first 10 natural numbers?",
     "para": ["Add the first 10 natural numbers.", "Calculate 1 plus 2 plus all the way up to 10.", "What is the sum from 1 to 10?"]},
    {"cat": "math",
     "seed": "How do you convert 100 degrees Fahrenheit to Celsius?",
     "para": ["What is 100 Fahrenheit in Celsius?", "Convert 100 degrees F to Celsius.", "How many degrees Celsius is 100 Fahrenheit?"]},
    {"cat": "math",
     "seed": "What is the probability of rolling a 6 on a die?",
     "para": ["What are the odds of rolling a 6 on a dice?", "Find the probability of getting 6 on one dice roll.", "What is the chance of rolling a six on a standard die?"]},
    {"cat": "math",
     "seed": "What is 20 percent of 85?",
     "para": ["Calculate 20% of 85.", "Find twenty percent of 85.", "What does 20% of 85 equal?"]},
    {"cat": "math",
     "seed": "What is the mean of 2, 4, 6, 8, and 10?",
     "para": ["Calculate the average of 2, 4, 6, 8, 10.", "Find the mean of the numbers 2 4 6 8 10.", "What is the arithmetic mean of 2, 4, 6, 8, 10?"]},
    {"cat": "math",
     "seed": "What is the perimeter of a rectangle 8 by 5?",
     "para": ["Calculate the perimeter of an 8 by 5 rectangle.", "Find the perimeter of a rectangle with sides 8 and 5.", "What is the perimeter of a rectangle that is 8 wide and 5 tall?"]},

    # ── Creative (12 groups) ─────────────────────────────────────────────────
    {"cat": "creative",
     "seed": "Write a haiku about the ocean.",
     "para": ["Compose a haiku about the sea.", "Create a haiku about ocean waves.", "Give me a haiku about the ocean."]},
    {"cat": "creative",
     "seed": "Suggest a name for a coffee shop.",
     "para": ["Give me a coffee shop name idea.", "What would be a good name for a café?", "Propose a creative name for a coffee shop."]},
    {"cat": "creative",
     "seed": "Write a short poem about autumn.",
     "para": ["Compose a poem about fall.", "Write a brief autumn poem.", "Give me a short poem about autumn leaves."]},
    {"cat": "creative",
     "seed": "Tell me a funny joke.",
     "para": ["Share a joke with me.", "Do you know any good jokes?", "Tell me something funny."]},
    {"cat": "creative",
     "seed": "Write a one-sentence story about a dragon.",
     "para": ["Give me a tiny story featuring a dragon.", "Compose a micro-story about a dragon.", "Tell me a very short dragon story in one sentence."]},
    {"cat": "creative",
     "seed": "Suggest a creative team name for software developers.",
     "para": ["What is a good developer team name?", "Give me a tech team name idea.", "Propose a catchy name for a software development team."]},
    {"cat": "creative",
     "seed": "Write a motivational quote.",
     "para": ["Give me an inspirational quote.", "Share a motivational saying with me.", "Compose an uplifting quote."]},
    {"cat": "creative",
     "seed": "Describe a sunset in one sentence.",
     "para": ["Write a one-line description of a sunset.", "Give me a vivid one-sentence sunset description.", "Describe a beautiful sunset briefly."]},
    {"cat": "creative",
     "seed": "Write a limerick about programming.",
     "para": ["Compose a programming limerick.", "Make up a funny limerick about coding.", "Write a limerick about software development."]},
    {"cat": "creative",
     "seed": "Suggest a plot for a short story.",
     "para": ["Give me a short story idea.", "What would make a good short story plot?", "Propose a creative short story concept."]},
    {"cat": "creative",
     "seed": "Write a tagline for a tech startup.",
     "para": ["Give me a startup tagline idea.", "Compose a catchy tech startup slogan.", "Suggest a tagline for a new technology company."]},
    {"cat": "creative",
     "seed": "Create a metaphor for learning.",
     "para": ["Give me a metaphor about learning.", "Describe learning using a metaphor.", "Invent a good metaphor for the process of learning."]},

    # ── Conversational (12 groups) ───────────────────────────────────────────
    {"cat": "conversational",
     "seed": "What should I have for dinner tonight?",
     "para": ["Any dinner suggestions for tonight?", "Help me decide what to eat for dinner.", "What is a good dinner idea for tonight?"]},
    {"cat": "conversational",
     "seed": "How do I stay productive while working from home?",
     "para": ["Tips for staying productive working from home?", "How can I be more productive when working remotely?", "What are good work from home productivity tips?"]},
    {"cat": "conversational",
     "seed": "What are some good books to read?",
     "para": ["Can you recommend some books?", "Suggest some books I should read.", "What books would you recommend reading?"]},
    {"cat": "conversational",
     "seed": "How do I improve my public speaking skills?",
     "para": ["Tips to get better at public speaking?", "How can I become a better public speaker?", "Give me advice on improving my public speaking."]},
    {"cat": "conversational",
     "seed": "What are the benefits of regular exercise?",
     "para": ["Why is regular exercise good for you?", "What health benefits come from exercising regularly?", "Tell me about the benefits of working out."]},
    {"cat": "conversational",
     "seed": "How do I manage stress effectively?",
     "para": ["Give me stress management tips.", "What are good ways to reduce stress?", "How can I deal with stress more effectively?"]},
    {"cat": "conversational",
     "seed": "What makes a good leader?",
     "para": ["What qualities make a good leader?", "Describe what makes someone a great leader.", "What traits define good leadership?"]},
    {"cat": "conversational",
     "seed": "How do I learn a new language quickly?",
     "para": ["Tips for learning a language fast?", "What is the fastest way to learn a new language?", "How can I pick up a new language quickly?"]},
    {"cat": "conversational",
     "seed": "What is the best way to save money?",
     "para": ["How should I go about saving money?", "Give me money-saving tips.", "What are effective ways to save more money?"]},
    {"cat": "conversational",
     "seed": "How do I improve my sleep quality?",
     "para": ["Tips for getting better sleep?", "How can I get higher quality sleep?", "What can I do to improve my sleep?"]},
    {"cat": "conversational",
     "seed": "What are some good hobbies to pick up?",
     "para": ["Suggest some hobbies I could start.", "What are fun hobbies to take up?", "Recommend hobbies for someone to try."]},
    {"cat": "conversational",
     "seed": "How do I become more confident?",
     "para": ["Give me confidence-building tips.", "What is the best way to build self-confidence?", "How can I become more confident in myself?"]},
]

# Truly novel queries: outside the corpus topics, likely to miss
_NOVEL_QUERIES: List[Tuple[str, str]] = [
    ("What are the main causes of climate change?", "factual"),
    ("How does the human immune system work?", "factual"),
    ("What is the history of the internet?", "factual"),
    ("Who was Nikola Tesla?", "factual"),
    ("What causes earthquakes?", "factual"),
    ("How does a nuclear reactor work?", "factual"),
    ("What is the theory of relativity?", "factual"),
    ("How do black holes form?", "factual"),
    ("What is CRISPR gene editing?", "factual"),
    ("Who was Marie Curie?", "factual"),
    ("How do you implement a binary search tree in Python?", "coding"),
    ("What is the difference between SQL and NoSQL databases?", "coding"),
    ("How does garbage collection work in Python?", "coding"),
    ("What is Docker and how does it work?", "coding"),
    ("Explain the CAP theorem.", "coding"),
    ("What is a microservices architecture?", "coding"),
    ("How does TCP/IP work?", "coding"),
    ("What is OAuth 2.0?", "coding"),
    ("What is a heap data structure?", "coding"),
    ("How do you implement a linked list in Python?", "coding"),
    ("What is 17 multiplied by 23?", "math"),
    ("Calculate the volume of a sphere with radius 3.", "math"),
    ("What is the integral of sin(x)?", "math"),
    ("What is a prime number?", "math"),
    ("What is the Fibonacci sequence?", "math"),
    ("Write a haiku about winter.", "creative"),
    ("Suggest a name for a bakery.", "creative"),
    ("Write a two-line poem about the moon.", "creative"),
    ("Tell me a riddle.", "creative"),
    ("Write a tagline for a fitness app.", "creative"),
    ("What are tips for better time management?", "conversational"),
    ("How do I make new friends as an adult?", "conversational"),
    ("What are the best ways to learn a new skill?", "conversational"),
    ("How do I deal with procrastination?", "conversational"),
    ("What are good morning routines?", "conversational"),
    ("How do I negotiate a higher salary?", "conversational"),
    ("What are some tips for healthy eating?", "conversational"),
    ("How do I build better habits?", "conversational"),
    ("What are signs of burnout?", "conversational"),
    ("How do I improve my memory?", "conversational"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass and percentile helper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    seq: int
    ts: float
    latency_ms: float
    hit: bool
    category: str
    variant: str       # seed | exact_repeat | paraphrase | novel
    query: str
    similarity: str = ""   # X-Cache-Similarity header value, if present
    error: bool = False


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ─────────────────────────────────────────────────────────────────────────────
# Request schedule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledRequest:
    query: str
    category: str
    variant: str


def build_schedule(n_requests: int, rng_seed: int = 42) -> List[ScheduledRequest]:
    rng = random.Random(rng_seed)

    seeds = [(g["seed"], g["cat"]) for g in QUERY_GROUPS]
    paraphrases = [(p, g["cat"]) for g in QUERY_GROUPS for p in g["para"]]

    schedule: List[ScheduledRequest] = []

    # Phase 1: warmup — one seed per group in random order (all misses)
    warmup = [ScheduledRequest(s, c, "seed") for s, c in seeds]
    rng.shuffle(warmup)
    schedule.extend(warmup)

    # Phase 2: mixed traffic
    n_mixed = n_requests - len(warmup)
    mixed: List[ScheduledRequest] = []

    for _ in range(n_mixed):
        r = rng.random()
        if r < 0.30:
            # Exact repeat of a warmup seed → near-100% hit
            q, cat = rng.choice(seeds)
            mixed.append(ScheduledRequest(q, cat, "exact_repeat"))
        elif r < 0.70:
            # Semantic paraphrase of a warmup seed → ~80% hit
            q, cat = rng.choice(paraphrases)
            mixed.append(ScheduledRequest(q, cat, "paraphrase"))
        else:
            # Novel query not in corpus → low hit rate
            q, cat = rng.choice(_NOVEL_QUERIES)
            mixed.append(ScheduledRequest(q, cat, "novel"))

    rng.shuffle(mixed)
    schedule.extend(mixed)
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Single request
# ─────────────────────────────────────────────────────────────────────────────

async def _send(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    host: str,
    model: str,
    req: ScheduledRequest,
    seq: int,
    timeout: float,
) -> Result:
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": req.query}],
            "stream": False,
        }
        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{host}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            hit = resp.headers.get("x-cache", "MISS").upper() == "HIT"
            similarity = resp.headers.get("x-cache-similarity", "")
            return Result(
                seq=seq,
                ts=t0,
                latency_ms=latency_ms,
                hit=hit,
                category=req.category,
                variant=req.variant,
                query=req.query,
                similarity=similarity,
                error=resp.status_code >= 500,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000
            return Result(
                seq=seq,
                ts=t0,
                latency_ms=latency_ms,
                hit=False,
                category=req.category,
                variant=req.variant,
                query=req.query,
                error=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring summary snapshot
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_summary(client: httpx.AsyncClient, host: str) -> Dict[str, Any]:
    try:
        r = await client.get(f"{host}/v1/monitoring/summary", timeout=10)
        return r.json()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_YELLOW = "\033[33m"
_RED   = "\033[31m"
_RESET = "\033[0m"
_NO_COLOR = not sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return text if _NO_COLOR else f"{code}{text}{_RESET}"


def _bar(fraction: float, width: int = 38) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f} s"
    return f"{ms:.0f} ms"


def _print_header(n: int, concurrency: int, host: str, model: str) -> None:
    w = 72
    border = "─" * w
    print(f"\n{border}")
    print(_c(f"  Semantic Cache Benchmark".center(w), _BOLD))
    print(f"  {n:,} requests  ·  concurrency {concurrency}  ·  model {model}".center(w))
    print(f"  {host}".center(w))
    print(border)


def _print_convergence(checkpoints: List[Tuple[int, float, int, int]]) -> None:
    """checkpoints: list of (n_done, hit_rate, n_hits, n_misses)"""
    print(f"\n{_c('Hit-Rate Convergence', _BOLD)}")
    first_nonzero = next((i for i, (_, hr, _, _) in enumerate(checkpoints) if hr > 0), None)
    for n_done, hit_rate, n_hits, n_misses in checkpoints:
        bar = _bar(hit_rate, width=30)
        pct = f"{hit_rate*100:5.1f}%"
        colour = _GREEN if hit_rate >= 0.50 else (_YELLOW if hit_rate >= 0.25 else _RED)
        note = ""
        if n_done == checkpoints[0][0] and hit_rate == 0:
            note = "  (cache warming)"
        elif n_done == checkpoints[-1][0]:
            note = "  ← converged"
        print(f"  after {n_done:>5,}  {_c(bar, colour)}  {_c(pct, colour)}{note}")


def _print_latency_table(results: List[Result]) -> None:
    hits  = [r.latency_ms for r in results if r.hit  and not r.error]
    misses = [r.latency_ms for r in results if not r.hit and not r.error]
    all_  = hits + misses

    def row(label: str, data: List[float]) -> str:
        if not data:
            return f"  {label:<10} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  (no data)"
        mean = sum(data) / len(data)
        return (
            f"  {label:<10}"
            f" {_fmt_ms(mean):>8}"
            f" {_fmt_ms(_pct(data, 50)):>8}"
            f" {_fmt_ms(_pct(data, 95)):>8}"
            f" {_fmt_ms(_pct(data, 99)):>8}"
        )

    print(f"\n{_c('Latency Percentiles', _BOLD)}")
    print(f"  {'':10} {'mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(_c(row("HIT",     hits),  _GREEN))
    print(_c(row("MISS",    misses), _YELLOW))
    print(     row("Overall", all_))

    if hits and misses:
        speedup = _pct(misses, 50) / _pct(hits, 50) if _pct(hits, 50) > 0 else 0
        print(f"\n  {_c(f'Cache hits are {speedup:.0f}× faster than misses (P50)', _BOLD)}")


def _print_cost_section(
    results: List[Result],
    summary_before: Dict[str, Any],
    summary_after: Dict[str, Any],
) -> None:
    n_total = len([r for r in results if not r.error])
    n_hits  = sum(1 for r in results if r.hit and not r.error)
    hit_rate = n_hits / n_total if n_total else 0.0

    # Delta from server-side monitoring
    tokens_before = summary_before.get("tokens_saved", 0) or 0
    tokens_after  = summary_after.get("tokens_saved", 0) or 0
    cost_before   = summary_before.get("cost_saved_usd", 0.0) or 0.0
    cost_after    = summary_after.get("cost_saved_usd", 0.0) or 0.0

    tokens_saved = tokens_after - tokens_before
    cost_saved   = cost_after - cost_before

    # Projected to 1 M requests
    projected_1m = (cost_saved / n_total * 1_000_000) if n_total and cost_saved > 0 else 0

    print(f"\n{_c('Cost Savings  (server-side estimate)', _BOLD)}")
    print(f"  Total requests:    {n_total:>10,}")
    print(f"  Cache hits:        {n_hits:>10,}  ({hit_rate*100:.1f}%)")
    if tokens_saved:
        print(f"  Tokens saved:      {tokens_saved:>10,}")
    if cost_saved:
        print(f"  Cost saved:        {'$'+f'{cost_saved:.4f}':>10}")
        if projected_1m:
            print(f"  Projected (1M):    {'$'+f'{projected_1m:,.0f}':>10}")
    elif not summary_after:
        print("  (monitoring endpoint unreachable — restart stack and rerun)")


def _print_variant_breakdown(results: List[Result]) -> None:
    variants = ["exact_repeat", "paraphrase", "novel", "seed"]
    print(f"\n{_c('Hit Rate by Traffic Mix', _BOLD)}")
    print(f"  {'Variant':<14} {'Requests':>8} {'Hits':>6} {'Hit Rate':>9}")
    print(f"  {'─'*14} {'─'*8} {'─'*6} {'─'*9}")
    for v in variants:
        rs = [r for r in results if r.variant == v and not r.error]
        if not rs:
            continue
        hits = sum(1 for r in rs if r.hit)
        hr = hits / len(rs) if rs else 0.0
        colour = _GREEN if hr >= 0.60 else (_YELLOW if hr >= 0.30 else _RED)
        print(f"  {v:<14} {len(rs):>8,} {hits:>6,} {_c(f'{hr*100:8.1f}%', colour)}")


def _save_csv(results: List[Result], path: str) -> None:
    fields = ["seq", "ts", "latency_ms", "hit", "category", "variant",
              "similarity", "error", "query"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "seq": r.seq,
                "ts": f"{r.ts:.3f}",
                "latency_ms": f"{r.latency_ms:.1f}",
                "hit": int(r.hit),
                "category": r.category,
                "variant": r.variant,
                "similarity": r.similarity,
                "error": int(r.error),
                "query": r.query,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def _run(
    host: str,
    n_requests: int,
    concurrency: int,
    model: str,
    timeout: float,
    output: str,
    rng_seed: int,
) -> None:
    _print_header(n_requests, concurrency, host, model)

    schedule = build_schedule(n_requests, rng_seed)
    n_warmup = len(QUERY_GROUPS)

    sem = asyncio.Semaphore(concurrency)
    results: List[Result] = []
    completed = 0
    checkpoint_every = max(100, n_requests // 20)
    checkpoints: List[Tuple[int, float, int, int]] = []

    # ── live progress ────────────────────────────────────────────────────────
    def _progress(done: int, total: int, phase: str) -> None:
        frac = done / total if total else 0
        bar  = _bar(frac, width=36)
        hits_so_far = sum(1 for r in results if r.hit and not r.error)
        total_so_far = sum(1 for r in results if not r.error)
        hr_str = f"  hit={hits_so_far/total_so_far*100:.1f}%" if total_so_far > 0 else ""
        line = f"\r  [{bar}] {done:>{len(str(total))}}/{total}  {phase}{hr_str}   "
        sys.stdout.write(line)
        sys.stdout.flush()

    # ── snapshot metrics before run ──────────────────────────────────────────
    async with httpx.AsyncClient() as probe:
        summary_before = await _fetch_summary(probe, host)
        if not summary_before:
            print(f"\n{_c('WARNING', _YELLOW)}: cannot reach {host}/v1/monitoring/summary")
            print(f"  Make sure the stack is running before continuing.\n")

    # ── warmup phase ─────────────────────────────────────────────────────────
    print(f"\n  Phase 1 / warmup — {n_warmup} seed queries (all misses, builds the cache)")
    async with httpx.AsyncClient() as client:
        tasks = [
            _send(client, sem, host, model, req, i, timeout)
            for i, req in enumerate(schedule[:n_warmup])
        ]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            completed += 1
            _progress(completed, n_requests, "warming")

    print()  # newline after progress bar

    # ── mixed phase ──────────────────────────────────────────────────────────
    n_mixed = n_requests - n_warmup
    print(f"\n  Phase 2 / mixed traffic — {n_mixed:,} requests")
    print(f"  (30% exact repeat · 40% paraphrase · 30% novel)")

    async with httpx.AsyncClient() as client:
        tasks = [
            _send(client, sem, host, model, req, n_warmup + i, timeout)
            for i, req in enumerate(schedule[n_warmup:])
        ]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            completed += 1
            _progress(completed, n_requests, "running")

            # checkpoint
            mixed_done = completed - n_warmup
            if mixed_done > 0 and (mixed_done % checkpoint_every == 0 or completed == n_requests):
                non_err = [x for x in results if not x.error]
                hits = sum(1 for x in non_err if x.hit)
                total = len(non_err)
                hr = hits / total if total else 0.0
                checkpoints.append((completed, hr, hits, total - hits))

    print()  # newline after progress bar

    # ── final snapshot ───────────────────────────────────────────────────────
    async with httpx.AsyncClient() as probe:
        summary_after = await _fetch_summary(probe, host)

    # ── report ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    _print_convergence(checkpoints)
    print(f"\n{'─'*72}")
    _print_latency_table(results)
    print(f"\n{'─'*72}")
    _print_cost_section(results, summary_before, summary_after)
    print(f"\n{'─'*72}")
    _print_variant_breakdown(results)

    errors = sum(1 for r in results if r.error)
    if errors:
        print(f"\n  {_c(f'{errors} request(s) errored', _RED)} — check stack logs")

    # ── CSV ──────────────────────────────────────────────────────────────────
    _save_csv(results, output)
    print(f"\n{'─'*72}")
    print(f"  {_c('Results saved to:', _BOLD)} {output}")
    print(f"  {_c('Live Grafana dashboard:', _BOLD)} http://localhost:3000")
    print(f"{'─'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(
        description="Semantic Cache Benchmark — hit rate, latency, cost savings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",        default="http://localhost:8000",
                   help="Base URL of the semantic cache API (default: %(default)s)")
    p.add_argument("--requests",    type=int, default=2000,
                   help="Total requests to send (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Max concurrent requests (default: %(default)s)")
    p.add_argument("--model",       default="gpt-4o-mini",
                   help="Model name to use in requests (default: %(default)s)")
    p.add_argument("--timeout",     type=float, default=60.0,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--output",      default=f"benchmark_{ts}.csv",
                   help="CSV output path (default: benchmark_<timestamp>.csv)")
    p.add_argument("--seed",        type=int, default=42,
                   help="RNG seed for reproducible request schedules (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.requests < len(QUERY_GROUPS):
        print(f"--requests must be >= {len(QUERY_GROUPS)} (warmup size). Got {args.requests}.")
        sys.exit(1)
    asyncio.run(_run(
        host=args.host,
        n_requests=args.requests,
        concurrency=args.concurrency,
        model=args.model,
        timeout=args.timeout,
        output=args.output,
        rng_seed=args.seed,
    ))


if __name__ == "__main__":
    main()
