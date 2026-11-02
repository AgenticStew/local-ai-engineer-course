#!/usr/bin/env python3
"""How much VRAM does a local LLM need? A first estimate, before you download anything."""
import argparse
import math

# bits/weight from llama.cpp's quantize README, measured on its Llama 3.1 8B example: a good estimate, not a constant.
# Real files vary a little (the Qwen3.8-27B UD-Q4_K_M in the video reports 4.82).
BITS = {"Q2_K": 3.16, "Q3_K_M": 4.00, "Q4_K_M": 4.89, "Q5_K_M": 5.70,
        "Q6_K": 6.56, "Q8_0": 8.50, "F16": 16.00}
GIB = 2**30


def number(kind, allow_zero=False):
    """argparse type: a finite number above zero (or zero and up), so bad input gets a clear error, not a verdict."""
    def check(text):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a {'whole ' if kind is int else ''}number: {text!r}")
        if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
            raise argparse.ArgumentTypeError(f"must be {'0 or more' if allow_zero else 'more than 0'}, got {text}")
        return v
    return check


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--params", type=number(float), required=True, help="billions of parameters, e.g. 27")
p.add_argument("--quant", default="Q4_K_M", choices=BITS)
p.add_argument("--context", type=number(int), default=32768, help="context length in tokens")
p.add_argument("--kv-kb", type=number(float, allow_zero=True), default=128,
               help="KV cache KiB per token. Read the real figure from llama.cpp's log. "
                    "128 suits a classic 32-layer model; Qwen3.8-27B is 64")
p.add_argument("--buffers", type=number(float, allow_zero=True), default=1.0, help="working buffers, GiB")
p.add_argument("--card", type=number(float), help="your card's VRAM, GB as sold")
p.add_argument("--in-use", type=number(float, allow_zero=True), default=1.5, help="VRAM your desktop already uses, GiB")
p.add_argument("--bandwidth", type=number(float), help="memory bandwidth in GB/s, for a speed ceiling")
a = p.parse_args()

file_gb = a.params * BITS[a.quant] / 8                 # the formula: parameters x bits / 8
weights = file_gb * 1e9 / GIB
kv = a.context * a.kv_kb * 1024 / GIB
need = weights + kv + a.buffers
if not math.isfinite(need) or not math.isfinite(file_gb):
    p.error("those numbers are too large to estimate")

print(f"Weights  {weights:5.1f} GiB  ({file_gb:.1f} GB on disk, {a.quant})")
print(f"Context  {kv:5.1f} GiB  ({a.context:,} tokens at {a.kv_kb:g} KiB each)")
print(f"Buffers  {a.buffers:5.1f} GiB")
print(f"Total    {need:5.1f} GiB")

if a.card is not None:
    spare = a.card - a.in_use - need
    verdict = ("fits" if spare >= 1 else "tight" if spare >= 0
               else "will not fit: lower --context or pick a smaller --quant")
    print(f"On a {a.card:g} GB card with {a.in_use:g} GiB in use: {spare:+.1f} GiB spare, {verdict}")
if a.bandwidth is not None:
    print(f"Speed ceiling at {a.bandwidth:g} GB/s: {a.bandwidth / file_gb:.0f} tokens/s (dense model, all in VRAM)")
