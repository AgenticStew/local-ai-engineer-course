# Episode 1: How much VRAM do I need to run an LLM?

`vram.py` is the calculator from the end of the video. It needs only Python 3, no GPU:

```bash
python3 vram.py --params 27 --kv-kb 64 --card 24 --bandwidth 1792
```

Every command below was run, in this order, on an RTX 5090 (32 GB) under WSL2 Ubuntu with llama.cpp
**0.8.0-dev (build 12033, commit 785322d17)** (`git checkout 785322d`). Readings are what that machine printed; yours will differ with your desktop and card.

## Build

You need the CUDA toolkit (`nvcc`), CMake, Git and a C++ compiler; setting those up is episode 7. Under WSL2, install
NVIDIA's `cuda-toolkit` package from its WSL-Ubuntu repository and never a Linux GPU driver (the Windows driver serves
WSL). `-j 24` suits a 16-core CPU with 64 GB of RAM; use a smaller number if `nvcc` runs out of memory.

```bash
sudo apt install libssl-dev          # so -hf can download over HTTPS
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 785322d
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 24   # an unbounded -j can run nvcc out of memory
export PATH="$PWD/build/bin:$PATH"
llama version
llama download -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --no-mmproj
```

Second terminal: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1`

## Predict, then measure (32K context)

```bash
llama serve -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M -c 32768 -np 1 --no-mmproj -lv 4 --log-file ep01.log
# Ctrl+C once: the memory breakdown is printed on exit (a second Ctrl+C skips it)
grep -E 'file size|kv_cache: size|memory_breakdown' ep01.log | tail -n 5
```

## Context (8K)

```bash
llama serve -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M -c 8192 -np 1 --no-mmproj
```

## What `--fit` picks (no model load)

```bash
llama fit-params -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
llama fit-params -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --fit-target 9216    # like a 24 GB card
llama fit-params -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --fit-target 17408   # like 16 GB
llama fit-params -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --fit-target 21504   # like 12 GB
```

`--fit-target` = (32,768 − the card's MiB) + 1,024, so a simulated card keeps the same 1 GiB margin a real one would.
`-ngl -1` in fit's output is `auto` in this build (`-ngl all`, or `-2`, forces every layer): fit left it unchanged because
everything fitted, and at load time auto puts every layer on the GPU. 4,096 is fit's default minimum context
(`--fit-ctx`), not a hard floor, and a `-c` you set yourself is never reduced. Fit can still fail if the model won't
fit even after moving layers into system RAM.
`-ngl` counts the output layer, and for this file also `blk.64`, an unused next-token-prediction block (`n_layer_all` = 65): `-ngl 56` puts 54 of the model's 64 layers on the GPU and 10 in system RAM (checked with a `-ngl 56` load: `load_tensors` assigns layers 0–9 to the CPU).

## Bandwidth

```bash
llama bench -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M -ngl 99,56 -r 3
```

## Readings from the video

| Reading | Value |
| --- | --- |
| llama.cpp build | 0.8.0-dev (build 12033, commit 785322d17) |
| VRAM in use before loading (desktop) | 2,749 MiB |
| `file size` log line | 15.32 GiB (4.82 BPW) |
| KV cache at 32K | 2,048.00 MiB, 16 layers (64 KiB per token) |
| Memory breakdown on exit (CUDA0) | model 14,674 · context 2,197 · compute 174 MiB |
| In use, 32K loaded | 20,889 MiB (model's share 18,140 MiB after the 2,749 MiB desktop) |
| In use, 8K loaded | 19,392 MiB (1,497 MiB less than 32K) |
| `fit-params`, 32 GB | `-c 229120 -ngl -1` |
| `fit-params --fit-target 9216` (24 GB) | `-c 100096 -ngl -1` |
| `fit-params --fit-target 17408` (16 GB) | `-c 4096 -ngl 56` |
| `fit-params --fit-target 21504` (12 GB) | `-c 4096 -ngl 36` |
| `llama bench` tg128, `-ngl 99` | 73.85 ± 0.19 tokens/s |
| `llama bench` tg128, `-ngl 56` | 19.67 ± 5.80 tokens/s |
