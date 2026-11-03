# Episode 2: What does Q4_K_M mean?

`gguf_mix.py` reads a GGUF's header and adds up what's inside it: every storage format, its share of the weights and
the file's real bits per weight. It needs only Python 3, with no GPU and no big download. Give it a local file or a
Hugging Face `resolve` URL (only the header is fetched, a few MB):

```bash
python3 gguf_mix.py https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf
python3 gguf_mix.py model.gguf --parts     # also each kind of tensor, and which importance matrix was used
python3 gguf_mix.py model.gguf --json      # everything, one entry per tensor
```

Every command below was run, in this order (except that each prompt run, and its check, ran straight after that quant's `llama bench`), on an RTX 5090 (32 GB) under WSL2 Ubuntu with llama.cpp **0.8.0-dev (build 12033, commit 785322d17)**
(`git checkout 785322d`; building it is in episode 1's folder). Readings are what that machine printed.

## Setup

```bash
export R=unsloth/Qwen3.8-27B-GGUF LC=~/src/llama.cpp     # LC = your llama.cpp checkout
mkdir -p logs
curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
python3 -m zipfile -e wikitext-2-raw-v1.zip .
{ head -c 45000 "$LC/src/llama-sampler.cpp"; cat "$LC/gguf-py/gguf/vocab.py"; head -c 45000 "$LC/src/llama-quant.cpp"; } > code.txt
```

`code.txt` is 134,852 bytes, 36,220 tokens for this model: enough for 60 chunks of 512.

## Read the headers (no GPU, no download)

```bash
B=https://huggingface.co/$R/resolve/main/Qwen3.8-27B
python3 gguf_mix.py $B-UD-Q4_K_M.gguf --parts
python3 gguf_mix.py $B-UD-Q8_K_XL.gguf
```

## The 16-bit reference

BF16 is 54.66 GB in two files, so it spills past a 32 GB card into system RAM (39 of 66 layers on the GPU here) and
runs slowly, which doesn't matter for this. Each run writes about 7.6 GB of logits.

```bash
llama download -hf $R:BF16 --no-mmproj
llama perplexity -hf $R:BF16 -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 --chunks 60 --kl-divergence-base prose.kld -lv 4 2>&1 | tee logs/base-prose.log
llama perplexity -hf $R:BF16 -f code.txt -c 512 -b 512 --chunks 60 --kl-divergence-base code.kld -lv 4 2>&1 | tee logs/base-code.log
```

`-lv 4` makes the log include llama.cpp's own `file size = … (… BPW)` line, which `gguf_mix.py` is checked against.

## Two plain Q4_K_M builds

```bash
BF16=$(find ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF -name '*BF16-00001-of-00002.gguf' | head -1)
llama quantize --dry-run "$BF16" Q4_K_M 2>&1 | tee logs/quantize-dry-run.log
curl -LO https://huggingface.co/$R/resolve/main/imatrix_unsloth.gguf
measure() { ls -l "$1.gguf"; python3 gguf_mix.py "$1.gguf"; for c in prose code; do llama perplexity -m "$1.gguf" -c 512 -b 512 --kl-divergence-base $c.kld --kl-divergence -lv 4 2>&1 | tee logs/kld-$1-$c.log; done; }
llama quantize "$BF16" plain-Q4_K_M.gguf Q4_K_M 2>&1 | tee logs/quantize-plain-Q4_K_M.log
measure plain-Q4_K_M
llama quantize --imatrix imatrix_unsloth.gguf "$BF16" plain-imatrix-Q4_K_M.gguf Q4_K_M 2>&1 | tee logs/quantize-plain-imatrix-Q4_K_M.log
measure plain-imatrix-Q4_K_M
```

## The ladder

```bash
for t in Q8_0 UD-Q6_K UD-Q4_K_M Q4_0 UD-Q3_K_XL UD-Q2_K_XL UD-IQ1_S; do
  for c in prose code; do
    llama perplexity -hf $R:$t -c 512 -b 512 --kl-divergence-base $c.kld --kl-divergence -lv 4 2>&1 | tee logs/kld-$t-$c.log
  done
  llama bench -hf $R:$t -ngl 99 -r 3 2>&1 | tee logs/bench-$t.log
done
grep -E 'Mean +KLD:|Same top p:' logs/kld-*.log
```

## One real prompt

At Q8_0, UD-Q4_K_M, UD-Q2_K_XL and UD-IQ1_S (and UD-IQ1_S again with `--presence-penalty 1.5`, saved as
`logs/prompt-UD-IQ1_S-penalty.txt`):

```bash
llama cli -hf $R:$t --no-mmproj --temp 0 --seed 42 -n 4000 -st -p "$(cat prompt.txt)" -o logs/prompt-$t.txt 2>&1 | tee logs/prompt-$t.log
python3 check_answer.py logs/prompt-*.txt
```

`llama cli` draws its output on the terminal, so `tee` saved nothing here: each run's conversation is in its `-o` file
(`logs/prompt-*.txt`), and what the screen showed is in `logs/ladder-terminal.cast`.

`check_answer.py` takes the answer after `Assistant:` (and after `[End thinking]`), strips code fences and runs the three
checks in a temporary folder with `timeout 5 python3 -I`, one check at a time. An empty answer, or one that fails to
run, counts as a failure.

| Run | Result | Checks passed | Finished thinking |
| --- | --- | --- | --- |
| Q8_0 | PASS | 3/3 | yes |
| UD-Q4_K_M | PASS | 3/3 | yes |
| UD-Q2_K_XL | PASS | 3/3 | yes |
| UD-IQ1_S | FAIL | 0/3 | no |
| UD-IQ1_S-penalty | FAIL | 0/3 | no |

## Readings

From `results.csv` (built from the logs in `logs/` by the episode's extraction script). Size is the size on disk
(Hugging Face's file size, or `ls -l` for the two builds made here). Real bits are from `gguf_mix.py`; llama.cpp BPW is
its own log line. The ± are llama.cpp's own uncertainties. Two files count as really different on a corpus only if the
gap is more than twice their two ± values added together and mean KLD points the same way.

| File | Size (GB) | Real bits | llama.cpp BPW | Same top token, prose | Same top token, code | Mean KLD, prose | Mean KLD, code | tg128 (t/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16-bit original | 54.66 | 16.00 | 16.0 | (reference) | (reference) |  |  |  |
| Q8_0 | 29.05 | 8.50 | 8.5 | 98.712 ± 0.091 % | 99.314 ± 0.067 % | 0.000884 ± 0.000064 | 0.000624 ± 0.000016 | 50.46 ± 0.30 |
| UD-Q6_K | 21.98 | 6.43 | 6.43 | 98.235 ± 0.106 % | 98.804 ± 0.088 % | 0.001904 ± 0.000066 | 0.002240 ± 0.000064 | 65.24 ± 0.44 |
| Plain Q4_K_M | 16.81 | 4.92 | 4.92 | 93.399 ± 0.201 % | 96.235 ± 0.154 % | 0.022511 ± 0.000445 | 0.020001 ± 0.000552 |  |
| Plain + importance matrix | 16.81 | 4.92 | 4.92 | 94.804 ± 0.179 % | 96.974 ± 0.138 % | 0.014578 ± 0.000292 | 0.013755 ± 0.000456 |  |
| UD-Q4_K_M | 16.46 | 4.82 | 4.82 | 96.020 ± 0.158 % | 97.268 ± 0.132 % | 0.008686 ± 0.000185 | 0.011400 ± 0.000358 | 80.99 ± 0.37 |
| Q4_0 | 16.06 | 4.70 | 4.7 | 92.967 ± 0.207 % | 96.203 ± 0.155 % | 0.025148 ± 0.000488 | 0.024238 ± 0.000650 | 81.90 ± 0.35 |
| UD-Q3_K_XL | 13.15 | 3.85 | 3.85 | 92.967 ± 0.207 % | 95.183 ± 0.173 % | 0.024488 ± 0.000594 | 0.033895 ± 0.001027 | 92.58 ± 0.39 |
| UD-Q2_K_XL | 9.83 | 2.87 | 2.87 | 87.222 ± 0.270 % | 91.797 ± 0.222 % | 0.082281 ± 0.001563 | 0.098747 ± 0.002437 | 107.51 ± 0.41 |
| UD-IQ1_S | 6.19 | 1.84 | 1.84 | 73.954 ± 0.355 % | 81.837 ± 0.312 % | 0.390525 ± 0.005046 | 0.417772 ± 0.006739 | 130.69 ± 0.52 |

## Files

- `gguf_mix.py`, `check_answer.py`, `prompt.txt`, `code.txt`, `results.csv`
- `logs/`: every perplexity, bench, quantize and prompt log, each prompt's output, and the three terminal recordings
  the video replays (`*-terminal.cast`, with the `nvidia-smi` pane beside each as `*-nvidia-smi.cast`; play them with
  `asciinema play`). The logits (`*.kld`, 7.6 GB each) and the models aren't included.
