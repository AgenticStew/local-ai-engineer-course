#!/usr/bin/env python3
"""What's really inside a GGUF: the format of every tensor, added up, read from the header alone.

  python3 gguf_mix.py FILE_OR_URL [--parts] [--json]

FILE_OR_URL  a local .gguf, or a Hugging Face "resolve" URL. For a URL only the header is fetched (HTTP range
             requests, a few MB), so you can check a file before downloading it. For a split model, give the
             first part (...-00001-of-0000N.gguf) and the other parts are read too.
--parts      also list each kind of tensor (attn_q, ffn_down, ...) and the formats it uses, plus the file's
             quantize.* metadata (which importance matrix was used)
--json       print everything as JSON instead, one entry per tensor

The arithmetic is llama.cpp's: a tensor's size is (elements / block size) x bytes per block, with the block
sizes from gguf-py/gguf/constants.py (GGML_QUANT_SIZES), and real bits per weight = 8 x total bytes / total
elements, as in llama.cpp's "file size = ... (... BPW)" line. Needs only Python 3: no GPU, no packages.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import urllib.request
from collections import Counter, defaultdict

# ggml type id: (name, weights per block, bytes per block). llama.cpp 785322d, gguf-py/gguf/constants.py
QK_K = 256
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", QK_K, 2 + 2 + QK_K // 16 + QK_K // 4), 11: ("Q3_K", QK_K, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", QK_K, 2 + 2 + QK_K // 2 + 12), 13: ("Q5_K", QK_K, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", QK_K, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16), 15: ("Q8_K", QK_K, 4 + QK_K + QK_K // 8),
    16: ("IQ2_XXS", QK_K, 2 + QK_K // 4), 17: ("IQ2_XS", QK_K, 2 + QK_K // 4 + QK_K // 32),
    18: ("IQ3_XXS", QK_K, 2 + QK_K // 4 + QK_K // 8), 19: ("IQ1_S", QK_K, 2 + QK_K // 8 + QK_K // 16),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", QK_K, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: ("IQ2_S", QK_K, 2 + QK_K // 4 + QK_K // 16), 23: ("IQ4_XS", QK_K, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", QK_K, QK_K // 8 + QK_K // 16 + QK_K // 32), 30: ("BF16", 1, 2),
    34: ("TQ1_0", QK_K, 2 + 4 * 13), 35: ("TQ2_0", QK_K, 2 + 64), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1),
}
# general.file_type -> the label llama.cpp prints (src/llama-model-loader.cpp, llama_ftype_name)
FTYPES = {
    0: "all F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K - Medium",
    11: "Q3_K - Small", 12: "Q3_K - Medium", 13: "Q3_K - Large", 14: "Q4_K - Small", 15: "Q4_K - Medium",
    16: "Q5_K - Small", 17: "Q5_K - Medium", 18: "Q6_K", 19: "IQ2_XXS - 2.0625 bpw", 20: "IQ2_XS - 2.3125 bpw",
    21: "Q2_K - Small", 22: "IQ3_XS - 3.3 bpw", 23: "IQ3_XXS - 3.0625 bpw", 24: "IQ1_S - 1.5625 bpw",
    25: "IQ4_NL - 4.5 bpw", 26: "IQ3_S - 3.4375 bpw", 27: "IQ3_S mix - 3.66 bpw", 28: "IQ2_S - 2.5 bpw",
    29: "IQ2_M - 2.7 bpw", 30: "IQ4_XS - 4.25 bpw", 31: "IQ1_M - 1.75 bpw", 32: "BF16", 36: "TQ1_0 - 1.69 bpw ternary",
    37: "TQ2_0 - 2.06 bpw ternary", 38: "MXFP4 MoE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0", 42: "F8_E4M3",
}


class Source:
    """Sequential reader over a local file or a URL. A URL is fetched in growing ranges, never in full."""

    def __init__(self, where: str):
        self.where, self.pos, self.buf, self.start = where, 0, b"", 0
        self.url = where.startswith(("http://", "https://"))
        self.f = None if self.url else open(where, "rb")
        self.chunk = 1 << 20

    def _fetch(self, need: int):
        while self.start + len(self.buf) < self.pos + need:
            lo = self.start + len(self.buf)
            req = urllib.request.Request(self.where, headers={"Range": f"bytes={lo}-{lo + self.chunk - 1}",
                                                              "User-Agent": "gguf_mix/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if r.status == 200 and lo:           # server ignored the range: it sent the whole file
                    raise SystemExit("the server doesn't support range requests; download the file instead")
            if not data:
                raise SystemExit(f"{self.where}: ended inside the header")
            if self.pos - self.start > (64 << 20):   # drop what has been parsed
                cut = self.pos - self.start
                self.buf, self.start = self.buf[cut:], self.start + cut
            self.buf += data
            self.chunk = min(self.chunk * 2, 16 << 20)

    def read(self, n: int) -> bytes:
        if self.f:
            b = self.f.read(n)
            if len(b) < n:
                raise SystemExit(f"{self.where}: ended inside the header")
            return b
        self._fetch(n)
        i = self.pos - self.start
        self.pos += n
        return self.buf[i:i + n]

    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def string(self): return self.read(self.u64()).decode("utf-8", "replace")


SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def value(s: Source, t: int):
    if t in SCALAR:
        fmt = SCALAR[t]
        return struct.unpack(fmt, s.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return s.string()
    if t == 9:
        et, n = s.u32(), s.u64()
        if et in SCALAR:                              # skip big numeric arrays in one read
            size = struct.calcsize(SCALAR[et])
            s.read(size * n)
            return f"[{n} values]"
        items = [value(s, et) for _ in range(n)]
        return items if n <= 16 else f"[{n} items]"
    raise SystemExit(f"unknown metadata type {t}")


def read_header(where: str) -> tuple[dict, list[dict]]:
    s = Source(where)
    if s.read(4) != b"GGUF":
        raise SystemExit(f"{where}: not a GGUF file")
    version = s.u32()
    if version < 2:
        raise SystemExit(f"{where}: GGUF version {version} is too old")
    n_tensors, n_kv = s.u64(), s.u64()
    meta = {}
    for _ in range(n_kv):
        k = s.string()
        meta[k] = value(s, s.u32())
    tensors = []
    for _ in range(n_tensors):
        name = s.string()
        dims = [s.u64() for _ in range(s.u32())]
        t = s.u32()
        s.u64()                                       # offset into the data section
        if t not in TYPES:
            raise SystemExit(f"{name}: unknown tensor type {t}")
        fmt, blk, bpb = TYPES[t]
        n = 1
        for d in dims:
            n *= d
        tensors.append({"name": name, "format": fmt, "shape": dims, "weights": n, "bytes": n // blk * bpb})
    return meta, tensors


def split_parts(where: str, meta: dict) -> list[str]:
    count = int(meta.get("split.count", 1) or 1)
    if count <= 1:
        return [where]
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf", where)
    if not m:
        raise SystemExit(f"{where} says it is part of {count} files, but its name has no -0000N-of-0000M")
    return [where[:m.start()] + f"-{i:05d}-of-{count:05d}.gguf" + where[m.end():] for i in range(1, count + 1)]


def part_of(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "", name).removesuffix(".weight")


def analyse(where: str) -> dict:
    meta, tensors = read_header(where)
    parts = split_parts(where, meta)
    for p in parts[1:]:
        tensors += read_header(p)[1]
    weights = sum(t["weights"] for t in tensors)
    nbytes = sum(t["bytes"] for t in tensors)
    by = defaultdict(lambda: {"weights": 0, "bytes": 0, "tensors": 0})
    for t in tensors:
        b = by[t["format"]]
        b["weights"] += t["weights"]; b["bytes"] += t["bytes"]; b["tensors"] += 1
    formats = []
    for fmt, b in sorted(by.items(), key=lambda kv: (-kv[1]["weights"], kv[0])):
        _, blk, bpb = next(v for v in TYPES.values() if v[0] == fmt)
        formats.append({"format": fmt, "bits_per_weight": 8 * bpb / blk, "share": b["weights"] / weights, **b})
    ft = meta.get("general.file_type")
    return {
        "file": re.sub(r"\?.*$", "", where).rsplit("/", 1)[-1],
        "files": len(parts),
        "label": FTYPES.get(ft, f"unknown ({ft})") if ft is not None else "not stated",
        "imatrix": any(k.startswith("quantize.imatrix.") for k in meta),
        "quantize_meta": {k: v for k, v in meta.items() if k.startswith("quantize.")},
        "architecture": meta.get("general.architecture"),
        "layers": meta.get(f"{meta.get('general.architecture')}.block_count"),
        "weights": weights, "bytes": nbytes, "bits_per_weight": 8 * nbytes / weights,
        "formats": formats, "tensors": tensors,
    }


def report(r: dict, parts: bool) -> str:
    out = [r["file"] + (f"  ({r['files']} files)" if r["files"] > 1 else ""),
           f"{'Label in the file':<21}{r['label']}",
           f"{'Importance matrix':<21}{'yes' if r['imatrix'] else 'no'}",
           f"{'Weights':<21}{r['weights'] / 1e9:.2f} B",
           f"{'Weight data':<21}{r['bytes'] / 1e9:.2f} GB",
           f"{'Real bits per weight':<21}{r['bits_per_weight']:.2f}",
           "",
           f"{'Format':<9}{'Bits/weight':>12}{'Share of weights':>18}{'Size':>11}{'Tensors':>9}"]
    for f in r["formats"]:
        out.append(f"{f['format']:<9}{f['bits_per_weight']:>12.2f}{f['share'] * 100:>17.1f}%"
                   f"{f['bytes'] / 1e9:>8.2f} GB{f['tensors']:>9}")
    if parts:
        groups = defaultdict(Counter)
        for t in r["tensors"]:
            groups[part_of(t["name"])][t["format"]] += 1
        out += ["", f"{'Part':<24}Formats (tensors)"]
        for name in sorted(groups, key=lambda g: (-sum(groups[g].values()), g)):
            mix = ", ".join(f"{fmt} {n}" for fmt, n in groups[name].most_common())
            out.append(f"{name:<24}{mix}")
        if r["quantize_meta"]:
            out += [""] + [f"{k} = {v}" for k, v in r["quantize_meta"].items()]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if len(args) != 1 or flags - {"--parts", "--json"}:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    r = analyse(args[0])
    if "--json" in flags:
        print(json.dumps(r, indent=1))
    else:
        print(report(r, "--parts" in flags))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
