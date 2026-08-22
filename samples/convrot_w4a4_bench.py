import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import hip

SHAPES = {
    "qkv": (21504, 5376),
    "mlp_up": (28672, 5376),
    "mlp_down": (5376, 14336),
    "attn_out": (5376, 7168),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck-root", type=Path, required=True)
    parser.add_argument("--shape", choices=["all", *SHAPES], default="all")
    parser.add_argument("--rows", type=int, default=9170)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


args = parse_args()
ck_root = args.ck_root.resolve()


if ck_root not in Path(ck.__file__).resolve().parents:
    raise RuntimeError(f"Imported comfy_kitchen from {ck.__file__}, expected {ck_root}")
if torch.version.hip is None or torch.cuda.get_device_properties(0).gcnArchName < "gfx1150":
    raise RuntimeError("This benchmark requires a gfx1150-or-newer ROCm device")


def check_quantizer():
    torch.manual_seed(0)
    weight = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    q_hip, scale_hip = hip.quantize_convrot_w4a4_weight(weight, 256)
    with ck.use_backend("eager"):
        q_eager, scale_eager = ck.quantize_convrot_w4a4_weight(weight, 256)

    def unpack(q):
        q = q.to(torch.int16)
        lo = q & 0xF
        hi = (q >> 4) & 0xF
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        return lo, hi

    lo_hip, hi_hip = unpack(q_hip)
    lo_eager, hi_eager = unpack(q_eager)
    delta = torch.maximum((lo_hip - lo_eager).abs(), (hi_hip - hi_eager).abs())
    torch.testing.assert_close(scale_hip, scale_eager, rtol=1e-2, atol=0)
    if delta.max().item() > 1:
        raise AssertionError(f"packed int4 delta {delta.max().item()} exceeds 1")
    print(json.dumps({"check": "passed", "max_int4_delta": delta.max().item()}))


def measure(fn):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def benchmark(name, n, k):
    m = args.rows
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    qweight = torch.randint(-128, 128, (n, k // 2), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01
    qact = torch.empty((m, k // 2), device="cuda", dtype=torch.int8)
    act_scale = torch.empty(m, device="cuda", dtype=torch.float32)

    def quantize():
        hip._C.convrot_quant_int4(
            hip._dl(x), hip._dl(qact), hip._dl(act_scale), m, k, 256, hip._stream(x)
        )

    def linear():
        return hip.convrot_w4a4_linear(x, qweight, weight_scale, None, 256)

    result = {"shape": name, "m": m, "n": n, "k": k}
    for label, fn in (("quant", quantize), ("full", linear)):
        samples = measure(fn)
        result[f"{label}_median_ms"] = statistics.median(samples)
        result[f"{label}_samples_ms"] = samples
    print(json.dumps(result), flush=True)
    return result


if args.check:
    check_quantizer()
else:
    selected = SHAPES.items() if args.shape == "all" else [(args.shape, SHAPES[args.shape])]
    results = []
    for name, (n, k) in selected:
        results.append(benchmark(name, n, k))
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "backend": str(ck_root),
                "device": torch.cuda.get_device_name(0),
                "arch": torch.cuda.get_device_properties(0).gcnArchName,
                "torch": torch.__version__,
                "results": results,
            }
        )
    )
