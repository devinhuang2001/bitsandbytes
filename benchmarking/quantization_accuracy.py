"""Compare reconstruction error and stored bytes at matched NF4/affine INT4 group sizes.

The affine reference follows the round-to-nearest quantizer in issue #2086:
integer zero points clipped to [0, 15], FP16 scales/zero points, and no calibration.
It is a benchmark reference, not a supported bitsandbytes storage or compute format.
"""

import argparse
import json
from pathlib import Path
import re

import torch

import bitsandbytes as bnb


def tensor_bytes(*tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def affine_int4(weight, blocksize):
    if weight.ndim != 2 or blocksize <= 0 or weight.shape[-1] % blocksize or weight.shape[-1] % 2:
        raise ValueError("Expected a 2-D weight with an even row length divisible by a positive blocksize")
    blocks = weight.float().reshape(weight.shape[0], -1, blocksize)
    lo = blocks.amin(-1, keepdim=True)
    hi = blocks.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    zero = (-lo / scale).round().clamp(0, 15)
    codes = (blocks / scale + zero).round().clamp(0, 15).to(torch.uint8).reshape(weight.shape)
    # Adjacent values occupy low/high nibbles; layout does not affect RMSE or storage size.
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scale = scale.squeeze(-1).half()
    zero = zero.squeeze(-1).half()
    # Reconstruct from packed codes and the actual stored FP16 metadata.
    unpacked = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(blocks.shape).float()
    restored = ((unpacked - zero.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)).reshape(weight.shape)
    return restored.to(weight.dtype), packed, scale, zero


def distribution_stats(weight, blocksize):
    blocks = weight.float().reshape(-1, blocksize)
    centered = blocks - blocks.mean(-1, keepdim=True)
    variance = centered.square().mean(-1)
    std = variance.sqrt()
    valid = variance > 0
    # Constant blocks contribute zero to normalized moments instead of NaN.
    kurtosis = torch.where(valid, centered.pow(4).mean(-1) / variance.square().clamp(min=1e-38), 0)
    skew = torch.where(valid, centered.pow(3).mean(-1) / std.pow(3).clamp(min=1e-38), 0)
    max_over_std = torch.where(valid, blocks.abs().amax(-1) / std.clamp(min=1e-38), 0)
    return {
        "mean_block_kurtosis": kurtosis.mean().item(),
        "mean_block_skew": skew.mean().item(),
        "mean_block_max_abs_over_std": max_over_std.mean().item(),
    }


@torch.no_grad()
def compare_weight(weight, blocksize):
    affine, packed, scale, zero = affine_int4(weight, blocksize)
    quantized, state = bnb.functional.quantize_nf4(weight, blocksize=blocksize, compress_statistics=False)
    nf4 = bnb.functional.dequantize_nf4(quantized, state)

    def measurement(restored, values, *metadata):
        packed_bytes = tensor_bytes(values)
        metadata_bytes = tensor_bytes(*metadata)
        return {
            "rmse": (restored.float() - weight.float()).square().mean().sqrt().item(),
            "packed_bytes": packed_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": packed_bytes + metadata_bytes,
        }

    return {
        "blocksize": blocksize,
        "distribution": distribution_stats(weight, blocksize),
        "nf4": measurement(nf4, quantized, state.absmax, state.code),
        "affine_int4": measurement(affine, packed, scale, zero),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint", type=Path, help="Local torch tensor state_dict (loaded with weights_only=True)"
    )
    source.add_argument("--model", help="Hugging Face model name or local model directory; requires transformers")
    parser.add_argument("--revision", default="main", help="Model revision; prefer a commit SHA for reproducibility")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blocksize", type=int, nargs="+", default=[32, 64, 128], choices=[32, 64, 128, 256])
    parser.add_argument("--pattern", default=r".*_proj\.weight$", help="Regex selecting 2-D weights by name")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float16)
        weights = model.state_dict()
        resolved_revision = getattr(model.config, "_commit_hash", None)
    else:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        resolved_revision = None

    records = []
    for name, source_weight in weights.items():
        if not re.search(args.pattern, name) or not isinstance(source_weight, torch.Tensor) or source_weight.ndim != 2:
            continue
        weight = source_weight.detach().to(device=args.device, dtype=torch.float16).contiguous()
        for blocksize in args.blocksize:
            if weight.shape[-1] % blocksize or weight.shape[-1] % 2:
                records.append({"name": name, "blocksize": blocksize, "skipped": "row length incompatible with group"})
                continue
            record = {"name": name, "shape": list(weight.shape), **compare_weight(weight, blocksize)}
            records.append(record)
            print(json.dumps(record), flush=True)
        del weight
    if not any("nf4" in record for record in records):
        raise SystemExit("No eligible weights measured; check --pattern and group sizes")

    report = {
        "source": str(args.checkpoint) if args.checkpoint else args.model,
        "requested_revision": args.revision if args.model else None,
        "resolved_revision": resolved_revision,
        "torch": str(torch.__version__),
        "bitsandbytes": bnb.__version__,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else args.device,
        "assumptions": {
            "reference_and_reconstruction_dtype": "float16",
            "rounding": "nearest; no activation calibration",
            "nf4_compress_statistics": False,
            "affine_metadata": "float16 scale and clipped integer zero point stored as float16 per group",
            "storage_accounting": "tensor payload bytes; includes NF4 codebook; excludes container/allocator overhead",
            "constant_block_statistics": "zero normalized moments for zero-variance blocks",
        },
        "results": records,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
