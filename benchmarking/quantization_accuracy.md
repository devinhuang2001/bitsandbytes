# Reconstruction accuracy at matched group sizes

`quantization_accuracy.py` supports the investigation in [issue #2086](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/2086).
It compares the existing NF4 format with a PyTorch affine INT4 reference at the
same block/group sizes, using FP16 source weights and FP16 reconstructions.
It does not benchmark inference speed or measure perplexity.

From the repository root, with bitsandbytes installed:

```bash
python -m benchmarking.quantization_accuracy \
  --model Qwen/Qwen2.5-0.5B --revision MODEL_COMMIT_SHA \
  --device cuda --blocksize 32 64 128 --output accuracy.json
```

The model option requires `transformers`. Alternatively, pass a local tensor
state dictionary with `--checkpoint weights.pt`. The default name pattern selects
projection weights ending in `_proj.weight`; use `--pattern` to select other
matrices. Row lengths incompatible with a group size are recorded as skipped.

The JSON contains a result for each selected layer and each group size:

- Reconstruction RMSE against the same FP16 source, with no activation calibration.
- Actual packed tensor, metadata, and total payload bytes. NF4 statistics are
  uncompressed FP32, and the 64-byte codebook is included for every weight.
  Affine metadata consists of FP16 scales and FP16 integer zero points.
- Mean per-block population kurtosis, skew, and maximum absolute value divided
  by standard deviation. Constant blocks contribute zero to these normalized
  statistics. These are block statistics, rather than moments of a whole layer.
- Package versions, device, model revision, shape, group size, and assumptions.

The affine reference uses the issue's quantization convention: a scale derived
from the block range, integer zero points clipped to `[0, 15]`, and rounding in
FP32 before storing metadata in FP16. Reconstruction uses the stored metadata.
In particular, clipping a zero point can saturate one-sided distributions;
constant blocks and FP16 scale underflow can also have high reconstruction error.
The reference is not a proposed bitsandbytes format or kernel.

Payload bytes exclude framework objects, serialization headers, and allocator
overhead. Matching group sizes does not force identical total bytes: the fixed
NF4 codebook is counted explicitly. RMSE rankings do not establish end-to-end
quality, throughput, or a universally preferable format.
