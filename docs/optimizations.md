# Optimizations

This document tracks optimization changes, their measured results, and whether they should stay enabled.

## F16 activation input and accumulation

Status: enabled for the CPU f16-weight matmul path.

Change:

- Convert the f32 activation vector to f16 inside the f16-weight dot-product kernel.
- Accumulate the f16-weight dot product in f16 lanes.
- Return the final row result as float so the rest of the model path remains unchanged.

Why it helps:

- F16 multiply and accumulation uses wider NEON half-precision vectors.
- The kernel does less f32 conversion and f32 arithmetic in the hot f16-weight matmul path.
- The measured perplexity movement is small enough to accept for the throughput gain.

Results:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.24101`, `tokens: 33`, `delta=+0.0029` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `6.62252 tok/s` vs baseline `5.11822 tok/s` |

Conclusion: this is a win. The small perplexity increase is acceptable for the measured throughput improvement.

## Int8 ARM dot-product matmul

Status: enabled for the CPU int8-weight matmul path.

Change:

- Quantize the activation vector to int8 once per int8 matmul.
- Use per-64-element activation scales to match the existing int8 weight scale groups.
- Use ARM `vdotq_s32` to compute int8 x int8 dot products into int32 lanes, then apply weight and activation dequantization scales.

Why it helps:

- MLP `gate_proj` and `up_proj` are the largest repeated matmuls in the current Q8F16 model.
- The old kernel widened int8 weights to f32 and used f32 FMA for every element.
- The new path uses native int8 dot-product instructions and reduces repeated conversion work.

Results:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.24294`, `tokens: 33`, `delta=+0.0049` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `7.88188 tok/s` vs Release control `7.12923 tok/s` |

Conclusion: this is a win. The per-group activation quantization variant keeps the test suite passing and improves throughput.

Note: a single activation scale for the whole matmul reached `7.9856 tok/s` and `Q8F16 PPL: 5.04486`, but failed `test logits multi top10` with a top-1 flip. Keep the per-group variant.

## Accelerate f32 matvec

Status: not enabled.

Experiment:

- Replace `matmul<float, float, float>` with Apple Accelerate `cblas_sgemv`.
- Link the macOS Accelerate framework.

Results:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.22887`, `tokens: 33`, `delta=-0.0092` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `7.06257 tok/s` vs Release control `7.12923 tok/s` |

Conclusion: do not keep this for the current Q8F16 path. It passes tests, but it is slightly slower than the existing f32 matmul control and most hot matmuls are f16 or int8 anyway.
