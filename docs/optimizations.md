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
