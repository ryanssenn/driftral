# F16 activations with f32 accumulation

Goal: reduce activation bandwidth by storing selected activations as f16 while keeping dot-product accumulation in f32 for numerical stability.

This is a future implementation note. Current behavior remains Q8F16 weights with f32 activations and f32 accumulation.

## Current state

The current hot path uses Q8F16 model weights:

- `fp16_t` linear weights for attention projections, down projection, embeddings, norms, and LM head when the model uses f16 linear weights.
- `int8_t` MLP gate/up weights with per-group scales.
- `Tensor<float>` activations and outputs for hidden state, q/k/v state, MLP intermediates, logits, and most scratch tensors.
- `float` accumulation inside the CPU matmul dot products.

The kernel API already has the right template shape for mixed precision:

```cpp
template <typename WeightT, typename ActivationT, typename AccumT>
void matmul(Tensor<AccumT>& xout, Tensor<WeightT>& w, Tensor<ActivationT>& x);
```

The current explicit combinations are:

```cpp
matmul<float, float, float>
matmul<fp16_t, float, float>
matmul<int8_t, float, float>
```

## Target math

For f16 activations, the preferred first target is:

```text
f16 weights x f16 activations -> f32 accumulation
int8 weights x f16 activations -> f32 accumulation
```

A dot product should keep the long reduction in f32:

```text
float acc = 0
for each element:
    acc += float(weight_i) * float(activation_i)
output = acc or f32_to_fp16(acc)
```

This keeps the numerically sensitive part of the operation in f32 while allowing activations to use half the memory bandwidth.

## Why keep f32 accumulators

Mistral matmul rows sum thousands of terms:

- Attention projections often reduce over `4096` values.
- MLP down projection reduces over `14336` values.
- LM head reduces over `4096` values for each vocab row.

Accumulating all of those products in f16 can introduce enough rounding error to move logits, alter top-k ordering, or increase perplexity. F32 accumulation is the safe default for preserving current behavior while experimenting with f16 storage and f16 multiply inputs.

## Where f16 activations are likely safe first

Good first candidates:

- `hidden_state`
- `q_state`, `k_state`, `v_state`
- `context`
- `mlp_gate`
- `mlp_up`

These tensors dominate bandwidth and feed matmul-heavy paths.

Keep f32 initially for:

- `logits`
- `probs`
- softmax workspace
- RMSNorm reductions
- any test/debug comparison output

The first implementation should prefer f16 inputs with f32 outputs before making outputs f16. That isolates the effect of f16 activation reads from the additional rounding caused by storing outputs as f16.

## Kernel combinations to add

Add explicit combinations rather than changing current kernels in place:

```cpp
dot_product<fp16_t, fp16_t, float>
dot_product<int8_t, fp16_t, float>

matmul<fp16_t, fp16_t, float>
matmul<int8_t, fp16_t, float>
```

Later, once accuracy is understood, add f16-output variants:

```cpp
matmul<fp16_t, fp16_t, fp16_t>
matmul<int8_t, fp16_t, fp16_t>
```

## Implementation path

1. Add a temporary f16 activation shadow path.
   - Convert the existing `Tensor<float>` activation input to a `Tensor<fp16_t>` scratch buffer before selected matmuls.
   - Keep matmul outputs as `Tensor<float>`.
   - Compare against current tests and perplexity.

2. Add f16 activation matmul specializations.
   - Implement `dot_product<fp16_t, fp16_t, float>`.
   - Implement `dot_product<int8_t, fp16_t, float>`.
   - Keep int8 grouped dequant inside the row kernel.

3. Move selected `InferenceState` tensors to f16.
   - Start with MLP intermediates or attention projection outputs.
   - Keep logits/probs f32.
   - Avoid broad tensor storage changes until the shadow path passes.

4. Decide output precision per tensor.
   - Use f32 output where accuracy matters most.
   - Test f16 output for hidden/intermediate tensors after f16 input is stable.

## Metal relevance

This matters most for Metal:

- f16 activations halve activation bandwidth.
- Apple GPUs are optimized for f16 throughput.
- f16 x f16 inputs with f32 accumulation are a strong first target.

On CPU, f16 activations may not be faster enough to justify the conversion cost. On Metal, keeping activations device-resident as f16 should reduce both memory traffic and kernel time.

## Verification

Correctness gates:

```bash
cmake --build build
./build/test_exec
./perplexity.sh --check
```

Performance gate:

```bash
./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0
```

Acceptance for early experiments:

- `./build/test_exec` passes.
- Perplexity delta stays near baseline.
- Greedy output remains stable for short prompts, or drift is documented.
- tok/s improves once conversion overhead is removed or activations are device-resident.

## Baseline results

These are current CPU/Q8F16 baseline results before f16 activation changes.

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.23808`, `tokens: 33`, `delta=+0.0000` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `5.11822 tok/s` |

## F16 accumulator experiment

Experiment: change `dot_product<fp16_t, float, float>` so the f16-weight matmul path accumulates internally in f16, while still returning/storing a float output. The int8 path was unchanged.

Result:

| Date | Command | Result |
|------|---------|--------|
| 2026-06-22 | `cmake --build build && ./build/test_exec` | Passed: `24 / 24` |
| 2026-06-22 | `./perplexity.sh --check` | `Q8F16 PPL: 5.24101`, `tokens: 33`, `delta=+0.0029` vs baseline `5.23808` |
| 2026-06-22 | `./build/qmog-cli mistral-7B-Q8F16.mog "Paris is the capital of" --temp 0` | `6.62252 tok/s` |

Conclusion: f16 accumulation improved one local greedy throughput run and the small perplexity movement is acceptable for this project. Use f16 accumulation in the CPU f16-weight matmul path.
