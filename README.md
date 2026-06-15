# Torchless

From-scratch C++ implementation of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU. It runs local text completion with its own tensors, quantized weight loader, BPE tokenizer, manual memory management with KV caching, and the full decoder architecture.

Educational project, meant for understanding how inference works by reading the code, not as a production inference engine.

<br>

![demo2](https://github.com/user-attachments/assets/1711dc3e-9ab2-4f73-8c35-b7ac3aabec55)

## How it works

If you are new to LLM internals, an inference engine is essentially a loop that predicts the next word in a sequence, adds it to the history, and repeats. Here is the full lifecycle:

### Loading
Before we can run any math, we need the weights. The `export_mistral.py` script takes the complex Hugging Face folder structure and packs the weights into a single standardized binary file. The C++ engine loads this entire file into RAM at startup so the data is mapped and ready for computation.

### Tokenization
The model performs math on numbers, not strings. When you type a prompt like "Paris is", the `tokenizer` breaks it down using byte-pair Encoding (BPE). It looks up these chunks in the Mistral vocabulary and converts them into a list of integer IDs (e.g. [1, 782, 312]).

### Transformer Loop
We feed these IDs into the model one by one. The goal is to update a single vector, the `hidden_state`, as it passes through the network.

* **Embedding:** We take the input `token ID`and look up its specific floating-point vector in the embedding table. This turns a simple integer into a dense vector representing the token's initial semantic meaning.
* **Layers:** This state travels through 32 identical layers. In every layer, we first apply `RMSNorm` to stabilize the numbers. Then the state enters the `attention` module. It projects the state into `query`, `key`, and `value` vectors. The query "looks back" at the Keys of previous tokens to find relevant information (values). We apply `RoPE` (Rotary Positional Embeddings) so the model understands relative distance between words, then store the key and value in the `KV Cache`. This cache acts as the model's short-term memory, saving us from recalculating the history for every new word.
* **MLP:** Finally, the state goes through the `feedforward` module (a SwiGLU block). If Attention gathers context from the past, the MLP processes that information. It projects the vector to a higher dimension (14,336) to untangle complex relationships, applies a non-linear activation (SiLU), and projects it back down.

### Prediction
After 32 layers of processing, the final `hidden_state` holds the "meaning" of the next predicted token. We project this vector against the entire vocabulary to get `logits` raw confidence scores for all 32,000 possible next tokens. We run a `softmax` operation to turn these scores into probabilities and `sample` the result (either choosing the most likely token or picking randomly based on the probability distribution). We decode that ID back into text, print it, and feed it back into the transformer.


# Running

#### Download Mistral 7B v0.1, Torchless

```
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/torchless.git
```

#### (Optional) Create Python virtual environment and download libraries

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Export the model with 8-bit quantization

```
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant int8
```

#### Compile project

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

#### Run
```
./torchless ../mistral.bin "Paris is the capital of"
```

If you run into issues that appear specific to your environment, feel free to open a GitHub issue.

# Testing

To check that the C++ code matches the real Mistral implementation, I validated each component separately rather than only checking end-to-end output.

First, the Python scripts in `scripts/test/mistral/` run individual pieces, attention, RMSNorm, RoPE, MLP, etc. using Hugging Face's Mistral with the actual weights. Each script dumps its output tensors into `test/mistral/expected.txt` as named float arrays.

Then the C++ tests in `test/mistral/` load those values and compare them against the output of the corresponding Torchless code. For example, an attention test copies a known `hidden_state` from the golden file, runs `Attention::forward`, and checks that Q/K/V and the output match. The same pattern is used for the tokenizer, CPU kernels (matmul, softmax, RoPE, SiLU), and each decoder module. Comparisons use a tolerance of ±0.05.

```bash
cd build
./test_exec
```

This expects `../mistral.bin` at the repo root. Which tests run depends on the quant mode of the loaded binary, f32 and int8 each have their own cases. int8 attention tests are not enabled yet.

# Roadmap

Full progress tracker: [ROADMAP.md](ROADMAP.md). Still todo: temperature scaling, terminal chat interface, fp8, SIMD, CUDA.


# Resources

#### Some of the material that helped me learn the theory or guided me build the engine

##### ML Theory
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)

##### Systems Internals
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [C++ Vtables](https://shaharmike.com/cpp/vtable-part1/)
- [Andrew Chan - yalm](https://andrewkchan.dev/posts/yalm.html)
- [Arseny Kapoulkine - LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/)
- [Maxime Labonne - Quantize llama models](https://medium.com/data-science/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)

##### Reference Implementations
- [Hugging Face - Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Arseny Kapoulkine - calm](https://github.com/zeux/calm/tree/main)
- [Georgi Gerganov - llama.cpp + ggml](https://github.com/ggml-org/llama.cpp/)
- [Andrej Karpathy - llama2.c quantization](https://github.com/karpathy/llama2.c/blob/master/export.py)


# License

MIT