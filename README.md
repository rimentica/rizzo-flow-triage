# rizzo-flow-triage

English · [Italiano](README.it.md)

[rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) on an **NVIDIA
Jetson Orin Nano Super**: the changes needed to run it on the board, the tests
I ran and the results.

rizzo-flow turns an LLM into a source of typed decisions: each allowed answer
becomes a letter, and the decision is read from the probabilities of those
letters, without generating tokens. Its README asks for reports from Linux and
new hardware; this repository is one, for a Jetson Orin Nano Super (8 GB shared
between CPU and GPU, 25 W mode, JetPack 6).

## What had to change

The board runs JetPack 6 (L4T R36.4.7, CUDA 12.6). Three obstacles, in the
order they showed up.

1. **The prebuilt runtime targets a newer CUDA.** On Linux arm64 with an NVIDIA
   GPU, `rizzo download` picks the official llama.cpp build
   `ubuntu-cuda-13.4-arm64`, which needs a CUDA 13 driver; JetPack 6 ships CUDA
   12.6. I did not try that build: I compiled llama.cpp from source at the
   commit rizzo-flow pins (`161755f`), with CUDA for the Orin GPU (`sm_87`) and
   only the `llama` shared library, and passed it to rizzo-flow with
   `RIZZO_LLAMA_DIR`, as its README describes. The build took 34 minutes on the
   board.
2. **The model was in memory twice.** On the Jetson, CPU and GPU share the same
   RAM. By default llama.cpp memory-maps the GGUF file (2.6 GB in page cache)
   and copies the weights to the GPU: here both copies live in the same 8 GB,
   and allocating the KV cache failed (`NvMapMemAllocInternalTagged ... error
   12`, `cudaMalloc failed: out of memory`) even with 2 GB free and a reduced
   context. Dropping the file from the page cache with `posix_fadvise` did not
   help: loading the model fills it again.
3. **Loading without mmap: the patch.** llama.cpp at that commit can read the
   model with direct I/O (`LLAMA_LOAD_MODE_DIRECT_IO = 4` in `llama.h`), which
   bypasses the page cache, but rizzo-flow does not expose the setting.
   `patch/rizzo-flow-load-mode.patch` adds four lines to
   `src/rizzo_flow/llama_cpp.py`: if the `RIZZO_LLAMA_LOAD_MODE` environment
   variable is set, its value goes into `model_params.load_mode` before the
   model is loaded. Without the variable nothing changes. With
   `RIZZO_LLAMA_LOAD_MODE=4` and `--ctx 1024` the server starts, with a 2.68 GB
   peak on the device.

```diff
         model_params.n_gpu_layers = GPU_LAYERS if chosen else 0
+        # Jetson and other integrated GPUs: with mmap the GGUF takes RAM twice (page
+        # cache + GPU copy) and cudaMalloc fails. RIZZO_LLAMA_LOAD_MODE=4 uses direct I/O.
+        if os.environ.get("RIZZO_LLAMA_LOAD_MODE"):
+            model_params.load_mode = int(os.environ["RIZZO_LLAMA_LOAD_MODE"])
         model = library.llama_model_load_from_file(str(gguf).encode(), model_params)
```

## Tests

The workload is the triage of internal IT requests written in Italian: three
typed decisions per request.

| Decision | Type | Values |
| --- | --- | --- |
| `reparto` (team) | choice | `salesforce`, `m365`, `report`, `it` |
| `tipo` (kind) | choice | `guasto` (incident), `richiesta` (request), `domanda` (question) |
| `urgente` (urgent) | boolean | blocks someone with a deadline today/tomorrow, or stops several people |

The three questions share the same state (the request text): the prefix is
computed once, then rizzo-flow reads only the logits of the allowed answer
letters. As a baseline, the same requests go through Qwen 3.5 1.9B Q8_0 on
Ollama, on the same board, which generates JSON constrained by a schema. Both
engines see exactly the same descriptions of teams and kinds, defined in
`triage.py`.

`dataset/richieste.jsonl`: 40 **synthetic** requests, written for this test,
10 per team; 19 incidents, 12 requests, 9 questions; 12 urgent. The labels were
written by the dataset author, not by a group of annotators: this is a
demonstration, not a benchmark.

## Results — 2026-10-12, Jetson Orin Nano Super, 25 W mode

| | rizzo-flow 4B Q4_K_M | Qwen 3.5 1.9B Q8_0 on Ollama |
| --- | ---: | ---: |
| How it decides | logits of the allowed answers | JSON generated with a schema |
| `reparto` (team) | **40/40** | **40/40** |
| `tipo` (kind) | **35/40** (87.5%) | 31/40 (77.5%) |
| `urgente` (urgent) | 35/40 (87.5%) | **39/40** (97.5%) |
| all three correct | 31/40 (77.5%) | 30/40 (75%) |
| ms per request, p50 / p95 | **1044 / 1068** | 1457 / 1587 |
| decisions per second | **2.86** | 2.05 |
| generated tokens | **0** | 1180 |
| probability for each option | yes | no |

rizzo-flow details (medians per request): state prefill 0.3 s, total inference
1.0 s, 144 prefix tokens shared by the three questions.

An honest reading:

- **The team is easy for both**: the descriptions name the systems.
- **rizzo-flow errs on the side of caution on urgency** (4 urgent requests
  missed: R14, R21, R26, R33; one false alarm, R07) and leans towards
  "request" for the kind. Qwen does the opposite: almost perfect urgency, more
  confusion on the kind (8 errors out of 9 involve "question", in both
  directions).
- **With 40 cases one request is worth 2.5 points**: the accuracy differences
  are not significant. The latency difference (about 30%) and the zero tokens
  are systematic.
- **Different models**: 4B against 1.9B. The comparison is between two ways of
  using an LLM on this hardware, not between two models of the same size.
- rizzo-flow probabilities are not calibrated (project README): to use them
  as thresholds you need labelled data from the domain.
- On the GPU in its README (RTX 5060 Ti) rizzo-flow reports about 50 ms for a
  short decision; here it is about 350 ms per decision: the Orin Nano is a
  25 W board.

Files: `risultati/rizzo-jetson-q4_k_m.json`, `risultati/ollama-qwen3.5-1.9b-q8_0.json`
(each row with expected and predicted values and timings; rizzo-flow also has
probabilities and `timing`).

## Setup on the Jetson

```bash
mkdir -p ~/rizzo-flow-lab && cd ~/rizzo-flow-lab
git clone https://github.com/rimentica/rizzo-flow-triage.git triage
git clone https://github.com/Rizzo-AI-Academy/rizzo-flow.git
git -C rizzo-flow checkout b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de
git -C rizzo-flow apply ../triage/patch/rizzo-flow-load-mode.patch
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 161755f29e415e2c33efe906e91843c068efd664
python3 -m pip install --user uv

# only libllama and ggml, shared, CUDA for Orin (sm_87)
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama -j 4

# Python environment and weights (fine-tuned 4B, Q4_K_M, 2.6 GB)
cd ../rizzo-flow
~/.local/bin/uv sync --locked
~/.local/bin/uv run rizzo download --only weights --quant q4_k_m
```

Everything lives in `~/rizzo-flow-lab`; nothing outside it is changed.

## Run

`triage.py` uses only the Python standard library.

```bash
cd ~/rizzo-flow-lab/rizzo-flow
RIZZO_LLAMA_DIR=~/rizzo-flow-lab/llama.cpp/build/bin RIZZO_LLAMA_LOAD_MODE=4 \
  ~/.local/bin/uv run rizzo serve --quant q4_k_m --ctx 1024 --device cuda

# from another terminal
cd ~/rizzo-flow-lab/triage
python3 triage.py --rizzo http://127.0.0.1:8017 --out risultati/rizzo-jetson-q4_k_m.json
# stop rizzo-flow, then, with the name of the model in your Ollama
python3 triage.py --ollama MODEL --out risultati/ollama.json
```

With 8 GB shared between CPU and GPU the two models do not fit together:
measure one engine at a time.

## Possible next steps

- Expose the load mode in rizzo-flow as a `--load-mode` option instead of a
  patch: proposed in [rizzo-flow#31](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/31), waiting for the maintainer.
- Make urgency more reliable with sharper instructions, or calibrate the
  probabilities on labelled requests.
- Try the smaller rizzo-flow model for latency.

## License

Code, dataset and results: Apache-2.0 (`LICENSE`). The patch in `patch/`
modifies a file of [rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow),
which is also Apache-2.0; the modified file carries a comment describing the
change. Model weights are not included: `rizzo download` fetches them.
