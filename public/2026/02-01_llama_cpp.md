---
tags: [llm-agents, llama-cpp, ollama]
summary: llama.cpp build and usage, router mode as a systemd service, reasoning and vision notes, a local model comparison (Nov 2026), Qwen agent deployment, Ollama.
---
# ___Llama.cpp___
***

# References
  - qwen2-vl-8b-instruct
  - image / pdf /
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
  - [langchain Llama.cpp](https://python.langchain.com/docs/integrations/chat/llamacpp/)
  - [Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF)
  - [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
  - [LLM Explorer](https://llm-explorer.com/)
  - [Streaming-STT-1.5B-GGUF](https://huggingface.co/mradermacher/Streaming-STT-1.5B-GGUF/tree/main)
  - [canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)
  - [qwen3-asr-0.6b-GGUF](https://huggingface.co/FlippyDora/qwen3-asr-0.6b-GGUF/tree/main)

***

# Build CPU or GPU
## Installation
  - [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
  - [gemma-3-270m-it-GGUF fp16](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/tree/main)
  - **GPU Configuration:** Should better build with CUDA, may disable it at runtime, or use `-ngl, --n-gpu-layers` for setting number of layers running on GPU.
  - **CUDA Capability:** Check GPU compute capability and set with `cmake -DCMAKE_CUDA_ARCHITECTURES`:
    ```sh
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    # NVIDIA GeForce RTX 2080 Ti, 7.5 -> -DCMAKE_CUDA_ARCHITECTURES=75
    ```
  - **Build Steps:**
    ```sh
    # Install cuda, at least ~9G storage
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    # sudo apt-get -y install cuda-toolkit-13-0
    sudo apt-get -y install cuda-toolkit-12-8  # For torch 2.8.0 requirement
    ```
    ```sh
    sudo apt-get install libcurl4-openssl-dev pciutils build-essential cmake curl libcurl4-openssl-dev git-all
    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp/
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release -j $(nproc)
    # cd build && sudo make install && cd -
    cmake --install build --prefix ~/.local

    LD_LIBRARY_PATH=$HOME/.local/lib/:$LD_LIBRARY_PATH llama-cli -m workspace/gemma-3-270m-it-F16.gguf
    ```
    **Earlier version**
    ```sh
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=g++-13
    cmake --build build -j
    ```
  - **Clone Models:**
    ```sh
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/gemma-3-270m
    ```
  - CUDA one should better specify `DCMAKE_CUDA_ARCHITECTURES`
    ```sh
    cd llama.cpp

    # Set up the correct CUDA environment variables
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # Clean previous build artifacts that might be cached
    rm -rf build
    # Configure with CUDA enabled, explicitly targeting architecture 80 (A100)
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80" -DLLAMA_CURL=OFF
    # Build using all available cores
    cmake --build build --config Release -j$(nproc)
    ```

***

# Basic guide
## Quick Start
  - [llama.cpp guide - Running LLMs locally, on any hardware, from scratch](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)
  - **Get SmolLM2 for experiment**
    ```sh
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
    cd SmolLM2-1.7B-Instruct/
    wget 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/model.safetensors?download=true' -O model.safetensors
    cd -
    ```
  - **Required files**
    - `config.json`: contains configuration/metadata of our model
    - `model.safetensors`: contains model weights
    - `tokenizer.json`: contains tokenizer data (mapping of text tokens to their ID’s, and other stuff). Sometimes this data is stored in tokenizer.model file instead.
    - `tokenizer_config.json`: contains tokenizer configuration (for example, special tokens and chat template)
  - **Create GGUF file from downloaded HuggingFace repository**
    ```sh
    cd llama.cpp
    python convert_hf_to_gguf.py ../SmolLM2-1.7B-Instruct --outfile ./SmolLM2.gguf  # 3.2G
    # INFO:hf-to-gguf:Loading model: SmolLM2-1.7B-Instruct
    # INFO:hf-to-gguf:Model architecture: LlamaForCausalLM
    # INFO:hf-to-gguf:gguf: indexing model part 'model.safetensors'
    # INFO:hf-to-gguf:heuristics detected bfloat16 tensor dtype, setting --outtype bf16
    # ...
    # INFO:gguf.gguf_writer:Writing the following files:
    # INFO:gguf.gguf_writer:SmolLM2.gguf: n_tensors = 218, total_size = 3.4G
    # Writing: 100%|█████████████████████████████████████████| 3.42G/3.42G [00:27<00:00, 126Mbyte/s]
    # INFO:hf-to-gguf:Model successfully exported to SmolLM2.gguf
    ```
  - **Quantizing the model** `{source gguf model} {target file} {quantization type} [number of parallel]`
    ```sh
    ./build/bin/llama-quantize -h
    # usage: ./build/bin/llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]
    #        [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file] [--prune-layers] [--keep-split] [--override-kv]
    #        model-f32.gguf [model-quant.gguf] type [nthreads]
    # ...
    ```
    - **Description:** in most cases shows either the example model’s size and perplexity, or the amount of bits per tensor weight (bpw) for that specific quantization.
    - **Perplexity:** is a metric that describes how certain the model is about it’s predictions. Lower perplexity -> model is more certain about it’s predictions -> model is more accurate.
    - **BPW:** is the “bits per weight” metric tells the average size of quantized tensor’s weight.
    - **Picking quantization type:** use the largest can fit in VRAM, unless it’s too slow.
    - **Memory requirements for the context:** besides from model weight size, should have at least 1GB memory for context. Can be controled by `--ctx-size` in `llama-server`
    ```sh
    ./build/bin/llama-quantize SmolLM2.gguf SmolLM2_Q8.gguf Q8_0 8  # 1.7G
    # ...
    # llama_model_quantize_impl: model size  =  3264.38 MiB
    # llama_model_quantize_impl: quant size  =  1734.38 MiB
    #
    # main: quantize time = 13286.27 ms
    # main:    total time = 13286.27 ms
    ```
  - **Run server**
    ```sh
    ./build/bin/llama-server --help
    #   Device 0: NVIDIA ..., compute capability ..., VMM: yes
    # ----- common params -----
    # --host HOST                             ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: 127.0.0.1) (env: LLAMA_ARG_HOST)
    # --port PORT                             port to listen (default: 8080) (env: LLAMA_ARG_PORT)
    # -m,    --model FNAME                    model path to load (env: LLAMA_ARG_MODEL)
    # -b,    --batch-size N                   logical maximum batch size (default: 2048) (env: LLAMA_ARG_BATCH)
    # -c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model) (env: LLAMA_ARG_CTX_SIZE)
    # -ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto) (env: LLAMA_ARG_N_GPU_LAYERS)
    ```
    ```sh
    llama-server -m SmolLM2.q8.gguf
    # ...
    # srv          init: init: chat template, thinking = 0
    # main: model loaded
    # main: server is listening on http://127.0.0.1:8080
    # main: starting the main loop...
    # srv  update_slots: all slots are idle
    ```
    Access the web UI on http://127.0.0.1:8080
  - **Tokenize / Detokenize**
    ```sh
    curl -X POST -H "Content-Type: application/json" -d '{"content": "hello world! this is an example message!"}' http://127.0.0.1:8080/tokenize
    # {"tokens":[28120,905,17,451,314,354,1183,3714,17]}
    curl -X POST -H "Content-Type: application/json" -d '{"tokens": [28120,905,17,451,314,354,1183,3714,17]}' http://127.0.0.1:8080/detokenize
    # {"content":"hello world! this is an example message!"}
    ```
  - **Other tools**
    - `./build/bin/llama-bench --model selected_model.gguf`: to benchmark the prompt processing and text generation speed of our llama.cpp build for a selected model.
    - `./build/bin/llama-cli --model ./SmolLM2_Q8.gguf --prompt "The highest mountain on earth"`: a simple CLI interface for the LLM to generate a completion for specified prompt, or chat with the LLM.
    - `ggml library` is the back-end for llama.cpp. This library contains the code for math operations used to run LLMs, and it supports many hardware accelerations that can enable to get the maximum LLM performance on specific hardware.
    ```sh
    cmake -S . -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/your/install/dir -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
    cmake --build build --config Release -j 16
    cmake --install build --config Release
    ./build/bin/llama-cli --list-devices
    # ggml_cuda_init: found 1 CUDA devices:
    #   Device 0: NVIDIA ..., compute capability ..., VMM: yes
    # Available devices:
    #   CUDA0: NVIDIA ...
    ```
  - **LLM configuration options and samplers available in llama.cpp**
    - **System Message**: recommendation is to put as much useful informations and precise behavior descriptions for your application as possible, to maximize the quality of LLM output.
    - **Temperature**: controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused. Keep it in 0.2-2.0 range for a start, and keep it positive and non-zero.
    - **Dynamic temperature**: Dynamic temperature sampling is an addition to temperature sampler. It tweaks the temperature of generated tokens based on their entropy. Entropy here can be understood as inverse of LLMs confidence in generated tokens. Lower entropy means that the LLM is more confident in it’s predictions, and therefore the temperature of tokens with low entropy should also be low. High entropy works the other way around. Effectively, this sampling can encourage creativity while preventing hallucinations at higher temperatures.
    - **Top-K**: keep only K most probable tokens. Higher values can result in more diverse text, because there’s more tokens to choose from when generating responses.
    - **Top-P**: also called nucleus sampling, limits the tokens to those that together have a cumulative probability of at least p. It means that the Top-P sampler takes a list of tokens and their probabilities as an input (note that the sum of their cumulative probabilities is by definition equal to 1), and returns tokens with highest probabilities from that list until the sum of their cumulative probabilities is greater or equal to p. Or, in other words, p value changes the % of tokens returned by the Top-P sampler. For example, when p is equal to 0.7, the sampler will return 70% of input tokens with highest probabilities.
    - **Min-P**: limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.
    - **Exclude Top Choices (XTC)**: instead of pruning the least likely tokens, under certain circumstances, it removes the most likely tokens from consideration. The parameters for XTC sampler are: `XTC threshold`: probability cutoff threshold for top tokens, in (0, 1) range. `XTC probability`: probability of XTC sampling being applied in [0, 1] range, where 0 = XTC disabled, 1 = XTC always enabled.
    - **Locally typical sampling (typical-P)**: sorts and limits tokens based on the difference between log-probability and entropy.
    - **DRY**: is used to prevent unwanted token repetition. Simplifying, it tries to detect repeating token sequences in generated text and reduces the probabilities of tokens that will create repetitions. The penalty for a token is calculated as multiplier * base ^ (n - allowed_length), where n is the length of the sequence before that token that matches the end of the input, and multiplier, base, and allowed_length are configurable parameters. If the length of the matching sequence is less than allowed_length, no penalty is applied.
    - **Mirostat**: is a funky sampling algorithm that overrides Top-K, Top-P and Typical-P samplers. It’s an alternative sampler that produces text with controlled perplexity (entropy), which means that we can control how certain the model should be in it’s predictions. This comes without side-effects of generating repeated text (as it happens in low perplexity scenarios) or incoherent output (as it happens in high perplexity scenarios). The configuration parameters for Mirostat are: `Mirostat version`: 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. `Mirostat learning rate (η, eta)`: specifies how fast the model converges to desired perplexity. `Mirostat target entropy (τ, tau)`: the desired perplexity. Depending on the model, it should not be too high, otherwise you may degrade it’s performance.
    - **Max tokens**: -1 makes the LLM generate until it decides it’s end of the sentence (by returning end-of-sentence token), or the context is full.
    - **Repetition penalty**: Repetition penalty algorithm (not to be mistaken with DRY) simply reduces the chance that tokens that are already in the generated text will be used again. Usually the repetition penalty algorithm is restricted to N last tokens of the context. In case of llama.cpp (i’ll simplify a bit), it works like that: first, it creates a frequency map occurrences for last N tokens. Then, the current logit bias for each token is divided by repeat_penalty value. By default it’s usually set to 1.0, so to enable repetition penalty it should be set to >1. Finally, frequency and presence penalties are applied based on the frequency map. The penalty for each token is equal to (token_count * frequency_penalty) + (presence_penalty if token_count > 0). The penalty is represented as logit bias, which can be in [-100, 100] range. Negative values reduce the probability of token appearing in output, while positive increase it. The configuration parameters for repetition penalty are: `Repeat last N`: Amount of tokens from the end of the context to consider for repetition penalty. `Repeat penalty`: repeat_penalty argument described above, if equal to 1.0 then the repetition penalty is disabled. `Presence penalty`: presence_penalty argument from the equation above. `Frequency penalty`: frequency_penalty argument from the equation above.
***

# Router mode
## models.ini presets as a user service
  - **Router mode**: one `llama-server` started with `--models-preset models.ini` serves every preset on one port. It spawns a child `llama-server` per model on first request, and `sleep-idle-seconds` unloads an idle one. `--models-max 2` keeps at most two resident, so a third evicts one.
  - **systemd user unit** `~/.config/systemd/user/llama-router.service`. The children use the router's own binary, so the build path matters:
    ```ini
    [Service]
    Environment=LD_LIBRARY_PATH=%h/workspace/llama.cpp/build/bin:/usr/local/cuda/lib64
    WorkingDirectory=%h/workspace/models
    ExecStart=%h/workspace/llama.cpp/build/bin/llama-server \
        --models-preset %h/workspace/models/models.ini \
        --models-max 2 --host 0.0.0.0 --port 8080
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=default.target
    ```
  - **Preset** example, a dense 27B with vision:
    ```ini
    [Qwen/Qwen3.8-27B]
    model = /home/USER/workspace/models/Qwen3.8-27B-Q8_0.gguf
    mmproj = /home/USER/workspace/models/mmproj-Qwen3.8-27B-BF16.gguf
    ctx-size = 65536
    flash-attn = true
    fit = on
    cache-type-k = q8_0
    cache-type-v = q8_0
    parallel = 1
    temp = 0.6
    top-p = 0.95
    top-k = 20
    sleep-idle-seconds = 300
    ```
  - **Daily commands**
    ```sh
    curl -s 'http://127.0.0.1:8080/v1/models?reload=1'   # re-read models.ini (hot reload, no restart)
    curl -s http://127.0.0.1:8080/v1/models               # presets and status: unloaded / loading / loaded / sleeping
    curl -s -X POST http://127.0.0.1:8080/models/unload -H 'Content-Type: application/json' -d '{"model": "Qwen/Qwen3.8-27B"}'
    curl -s http://127.0.0.1:8080/health
    journalctl --user -u llama-router -n 50 --no-pager
    ```
  - **Switch the service to a new build**: a plain restart keeps the binary named in `ExecStart`. Point it at the new build first. A restart drops all loaded models, which reload on their next request.
    ```sh
    cd ~/workspace/llama.cpp && git pull
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release -j $(nproc)

    f=~/.config/systemd/user/llama-router.service
    cp $f $f.bak-$(date +%Y%m%d)
    sed -i 's#OLD/llama.cpp/build/bin#NEW/llama.cpp/build/bin#g' $f   # both LD_LIBRARY_PATH and ExecStart
    systemctl --user daemon-reload && systemctl --user restart llama-router
    systemctl --user status llama-router --no-pager | head -5
    ```
  - **Memory**: `fit = on` sizes layers to the free memory at load time. A standalone server started with `-ngl 99` doesn't, so it can fail to reload after sleeping if something else took the GPU meanwhile. Never load two copies of the same large model: with two 48 GB GPUs and 62 GB of host RAM, a second 35B copy got the whole router OOM-killed by systemd-oomd.
## Reasoning control
  - **Per request**, to turn thinking off for templates that support it (Qwen3.x, Gemma 4):
    ```json
    {"chat_template_kwargs": {"enable_thinking": false}}
    ```
    Gemma-4-26B-A4B with thinking on reasoned for 40 min (~380k chars) on a 6-image request and never answered. With it off, it answers in seconds.
  - **Server flags**: `--reasoning on|off|auto`, `--reasoning-budget N` (token cap on thinking; 0 ends it at once), `--reasoning-format deepseek` (thinking goes to `message.reasoning_content`).
  - **Client side**: when streaming, stop after N reasoning characters with no answer yet. A stuck model then fails fast instead of hanging.
  - **Context per slot**: `-c` is split across `-np` slots. A long reasoner that runs out of slot context returns an empty answer with no error, so give it `-c` = per-request need × `-np`.
## Vision models
  - **mmproj**: vision needs the model's projector, `--mmproj mmproj-*.gguf` or `mmproj =` in a preset. Without one the model is text-only. `/v1/models` on a standalone server lists `"capabilities": ["multimodal"]`.
  - **Small transcriber for a text-only model**: Qwen3-VL-4B (Q4_K_M, greedy, ctx 16k) transcribes a screenshot page into Markdown tables and verbatim text, Chinese included, in ~13 s a page. Ask it to tag only marks a person drew (red boxes, arrows) and to treat UI badges as plain text, or it tags everything. Cap `max_tokens` and cut repeated lines, since greedy decoding can loop.
  - **Muse Glimmer-30B** (Meta, Oct 2026) needs a build with `LLM_ARCH_MUSE_GLIMMER` (mid-Nov 2026 or later). The router's older build can't load it.
    ```sh
    D=~/workspace/models/Muse-Glimmer-30B-GGUF
    llama-server --port 18030 --alias Muse-Glimmer-30B \
        -m $D/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj $D/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
        -md $D/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld 99 \
        --jinja -c 65536 -np 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --fit on \
        --temp 1.0 --top-p 0.95 --top-k 64 --sleep-idle-seconds 300
    ```
    Reasoning effort goes in the system prompt as a line `Reasoning effort: high`. ~23 GB over two GPUs, ~43 tok/s with the DFlash drafter on RTX 8000.
## Local model comparison, Nov 2026
  - **Task**: turn a chat export (6 page images of screenshots) into a short task brief for a coding agent. The benchmark is 5 private cases scored by regex checks: exact values, seen vs guessed, which messages count, no invented asks.
  - **Hardware**: 2× Quadro RTX 8000 (48 GB, sm_75). Times are wall time per brief, the first one after a load included.

    | model | score | mean time | range |
    |---|---|---|---|
    | Gemini 3.8 Flash (cloud, reference) | ~95% | ~25 s | 15–40 s |
    | Qwen3.8-27B dense Q8_0, temp 0.3 | 85% | ~13 min | 8–20 min |
    | Qwen3.6-27B Fable fine-tune Q4_K_M, temp 0.3 | 81% | ~110 s | 27–290 s |
    | Muse Glimmer-30B Q4_K_M + DFlash drafter | 80% | ~106 s | 74–150 s |
    | Muse Glimmer-30B, `Reasoning effort: high` | 78% | ~100 s | 83–139 s |
    | Qwen3.6-35B-A3B Q8, temp 0.3 | 74% | ~100 s | 33–350 s |
    | Fable + a second "check your draft" pass | 64% | ~9 min | 2–27 min |
    | Nemotron-3.5-30B-A3B (text-only) + Qwen3-VL-4B transcripts | 64% | ~210 s | 153–340 s |
    | Gemma-4-26B-A4B, thinking off | 58% | 8 s | 5–11 s |
  - **Findings**
    - Every model read the screenshots correctly. The gap to the cloud model is judgement: which messages count, and keeping a guess apart from a fact.
    - More reasoning didn't help. High effort and a self-review pass both scored lower, and the review pass once looped in its thinking.
    - Temperature 0.3 beat the default ~1.0 for the Qwen3.6 models, scoring higher and running ~3× faster.
    - The dense Q8 27B was the most accurate local model, but 5–10× slower than the MoE / Q4 ones on these older GPUs.
***

# Qwen agent
## Deployment
  ```sh
  cd ~/workspace/llama.cpp && ./build/bin/llama-server -m ../qwen_rag/Qwen3-0.6B-Q8_0.gguf --ctx-size 65536
  cd ~/workspace/qwen_rag && python rag_webui.py
  ```
***

# Ollama
- [ollama](https://ollama.com/)
- install
  ```sh
  curl -fsSL https://ollama.com/install.sh | sh
  ```
```sh
ollama pull nemotron-3-nano:4b

ollama create nemotron3_4b -f modelfile_nemotron3
ollama run nemotron3_4b
```
```sh
pip install easyocr
python ocr.py image.png
```
```py
import easyocr

reader = easyocr.Reader(["en"])
results = reader.readtext(image_path, detail=0)
print("\n".join(results))
```

<!-- related -->
# Related
- [claude](02-28_claude.md)
- [OpenClaw](01-29_openclaw.md)
- [llama2](../2023/08-06_llama2.md)
- [langgraph](05-08_langgraph.md)
- [learn claude code](07-27_learn_claude_code.md)
***
<!-- /related -->
