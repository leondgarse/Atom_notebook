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

# Build CPU
  - [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
  - [gemma-3-270m-it-GGUF fp16](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/tree/main)
  - Should better build with CUDA, may disable it at runtime, or use `-ngl, --n-gpu-layers` for setting number of layers running on GPU.
  - Check GPU compute capability and set with `cmake`: `nvidia-smi --query-gpu=name,compute_cap --format=csv`
  ```sh
  # Install cuda, at least ~9G storage
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  # sudo apt-get -y install cuda-toolkit-13-0
  sudo apt-get -y install cuda-toolkit-12-8  # For torch 2.8.0 requirement

  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release
  ```
  ```sh
  sudo apt-get install libcurl4-openssl-dev
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp/
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build --config Release -j $(nproc)
  cd build && sudo make install && cd -

  LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH llama-cli -m workspace/gemma-3-270m-it-F16.gguf
  ```
  ```sh
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/gemma-3-270m
  ```
# Basic guide
  - [llama.cpp guide - Running LLMs locally, on any hardware, from scratch](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)
  - **Get SmolLM2 for experiment**
    ```sh
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
    cd SmolLM2-1.7B-Instruct/
    wget 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/model.safetensors?download=true' -O model.safetensors
    cd -
    ```
  - **Required files**
    - `config.json`  contains configuration/metadata of our model
    - `model.safetensors` contains model weights
    - `tokenizer.json` contains tokenizer data (mapping of text tokens to their ID’s, and other stuff). Sometimes this data is stored in tokenizer.model file instead.
    - `tokenizer_config.json` contains tokenizer configuration (for example, special tokens and chat template)
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
  - **quantizing the model** `{source gguf model} {target file} {quantization type} [number of parallel]`
    ```sh
    ./build/bin/llama-quantize -h
    # usage: ./build/bin/llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]
    #        [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file] [--prune-layers] [--keep-split] [--override-kv]
    #        model-f32.gguf [model-quant.gguf] type [nthreads]
    # ...
    # Allowed quantization types:
    #    2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B
    #    3  or  Q4_1    :  4.78G, +0.4511 ppl @ Llama-3-8B
    #   38  or  MXFP4_MOE :  MXFP4 MoE
    #    8  or  Q5_0    :  5.21G, +0.1316 ppl @ Llama-3-8B
    #    9  or  Q5_1    :  5.65G, +0.1062 ppl @ Llama-3-8B
    #   19  or  IQ2_XXS :  2.06 bpw quantization
    #   20  or  IQ2_XS  :  2.31 bpw quantization
    #   28  or  IQ2_S   :  2.5  bpw quantization
    #   29  or  IQ2_M   :  2.7  bpw quantization
    #   24  or  IQ1_S   :  1.56 bpw quantization
    #   31  or  IQ1_M   :  1.75 bpw quantization
    #   36  or  TQ1_0   :  1.69 bpw ternarization
    #   37  or  TQ2_0   :  2.06 bpw ternarization
    #   10  or  Q2_K    :  2.96G, +3.5199 ppl @ Llama-3-8B
    #   21  or  Q2_K_S  :  2.96G, +3.1836 ppl @ Llama-3-8B
    #   23  or  IQ3_XXS :  3.06 bpw quantization
    #   26  or  IQ3_S   :  3.44 bpw quantization
    #   27  or  IQ3_M   :  3.66 bpw quantization mix
    #   12  or  Q3_K    : alias for Q3_K_M
    #   22  or  IQ3_XS  :  3.3 bpw quantization
    #   11  or  Q3_K_S  :  3.41G, +1.6321 ppl @ Llama-3-8B
    #   12  or  Q3_K_M  :  3.74G, +0.6569 ppl @ Llama-3-8B
    #   13  or  Q3_K_L  :  4.03G, +0.5562 ppl @ Llama-3-8B
    #   25  or  IQ4_NL  :  4.50 bpw non-linear quantization
    #   30  or  IQ4_XS  :  4.25 bpw non-linear quantization
    #   15  or  Q4_K    : alias for Q4_K_M
    #   14  or  Q4_K_S  :  4.37G, +0.2689 ppl @ Llama-3-8B
    #   15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B
    #   17  or  Q5_K    : alias for Q5_K_M
    #   16  or  Q5_K_S  :  5.21G, +0.1049 ppl @ Llama-3-8B
    #   17  or  Q5_K_M  :  5.33G, +0.0569 ppl @ Llama-3-8B
    #   18  or  Q6_K    :  6.14G, +0.0217 ppl @ Llama-3-8B
    #    7  or  Q8_0    :  7.96G, +0.0026 ppl @ Llama-3-8B
    #    1  or  F16     : 14.00G, +0.0020 ppl @ Mistral-7B
    #   32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
    #    0  or  F32     : 26.00G              @ 7B
    #           COPY    : only copy tensors, no quantizing
    ```
    - **Description** that in most cases shows either the example model’s size and perplexity, or the amount of bits per tensor weight (bpw) for that specific quantization.
    - **Perplexity** is a metric that describes how certain the model is about it’s predictions. Lower perplexity -> model is more certain about it’s predictions -> model is more accurate.
    - **BPW** is the “bits per weight” metric tells the average size of quantized tensor’s weight.
    - **Picking quantization type** use the largest can fit in VRAM, unless it’s too slow.
    - **Memory requirements for the context** besides from model weight size, should have at least 1GB memory for context. Can be controled by `--ctx-size` in `llama-server`
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
  - **tokenize / detokenize**
    ```sh
    curl -X POST -H "Content-Type: application/json" -d '{"content": "hello world! this is an example message!"}' http://127.0.0.1:8080/tokenize
    # {"tokens":[28120,905,17,451,314,354,1183,3714,17]}
    curl -X POST -H "Content-Type: application/json" -d '{"tokens": [28120,905,17,451,314,354,1183,3714,17]}' http://127.0.0.1:8080/detokenize
    # {"content":"hello world! this is an example message!"}
    ```
  - **Other tools**
  - `./build/bin/llama-bench --model selected_model.gguf` to benchmark the prompt processing and text generation speed of our llama.cpp build for a selected model.
  - `./build/bin/llama-cli --model ./SmolLM2_Q8.gguf --prompt "The highest mountain on earth"` a simple CLI interface for the LLM to generate a completion for specified prompt, or chat with the LLM.
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
# Qwen agent

```sh
cd ~/workspace/llama.cpp && ./build/bin/llama-server -m ../qwen_rag/Qwen3-0.6B-Q8_0.gguf --ctx-size 65536
cd ~/workspace/qwen_rag && python rag_webui.py
```
