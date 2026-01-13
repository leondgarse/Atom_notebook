  - qwen2-vl-8b-instruct
  - image / pdf /
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [langchain Llama.cpp](https://python.langchain.com/docs/integrations/chat/llamacpp/)
***

# Build CPU
- [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [gemma-3-270m-it-GGUF fp16](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/tree/main)
```sh
sudo apt-get install libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp/
cmake -B build && cmake --build build --config Release
cd build && sudo make install && cd -

LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH llama-cli -m workspace/gemma-3-270m-it-F16.gguf
```
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
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/gemma-3-270m
```
