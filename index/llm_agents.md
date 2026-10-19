# LLMs & AI Agents

GPT / Llama training, LLM serving (vLLM, llama.cpp), agent frameworks and coding agents.

Back to [all indexes](README.md).

## Models & training

- **[gpt](../public/2023/02-27_gpt.md)** (2023-02-27, 23 KB) — GPT from scratch — nanoGPT tokenizer, model, HF weight loading, Shakespeare training, TFDS, Llama2.  
  [NanoGPT](../public/2023/02-27_gpt.md#nanogpt) · [Prepare](../public/2023/02-27_gpt.md#prepare) · [Tokennizer](../public/2023/02-27_gpt.md#tokennizer) · [Model definition](../public/2023/02-27_gpt.md#Model%20definition) · [Model load statedict from huggingface](../public/2023/02-27_gpt.md#Model%20load%20statedict%20from%20huggingface) · [Model evaluation](../public/2023/02-27_gpt.md#Model%20evaluation) · [Shakespeare datasets](../public/2023/02-27_gpt.md#Shakespeare%20datasets) · [Optimizer](../public/2023/02-27_gpt.md#optimizer)
- **[llama2](../public/2023/08-06_llama2.md)** (2023-08-06, 1 KB) — Minimal Chinese Llama 2 inference with HF transformers.
- **[llm train](../public/2024/03-23_llm_train.md)** (2024-03-23, 13 KB) — Training a small LLM with Hugging Face — tokenizer, data, pretraining and LoRA.  
  [Huggingface train](../public/2024/03-23_llm_train.md#Huggingface%20train) · [Huggingface lora](../public/2024/03-23_llm_train.md#Huggingface%20lora)

## Serving

- **[vllm](../public/2025/07-05_vllm.md)** (2025-07-05, 51 KB) — vLLM architecture (EngineCore, executor, KV cache, scheduler, prefix caching), vLLM Ascend, offline inference.  
  [vLLM](../public/2025/07-05_vllm.md#__vLLM__) · [vLLM 架构](../public/2025/07-05_vllm.md#vLLM%20架构) · [VLLM Ascend](../public/2025/07-05_vllm.md#VLLM%20Ascend) · [Offline inference](../public/2025/07-05_vllm.md#Offline%20inference)
- **[Llama.cpp](../public/2026/02-01_llama_cpp.md)** (2026-02-01, 15 KB) — llama.cpp build and usage, Qwen agent deployment, Ollama.  
  [References](../public/2026/02-01_llama_cpp.md#references) · [Build CPU or GPU](../public/2026/02-01_llama_cpp.md#Build%20CPU%20or%20GPU) · [Basic guide](../public/2026/02-01_llama_cpp.md#Basic%20guide) · [Qwen agent](../public/2026/02-01_llama_cpp.md#Qwen%20agent) · [Ollama](../public/2026/02-01_llama_cpp.md#ollama)
- **[server chat](../public/2026/03-11_server_chat.md)** (2026-03-11, 1 KB) — Repositories for a self-hosted server chat setup.

## Agents & tools

- **[OpenClaw](../public/2026/01-29_openclaw.md)** (2026-01-29, 69 KB) — OpenClaw agent — install, llama.cpp provider, gateway service, security, Discord and Obsidian integration.  
  [Requirements](../public/2026/01-29_openclaw.md#requirements) · [Install openclaw via npm](../public/2026/01-29_openclaw.md#Install%20openclaw%20via%20npm) · [Configuration and set up a custom llama.cpp pro…](../public/2026/01-29_openclaw.md#Configuration%20and%20set%20up%20a%20custom%20llama%20cpp%20provider) · [Install gateway and start as a systemd user ser…](../public/2026/01-29_openclaw.md#Install%20gateway%20and%20start%20as%20a%20systemd%20user%20service) · [Diagnostics](../public/2026/01-29_openclaw.md#diagnostics) · [security](../public/2026/01-29_openclaw.md#security) · [Discord](../public/2026/01-29_openclaw.md#discord) · [Obsidian](../public/2026/01-29_openclaw.md#obsidian)
- **[Gemini](../public/2026/02-02_gemini_cli.md)** (2026-02-02, 13 KB) — Gemini CLI and Antigravity — rules, workflows, agent skills, MCP, NotebookLM MCP.  
  [Get Started](../public/2026/02-02_gemini_cli.md#Get%20Started) · [Anti-gravity](../public/2026/02-02_gemini_cli.md#anti-gravity) · [SKILLs](../public/2026/02-02_gemini_cli.md#skills)
- **[claude](../public/2026/02-28_claude.md)** (2026-02-28, 11 KB) — Claude Code setup — install, Ghostty, local models via Ollama/llama.cpp/litellm, settings, learning resources.  
  [Download](../public/2026/02-28_claude.md#download) · [Extract to a local directory (no sudo needed)](../public/2026/02-28_claude.md#Extract%20to%20a%20local%20directory%20no%20sudo%20needed) · [Make sure ~/.local/bin is in your PATH](../public/2026/02-28_claude.md#Make%20sure%20local%20bin%20is%20in%20your%20PATH) · [Desktop entry](../public/2026/02-28_claude.md#Desktop%20entry) · [Icons](../public/2026/02-28_claude.md#icons) · [Terminfo (needed for proper terminal behavior)](../public/2026/02-28_claude.md#Terminfo%20needed%20for%20proper%20terminal%20behavior) · [Learning](../public/2026/02-28_claude.md#learning)
- **[langgraph](../public/2026/05-08_langgraph.md)** (2026-05-08, 47 KB) — LangGraph — local model setup, Graph API, an email agent walkthrough, workflow patterns and agents.  
  [A basic email agent](../public/2026/05-08_langgraph.md#A%20basic%20email%20agent) · [Workflows](../public/2026/05-08_langgraph.md#workflows) · [Capabilities](../public/2026/05-08_langgraph.md#capabilities)
- **[agno](../public/2026/05-10_agno.md)** (2026-05-10, 8 KB) — Agno agent framework — getting started, working with Claude, Dash, AgentOS runtime and setup.  
  [Starting](../public/2026/05-10_agno.md#starting) · [Work with claude](../public/2026/05-10_agno.md#Work%20with%20claude) · [Dash](../public/2026/05-10_agno.md#dash) · [Runtime](../public/2026/05-10_agno.md#runtime) · [Setup](../public/2026/05-10_agno.md#setup) · [Others](../public/2026/05-10_agno.md#others)
- **[learn claude code](../public/2026/07-27_learn_claude_code.md)** (2026-07-27, 9 KB) — Learning how coding agents work — the tool-use loop and step-by-step agent sessions with the Anthropic SDK.  
  [S01](../public/2026/07-27_learn_claude_code.md#s01) · [S05](../public/2026/07-27_learn_claude_code.md#s05)
- **[Deepseek Harness](../public/2026/08-20_deepseek_harness.md)** (2026-08-20, 1 KB) — DeepSeek Harness (dsh) install and run.

## See also

- [Deep Learning Frameworks](dl_frameworks.md)
- [NUS MSc Coursework](nus_courses.md)
