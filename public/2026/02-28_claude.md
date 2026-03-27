- [claude docs](https://code.claude.com/docs/en/overview)
```sh
curl -fsSL https://claude.ai/install.sh | bash

cd your-project
claude
```

```sh
pip install claude-agent-sdk
```
| Command             | What it does                                           | Example                             |
| ------------------- | ------------------------------------------------------ | ----------------------------------- |
| `claude`            | Start interactive mode                                 | `claude`                            |
| `claude "task"`     | Run a one-time task                                    | `claude "fix the build error"`      |
| `claude -p "query"` | Run one-off query, then exit                           | `claude -p "explain this function"` |
| `claude -c`         | Continue most recent conversation in current directory | `claude -c`                         |
| `claude -r`         | Resume a previous conversation                         | `claude -r`                         |
| `claude commit`     | Create a Git commit                                    | `claude commit`                     |
| `/clear`            | Clear conversation history                             | `/clear`                            |
| `/help`             | Show available commands                                | `/help`                             |
| `exit` or `Ctrl+C`  | Exit Claude Code                                       | `exit`                              |
## Ghostty
  The script downloads a .deb from GitHub releases and installs it via apt-get install (which
  requires sudo). To install without sudo, you can extract the .deb manually:

  Step 1: Find the right .deb URL
  curl -s https://api.github.com/repos/mkasberg/ghostty-ubuntu/releases/latest \
    | grep "browser_download_url.*$(dpkg --print-architecture).*$(lsb_release -cs).*\.deb" \
    | cut -d '"' -f 4

  Step 2: Download and extract the .deb without installing
  # Download
  curl -L -o ghostty.deb "<URL from step 1>"

  # Extract to a local directory (no sudo needed)
  mkdir -p ~/.local/ghostty-extract
  dpkg-deb -x ghostty.deb ~/.local/ghostty-extract

  Step 3: Set up the binary
  mkdir -p ~/.local/bin
  cp ~/.local/ghostty-extract/usr/bin/ghostty ~/.local/bin/

  # Make sure ~/.local/bin is in your PATH
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc

  Step 4 (optional): Install desktop/icon files
  # Desktop entry
  mkdir -p ~/.local/share/applications
  cp ~/.local/ghostty-extract/usr/share/applications/*.desktop ~/.local/share/applications/
  2>/dev/null

  # Icons
  cp -r ~/.local/ghostty-extract/usr/share/icons ~/.local/share/ 2>/dev/null

  # Terminfo (needed for proper terminal behavior)
  mkdir -p ~/.terminfo
  cp -r ~/.local/ghostty-extract/usr/share/terminfo/* ~/.terminfo/ 2>/dev/null

  Caveats: This works only if all of Ghostty's shared library dependencies are already
  satisfied on your system. If ghostty fails to launch due to missing .so libraries, those
  would need to be installed system-wide (requiring sudo via apt-get install).
## Ollama
  - **Cloud**
    ```sh
    export OLLAMA_CONTEXT_LENGTH=64000
    export ANTHROPIC_AUTH_TOKEN=ollama
    export ANTHROPIC_BASE_URL=http://localhost:11434
    claude --model glm-4.7:cloud  # glm-4.7-flash seems not working, cannot modify files

    alias Claude="ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude --model glm-4.7:cloud"
    ```
  - **Locally** Cluade needs context length at lest 64k [Claude Code Manual setup](https://docs.ollama.com/integrations/claude-code#manual-setup).
    ```sh
    # Set context length at 64k
    sudo sed -i '/\[Service\]/a Environment="OLLAMA_CONTEXT_LENGTH=64000"' /etc/systemd/system/ollama.service
    # Set KV Cache to 8-bit and enable Flash Attention
    sudo sed -i '/\[Service\]/a Environment="OLLAMA_FLASH_ATTENTION=1"' /etc/systemd/system/ollama.service
    # sudo sed -i '/\[Service\]/a Environment="OLLAMA_KV_CACHE_TYPE=q8_0"' /etc/systemd/system/ollama.service
    sudo sed -i '/\[Service\]/a Environment="OLLAMA_NUM_PARALLEL=2"' /etc/systemd/system/ollama.service

    sudo systemctl daemon-reload && sudo systemctl restart ollama
    ```
    ```sh
    ollama run glm-4.7-flash  # Not a must

    export ANTHROPIC_AUTH_TOKEN=ollama
    export ANTHROPIC_BASE_URL=http://localhost:11434
    claude --model glm-4.7-flash

    alias Claude="ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude --model glm-4.7-flash"
    ```
## Llama cpp
- [How to Run Local LLMs with Claude Code](https://unsloth.ai/docs/basics/claude-code)
  ```sh
  llama-server --model ~/workspace/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf --alias "unsloth/GLM-4.7-Flash" \
  --temp 1.0 --top-p 0.95 --min-p 0.01 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --fit on --batch-size 4096 --ubatch-size 1024 \
  --port 8080 --ctx-size 131072 #change as required

  # or
  llama-server --model ~/workspace/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf --alias "unsloth/GLM-4.7-Flash" \
  --temp 1.0 --top-p 0.95 --min-p 0.01 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --fit on --batch-size 2048 --ubatch-size 512 \
  --parallel 2 --cont-batching \
  --port 8080 --ctx-size 65536 #change as required
  ```
  ```sh
  llama-server --models-preset models.ini --port 8080 --sleep-idle-seconds 300


  --temp 1.0 --top-p 0.95 --min-p 0.01 --kv-unified --fit on --parallel 2 --cont-batching
  --port 8080 --ctx-size 65536 #change as required
  ```
  ```sh
  export ANTHROPIC_BASE_URL="http://localhost:8080"
  export ANTHROPIC_API_KEY="sk-no-key-required" ## or 'sk-1234'
  claude --model unsloth/GLM-4.7-Flash
  ```
  ```sh
  CLAUDE_PROMPT=$(printf '%s' '<base64>' | base64 -d)
  claude --output-format stream-json --verbose --dangerously-skip-permissions
  -p "$CLAUDE_PROMPT"  [--resume <session_id>]
  ```
- **llama.cpp router mode to auto start and offload on inactive**. Create `models.ini` file. Replace `$HOME` with actual path.
  ```sh
  [unsloth/GLM-4.7-Flash]
  model = $HOME/workspace/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf

  # Context & Memory Optimizations
  ctx-size = 131072
  batch-size = 1024
  ubatch-size = 256
  flash-attn = true
  kv-unified = true
  fit = on

  # KV Cache Quantization (q8_0 or q4_0)
  cache-type-k = q8_0
  cache-type-v = q8_0

  # Performance & Architecture Flags. Note: Keeping parallel at 1 to prevent OOM
  parallel = 1
  cont-batching = true
  mlock = true

  # Sampling Parameters (Optimized for Coding/GLM-4)
  temp = 1.0
  top-p = 0.95
  min-p = 0.01

  # Router Specifics. This ensures it unloads from GPU after 5 mins of inactivity
  sleep-idle-seconds = 300
  ```
  ```sh
  llama-server --models-preset ~/workspace/models/models.ini --port 8080
  ```
- **Set llama-server as a service with router mode**. Create `/etc/systemd/system# cat llama-router.service`. Replace `$HOME` with actual path.
  ```sh
  [Unit]
  Description=Llama.cpp Router Service (Root with CUDA)
  After=network.target nvidia-persistenced.service

  [Service]
  User=root
  Group=root

  # We combine your local libs and the system CUDA libs, separated by a colon
  Environment="LD_LIBRARY_PATH=$HOME/.local/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
  Environment="PATH=$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  # Working directory
  WorkingDirectory=$HOME/workspace/models

  # Executable path
  ExecStart=$HOME/.local/bin/llama-server \
      --models-preset $HOME/workspace/models/models.ini \
      --host 0.0.0.0 \
      --port 8080

  # Reliability
  Restart=always
  RestartSec=5
  LimitNOFILE=65535

  [Install]
  WantedBy=multi-user.target
  ```
  ```sh
  sudo systemctl daemon-reload && sudo systemctl restart ollama

  # log
  sudo journalctl -u llama-router -f
  ```
## Claude private setting
```sh
cp ~/.claude/settings.json ~/.claude/settings.default.json
cp ~/.claude/settings.json ~/.claude/settings.private.json

# Edit in ~/.claude/settings.private.json
python3 -c "
import json
from pathlib import Path
p = Path.home() / '.claude/settings.private.json'
p.parent.mkdir(parents=True, exist_ok=True)
s = json.loads(p.read_text()) if p.exists() else {}
s.setdefault('env', {}).update({
  'CLAUDE_CODE_ENABLE_TELEMETRY':'0',
  'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
  'CLAUDE_CODE_ATTRIBUTION_HEADER':'0'
})
s.setdefault('attribution', {}).update({
  'commit':'',
  'pr':'',
})
s['plansDirectory'] = './plans'
s['prefersReducedMotion'] = True
s['terminalProgressBarEnabled'] = False
s['effortLevel'] = 'high'
p.write_text(json.dumps(s, indent=2))
"

cat ~/.claude/settings.private.json

# In ~/.bashrc, use private setting before starting claude, and link back at exit.
Claude () {
  ln -sf ~/.claude/settings.private.json ~/.claude/settings.json
  trap 'ln -sf ~/.claude/settings.default.json ~/.claude/settings.json' EXIT INT TERM
  CLAUDE_CODE_OAUTH_TOKEN="" ANTHROPIC_BASE_URL="http://127.0.0.1:8080" ANTHROPIC_API_KEY=
  ln -sf ~/.claude/settings.default.json ~/.claude/settings.json
  trap - EXIT INT TERM
}
```
