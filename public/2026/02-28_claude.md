---
tags: [llm-agents, claude-code, agents]
summary: Claude Code setup — install, Ghostty, local models via Ollama/llama.cpp/litellm, settings, Garmin MCP connector for claude.ai via Tailscale Funnel, learning resources.
---
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
| `claude "commit"`   | Create a Git commit (a prompt, not a subcommand)       | `claude "commit"`                   |
| `/clear`            | Clear conversation history                             | `/clear`                            |
| `/help`             | Show available commands                                | `/help`                             |
| `exit` or `Ctrl+C`  | Exit Claude Code                                       | `exit`                              |
## Ghostty
The install script downloads a .deb from GitHub releases and installs it with `apt-get install`, which needs sudo. To install without sudo, extract the .deb manually:
```sh
# Step 1: find the right .deb URL
curl -s https://api.github.com/repos/mkasberg/ghostty-ubuntu/releases/latest \
  | grep "browser_download_url.*$(dpkg --print-architecture).*$(lsb_release -cs).*\.deb" \
  | cut -d '"' -f 4

# Step 2: download and extract the .deb without installing
curl -L -o ghostty.deb "<URL from step 1>"
mkdir -p ~/.local/ghostty-extract
dpkg-deb -x ghostty.deb ~/.local/ghostty-extract

# Step 3: set up the binary, and make sure ~/.local/bin is in PATH
mkdir -p ~/.local/bin
cp ~/.local/ghostty-extract/usr/bin/ghostty ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Step 4 (optional): desktop entry, icons, terminfo (terminfo is needed for proper terminal behavior)
mkdir -p ~/.local/share/applications
cp ~/.local/ghostty-extract/usr/share/applications/*.desktop ~/.local/share/applications/ 2>/dev/null
cp -r ~/.local/ghostty-extract/usr/share/icons ~/.local/share/ 2>/dev/null
mkdir -p ~/.terminfo
cp -r ~/.local/ghostty-extract/usr/share/terminfo/* ~/.terminfo/ 2>/dev/null
```
Caveat: this works only if all of Ghostty's shared library dependencies are already installed. If ghostty fails to launch because of missing .so libraries, those need a system-wide `apt-get install`, which needs sudo.
## Ollama
- **Cloud**
  ```sh
  export OLLAMA_CONTEXT_LENGTH=64000
  export ANTHROPIC_AUTH_TOKEN=ollama
  export ANTHROPIC_BASE_URL=http://localhost:11434
  claude --model glm-4.7:cloud  # glm-4.7-flash seems not working, cannot modify files

  alias Claude="ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude --model glm-4.7:cloud"
  ```
- **Locally** Claude needs context length at least 64k [Claude Code Manual setup](https://docs.ollama.com/integrations/claude-code#manual-setup).
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

  # Other flags used with it, normally set per model in models.ini:
  # --temp 1.0 --top-p 0.95 --min-p 0.01 --kv-unified --fit on --parallel 2 --cont-batching --ctx-size 65536
  ```
  ```sh
  llama-server -hf unsloth/Qwen3.6-27B-GGUF:UD-Q8_K_XL --host 0.0.0.0 --port 8095 --ctx-size 512000 --no-mmap \
  --parallel 2 --flash-attn on --n-gpu-layers 999 --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0.00 --repeat-penalty 1.0 --presence-penalty 0.0
  ```
  ```sh
  export ANTHROPIC_BASE_URL="http://localhost:8080"
  export ANTHROPIC_API_KEY="sk-no-key-required" ## or 'sk-1234'
  claude --model unsloth/GLM-4.7-Flash
  ```
  ```sh
  CLAUDE_PROMPT=$(printf '%s' '<base64>' | base64 -d)
  claude --output-format stream-json --verbose --dangerously-skip-permissions \
  -p "$CLAUDE_PROMPT"  [--resume <session_id>]
  ```
- **llama.cpp router mode to auto start and offload on inactive**. Create `models.ini` file. Replace `$HOME` with actual path.
  ```ini
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
- **Set llama-server as a service with router mode**. A systemd *user* unit, no root needed: `~/.config/systemd/user/llama-router.service`. `%h` is the home directory. `--models-max 2` keeps two models loaded; a third evicts one. Idle unload is per model, via `sleep-idle-seconds` in `models.ini`.
  ```ini
  [Unit]
  Description=llama.cpp router (models.ini presets on :8080)
  After=network.target

  [Service]
  # The binary's own lib dir first, or the loader picks up older libllama/libmtmd from ~/.local/lib
  Environment=LD_LIBRARY_PATH=/mnt/data/gwwang/build/llama.cpp/build/bin:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
  Environment=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  WorkingDirectory=%h/workspace/models
  ExecStart=/mnt/data/gwwang/build/llama.cpp/build/bin/llama-server \
      --models-preset %h/workspace/models/models.ini \
      --models-max 2 \
      --host 0.0.0.0 \
      --port 8080
  Restart=always
  RestartSec=5
  LimitNOFILE=65535

  [Install]
  WantedBy=default.target
  ```
  ```sh
  loginctl enable-linger $USER  # keep it running without a login session
  systemctl --user daemon-reload
  systemctl --user enable --now llama-router
  systemctl --user restart llama-router

  # log
  journalctl --user -u llama-router -f

  # apply models.ini edits without a restart: changed models unload, others keep running
  curl -s 'http://127.0.0.1:8080/v1/models?reload=1'
  ```
  ```sh
  curl http://localhost:8080/health
  curl http://localhost:8080/models
  curl http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "unsloth/GLM-4.7-Flash",
    "messages": [
      {"role": "user", "content": "Hello! Can you help me with a Python function to calculate fibonacci numbers?"}
    ],
    "stream": false,
    "max_tokens": 500
  }'

  curl http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Jackrong/Qwopus3.6",
    "messages": [
      {"role": "user", "content": "Hello! Can you help me with a Python function to calculate fibonacci numbers?"}
    ],
    "stream": false,
    "max_tokens": 500
  }'

  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jackrong/Qwopus3.6",
      "messages": [{"role": "user", "content": "What is 2 times 3?"}],
      "tools": [{
        "type": "function",
        "function": {
          "name": "multiply",
          "description": "Multiply two integers",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {"type": "integer"},
              "b": {"type": "integer"}
            },
            "required": ["a", "b"]
          }
        }
      }]
    }'
  ```
## litellm proxy
```sh
pip install 'litellm[proxy]'
```
**Then a config.yaml**
```yml
model_list:
  - model_name: local-glm
    litellm_params:
      model: openai/local
      api_base: http://localhost:8080/v1
      api_key: not-needed
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.0-flash
      api_key: os.environ/GOOGLE_API_KEY
```
```sh
litellm --config config.yaml --port 4000
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
    CLAUDE_CODE_OAUTH_TOKEN="" ANTHROPIC_BASE_URL="http://127.0.0.1:8080" ANTHROPIC_API_KEY="sk-no-key-required" claude "$@"
    ln -sf ~/.claude/settings.default.json ~/.claude/settings.json
    trap - EXIT INT TERM
  }
  ```
## Messaging another session
- Sessions message each other through the `ListAgents` / `SendMessage` tools. The address is the session **name** that `ListAgents` prints, like `keystone-ai-system-c7`, or `name [ref]` when two share a name. It is **not** the session UUID that `claude -r` takes, and not the title shown in the tmux pane corner, which is the auto-generated conversation summary (`ai-title`).
- `ListAgents` never prints UUIDs, so "message session 6123308c-..." makes the sender ask which one you mean. Give it the name instead.
- The name belongs to the running process. A resumed session may get a new name, so look it up each time.
- Each live session registers itself in `~/.claude/sessions/<pid>.json`, which holds `name`, `sessionId`, `tmux` pane, `status` and `cwd`. That maps a UUID or a pane to a name without touching the busy session. It is an internal format and may change after an upgrade.
  ```sh
  jq -r '[.name, .sessionId, .tmux, .status] | @tsv' ~/.claude/sessions/*.json
  tmux list-panes -a -F '#{pane_id} #{pane_title}'  # pane -> title, to match the tmux column
  ```
- **ListAgents shell command**, saved as `~/.claude-sessions.sh` and sourced from `~/.bashrc` (works in zsh too). `ListAgents [filter]` lists live sessions; the filter matches any column, case-insensitive: a UUID prefix, a name or part of a title.
  ```sh
  ListAgents() {
    local f pid pane title
    {
      printf 'NAME\tSTATUS\tPANE\tTITLE\tSESSION_ID\tCWD\n'
      for f in "$HOME"/.claude/sessions/*.json; do
        [ -e "$f" ] || continue
        pid=$(jq -r '.pid' "$f")
        kill -0 "$pid" 2>/dev/null || continue
        pane=$(jq -r '.tmux // "" | sub(".*\\.";"")' "$f")
        title=""
        if [ -n "$pane" ] && command -v tmux >/dev/null 2>&1; then
          title=$(tmux display-message -p -t "$pane" '#{pane_title}' 2>/dev/null | sed 's/^✳ *//')
        fi
        jq -r --arg pane "$pane" --arg title "$title" \
          '[.name // "-", .status // "-", ($pane | if . == "" then "-" else . end), ($title | if . == "" then "-" else . end), .sessionId, .cwd] | @tsv' "$f"
      done
    } | if [ -n "$1" ]; then awk -F'\t' -v q="$1" 'NR==1 || index(tolower($0), tolower(q))'; else cat; fi \
      | column -t -s "$(printf '\t')"
  }
  ```
  ```sh
  printf '\n[ -f ~/.claude-sessions.sh ] && . ~/.claude-sessions.sh\n' >> ~/.bashrc && . ~/.bashrc

  $ ListAgents "po compare"
  NAME                   STATUS  PANE  TITLE                          SESSION_ID                            CWD
  keystone-ai-system-c7  shell   %47   Customer PO compare benchmark  6123308c-e32f-427a-9c6c-47f86891d139  /mnt/data/gwwang/...
  ```
- Then tell the sender: "message `keystone-ai-system-c7` ...", or "look up session 6123308c in `~/.claude/sessions/*.json` to get its name".
- The `ListAgents` tool can also list your account's sessions on other machines or in the cloud, when Remote Control is connected. Those rows show a title instead of a `<dir>-xx` name and are not in the local registry.
## Garmin MCP connector for claude.ai
- **Goal**: use Garmin data on claude.ai (web and mobile). claude.ai custom connectors take a **remote HTTPS URL** only, so a local stdio MCP server (`npx ...` / `uvx ...`) doesn't work there. Run an HTTP MCP server locally and expose it publicly.
  ```text
  claude.ai --HTTPS--> Tailscale Funnel (https://<mac>.<tailnet>.ts.net/<secret>/mcp)
            --> server.py on 127.0.0.1:8787 --Bearer token--> personal.connectapi.garmin.com
  ```
- **Garmin Personal API** (official, for your own data): create a key at `https://connect.garmin.com/app/settings/personalApiToken` (expires after 30/90/180/365 days, shown once). The key is exchanged for a **24h access token** (a raw JWT, no refresh token). Request a new one with the key when it expires.
  ```sh
  # key -> access token
  curl -X POST 'https://personal.connectapi.garmin.com/personalapitoken-service/personal-api/personal-api-key/token' \
    -H "Authorization: ApiKey $GARMIN_API_KEY"
  # token -> data
  curl 'https://personal.connectapi.garmin.com/usersummary-service/usersummary/daily?calendarDate=2026-11-14' \
    -H "Authorization: Bearer $TOKEN"
  ```
  - China accounts use `https://personal.connectapi.garmin.cn`.
  - The Swagger docs page (`developer.garmin.com/gc-developer-program/personal/swagger-docs/`) is rendered in the browser. The OpenAPI spec is embedded as `JSON.parse('{"openapi":"3.0.1",...}')` in the page's `component---src-pages-gc-developer-program-personal-swagger-docs-js-*.js` bundle. It has 25 read-only GET endpoints: daily summaries, sleep, HRV, heart rate, HR zones, body battery, stress/steps/hydration stats, SpO2, fitness age, weight, weight goals, nutrition, menstrual cycle, activity list/detail/workouts, activity file download, profile settings.
- **MCP server**: Python `mcp` SDK **v2** (`from mcp.server.mcpserver import MCPServer`; v1's `FastMCP` was renamed) plus `httpx`. 21 `@mcp.tool` functions, one per endpoint. It caches the token and refreshes it 5 minutes before the JWT `exp`, or on a 401. Responses are trimmed: the activity list keeps key fields only (63 KB down to 14 KB) and anything over 120k chars is truncated.
  ```sh
  mkdir connectors && cd connectors
  uv init --bare --name garmin-mcp && uv add "mcp[cli]" httpx
  # .env: MCP_PATH_SECRET=<python3 -c 'import secrets;print(secrets.token_urlsafe(24))'>, PORT=8787
  set -a; source .env; set +a; uv run python server.py
  ```
  - Serve at `/<secret>/mcp`. claude.ai connectors support OAuth or no auth, not a static API-key header, so the unguessable path is the access control. Every other path returns 404.
  - Disable the SDK's DNS-rebinding guard (`TransportSecuritySettings(enable_dns_rebinding_protection=False)`). Otherwise the public Host header is rejected, because only localhost is allowed by default.
  - Use stateful mode (sessions) with SSE responses, which are the SDK defaults. Set a non-empty `version=` on the server.
- **Tailscale Funnel** (the commands actually run, Homebrew CLI on macOS):
  ```sh
  sudo brew services start tailscale   # start tailscaled as root, also starts at boot
  tailscale up                         # prints a login URL, then the machine joins the tailnet
  tailscale funnel --bg 8787           # first time prints "Funnel is not enabled on your tailnet"
                                       #   plus a https://login.tailscale.com/f/funnel?node=... link; open it and click Enable
  tailscale funnel --bg 8787           # rerun: "Available on the internet: https://<mac>.<tailnet>.ts.net/ |-- proxy http://127.0.0.1:8787"
  tailscale funnel status              # check
  tailscale funnel --https=443 off     # turn it off
  ```
  - The Funnel config persists across reboots. Only `server.py` has to be started again.
  - The first HTTPS request may fail for a few seconds while the certificate is issued.
- **claude.ai**: Settings → Connectors → Add custom connector → URL `https://<mac>.<tailnet>.ts.net/<secret>/mcp` → "Don't need login" → Connect. Also works in the mobile apps.
- **Pitfall: `*.trycloudflare.com` (Cloudflare quick tunnel) is rejected by claude.ai.** It shows `无法连接到 Garmin ... ofid_...`. The server log shows a single `initialize` from Anthropic's servers (UA `python-httpx/0.28.1`, clientInfo `Anthropic`) answered with 200, and then nothing more. How it was narrowed down:
  - MCP v1 and v2 Python clients through the same public URL worked fully.
  - `https://mcp.deepwiki.com/mcp` connected on the same account.
  - The same server via `ssh -R 80:localhost:8787 nokey@localhost.run` (`*.lhr.life`) connected.
  - So the domain is the problem. Use Tailscale Funnel, an ngrok static domain, or a Cloudflare *named* tunnel on your own domain.
- **Test a remote MCP URL locally**:
  ```py
  from mcp import ClientSession
  from mcp.client.streamable_http import streamablehttp_client   # mcp<2; v2: streamable_http_client
  async with streamablehttp_client(URL) as (r, w, _):
      async with ClientSession(r, w) as s:
          await s.initialize(); print((await s.list_tools()).tools)
  ```
***

# Learning
## Workflow
- [Claude Basic workflows](https://platform.claude.com/cookbook/patterns-agents-basic-workflows)


| Model                       | Size | GPU Usage |
| --------------------------- | ---- | --------- |
| GLM-4.7-Flash-UD-Q4_K_XL    | 30B  | 20596MiB  |
| Qwopus3.6-v1-preview-Q4_K_M | 27B  | 20186MiB  |

<!-- related -->
# Related
- [learn claude code](07-27_learn_claude_code.md)
- [agno](05-10_agno.md)
- [OpenClaw](01-29_openclaw.md)
- [Antigravity](02-02_antigravity_cli.md)
- [Llama.cpp](02-01_llama_cpp.md)
***
<!-- /related -->
