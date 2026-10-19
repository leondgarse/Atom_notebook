---
tags: [llm-agents, deepseek, agents]
summary: DeepSeek Harness (dsh) install and run.
---
# ___Deepseek Harness___
***

- [Github deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
```sh
npx @deepseek-ai/dsh web

# Or
npm install -g @deepseek-ai/dsh
dsh web
```
Or from source
```sh
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```
system service, `sudo vi /etc/systemd/system/dsh-web@.service`:
```java
[Unit]
Description=DeepSeek Harness Web UI (%i)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=%i
Group=%i
WorkingDirectory=/home/%i
Environment=HOME=/home/%i
Environment=DEEPSEEK_API_KEY=sk-...
ExecStart=/usr/bin/dsh web --no-open
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Then
```sh
sudo systemctl daemon-reload
sudo systemctl enable --now dsh-web@`whoami`.service   # starts now and at every boot
journalctl -u dsh-web@`whoami`.service -f              # follow its logs
```
