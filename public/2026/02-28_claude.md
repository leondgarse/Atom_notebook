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
