---
tags: [shell-tools, macos, terminal]
summary: Moving from Ubuntu to macOS — keyboard shortcuts, Finder, input method, window management, Karabiner remaps, Dock, Homebrew apps, zsh/terminal, ssh and git case-insensitivity.
---
# ___macOS___
***

# Keyboard
## Modifier keys on a PC keyboard
- **Win = ⌘ Command, Alt = ⌥ Option.** Symbols used below: ⌘ Command, ⌥ Option, ⇧ Shift, Ctrl Control.
- On a Mac keyboard the order next to the spacebar is `Ctrl ⌥ ⌘ [space]`; on a PC keyboard ⌘ (Win) is one key further away. To swap: System Settings → Keyboard → Keyboard Shortcuts → Modifier Keys → pick the keyboard → Option → Command, Command → Option.
- Almost every Linux `Ctrl+X` shortcut is `⌘X` on macOS; `Ctrl+C` in the terminal still interrupts.
## Everyday shortcuts

| Action                                        | Keys                                           |
| --------------------------------------------- | ---------------------------------------------- |
| Quit app / close window only                  | `⌘Q` / `⌘W` (the app keeps running after `⌘W`) |
| Hide app / minimize window                    | `⌘H` / `⌘M`                                    |
| App switcher / windows of the current app     | `⌘Tab` / ``⌘` ``                               |
| Force Quit                                    | `⌘⌥Esc`                                        |
| Mission Control / App Exposé / switch desktop | `Ctrl+↑` / `Ctrl+↓` / `Ctrl+←/→`               |
| Full screen                                   | `Ctrl⌘F`                                       |
| Spotlight (launch apps) / Apps grid           | `⌘Space` / `⌘Space` then `⌘1`                  |
| Lock screen                                   | `Ctrl⌘Q`                                       |
| Screenshot area / full / menu with recording  | `⌘⇧4` / `⌘⇧3` / `⌘⇧5`                          |
| Emoji picker                                  | `Ctrl⌘Space`                                   |
| Settings of the current app                   | `⌘,`                                           |
## Text editing (most apps, not the terminal)

| Linux | macOS | Action |
| --- | --- | --- |
| `Home` / `End` | `⌘←` / `⌘→` | line start / end |
| `Ctrl+Home` / `Ctrl+End` | `⌘↑` / `⌘↓` | document start / end |
| `Ctrl+←` / `Ctrl+→` | `⌥←` / `⌥→` | previous / next word |
| `Ctrl+Backspace` | `⌥Backspace` | delete previous word |
| — | `⌘Backspace` | delete to line start |
| add `Shift` | add `⇧` | select instead of move |

- In most Mac apps `Home` / `End` alone only scroll the view; the cursor stays.
- Pulsar keymap (`~/.pulsar/keymap.cson`) to keep the Linux habits:
  ```coffee
  'atom-text-editor':
    'ctrl-home': 'core:move-to-top'
    'ctrl-end': 'core:move-to-bottom'
    'ctrl-shift-home': 'core:select-to-top'
    'ctrl-shift-end': 'core:select-to-bottom'
    'ctrl-b': 'markdown-writer:toggle-bold-text'
    'ctrl-i': 'markdown-writer:toggle-italic-text'
  ```
  A `'.platform-linux atom-text-editor'` selector does nothing on macOS.
## Terminal tabs (Terminal.app and Ghostty)

| Action | Keys |
| --- | --- |
| Go to tab N | `⌘1` … `⌘9` |
| Next / previous tab | `⌘⇧]` / `⌘⇧[`, `Ctrl+Tab` / `Ctrl+⇧Tab` |
| New tab / close tab / new window | `⌘T` / `⌘W` / `⌘N` |
| Copy / paste | `⌘C` / `⌘V` |

- **⌘PageUp / ⌘PageDown for tabs** — Ghostty, in `~/Library/Application Support/com.mitchellh.ghostty/config.ghostty`, reload with `⌘⇧,`:
  ```sh
  keybind = super+page_up=previous_tab
  keybind = super+page_down=next_tab
  ```
  Terminal.app binds menu items; the app must be restarted (`⌘Q`) to pick it up:
  ```sh
  defaults write com.apple.Terminal NSUserKeyEquivalents -dict-add "Show Previous Tab" '@\UF72C' "Show Next Tab" '@\UF72D'
  ```
  `@` = ⌘, `\UF72C` / `\UF72D` = PageUp / PageDown. The same mechanism (System Settings → Keyboard → Keyboard Shortcuts → App Shortcuts) binds any menu item of any app.
- **⌘Backspace wipes the whole line in Ghostty** — its default binding sends `Ctrl+U`. Easy to hit when used to `Alt+Backspace` for deleting a word. Make both delete the previous word instead (`ESC DEL`, zsh `backward-kill-word`):
  ```sh
  keybind = super+backspace=text:\x1b\x7f
  keybind = alt+backspace=text:\x1b\x7f
  ```
  Use `keybind = super+backspace=ignore` to make it do nothing. `ghostty +list-keybinds --default` lists all built-in bindings. `Ctrl+U` / `Ctrl+W` still clear the line / a word.
***

# Finder
## Shortcuts

| Action | Keys |
| --- | --- |
| Open selected folder / file | `⌘↓` or `⌘O` (Enter **renames**) |
| Parent folder / back / forward | `⌘↑` / `⌘[` / `⌘]` |
| Go to a typed path (like `Ctrl+L` in Nautilus) | `⌘⇧G` |
| Copy full path of the selection | `⌥⌘C` |
| Show hidden files | `⌘⇧.` |
| Move to Trash | `⌘Backspace` |
| Quick Look preview | `Space` |
| Icon / list view | `⌘1` / `⌘2` |
| View options | `⌘J` |
| Info / live Inspector (total size of a selection) | `⌘I` / `⌥⌘I` |
| Preview pane (size, dates of the selection) | `⇧⌘P` |
| Status bar / path bar | `⌘/` / `⌥⌘P` |

- ⌘-click the window title for a menu of parent folders; dragging a file into the terminal pastes its escaped path.
- Selected file size: the status bar only shows item count and free space. Use the preview pane, list view with the Size column (`⌘J` → "Calculate all sizes"), or `⌘J` → "Show item info" in icon view.
- Icons on a grid: `⌘J` → Sort By "Snap to Grid" or "Name", then "Use as Defaults". One-off: right-click → Clean Up By → Name. The Desktop has its own `⌘J` settings.
- Full path in the title bar and the path bar:
  ```sh
  defaults write com.apple.finder _FXShowPosixPathInTitle -bool true
  defaults write com.apple.finder ShowPathbar -bool true
  killall Finder
  ```
- Quick Look renders Markdown with `brew install --cask qlmarkdown`; it also previews html, pdf and most other files with `Space`.
***

# Input method
- Add Pinyin: System Settings → Keyboard → Text Input → Input Sources → Edit… → **+** → Chinese, Simplified → Pinyin – Simplified.
- Switch key: Keyboard Shortcuts… → Input Sources → "Select the previous input source" (`Ctrl+Space`, off by default), or in Input Sources turn on "Use Caps Lock to switch to and from ABC".
- In Pinyin: `Shift` toggles Chinese / English, `Space` takes the first candidate, `-` / `=` page candidates.
- Per-window input method: Input Sources → Edit… → "Automatically switch to a document's input source". Per-app defaults: `brew install --cask input-source-pro`.
***

# Window management
- No built-in maximize shortcut for PC keyboards: `fn Ctrl F` (Fill) / `fn Ctrl R` (restore) need the Apple 🌐 key. Double-click the title bar or hover the green button instead.
- **Rectangle** (`brew install --cask rectangle`): `Ctrl⌥Enter` maximize, `Ctrl⌥Backspace` restore, `Ctrl⌥←/→` halves.
- **AltTab** (`brew install --cask alt-tab`): Windows-style `⌥Tab` window switcher with previews. Needs Accessibility + Screen Recording permission. The 14-day trial is for Pro (search, styles, more shortcuts); the free version keeps working.
- macOS cannot rebind `⌘Tab` itself or bind a lone modifier key; those need a helper app (AltTab, Karabiner).
***

# Karabiner-Elements
- `brew install --cask karabiner-elements` (pkg, asks for the password), then allow the driver extension and Input Monitoring. Config `~/.config/karabiner/karabiner.json` reloads automatically.
- **A keyboard that also reports a pointing device is ignored by default** — rules silently do nothing. Check `/var/log/karabiner/core_service.log` for "grabbed", and enable the device in the profile:
  ```json
  "devices": [{
    "identifiers": {"is_keyboard": true, "is_pointing_device": true, "product_id": 544, "vendor_id": 20480},
    "ignore": false
  }]
  ```
  `/Library/Application Support/org.pqrs/Karabiner-Elements/bin/karabiner_cli --list-connected-devices` lists the ids.
- **Tap ⌘ alone → Mission Control** (like the Super key in GNOME); holding ⌘ with other keys is unchanged:
  ```json
  {
    "type": "basic",
    "from": {"key_code": "left_command", "modifiers": {"optional": ["any"]}},
    "to": [{"key_code": "left_command", "lazy": true}],
    "to_if_alone": [{"key_code": "up_arrow", "modifiers": ["left_control"]}],
    "parameters": {"basic.to_if_alone_timeout_milliseconds": 300}
  }
  ```
- **⌘⌥N → open / focus the Nth app** (like Super+N on the Ubuntu dock), one manipulator per key:
  ```json
  {
    "type": "basic",
    "from": {"key_code": "2", "modifiers": {"mandatory": ["command", "option"]}},
    "to": [{"shell_command": "open -a 'Microsoft Edge'"}]
  }
  ```
  `open -a` launches the app or brings it to the front. If it opens a second instance, the running copy is from a mounted `.dmg` — quit it, eject the image, run the one in `/Applications`.
- **F5 reloads in browsers**: Mac browsers use `⌘R`, and with standard F-keys macOS Dictation grabs F5 (stray "I" typed, black focus box in the page):
  ```json
  {
    "type": "basic",
    "from": {"key_code": "f5"},
    "to": [{"key_code": "r", "modifiers": ["left_command"]}],
    "conditions": [{"type": "frontmost_application_if",
                    "bundle_identifiers": ["^com\\.microsoft\\.edgemac$", "^com\\.google\\.Chrome$", "^com\\.apple\\.Safari$", "^org\\.mozilla\\.firefox$"]}]
  }
  ```
- F-keys: Karabiner maps F1–F12 to Apple media actions (F5 = Dictation). Real F-keys: System Settings → Keyboard → Keyboard Shortcuts → Function Keys → "Use F1, F2, etc. keys as standard function keys". Dictation's own shortcut: Keyboard → Dictation → Shortcut. Karabiner → EventViewer shows what a key really sends.
***

# Dock, mouse, system
- Dock auto-hide shows up slowly (0.5 s delay + 0.5 s animation):
  ```sh
  defaults write com.apple.dock autohide-delay -float 0
  defaults write com.apple.dock autohide-time-modifier -float 0.25
  killall Dock
  # undo: defaults delete com.apple.dock autohide-delay; defaults delete com.apple.dock autohide-time-modifier; killall Dock
  ```
  `⌥⌘D` toggles auto-hide.
- Mouse scroll direction like Ubuntu: System Settings → Mouse → turn off "Natural scrolling" (shared with the trackpad; LinearMouse / Scroll Reverser to split them).
- Default browser: System Settings → Desktop & Dock → Default web browser.
***

# Apps
- Prefer Homebrew casks: no Apple ID, no `.dmg` mistakes, easy removal.
  ```sh
  brew install --cask microsoft-edge obsidian
  brew uninstall --cask --zap alt-tab  # --zap also removes settings and caches
  brew list --cask
  ```
- Without brew: drag the app from `/Applications` to the Trash (`⌘Backspace`); App Store apps can also be removed from the Apps grid while holding `⌥`. Leftovers live in `~/Library/Application Support`, `Preferences`, `Caches`; `brew install --cask appcleaner` finds them. Apps with drivers (Karabiner, Docker/OrbStack, VPNs) need their own uninstaller.
- Installing from a `.dmg`: drag to Applications, **eject**, then open from Applications. Running from the mounted image makes `open -a` start a second copy.
- Edge vertical tabs: `⌘⇧,`. Safari has no real vertical tabs; Firefox has them under Settings → General → Browser Layout.
- **Slow App Store downloads** (50 KB/s): the network's DNS sent the Apple CDN hostnames to a slow edge. Find the connection with `nettop -L 1 -n -m tcp -p appstoreagent`, then pin the hostname in `/etc/hosts` to a faster Apple edge (`curl --resolve host:443:IP` to test it). App Store apps download from `iosapps.itunes.apple.com`; OS updates from `swcdn.apple.com`, `updates.cdn-apple.com`, `mesu.apple.com`. Remove the pins afterwards — a stale IP breaks updates.
- Markdown editors: macOS has none built in. Typora (WYSIWYG, File Tree `⌃⌘1`), Obsidian (folder of notes), MarkEdit (native, lightweight), VS Code (`⌘K V` preview), Zettlr.
***

# Terminal and zsh
## free -h
- No `free` on macOS:
  ```sh
  top -l 1 -n 0 | grep PhysMem  # PhysMem: 15G used (1979M wired, 5455M compressor), 172M unused.
  sysctl vm.swapusage           # total = 3072.00M  used = 1933.94M  free = 1138.06M  (encrypted)
  memory_pressure | tail -1     # System-wide memory free percentage: 52%
  vm_stat                       # page counts, like /proc/meminfo
  ```
- "Unused" is always small: free RAM goes to file cache, and idle pages are compressed before swapping. Memory pressure (or Activity Monitor → Memory → Memory Pressure graph) is the real health check.
- `free` function for `~/.zshrc`, used = app + wired + compressed as in Activity Monitor:
  ```sh
  free() {
      { sysctl -n hw.memsize; vm_stat; sysctl -n vm.swapusage; } | awk '
          function h(b) { split("B Ki Mi Gi Ti", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ }; return sprintf(i > 3 ? "%.1f%s" : "%.0f%s", b, u[i]) }
          NR == 1 { total = $1; next }
          /page size of/ { ps = $8 }
          { gsub(/\./, "", $NF); v = $NF }
          /^Pages free/ { fr = v } /^Pages speculative/ { sp = v } /^Pages wired/ { wi = v }
          /^Pages purgeable/ { pu = v } /^Anonymous pages/ { an = v } /^File-backed/ { fb = v }
          /occupied by compressor/ { co = v }
          /^total =/ { st = $3; su = $6; sf = $9 }
          END {
              used = (an - pu + wi + co) * ps; cache = (fb + pu) * ps
              printf "%-6s %10s %10s %10s %10s %10s\n", "", "total", "used", "free", "buff/cache", "available"
              printf "%-6s %10s %10s %10s %10s %10s\n", "Mem:", h(total), h(used), h((fr + sp) * ps), h(cache), h(total - used)
              m = 1024 * 1024; sub(/M/, "", st); sub(/M/, "", su); sub(/M/, "", sf)
              printf "%-6s %10s %10s %10s\n", "Swap:", h(st * m), h(su * m), h(sf * m)
          }'
  }
  free
  #             total       used       free buff/cache  available
  # Mem:       16.0Gi     12.9Gi       89Mi      2.5Gi      3.1Gi
  # Swap:       3.0Gi      1.9Gi      1.1Gi
  ```
## Up arrow history search with the cursor at the end
- `history-beginning-search-backward` keeps the cursor where it was (column 0 on an empty line). The `*-line-or-beginning-search` widgets do the same prefix search and move to the end, like bash `history-search-backward`:
  ```sh
  autoload -U up-line-or-beginning-search down-line-or-beginning-search
  zle -N up-line-or-beginning-search; zle -N down-line-or-beginning-search
  bindkey '^[[A' up-line-or-beginning-search;  bindkey '^[[B' down-line-or-beginning-search
  bindkey '^[OA' up-line-or-beginning-search;  bindkey '^[OB' down-line-or-beginning-search  # application-mode arrows (tmux, some apps)
  ```
***

# SSH: Too many authentication failures
- `ssh -v` shows every agent key being offered, then `Received disconnect ... Too many authentication failures` before any password prompt. The server's `MaxAuthTries` (default 6) counts each rejected key; with 6 keys in the agent the password step is never reached. It worked on Ubuntu because fewer keys were loaded there.
- Fix per host in `~/.ssh/config`:
  ```sh
  # password login: offer no keys
  Host my-password-server
      PubkeyAuthentication no
      PreferredAuthentications keyboard-interactive,password

  # key login: offer only the right key
  Host my-key-server
      IdentityFile ~/.ssh/id_rsa_xxx
      IdentitiesOnly yes
  ```
- `ssh-add -l` lists agent keys, `ssh-add -D` clears them. With `AddKeysToAgent yes` keys are re-added on first use. `ssh -G host` prints the effective config. Paths copied from Linux (`/home/<user>/...`) must become `~/...`.
- Ghostty's "Failed to install terminfo" on ssh is its shell integration copying `xterm-ghostty` to the server; if a server complains about the terminal, use `TERM=xterm-256color ssh ...`.
***

# Git: case-insensitive filenames
- APFS is case-insensitive: `readme.md` and `README.md` are one file. A repo created on Linux with both breaks `git pull --rebase`, and even `git rebase --abort`:
  ```sh
  error: The following untracked working tree files would be overwritten by merge:
      readme.md
  ```
- Recovery when the remote only deleted `readme.md` and a local commit still has both: leave the rebase without checking out (`git rebase --quit`), re-apply the local commit's diff without the file, then point the branch at it:
  ```sh
  git rebase --quit  # HEAD stays on the remote commit
  git diff --binary <remote> <local> -- . ':(exclude)readme.md' | git apply --index
  git commit -C <local>
  git diff --name-status <local> HEAD  # only: D readme.md
  git branch -f master HEAD && git switch master
  ```
- Never create names that differ only in case; rename through a temp name (`git mv a tmp && git mv tmp A`).
***

<!-- related -->
# Related
- [Github](../2017/01-14_github.md)
- [Shortcuts for Atom (text editor) 1.0 (win/en](../2017/07-07_atom_usage.md)
***
<!-- /related -->
