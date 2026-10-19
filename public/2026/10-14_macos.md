---
tags: [shell-tools, macos, terminal]
summary: macOS tips — Terminal.app and Ghostty tab shortcuts and other small setup notes.
---
1. Terminal tabs (Terminal.app and Ghostty)

┌─────────────────────┬──────────────────────────────────────────┐
│       Action        │                   Keys                   │
├─────────────────────┼──────────────────────────────────────────┤
│ Go to tab N         │ ⌘1 … ⌘9                                  │
├─────────────────────┼──────────────────────────────────────────┤
│ Next / previous tab │ ⌘⇧] / ⌘⇧[ (or Ctrl+Tab / Ctrl+Shift+Tab) │
├─────────────────────┼──────────────────────────────────────────┤
│ New tab / close tab │ ⌘T / ⌘W                                  │
├─────────────────────┼──────────────────────────────────────────┤
│ New window          │ ⌘N                                       │
├─────────────────────┼──────────────────────────────────────────┤
│ Copy / paste        │ ⌘C / ⌘V (no more Ctrl+Shift+C/V)         │
├─────────────────────┼──────────────────────────────────────────┤
│ Interrupt a process │ Ctrl+C (the same as before)              │
└─────────────────────┴──────────────────────────────────────────┘

Your tmux prefix C-s works the same as before.

"Win+N opens the Nth Dock app": macOS has no built-in shortcut for this. Here's what to use instead:
- ⌘Tab: the app switcher. Hold ⌘ and tap Tab to cycle through apps. This is the closest match to your old habit.
- ⌘ (backtick): cycles through windows of the current app, for example between two Chrome windows.
- If you really want "a key opens app X", you can set a global hotkey per app in Raycast (brew install --cask raycast) or
  Hammerspoon. For example, Ctrl+Option+1 → Ghostty, Ctrl+Option+2 → Chrome.

2. Launching apps (for example WeChat)

- ⌘Space opens Spotlight. Type wechat and press Enter. This is how most Mac users launch apps.
- In recent macOS versions, Spotlight also has an Apps view, roughly the GNOME "Show Apps" grid. Open Spotlight, then press ⌘1. On a
  Mac keyboard F4 also opens it.
- WeChat isn't installed yet: brew install --cask wechat

3. Other everyday shortcuts

Windows and apps
- ⌘Q quits the app. ⌘W only closes the window, and the app keeps running. This is a big difference from Linux.
- ⌘H hides the app, ⌘M minimizes the window.
- ⌘⌥Esc opens Force Quit, like killing a frozen app.
- Ctrl+↑ opens Mission Control (all windows). Ctrl+←/→ switches desktops.
- Window snapping isn't great out of the box. Install rectangle (it's on your to-do list) to get Ctrl+Option+←/→ for half-screen.

Text editing (in all apps, mostly not in the terminal)
- ⌘← / ⌘→ jumps to the start/end of the line (Home/End)
- ⌥← / ⌥→ moves one word at a time (was Ctrl+←/→)
- ⌘↑ / ⌘↓ jumps to the top/bottom of the document
- ⌥Backspace deletes a word, ⌘Backspace deletes to the start of the line

System
- Ctrl+Space switches input method, for example English ↔ Chinese Pinyin
- Ctrl+⌘Q locks the screen
- ⌘⇧4 takes an area screenshot, ⌘⇧3 the full screen, ⌘⇧5 opens the menu with recording
- Ctrl+⌘Space opens the emoji picker

Finder (the file manager)
- ⌘⇧G goes to a path you type, like Ctrl+L in Nautilus
- ⌘⇧. shows or hides hidden files
- Enter renames a file. ⌘O or ⌘↓ opens it.
- ⌘Backspace moves a file to the Trash
- Space shows a Quick Look preview

Tip: swap Win and Alt

On a Mac keyboard the order next to the spacebar is Ctrl ⌥ ⌘ [space], so ⌘ sits right beside the spacebar and is easy to reach with
your thumb. On your PC keyboard ⌘ (Win) is one key further away. You can swap them in System Settings → Keyboard → Keyboard
Shortcuts → Modifier Keys: pick your keyboard, then set Option → Command and Command → Option. Many PC-keyboard users do this
because almost every shortcut uses ⌘.
