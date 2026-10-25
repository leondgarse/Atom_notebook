---
tags:
  - shell-tools
  - obsidian
  - markdown
summary: Obsidian setup on macOS and Linux — hub vault of symlinks, per-vault settings and hotkeys, wide-text CSS, Pulsar-like editing (2-space Tab, fold to headings, move / join lines, multi-cursor, table editing), own plugins, rendering indented tables, Smart Connections and private notes.
---
# ___Obsidian___
***

# Vaults and files
- A vault is one root folder per window; no "Add Project Folder". To combine folders, make a hub vault of symlinks (linked folders must not overlap):
  ```sh
  mkdir -p ~/obsidian_projects && ln -s ~/Atom_notebook ~/obsidian_projects/Atom_notebook
  ```
- Plugins, themes, settings and hotkeys live in `<vault>/.obsidian/` — per vault. Copy `.obsidian` to reuse a setup, or use a single hub vault.
- Other file types (`.py` ...) are hidden: Settings → Files and links → "Detect all file extensions". They still open in the default app unless a code-editor plugin is installed.
- Open in a new tab: `⌘`-click / `Ctrl`-click (`⌘⌥` / `Ctrl+Alt`-click for a split); pinned tabs are never replaced.
- Hotkey file `<vault>/.obsidian/hotkeys.json`: `Mod` is ⌘ on macOS and `Ctrl` on Linux, `Ctrl` is always Control. Single combos only, no Pulsar-style chords like `Ctrl+K Ctrl+1`. Reload with `⌘P` / `Ctrl+P` → "Reload app without saving":
  ```json
  {"app:toggle-left-sidebar": [{"modifiers": ["Ctrl"], "key": "\\"}]}
  ```
- Wider text: keep Settings → Editor → "Readable line length" on and add `<vault>/.obsidian/snippets/wide.css`, then Appearance → CSS snippets → refresh → enable:
  ```css
  body { --file-line-width: 1200px; }
  ```
***

# Pulsar-like editing
- **Tab = 2 spaces**: since 1.8, Tab with "Indent using tabs" off always inserts 4 spaces; "Tab indent size" is only the display width of a tab character. Own plugin [Indent with Spaces](https://github.com/leondgarse/obsidian-indent-with-spaces) (id `two-space-indent`) makes Tab / Shift+Tab indent by N spaces (Settings → Indent with Spaces → Spaces per indent, default 2), at line start, on list items (even mid-line) and on selected lines. Tab in the middle of plain text and in Live Preview table cells (`.table-cell-wrapper` editors) is left to Obsidian.
- **Table editing like Atom's markdown-table-editor**: community plugin "Advanced Tables" (`table-editor-obsidian`, same engine) — auto-format column widths (CJK counted double), Tab on a header-only row inserts `| --- |`, Tab / Shift+Tab next / previous cell, Enter new row. Works on indented tables too. Its own Tab / Enter bindings are off in Live Preview, so Indent with Spaces runs its "Next cell" / "Previous cell" commands on `|` lines instead; Enter for a new row still works only in Source mode. Keep Format type "Normal".
- **Fold to headings**: no built-in "fold to level N". Own plugin [fold-unfold-headings](https://github.com/leondgarse/obsidian-fold-unfold-headings); built-in "Unfold all headings and lists" may not clear its folds, use its own "Unfold all".
- **Join lines, duplicate / delete line**: no built-in commands; community plugin "Code Editor Shortcuts" (`obsidian-editor-shortcuts`). Its defaults for copy line up / down, insert line below, select line and select word were removed in Hotkeys, so Obsidian's own `Ctrl+Enter`, `Ctrl+L`, `Ctrl+D` stay.
- Move line and add cursor exist as built-in commands but have no default hotkeys; bound in Hotkeys, see [Hotkeys](#hotkeys).
- **Own plugins**: [indent-with-spaces](https://github.com/leondgarse/obsidian-indent-with-spaces), [fold-unfold-headings](https://github.com/leondgarse/obsidian-fold-unfold-headings), [indented-table-render](https://github.com/leondgarse/obsidian-indented-table-render). Each `<vault>/.obsidian/plugins/<id>/` folder is its own git repo (`main.js` + `manifest.json`, no build step). Release: bump `version` in `manifest.json` and `versions.json`, commit, `git tag 0.2.1 && git push origin main 0.2.1`; a GitHub Actions workflow creates the release with artifact attestations (recommended by the community directory). Tag = exact version, no `v`. Submit at [community.obsidian.md](https://community.obsidian.md) (replaced the `obsidian-releases` PR in 2026); plugin `id` must not contain "obsidian". After publishing, only the display `name` can change; the `id` (and its folder) stays, or installed copies stop getting updates.
***

# Hotkeys
- Linux keys; on macOS `Ctrl` → `⌘` and `Alt` → `⌥`, except `Ctrl+Tab` / `Ctrl+Shift+Tab` and the `Ctrl+\` binding, which stay Control.
- From Obsidian 1.14.5 defaults, Code Editor Shortcuts 1.14.0 defaults, own plugins, and `hotkeys.json`. Change in Settings → Hotkeys; defaults aren't written to `hotkeys.json`, only changes are.

| General      |                                        |
| ------------ | -------------------------------------- |
| Ctrl+P       | Command palette                        |
| Ctrl+O       | Quick switcher (open file by name)     |
| Ctrl+,       | Settings                               |
| Ctrl+E       | Toggle Reading view / editing view     |
| Ctrl+\       | Toggle left sidebar (`hotkeys.json`)   |
| Ctrl+G       | Graph view                             |
| Ctrl+;       | Add file property                      |
| F1           | Help                                   |

| Files and Tabs              |                          |
| --------------------------- | ------------------------ |
| Ctrl+N                      | New note                 |
| Ctrl+Shift+N                | New note to the right    |
| Ctrl+S                      | Save                     |
| F2                          | Rename file (edit title) |
| Ctrl+T                      | New tab                  |
| Ctrl+W                      | Close tab                |
| Ctrl+Shift+W                | Close window             |
| Ctrl+Shift+T                | Reopen closed tab        |
| Ctrl+Tab / Ctrl+Shift+Tab   | Next / previous tab      |
| Ctrl+PageDown / Ctrl+PageUp | Next / previous tab      |
| Ctrl+9                      | Last tab                 |

| Editing Lines    |                                                   |
| ---------------- | ------------------------------------------------- |
| Ctrl+Up / Down   | Move line up / down (`hotkeys.json`)              |
| Ctrl+Shift+D     | Duplicate line (Code Editor Shortcuts)            |
| Ctrl+Shift+K     | Delete line (Code Editor Shortcuts)               |
| Ctrl+D           | Delete paragraph                                  |
| Ctrl+J           | Join lines (Code Editor Shortcuts)                |
| Ctrl+Shift+Enter | Insert line above (Code Editor Shortcuts)         |
| Tab / Shift+Tab  | Indent / outdent by 2 spaces (Indent with Spaces) |
|                  | at line start, list items, selected lines         |
| Ctrl+/           | Toggle comment `%% %%`                            |
| Ctrl+L           | Toggle checklist `- [ ]`                          |

| Cursors and Selection |                                                                  |
| --------------------- | ---------------------------------------------------------------- |
| Alt+Shift+Up / Down   | Add cursor above / below (`hotkeys.json`, Code Editor Shortcuts) |
| Ctrl+Shift+L          | Select all occurrences (Code Editor Shortcuts)                   |

| Tables            |                                                                |
| ----------------- | -------------------------------------------------------------- |
| Tab / Shift+Tab   | Next / previous cell (Advanced Tables, via Indent with Spaces) |
| Tab on header row | Insert `\| --- \|` delimiter row and format                    |
| Enter             | New row (Advanced Tables, Source mode only)                    |

| Formatting and Links       |                                    |
| -------------------------- | ---------------------------------- |
| Ctrl+B                     | Bold                               |
| Ctrl+I                     | Italic                             |
| Ctrl+K                     | Insert Markdown link               |
| Alt+Enter                  | Follow link under cursor           |
| Ctrl+Enter                 | Open link in new tab               |
| Ctrl+Alt+Enter             | Open link to the right             |
| Ctrl+Alt+Shift+Enter       | Open link in new window            |

| Folding          |                                                    |
| ---------------- | -------------------------------------------------- |
| Ctrl+Alt+0       | Fold all content to its own heading (Outline-like) |
| Ctrl+Alt+1 ... 5 | Fold to heading level 1 ... 5                      |
| Ctrl+Alt+U       | Unfold all                                         |
|                  |                                                    |

| Find and Replace |                                  |
| ---------------- | -------------------------------- |
| Ctrl+F           | Find in current file             |
| Ctrl+H           | Find and replace in current file |
| Ctrl+Shift+F     | Search in vault                  |
|                  |                                  |
***
# Markdown that Obsidian doesn't render
- Indented tables (`  | a | b |`) and tables right after a text line: Obsidian's parser needs them flush-left with a blank line before. Atom-style notes (everything indented 2 spaces under headings) were converted once with a script: dedent a section only if all its lines are indented ≥ 2, keep code inside column-0 fences, expand leading tabs first; verified by rendering every note with `markdown-it-py` before / after.
- Tables nested in list items stay indented: Reading view renders them, Live Preview leaves them as text. Own plugin [indented-table-render](https://github.com/leondgarse/obsidian-indented-table-render) renders them in Live Preview (cursor in the table or a click shows the source) and fixes leftovers in Reading view, without changing the note:
  - Nested in list items at any indent, or 1–3 spaces under a heading; 4+ spaces outside a list stays a code block.
  - Delimiter cells with 1+ dashes (`|-|-|`); header and delimiter must have the same number of cells.
  - Community "Nested Markdown Renderer" needs `---` (3+ dashes) and a list-item parent, and found 70 of 115 indented tables here; "Indented Table" only styles the pipe text.
***

# AI plugins and private notes
- Smart Connections embeds notes locally (transformers model, stored in `<vault>/.smart-env/`); its Smart Chat sends retrieved notes to the configured chat model. Exclude secret folders in Settings → Smart Connections → Excluded folders.
- Notes with passwords: keep them out of git (`.gitignore`), out of AI indexing, and ideally move the secrets to the Passwords app.
***

<!-- related -->
# Related
- [Markdown grammar](../2017/05-25_Markdown_grammar.md)
- [Shortcuts for Atom (text editor) 1.0 (win/en](../2017/07-07_atom_usage.md)
- [macOS](10-14_macos.md)
- [Vim使用](../2013_Basic/05-12_Vim使用.md)
- [Ubuntu 使用与 shell 命令](../2013_Basic/05-12_Ubuntu_Usage.md)
***
<!-- /related -->
