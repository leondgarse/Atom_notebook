# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Workspace Structure

- `~/.openclaw/workspace/` — Main workspace for memory and knowledge management
  - `memory/YYYY-MM-DD.md` — Daily session logs
  - `second-brain/` — Structured knowledge base (journal, concepts, documents)
  - `directives/` — SOPs and workflows
  - `skills/` — Custom skills

- `~/Atom_notebook/` — This Obsidian vault for knowledge graph visualization
  - `memory/YYYY-MM-DD.md` — Daily session logs
  - `second-brain/` — Structured knowledge base (journal, concepts, documents)
  - `public/` — Public-facing content for GitHub Pages
  - `private/` — Sensitive content (not published)

## Obsidian Vault Locations

**Primary vault:** `~/Atom_notebook` — Point Obsidian here for the knowledge graph

**Secondary workspace:** `~/.openclaw/workspace` — Used by OpenClaw for file injection at session start

This ensures:
- Knowledge graph connects all memory components in one vault
- OpenClaw can read files from the workspace
- Files are in one location for semantic search
- Easy access to daily logs and structured knowledge

## Skills

- `skills/obsidian-openclaw-memory/` — Memory system integration skill

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.