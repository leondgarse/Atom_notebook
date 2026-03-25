---
name: obsidian-openclaw-memory
description: Set up Obsidian + OpenClaw as a living AI memory system. Use when helping users configure their workspace so their AI assistant remembers context across sessions, builds a knowledge graph, and proactively maintains memory. Covers file structure, Obsidian vault setup, QMD semantic search, and heartbeat-based memory maintenance.
---

# Obsidian + OpenClaw Memory System

## Overview

The AI doesn't *have* memory — it *reads* memory. OpenClaw injects workspace files into the system prompt at session start, giving the AI persistent context across sessions. Obsidian visualizes the knowledge graph. QMD provides semantic search so the AI finds relevant context without loading everything.

**Three components:**
- **OpenClaw** — reads workspace files at session start (injected into system prompt)
- **Obsidian** — vault pointed at the workspace; Graph View shows connections between files
- **QMD** — on-device semantic search; find relevant files without loading them all

## File Structure

See `references/file-structure.md` for the full annotated file tree.

Core files:

| File/Folder | Purpose |
|---|---|
| `MEMORY.md` | Curated long-term memory (distilled from daily logs) |
| `memory/YYYY-MM-DD.md` | Raw daily session logs |
| `second-brain/` | Structured knowledge base (concepts, journal, documents) |
| `directives/` | SOPs and workflows |
| `HEARTBEAT.md` | Drives proactive AI behavior between sessions |
| `AGENTS.md` | How the AI should operate in this workspace |
| `USER.md` | Context about the human |
| `SOUL.md` | AI persona and tone |

## How OpenClaw Reads Files

OpenClaw's workspace injection reads files listed in its config and prepends them to the system prompt. This means:
- Files in the workspace root are always available
- The AI "wakes up" each session already knowing what's in those files
- Updating a file = updating what the AI knows next session

Key principle: **write important things to files, not just say them in chat.**

## Obsidian Setup

1. **Create vault** pointing to your `~/clawd` workspace folder (File → Open Vault → Open Folder as Vault)
2. **Enable Graph View** (Ctrl/Cmd+G) — see how memory files link to each other
3. **Install plugins:**
   - **Dataview** — query memory files like a database (`TABLE, LIST, TASK` queries)
   - **Templater** — daily note templates for `memory/YYYY-MM-DD.md`
4. **Daily note template** (via Templater):
   ```
   # <% tp.date.now("YYYY-MM-DD") %>
   
   ## Session Log
   
   ## Decisions Made
   
   ## Things to Remember
   ```
5. **Dataview query** to surface recent memories:
   ```dataview
   LIST FROM "memory" SORT file.name DESC LIMIT 7
   ```

## QMD Semantic Search Setup

```bash
# Add workspace to QMD index
qmd collection add ~/clawd --name clawd

# Generate embeddings (run after adding new files)
qmd embed

# Search from within OpenClaw
mcporter call qmd.vsearch query="what did we decide about X"
mcporter call qmd.query query="project status"
```

This lets the AI find relevant context without loading every file into the context window.

## Heartbeat-Based Memory Maintenance

`HEARTBEAT.md` drives proactive AI behavior. OpenClaw polls it on a schedule and acts on what it finds.

Example `HEARTBEAT.md`:
```markdown
## Memory Maintenance
- [ ] Review memory/ files from last 3 days
- [ ] Distill key insights into MEMORY.md
- [ ] Remove outdated entries from MEMORY.md
```

The AI will pick this up, review recent logs, and update long-term memory — like a human reviewing their journal and updating their mental model.

**Schedule:** Add a heartbeat cron or configure OpenClaw's heartbeat interval. Every few days is sufficient for memory distillation.

## Best Practices

1. **Write it down** — if you want the AI to remember something next session, say "write this to memory" or update `memory/YYYY-MM-DD.md` directly
2. **Keep MEMORY.md curated** — it's the distilled essence, not a dump. Short, high-signal entries.
3. **Daily logs are raw** — `memory/YYYY-MM-DD.md` is for raw session notes; don't worry about formatting
4. **Use directives/ for SOPs** — repeatable workflows go here so the AI can follow them consistently
5. **Link files in Obsidian** — use `[[MEMORY]]` wiki-links to build the graph view
6. **Re-embed after adding files** — run `qmd embed` after adding significant new content

## Architecture Diagram

See `assets/architecture.png` for a visual overview of how the three components interact.
