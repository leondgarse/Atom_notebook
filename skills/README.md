# Custom Skills

This directory contains custom skills that extend my capabilities beyond the built-in OpenClaw skills.

## Structure

- Each skill is a directory with:
  - `SKILL.md` — the main skill definition
  - Optional additional files (examples, references, assets)

## Usage

When a skill matches a task, I read its `SKILL.md` and follow the instructions.

## Currently Available

- `obsidian-openclaw-memory/` — Setup Obsidian + OpenClaw as a living AI memory system with knowledge graph

## Adding Skills

To add a new skill:
1. Create a directory in `skills/`
2. Add `SKILL.md` with the skill definition
3. Update the workspace structure and AGENTS.md as needed
4. Test the skill by triggering the task

See `~/.nvm/versions/node/v20.20.0/lib/node_modules/openclaw/docs` for OpenClaw skill documentation.