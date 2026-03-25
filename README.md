# Atom Notebook

Personal knowledge base built with Obsidian + OpenClaw for persistent AI memory.

## What This Is

This is an Obsidian vault containing:
- Daily session logs and notes
- Structured summaries of important discussions
- Concept files for frameworks and ideas
- SOPs and workflows
- AI persona and user context
- Memory system integration

## Structure

```
Atom_notebook/
├── MEMORY.md                    # Curated long-term memory
├── AGENTS.md                    # AI operating principles
├── SOUL.md                      # AI persona and tone
├── USER.md                      # Context about Leon
├── TOOLS.md                     # Local setup notes
├── HEARTBEAT.md                 # Proactive behavior checklist
├── memory/                      # Daily session logs
│   └── YYYY-MM-DD.md
├── second-brain/                # Structured knowledge
│   ├── journal/                # Daily summaries
│   ├── concepts/               # Deep dives
│   └── documents/              # Working docs
├── directives/                  # SOPs and workflows
└── skills/                      # Custom skills
    └── obsidian-openclaw-memory/
```

## Obsidian Setup

1. **Open this folder as a vault:**
   - In Obsidian: `File → Open Folder as Vault`
   - Navigate to: `~/Atom_notebook`
   - Confirm

2. **Enable Graph View:**
   - Shortcut: `Ctrl/Cmd+G`
   - See connections between files

3. **Install plugins (optional but recommended):**
   - **Dataview** — Query memory files
   - **Templater** — Daily note templates
   - **Obsidian Graph Analysis** — Visual graph insights

4. **Configure vault:**
   - Go to `Settings → Files & Links`
   - Set daily notes folder to: `memory`

## How It Works

**OpenClaw Integration:**
- OpenClaw reads files at session start
- Updated files = AI knows more next session
- Key principle: Write important things to files, not just say them

**Daily Flow:**
1. AI writes raw notes to `memory/YYYY-MM-DD.md`
2. AI writes structured summary to `second-brain/journal/YYYY-MM-DD.md`
3. AI periodically updates `MEMORY.md` with distilled insights

**Obsidian:**
- Point vault at `~/Atom_notebook`
- Use `[[wiki-links]]` to connect files
- Build knowledge graph with proper linking
- Dataview for querying and analysis

## Git Repository

This repo is a GitHub Pages site.
- `public/` — Published content
- `private/` — Sensitive content (not published)

## Usage

**When a session ends:**
- Check `memory/YYYY-MM-DD.md` for today's notes
- Read `second-brain/journal/YYYY-MM-DD.md` for structured summary
- Update `MEMORY.md` if new insights worth keeping

**For research:**
- Search daily logs via Obsidian search
- Query with Dataview: `LIST FROM "memory" WHERE file.mtime > date(today)`
- Use wiki-links to find related concepts

**For reference:**
- `directives/*.md` — Standard workflows
- `second-brain/concepts/*.md` — Deep dives on ideas
- `USER.md` — Context about Leon

## Resources

- **OpenClaw Docs:** https://docs.openclaw.ai
- **Obsidian Docs:** https://docs.obsidian.md
- **QMD Docs:** (TBD)
- **GitHub:** https://github.com/Samin12/obsidian-openclaw-memory

## Maintenance

**Periodic (every few days):**
- Review `memory/` from last 3 days
- Distill key insights to `MEMORY.md`
- Remove outdated entries

**Heartbeat checks:**
- Review `HEARTBEAT.md` for tasks
- Check for new messages/mentions
- Update tools notes if needed

---

*Built with Obsidian + OpenClaw for persistent AI memory.*