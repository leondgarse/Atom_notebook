# File Structure Reference

Annotated file tree for the Obsidian + OpenClaw memory workspace.

```
~/clawd/
│
│  ── CORE IDENTITY FILES ──────────────────────────────────────────
│
├── AGENTS.md            # Operating principles for the AI agent
│                        # How to behave, tool usage, task tracking
│                        # Multi-agent rules, safety guidelines
│
├── SOUL.md              # AI persona, tone, and values
│                        # Who the AI is, how it speaks
│                        # Guides every response
│
├── IDENTITY.md          # Name, avatar, vibe
│                        # Short — just the core identity facts
│
├── USER.md              # About the human
│                        # Name, timezone, goals, business context
│                        # What to call them, what they care about
│
├── TOOLS.md             # Local tool notes (API keys, credentials, hints)
│                        # Skills are generic; TOOLS.md is personal
│
│  ── MEMORY FILES ─────────────────────────────────────────────────
│
├── MEMORY.md            # ★ LONG-TERM MEMORY ★
│                        # Curated, distilled insights from daily logs
│                        # Like a human's long-term memory
│                        # Updated periodically (not every session)
│                        # ONLY load in main session (privacy)
│
├── memory/
│   ├── 2026-03-05.md    # Raw daily session log
│   ├── 2026-03-04.md    # One file per day
│   └── heartbeat-state.json  # Tracks last check times (email, calendar, etc.)
│                        # Prevents redundant checks during heartbeats
│
│  ── KNOWLEDGE BASE ────────────────────────────────────────────────
│
├── second-brain/
│   ├── README.md        # Index / navigation file for the knowledge base
│   ├── journal/
│   │   └── 2026-03-05.md  # Daily summaries of important discussions
│   ├── concepts/
│   │   └── example.md   # Deep dives on important ideas / frameworks
│   └── documents/       # Working documents, strategies, plans
│
│  ── DIRECTIVES / SOPs ─────────────────────────────────────────────
│
├── directives/
│   ├── example-sop.md   # Step-by-step workflow for repeatable tasks
│   └── ...              # One file per workflow/SOP
│
│  ── PROACTIVE BEHAVIOR ────────────────────────────────────────────
│
├── HEARTBEAT.md         # Checklist for proactive AI behavior
│                        # Checked on heartbeat polls
│                        # Example: "review memory, check email"
│
│  ── EXECUTION & AUTOMATION ────────────────────────────────────────
│
├── execution/           # Deterministic Python scripts
│   └── example.py       # Scripts the AI can call vs. rewriting each time
│
├── trigger/             # Trigger.dev serverless tasks
│   └── example.ts       # Webhook receivers, scheduled jobs
│
│  ── SKILLS ────────────────────────────────────────────────────────
│
├── skills/
│   └── my-skill/        # Custom skills extend AI capabilities
│       └── SKILL.md     # Each skill is self-contained
│
│  ── SCRATCH / TEMP ────────────────────────────────────────────────
│
├── .tmp/                # Intermediate files (always regenerated, not committed)
│
│  ── PROJECTS ──────────────────────────────────────────────────────
│
└── projects/
    └── my-project/      # One folder per project
        └── ...          # Project-specific files, code, assets
```

---

## File Lifecycle

### MEMORY.md (Long-Term Memory)
- **Written:** Periodically, when heartbeat distillation runs
- **Read:** Every main session (injected into system prompt)
- **Content:** Key decisions, important context, lessons learned
- **Keep:** Short and high-signal. Remove outdated entries.

### memory/YYYY-MM-DD.md (Daily Logs)
- **Written:** During each session — raw notes, decisions, things to remember
- **Read:** AI reads last 1-2 days for recent context
- **Content:** Unformatted notes, session summaries, quick captures
- **Keep:** All logs (they're cheap, useful for distillation)

### second-brain/journal/YYYY-MM-DD.md (Daily Summaries)
- **Written:** End of significant conversations
- **Read:** When researching past work or context
- **Content:** Structured summary: discussion topics, decisions, open questions
- **Keep:** Permanent record

### HEARTBEAT.md (Proactive Checklist)
- **Written:** When you want the AI to do something proactively
- **Read:** On every heartbeat poll
- **Content:** Short checklist of tasks, checks, reminders
- **Keep:** Small — it burns tokens on every heartbeat

### directives/ (SOPs)
- **Written:** When a workflow is established and should be repeatable
- **Read:** When AI executes that specific workflow
- **Content:** Step-by-step instructions, inputs, outputs, edge cases
- **Keep:** Update as workflows evolve; add lessons learned

---

## Obsidian-Specific Tips

- Use `[[filename]]` wiki-links to create edges in the Graph View
- Tag files with `#memory`, `#concept`, `#directive` for filtering
- Dataview plugin can query across all files: `TABLE file.mtime FROM "memory" SORT file.mtime DESC`
- Set `memory/` as the daily notes folder in Obsidian settings
- Use canvas feature for visual planning that links back to notes
