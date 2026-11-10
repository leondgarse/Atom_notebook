---
tags: [llm-agents, antigravity, gemini, agents]
summary: Antigravity CLI (agy) — CLI usage, settings & permissions, rules, workflows, agent skills, hooks, MCP, NotebookLM MCP.
---
# ___Antigravity___
***

# Get Started
## Antigravity CLI (agy)
- **Documentation:** [Antigravity Docs](https://antigravity.google/docs) · [CLI Reference](https://antigravity.google/docs/cli/reference) · [Agent Permissions & Security](https://antigravity.google/docs/permissions)
- **Install & Launch:**
  ```sh
  # macOS / Linux (installs to ~/.local/bin/agy)
  curl -fsSL https://antigravity.google/cli/install.sh | bash

  # Configure environment paths and shell settings
  agy install

  # Start interactive TUI session
  agy

  # Update CLI to latest version
  agy update
  ```

| Command / Flag                              | Description                                                                 | Example                                         |
| ------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------- |
| `agy`                                       | Start interactive TUI session                                               | `agy`                                           |
| `-p`, `--print`, `--prompt "<query>"`       | Run a single prompt non-interactively and print the response                | `agy -p "explain this function"`                |
| `-i`, `--prompt-interactive "<prompt>"`     | Run an initial prompt interactively and continue the session                | `agy -i "fix the build error"`                  |
| `-c`, `--continue`                          | Continue the most recent conversation                                       | `agy -c`                                        |
| `--conversation <id>`                       | Resume a previous conversation by ID                                        | `agy --conversation <uuid>`                     |
| `--model <model-id>`                        | Select model for the current CLI session                                    | `agy --model gemini-4-argon-medium`             |
| `--effort <level>`                          | Set reasoning effort (`low`, `medium`, `high`, `xhigh`, `max`)              | `agy --effort high`                             |
| `--mode <mode>`                             | Set agent execution mode (`accept-edits`, `auto`, `plan`)                   | `agy --mode auto`                               |
| `--add-dir <path>`                          | Add a directory to the workspace (repeatable)                               | `agy --add-dir ../shared`                       |
| `--sandbox`                                 | Run in a sandbox with terminal restrictions enabled                         | `agy --sandbox`                                 |
| `--dangerously-skip-permissions`            | Auto-approve all tool permission requests without prompting                 | `agy --dangerously-skip-permissions`            |
| `--output-format <fmt>`                     | Output format for print mode (`text`, `json`, `stream-json`)                | `agy -p "review" --output-format json`          |
| `agy models`                                | List available models                                                       | `agy models`                                    |
| `agy agents`                                | List available agents                                                       | `agy agents`                                    |
| `agy mcp <subcommand>`                      | Manage MCP servers (`add`, `remove`, `list`, `enable`, `disable`)           | `agy mcp list`                                  |
| `agy plugin <subcommand>`                   | Manage plugins (`list`, `install`, `uninstall`, `enable`, `disable`)        | `agy plugin list`                               |
| `agy remote-control <subcommand>`           | Manage the remote-control background daemon (`start`, `status`, `stop`)     | `agy remote-control status`                     |

- **Interactive TUI Slash Commands:**
  - `/help` — Show all available slash commands
  - `/settings` or `/config` — Open the interactive settings overlay
  - `/permissions` — Open the interactive Permissions Manager (`Project`, `Shared`, or `Global` scope)
  - `/skills` — List and reload available skills
  - `/agents` — Open the Agent Manager to switch agents or monitor subagents
  - `/context` — Inspect loaded rule/config files (`GEMINI.md`, `AGENTS.md`) and token usage
  - `/codesearch` (alias `/cs`) — Fast regex/literal search across the codebase
  - `/plan` — Step-by-step research and implementation planning before execution
  - `/grill-me` — Interactive interview to align on design decisions before building a plan
  - `/goal` — Autonomous long-running execution until the goal is fully achieved
  - `/boost` — Deep thinking, multi-perspective planning, and rigorous verification
  - `/teamwork` — Coordinate a team of autonomous subagents on a large project
  - `/schedule` — Run an instruction on a recurring cron schedule or one-time timer
  - `/learn` — Persist corrections or complex setups into reusable customizations
  - `/compact` — Summarize and compact conversation history to free context window
  - `/btw` — Ask a side question without polluting the main task's context
  - `/resume` — List and resume previous conversations
  - `/exit` or `/quit` (or `Ctrl+D Ctrl+D`) — Exit the CLI
- **Global Configuration & Permissions (`~/.gemini/antigravity-cli/settings.json`):**
  - Rules use `action(target)` syntax (`command(...)`, `unsandboxed(...)`, `read_file(...)`, `write_file(...)`, `read_url(...)`, `execute_url(...)`, `mcp(...)`).
  - Evaluation precedence is strictly **`deny` > `ask` > `allow`**, so you can allow `"command(*)"` globally while blocking destructive commands in `deny` and prompting for sensitive commands in `ask`:
  ```json
  {
    "allowNonWorkspaceAccess": true,
    "colorScheme": "tokyo night",
    "permissions": {
      "allow": [
        "command(*)",
        "read_file(*)",
        "write_file(*)"
      ],
      "deny": [
        "command(rm -rf)",
        "command(sudo)",
        "write_file(.git/)"
      ],
      "ask": [
        "command(git push)",
        "command(rm)"
      ]
    },
    "trustedWorkspaces": [
      "~/workspace"
    ]
  }
  ```
- **Permission Presets (App / IDE / CLI):**
  - **Request Review:** Terminal sandbox disabled; always asks before running commands or modifying files; file access restricted to workspace.
  - **Default:** Terminal sandbox enabled; commands allowed in sandbox and prompted outside; file access restricted to workspace + temp dirs.
  - **Auto:** Terminal sandbox enabled; automatically evaluates and approves safe, reversible commands outside the sandbox while prompting for risky actions.
  - **Turbo:** Unrestricted (`always-proceed`); disables sandbox, grants full filesystem access, and runs all commands without prompting.
- **Installing NotebookLM MCP:** Copy and paste the below into `mcp_config.json` (`~/.gemini/config/mcp_config.json` or `~/.gemini/antigravity/mcp_config.json`):
  ```sh
  cat ~/.gemini/config/mcp_config.json
  # {
  #   "mcpServers": {
  #     "notebooklm": {
  #       "command": "npx",
  #       "args": [
  #         "notebooklm-mcp@latest"
  #       ]
  #     }
  #   }
  # }
  ```
- **AntiGravity Skills Creator:** `antigravity-skill-creator.md`
  ```md
  # Antigravity Skill Creator System Instructions
  You are an expert developer specializing in creating "Skills" for the Antigravity agent environment. Your goal is to generate high-quality, predictable, and efficient `.agents/skills/` (or `.agent/skills/`) directories based on user requirements.

  1. Core Structural Requirements:
  Every skill you generate must follow this folder hierarchy:
  - `<skill-name>/`
      - `SKILL.md` (Required: Main logic and instructions)
      - `scripts/` (Optional: Helper scripts)
      - `examples/` (Optional: Reference implementations)
      - `resources/` (Optional: Templates or assets)
      - `references/` (Optional: Detailed documentation or manuals)

  2. YAML Frontmatter Standards:
  The `SKILL.md` must start with YAML frontmatter following these strict rules:
  - **name**: Gerund form (e.g., `testing-code`, `managing-databases`). Max 64 chars. Lowercase, numbers, and hyphens only. No "claude" or "anthropic" in the name.
  - **description**: Written in **third person**. Must include specific triggers/keywords. Max 1024 chars. (e.g., "Extracts text from PDFs. Use when the user mentions document processing or PDF files.")

  3. Writing Principles (The "Claude Way"):
  When writing the body of `SKILL.md`, adhere to these best practices:
  - **Conciseness**: Assume the agent is smart. Do not explain what a PDF or a Git repo is. Focus only on the unique logic of the skill.
  - **Progressive Disclosure**: Keep `SKILL.md` under 500 lines. If more detail is needed, link to secondary files (e.g., `[See ADVANCED.md](references/ADVANCED.md)`) only one level deep.
  - **Forward Slashes**: Always use `/` for paths, never `\`.
  - **Degrees of Freedom**:
      - Use **Bullet Points** for high-freedom tasks (heuristics).
      - Use **Code Blocks** for medium-freedom (templates).
      - Use **Specific Bash Commands** for low-freedom (fragile operations).

  4. Workflow & Feedback Loops:
  For complex tasks, include:
  - **Checklists**: A markdown checklist the agent can copy and update to track state.
  - **Validation Loops**: A "Plan-Validate-Execute" pattern. (e.g., Run a script to check a config file BEFORE applying changes).
  - **Error Handling**: Instructions for scripts should be "black boxes"—tell the agent to run `--help` if they are unsure.

  5. Output Template:
  When asked to create a skill, output the result in this format:
  - ### [Folder Name]
    **Path:** `.agents/skills/[skill-name]/`
  - ### [SKILL.md]
    (Markdown content with YAML frontmatter)
  ```
***

# Anti-gravity
## Documentation
- [Google Antigravity Documentation](https://antigravity.google/docs) · [Skills](https://antigravity.google/docs/skills) · [Rules & Workflows](https://antigravity.google/docs/rules-workflows) · [Hooks](https://antigravity.google/docs/hooks) · [Plugins](https://antigravity.google/docs/plugins) · [MCP](https://antigravity.google/docs/mcp)

| Customization Type | Config File / Folder                  | Scope                     | Purpose                                                                           |
| ------------------ | ------------------------------------- | ------------------------- | --------------------------------------------------------------------------------- |
| **Rules**          | `GEMINI.md`, `AGENTS.md`, `rules/*.md`| Contextual / Hierarchical | Enforcing coding styles, API restrictions, and local guidelines                   |
| **Skills**         | `skills/<name>/SKILL.md`              | On-Demand (Progressive)   | Teaching multi-step procedures, runbooks, and tool workflows                      |
| **Workflows**      | `workflows/<name>.md`                 | Slash Command (`/name`)   | Repeatable step-by-step prompt trajectories                                       |
| **Plugins**        | `plugins/<name>/plugin.json`          | Bundle                    | Packaging related skills, rules, hooks, and MCP configs into a single unit        |
| **Hooks**          | `hooks.json`                          | Lifecycle Event           | Running shell commands at agent lifecycle points (`PreToolUse`, `PostToolUse`)    |
| **MCP Servers**    | `mcp_config.json`                     | Tool Integration          | Connecting the agent to external services and custom tool providers               |

- **Discovery Locations & Precedence (Highest to Lowest):**
  1. **Workspace Project:** `.agents/` (or `.agent/`, `_agents/`, `_agent/`) and hierarchical `GEMINI.md` / `AGENTS.md` walking up from CWD to the repo root.
  2. **Declared Workspace Configs:** Explicitly listed in `skills.json` or `plugins.json` in the workspace.
  3. **Global Discovery:** `~/.gemini/config/` (and `~/.gemini/GEMINI.md`).
  4. **Built-in Customizations:** Bundled default skills mounted by name.
  5. **Global Declared Configs:** Explicitly listed in global JSON configs.
## Rules
Rules are manually defined constraints for the Agent to follow, at both the local and global levels. Rules allow users to guide the agent to follow behaviors particular to their own use cases and style.

- **Getting started with Rules in the Editor UI:**
  1. Open the Customizations panel via the `"..."` dropdown at the top of the editor's agent panel.
  2. Navigate to the **Rules** panel.
  3. Click **+ Global** to create new Global Rules, or **+ Workspace** to create new Workspace-specific rules.
- **Locations & Limits:**
  - **Global Rules:** live in `~/.gemini/GEMINI.md` (or `~/.gemini/config/rules/`) and are applied across all workspaces.
  - **Directory Rules (`GEMINI.md` / `AGENTS.md`):** placed in any directory; the agent walks up from the current working directory to the repository root and loads them automatically.
  - **Workspace Rules:** live in the `.agents/rules/` (or `.agent/rules/`) folder of your workspace or git root.
  - **Size Limits:** Per-file limit is `24 KB` (`24,000` bytes; legacy UI limit `12,000` chars). Always-on and global rules share a `20,000`-token aggregate budget (over-budget rules are demoted to file path pointers read on demand).
- **Rule Activation Modes:**
  - **Manual:** Activated via `@` mention in the Agent's input box.
  - **Always On:** Always applied to the context.
  - **Model Decision:** Based on a natural language description of the rule, the model decides whether to apply it.
  - **Glob:** Applied automatically when working with files matching the glob pattern (e.g., `*.js`, `src/**/*.ts`).
- **@ Mentions:**
  - Reference other files using `@filename` (or `@[label](path)`) in a Rules file. Relative paths resolve relative to the Rules file; `/path/to/file.md` attempts absolute resolution first, falling back to `workspace/path/to/file.md`.
## Workflows
Workflows enable you to define a series of steps to guide the Agent through a repetitive set of tasks, such as deploying a service or responding to PR comments. These Workflows are saved as markdown files, allowing you to have an easy repeatable way to run key processes. Once saved, Workflows can be invoked in Agent via a slash command with the format `/workflow-name`.

While Rules provide models with guidance by providing persistent, reusable context at the prompt level, Workflows provide a structured sequence of steps or prompts at the trajectory level, guiding the model through a series of interconnected tasks or actions.

- **Creating and running a Workflow:**
  1. Open the Customizations panel via the `"..."` dropdown at the top of the editor's agent panel.
  2. Navigate to the **Workflows** panel.
  3. Click **+ Global** to create a global workflow across all workspaces, or **+ Workspace** to create a workspace-specific workflow.
  4. Invoke it in Agent using `/workflow-name`. Workflows can chain other workflows (e.g., `/workflow-1` can include instructions like `"Call /workflow-2"` and `"Call /workflow-3"`).
- **Workflow Details:**
  - Saved as Markdown files containing a title, a description, and a series of steps.
  - Limited to 12,000 characters each.
  - **Agent-Generated Workflows:** Ask the Agent to generate a Workflow for you after working through a multi-step task in conversation.
## Agent Skills
Skills are an open standard for extending agent capabilities. A skill is a folder containing a `SKILL.md` file with instructions that the agent can follow when working on specific tasks.

- **What are skills?** Reusable packages of knowledge that extend what the agent can do:
  - Instructions for how to approach a specific type of task
  - Best practices and conventions to follow
  - Optional scripts and resources the agent can use
- **Where skills live:**
  - **Workspace-specific:** `<workspace-root>/.agents/skills/<skill-folder>/` (or `.agent/skills/<skill-folder>/`) — for project-specific workflows like team deployment or testing conventions.
  - **Global (all workspaces):** `~/.gemini/config/skills/<skill-folder>/` (or `~/.gemini/antigravity/skills/<skill-folder>/`) — for personal utilities or general-purpose tools everywhere.
- **Creating a skill:**
  1. Create a folder for your skill in one of the skill directories:
     ```text
     .agents/skills/my-skill/
     ├── SKILL.md          # Required: Main instructions with YAML frontmatter
     ├── scripts/          # Optional: Helper scripts and utilities
     ├── examples/         # Optional: Reference implementations
     ├── resources/        # Optional: Templates and assets
     └── references/       # Optional: Detailed documentation loaded on demand
     ```
  2. Add YAML frontmatter at the top of `SKILL.md`:
     ```yaml
     ---
     name: my-skill
     description: Helps with a specific task. Use when you need to do X or Y.
     ---
     ```
- **Frontmatter fields:**
  - `name` (required): Unique identifier for the skill (lowercase, hyphens for spaces). Defaults to the folder name if omitted.
  - `description` (required): Clear description of **what** the skill does and **when** to use it. Write in third person and include trigger keywords (e.g., `"Generates unit tests for Python code using pytest conventions."`).
- **How the agent uses skills (Progressive Disclosure):**
  1. **Discovery:** At conversation start, only skill names and descriptions are injected into context.
  2. **Activation:** When a skill matches the user's task, the agent reads the full `SKILL.md` content (and any linked `references/` files on demand).
  3. **Execution:** The agent follows the skill's instructions.
- **Best practices:**
  - **Keep skills focused:** Each skill should do one thing well.
  - **Write clear descriptions:** Make it specific about what the skill does and when it's useful.
  - **Use scripts as black boxes:** Encourage the agent to run helper scripts with `--help` first rather than reading their entire source code.
  - **Include decision trees:** Help the agent choose the right path based on the situation.
- **Example: A code review skill**
  ```yaml
  ---
  name: code-review
  description: Reviews code changes for bugs, style issues, and best practices. Use when reviewing PRs or checking code quality.
  ---
  ```
## Hooks
Lifecycle hooks (`hooks.json` in `.agents/hooks.json` or `~/.gemini/config/hooks.json`) execute external shell commands at specific points in the agent loop. Hook scripts receive a JSON payload on `stdin` and return a JSON response on `stdout`.

- **Supported Lifecycle Events:**
  - `SessionStart`: Runs once when a new conversation session starts (can inject initial system messages via `injectSteps`).
  - `PreToolUse`: Runs before a tool executes (matched by regex `matcher`, e.g., `"run_command"`); returns `"decision": "allow" | "deny" | "ask" | "force_ask"` and optional argument `overwrite`.
  - `PostToolUse`: Runs after a tool completes (e.g., auto-linting or formatting).
  - `PreInvocation` / `PostInvocation`: Runs before/after model calls to inject steps or force continuation/termination.
  - `Stop`: Runs when the execution loop terminates; can return `{"decision": "continue", "reason": "..."}` to keep the agent working if goals aren't met.
  ```json
  {
    "lint-checker": {
      "PostToolUse": [
        {
          "matcher": "run_command",
          "hooks": [
            {
              "type": "command",
              "command": "./scripts/lint.sh",
              "timeout": 10
            }
          ]
        }
      ]
    }
  }
  ```
## MCP Integration
Antigravity supports the Model Context Protocol (MCP), an open standard that connects the agent to local tools, databases, and remote services.

- **Core Features:**
  1. **Context Resources:** Read data directly from connected MCP servers (e.g., inspect live Neon/Supabase schemas when writing SQL, or pull build logs from Netlify/Heroku).
  2. **Custom Tools:** Execute actions defined by connected servers (e.g., `"Create a Linear issue for this TODO"`, `"Search Notion or GitHub for authentication patterns"`).
- **Connecting via the Editor MCP Store:**
  1. Open the **MCP Store** panel via the `"..."` dropdown at the top of the editor's agent panel.
  2. Browse & click **Install** on any supported server and follow on-screen authentication prompts.
- **Connecting Custom MCP Servers (`mcp_config.json`):**
  - Global config lives at `~/.gemini/config/mcp_config.json` (or via **Manage MCP Servers → View raw config** in the editor, or `agy mcp add` in the CLI).
  - Supports **Stdio** (local command-line executables) and **Streamable HTTP** (`url` pointing to `/mcp` remote endpoint):
  ```json
  {
    "mcpServers": {
      "sqlite-helper": {
        "command": "sqlite-mcp-server",
        "args": ["/path/to/database.db"],
        "env": {
          "DB_READONLY": "true"
        }
      },
      "remote-service": {
        "url": "https://mcp.example.com/mcp"
      }
    }
  }
  ```
***

# SKILLs
## NotebookLM MCP
```md
---
name: notebooklm-research
description: Performs deep research and knowledge synthesis using NotebookLM MCP. Use when the user mentions research, learning materials, creating study guides, generating audio/video overviews, or needs to gather comprehensive information on a topic from web or Google Drive sources.
---

# NotebookLM Research & Knowledge Synthesis

## When to use this skill

- User asks to "research [topic]"
- User wants to create learning materials (study guides, flashcards, quizzes)
- User needs to synthesize information from multiple sources
- User mentions "deep dive" or "comprehensive analysis"
- User wants audio/video overviews or podcasts on a topic
- User needs to organize knowledge into notebooks
- User wants to search Google Drive for relevant documents

## Core Workflow Patterns

### Pattern 1: Deep Research (Recommended)
1. Start research: `research_start` (mode: deep, ~5min, ~40 sources)
2. Poll status: `research_status` (max_wait: 300, poll_interval: 30)
3. Import sources: `research_import` (imports all discovered sources)
4. Generate artifacts: audio_overview, slide_deck, report, etc.

### Pattern 2: Fast Research
1. Start research: `research_start` (mode: fast, ~30s, ~10 sources)
2. Poll status: `research_status` (max_wait: 60, poll_interval: 10)
3. Import sources: `research_import`

### Pattern 3: Manual Source Addition
1. Create notebook: `notebook_create`
2. Add sources: `notebook_add_url`, `notebook add text`, `notebook add drive`
3. Query or generate artifacts
```
***

<!-- related -->
# Related
- [langgraph](05-08_langgraph.md)
- [agno](05-10_agno.md)
- [learn claude code](07-27_learn_claude_code.md)
- [OpenClaw](01-29_openclaw.md)
- [claude](02-28_claude.md)
***
<!-- /related -->
