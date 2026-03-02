# ___Gemini___
***

# Get Started
## Gemini CLI
  - **Gemini CLI Get Started** [Gemini CLI Docs](https://geminicli.com/docs/get-started/)
    ```sh
    npm install -g @google/gemini-cli

    gemini -m gemini-2.5-flash-lite
    ```
  - **Installing NotbookLM MCP** Copy and paste the below 
    ```sh
    cat ~/.gemini/antigravity/mcp_config.json
    # {
    #   "mcpServers": {
    #     "notebooklm": {
    #       "command": "npx",
    #       "args": [
    #         "notebooklm-mcp@latest"
    #       ]
    #     }
    # }
    ```
  - **AntiGravity Skills Creator** `antigravity-skill-creator.md`
    ```md
    #Antigravity Skill Creator System Instructions
    You are an expert developer specializing in creating "Skills" for the Antigravity agent environment. Your goal is to generate high-quality, predictable, and efficient `.agent/skills/` directories based on user requirements.

    1. Core Structural Requirements:
    Every skill you generate must follow this folder hierarchy:
    - `<skill-name>/`
        - `SKILL.md` (Required: Main logic and instructions)
        - `scripts/` (Optional: Helper scripts)
        - `examples/` (Optional: Reference implementations)
        - `resources/` (Optional: Templates or assets)

    2. YAML Frontmatter Standards:
    The `SKILL.md` must start with YAML frontmatter following these strict rules:
    - **name**: Gerund form (e.g., `testing-code`, `managing-databases`). Max 64 chars. Lowercase, numbers, and hyphens only. No "claude" or "anthropic" in the name.
    - **description**: Written in **third person**. Must include specific triggers/keywords. Max 1024 chars. (e.g., "Extracts text from PDFs. Use when the user mentions document processing or PDF files.")

    3. Writing Principles (The "Claude Way"):
    When writing the body of `SKILL.md`, adhere to these best practices:
    - **Conciseness**: Assume the agent is smart. Do not explain what a PDF or a Git repo is. Focus only on the unique logic of the skill.
    - **Progressive Disclosure**: Keep `SKILL.md` under 500 lines. If more detail is needed, link to secondary files (e.g., `[See ADVANCED.md](ADVANCED.md)`) only one level deep.
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
      **Path:** `.agent/skills/[skill-name]/`
    - ### [SKILL.md]
      (Markdown content with YAML frontmatter)
    ```
***

# Anti-gravity
## Documentation
  - [Google Antigravity Documentation](https://antigravity.google/docs/get-started)
## Rules
  Rules are manually defined constraints for the Agent to follow, at both the local and global levels. Rules allow users to guide the agent to follow behaviors particular to their own use cases and style.

  To get started with Rules:
  1. Open the Customizations panel via the "..." dropdown at the top of the editor's agent panel.
  2. Navigate to the Rules panel.
  3. Click + Global to create new Global Rules, or + Workspace to create new Workspace-specific rules.

  A Rule itself is simply a Markdown file, where you can input the constraints to guide the Agent to your tasks, stack, and style.
  - Rules files are limited to 12,000 characters each.
  - **Global Rules:** live in `~/.gemini/GEMINI.md` and are applied across all workspaces.
  - **Workspace Rules:** live in the `.agent/rules` folder of your workspace or git root.

  At the rule level you can define how a rule should be activated:
  - **Manual:** The rule is manually activated via at mention in Agent’s input box.
  - **Always On:** The rule is always applied.
  - **Model Decision:** Based on a natural language description of the rule, the model decides whether to apply the rule.
  - **Glob:** Based on the glob pattern you define (e.g., `.js`, `src/**/.ts`), the rule will be applied to all files that match the pattern.

  **@ Mentions:**
  You can reference other files using `@filename` in a Rules file. If filename is a relative path, it will be interpreted relative to the location of the Rules file. If filename is an absolute path, it will be resolved as a true absolute path, otherwise it will be resolved relative to the repository. For example, `@/path/to/file.md` will first attempt to be resolved to `/path/to/file.md`, and if that file does not exist, it will be resolved to `workspace/path/to/file.md`.
## Workflows
  Workflows enable you to define a series of steps to guide the Agent through a repetitive set of tasks, such as deploying a service or responding to PR comments. These Workflows are saved as markdown files, allowing you to have an easy repeatable way to run key processes. Once saved, Workflows can be invoked in Agent via a slash command with the format `/workflow-name`.

  While Rules provide models with guidance by providing persistent, reusable context at the prompt level, Workflows provide a structured sequence of steps or prompts at the trajectory level, guiding the model through a series of interconnected tasks or actions.

  To create a workflow:
  1. Open the Customizations panel via the "..." dropdown at the top of the editor's agent panel.
  2. Navigate to the Workflows panel.
  3. Click the + Global button to create a new global workflow that can be accessed across all your workspaces, or click the + Workspace button to create a workflow specific to your current workspace.
  4. To execute a workflow, simply invoke it in Agent using the `/workflow-name` command. You can call other Workflows from within a workflow! For example, `/workflow-1` can include instructions like “Call /workflow-2” and “Call /workflow-3”. Upon invocation, Agent sequentially processes each step defined in the workflow, performing actions or generating responses as specified.

  - Workflows are saved as markdown files and contain a title, a description and a series of steps with specific instructions for Agent to follow.
  - Workflow files are limited to 12,000 characters each.
  - **Agent-Generated Workflows:** You can also ask Agent to generate Workflows for you! This works particularly well after manually working with Agent through a series of steps since it can use the conversation history to create the Workflow.
## Agent Skills
  Skills are an open standard for extending agent capabilities. A skill is a folder containing a `SKILL.md` file with instructions that the agent can follow when working on specific tasks.

  **What are skills?**
  Skills are reusable packages of knowledge that extend what the agent can do. Each skill contains:
  - Instructions for how to approach a specific type of task
  - Best practices and conventions to follow
  - Optional scripts and resources the agent can use

  When you start a conversation, the agent sees a list of available skills with their names and descriptions. If a skill looks relevant to your task, the agent reads the full instructions and follows them.

  **Where skills live**
  Antigravity supports two types of skills:
  - `<workspace-root>/.agent/skills/<skill-folder>/`: Workspace-specific
  - `~/.gemini/antigravity/skills/<skill-folder>/`: Global (all workspaces)

  - Workspace skills are great for project-specific workflows, like your team's deployment process or testing conventions.
  - Global skills work across all your projects. Use these for personal utilities or general-purpose tools you want everywhere.

  **Creating a skill**
  To create a skill:
  1. Create a folder for your skill in one of the skill directories
  2. Add a `SKILL.md` file inside that folder
     - `.agent/skills/└─── my-skill/└─── SKILL.md`
  3. Every skill needs a `SKILL.md` file with YAML frontmatter at the top:
     ```yaml
     ---
     name: my-skill
     description: Helps with a specific task. Use when you need to do X or Y.
     ---
     ```

  **Frontmatter fields**
  - `name`: A unique identifier for the skill (lowercase, hyphens for spaces). Defaults to the folder name if not provided.
  - `description`: A clear description of what the skill does and when to use it. This is what the agent sees when deciding whether to apply the skill.

  **Tip:** Write your description in third person and include keywords that help the agent recognize when the skill is relevant. For example: "Generates unit tests for Python code using pytest conventions."

  **Skill folder structure**
  While `SKILL.md` is the only required file, you can include additional resources:
  - `SKILL.md`: Main instructions (required)
  - `scripts/`: Helper scripts (optional)
  - `examples/`: Reference implementations (optional)
  - `resources/`: Templates and other assets (optional)

  **How the agent uses skills**
  Skills follow a progressive disclosure pattern:
  1. **Discovery:** When a conversation starts, the agent sees a list of available skills with their names and descriptions
  2. **Activation:** If a skill looks relevant to your task, the agent reads the full `SKILL.md` content
  3. **Execution:** The agent follows the skill's instructions while working on your task

  **Best practices**
  - **Keep skills focused:** Each skill should do one thing well. Instead of a "do everything" skill, create separate skills for distinct tasks.
  - **Write clear descriptions:** The description is how the agent decides whether to use your skill. Make it specific about what the skill does and when it's useful.
  - **Use scripts as black boxes:** If your skill includes scripts, encourage the agent to run them with `--help` first rather than reading the entire source code. This keeps the agent's context focused on the task.
  - **Include decision trees:** For complex skills, add a section that helps the agent choose the right approach based on the situation.

  **Example: A code review skill**
  ```yaml
  ---
  name: code-review
  description: Reviews code changes for bugs, style issues, and best practices. Use when reviewing PRs or checking code quality.
  ---
  ```
  (Followed by review checklist and feedback guidelines)
## MCP Integration
  Antigravity supports the Model Context Protocol (MCP), a standard that allows the editor to securely connect to your local tools, databases, and external services. This integration provides the AI with real-time context beyond just the files open in your editor.

  What is MCP?
  MCP acts as a bridge between Antigravity and your broader development environment. Instead of manually pasting context (like database schemas or logs) into the editor, MCP allows Antigravity to fetch this information directly when needed.

  Core Features
  1. Context Resources
  The AI can read data from connected MCP servers to inform its suggestions.

  Example: When writing a SQL query, Antigravity can inspect your live Neon or Supabase schema to suggest correct table and column names.

  Example: When debugging, the editor can pull in recent build logs from Netlify or Heroku.

  2. Custom Tools
  MCP enables Antigravity to execute specific, safe actions defined by your connected servers.

  Example: "Create a Linear issue for this TODO."

  Example: "Search Notion or GitHub for authentication patterns."

  How to Connect
  Connections are managed directly through the built-in MCP Store.

  Access the Store: Open the MCP Store panel within the "..." dropdown at the top of the editor's side panel.
  Browse & Install: Select any of the supported servers from the list and click Install.
  Authenticate: Follow the on-screen prompts to securely link your accounts (where applicable).
  Once installed, resources and tools from the server are automatically available to the editor.

  Connecting Custom MCP Servers
  To connect to a custom MCP server:

  Open the MCP store via the "..." dropdown at the top of the editor's agent panel.
  Click on "Manage MCP Servers"
  Click on "View raw config"
  Modify the mcp_config.json with your custom MCP server configuration.
***

# SKILLs
## NotbookLM MCP
```
**name:** notebooklm-research
**description:** Performs deep research and knowledge synthesis using NotebookLM MCP. Use when the user mentions research, learning materials, creating study guides, generating audio/video overviews, or needs to gather comprehensive information on a topic from web or Google Drive sources.

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
