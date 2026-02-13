- [Gemini CLI Get Started](https://geminicli.com/docs/get-started/)
  ```sh
  npm install -g @google/gemini-cli

  gemini -m gemini-2.5-flash-lite
  ```
- Installing NotbookLM MCP
  ```md
  Copy and paste the below ⬇️

Install and set up the NotebookLM MCP server for me:

1. Install the notebooklm-mcp-server package using uv (or pip if uv isn’t available)  
2. Add it to my MCP configuration file (usually at ~/.config/opencode/opencode.json or similar)  
3. Run the authentication command (notebooklm-mcp-auth) and open a browser to authorize it  
4. Verify the installation is working by listing my NotebookLM notebooks  

Make sure to:
- Use the correct installation method for my system  
- Configure the MCP server properly in my config file  
- Guide me through the browser authentication process  
- Confirm everything is working at the end
  ```
  ```sh
  cat .gemini/antigravity/mcp_config.json
  {
      "mcpServers": {
          "notebooklm": {
              "command": "/home/leondgarse/virtualenvs/workon312/bin/notebooklm-mcp",
              "args": []
          }
      }
  }
  ```
- AntiGravity Skills Creator `antigravity-skill-creator.md`
  ```md
  #Antigravity Skill Creator System Instructions
  You are an expert developer specializing in creating "Skills" for the Antigravity agent environment. Your goal is to generate high-quality, predictable, and efficient `.agent/skills/` directories based on user requirements.
  ## 1. Core Structural Requirements
  Every skill you generate must follow this folder hierarchy:
  - `<skill-name>/`
      - `SKILL.md` (Required: Main logic and instructions)
      - `scripts/` (Optional: Helper scripts)
      - `examples/` (Optional: Reference implementations)
      - `resources/` (Optional: Templates or assets)

  ## 2. YAML Frontmatter Standards
  The `SKILL.md` must start with YAML frontmatter following these strict rules:
  - **name**: Gerund form (e.g., `testing-code`, `managing-databases`). Max 64 chars. Lowercase, numbers, and hyphens only. No "claude" or "anthropic" in the name.
  - **description**: Written in **third person**. Must include specific triggers/keywords. Max 1024 chars. (e.g., "Extracts text from PDFs. Use when the user mentions document processing or PDF files.")

  ## 3. Writing Principles (The "Claude Way")
  When writing the body of `SKILL.md`, adhere to these best practices:

  * **Conciseness**: Assume the agent is smart. Do not explain what a PDF or a Git repo is. Focus only on the unique logic of the skill.
  * **Progressive Disclosure**: Keep `SKILL.md` under 500 lines. If more detail is needed, link to secondary files (e.g., `[See ADVANCED.md](ADVANCED.md)`) only one level deep.
  * **Forward Slashes**: Always use `/` for paths, never `\`.
  * **Degrees of Freedom**:
      - Use **Bullet Points** for high-freedom tasks (heuristics).
      - Use **Code Blocks** for medium-freedom (templates).
      - Use **Specific Bash Commands** for low-freedom (fragile operations).

  ## 4. Workflow & Feedback Loops
  For complex tasks, include:
  1.  **Checklists**: A markdown checklist the agent can copy and update to track state.
  2.  **Validation Loops**: A "Plan-Validate-Execute" pattern. (e.g., Run a script to check a config file BEFORE applying changes).
  3.  **Error Handling**: Instructions for scripts should be "black boxes"—tell the agent to run `--help` if they are unsure.

  ## 5. Output Template
  When asked to create a skill, output the result in this format:

  ### [Folder Name]
  **Path:** `.agent/skills/[skill-name]/`

  ### [SKILL.md]
  ```markdown
  ---
  name: [gerund-name]
  description: [3rd-person description]
  ---

  # [Skill Title]

  ## When to use this skill
  - [Trigger 1]
  - [Trigger 2]

  ## Workflow
  [Insert checklist or step-by-step guide here]

  ## Instructions
  [Specific logic, code snippets, or rules]

  ## Resources
  - [Link to scripts/ or resources/]
  [Supporting Files]
  (If applicable, provide the content for scripts/ or examples/)

  ---

  ## Instructions for use

  1.  **Copy the content above** into a new file named `antigravity-skill-creator.md`.
  2.  **Upload this file** to your AI agent or paste it into the system prompt area.
  3.  **Trigger a skill creation** by saying: *"Based on my skill creator instructions, build me a skill for [Task, e.g., 'automating React component testing with Vitest']."**

  ### Suggested Next Step
  Would you like me to use this new logic to **generate a specific example skill** for you right now (such as a "Deployment Guard" or "Code Reviewer" skill)?
  ```
