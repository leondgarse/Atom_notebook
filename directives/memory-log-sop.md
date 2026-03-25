# Memory Log Creation SOP

## Overview
Automated procedure to capture daily session notes and maintain memory structure.

## Prerequisites
- Session is ending
- Have time to review the conversation and decisions made
- Ready to write to files

## Steps

1. **Review Conversation**
   - Review main session messages and any Discord group chats
   - Identify key decisions made
   - Note important context or information shared
   - Recognize mistakes or lessons learned

2. **Update Daily Log**
   - Read `memory/YYYY-MM-DD.md` (create if doesn't exist)
   - Append today's session notes in a structured format:
     - Session context (what were we working on?)
     - Decisions made (what did we agree on?)
     - Things to remember (key context, context about Leon, etc.)
     - Lessons learned (mistakes, improvements)

3. **Update Journal Summary** (if discussion was significant)
   - Read `second-brain/journal/YYYY-MM-DD.md`
   - Add structured summary:
     - Session summary (checklist of what was done)
     - Key discussions (important topics)
     - Decisions made
     - Open questions (if any)

4. **Review and Refine**
   - Check if any updates needed to `MEMORY.md`
   - Remove outdated info from long-term memory if no longer relevant
   - Add new insights to curated long-term memory if valuable

5. **Commit Changes**
   - Use git to commit memory file updates
   - Add commit message describing what was captured

## Outputs
- Updated `memory/YYYY-MM-DD.md` with session notes
- Updated `second-brain/journal/YYYY-MM-DD.md` with summary
- Updated `MEMORY.md` with distilled insights (periodically)

## When to Use
- At the end of a significant session
- When multiple decisions or important context were shared
- When you want to ensure nothing gets forgotten
- When transitioning to a new project or topic