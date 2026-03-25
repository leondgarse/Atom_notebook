# Workflows and Standard Operating Procedures

This directory contains repeatable workflows that the AI should follow consistently.

## Structure

- Each SOP is a `.md` file
- Title describes the workflow
- Content is step-by-step instructions

## Example SOP Template

```markdown
# [Workflow Name]

## Overview
Brief description of what this workflow does.

## Prerequisites
- [ ] Condition A
- [ ] Condition B

## Steps
1. Step one description
2. Step two description
3. Edge case handling

## Outputs
What the AI produces when this workflow is complete.

## When to Use
Context where this workflow applies.
```

## Example: Memory Log Creation

When a session ends, create a daily log:
1. Read `memory/YYYY-MM-DD.md` from today
2. Append session notes, decisions made, things remembered
3. Update `second-brain/journal/YYYY-MM-DD.md` with structured summary
4. Mark task complete in heartbeat check