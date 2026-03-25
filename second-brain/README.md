# Second Brain - Knowledge Base

This directory contains structured knowledge about projects, concepts, and important discussions.

## Structure

- **journal/** — Daily summaries of important conversations and decisions
- **concepts/** — Deep dives on ideas, frameworks, and frameworks
- **documents/** — Working documents, strategies, and plans

## Purpose

While `memory/YYYY-MM-DD.md` contains raw session logs and quick captures, this directory is for:
- Structured summaries that are easier to search and query
- Deep-dive content that deserves more than a quick note
- Documents that multiple people (or future you) might reference
- Knowledge graph nodes that show up in Obsidian's Graph View

## Usage

When something important happens:
1. Capture the raw note in `memory/YYYY-MM-DD.md` (immediate, unstructured)
2. Write a structured summary in `second-brain/journal/YYYY-MM-DD.md` (if discussion was significant)
3. Create a concept file in `second-brain/concepts/` if it introduces a new framework or idea
4. Link files using `[[wiki-links]]` to build the knowledge graph

## Dataview Example

Query recent journal entries:
```dataview
TABLE file.mtime AS "Last Updated"
FROM "second-brain/journal"
SORT file.mtime DESC
```

Query concepts:
```dataview
LIST FROM "second-brain/concepts"
SORT file.name DESC
```