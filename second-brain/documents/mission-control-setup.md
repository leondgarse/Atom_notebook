# Mission Control Setup Guide

## Overview

Mission Control is an AI agent orchestration dashboard for managing AI agent fleets, dispatching tasks, tracking costs, and coordinating multi-agent workflows.

## Installation

### Location
- **Repository:** `~/workspace/mission-control`
- **Port:** 3001 (running in standalone mode)
- **URL:** http://localhost:3001

### Credentials (from .env auto-generation)
- **Admin User:** admin
- **Admin Password:** VrfC8SMT7Aiq0Ye7n3aPkv3p
- **API Key:** 774206d331989d2220de8bf61d05f3cb

### Setup Process
1. Visit http://localhost:3001/setup
2. Create admin account (can use the credentials above)
3. Configure OpenClaw integration

## OpenClaw Integration

### Configuration
Mission Control is configured to connect to OpenClaw:
- **OPENCLAW_HOME:** `/home/leondgarse/.openclaw`
- **Mode:** Standalone (no gateway required)
- **Gateway Port:** 18789 (configured but optional)
- **Coordinator Agent:** `coordinator`

### Standalone Mode
Set `NEXT_PUBLIC_GATEWAY_OPTIONAL=true` in `.env` for standalone operation:
- Core CRUD features work (agents, tasks, skills, logs)
- Live gateway events disabled
- Task board, projects, and cost tracking functional

### If Gateway Connection Needed
1. Install OpenClaw gateway
2. Update `.env`:
   ```bash
   OPENCLAW_GATEWAY_HOST=127.0.0.1
   OPENCLAW_GATEWAY_PORT=18789
   OPENCLAW_GATEWAY_TOKEN=your-token-here
   ```

## Usage

### Web Interface
- **Dashboard:** http://localhost:3001
- **Setup:** http://localhost:3001/setup
- **API Docs:** http://localhost:3001/api/docs

### Key Features
1. **Agent Management:** Register agents, view status, manage roles
2. **Task Board:** Kanban-style task tracking (inbox → assigned → in progress → review → quality review → done)
3. **Cost Tracking:** Token usage dashboard with per-model breakdowns
4. **Memory Browser:** Explore agent memory and knowledge graph
5. **Skills Hub:** Browse, install, and manage agent skills
6. **Security Monitoring:** Trust scoring, secret detection, MCP auditing

### API Integration

**Register an Agent:**
```bash
curl -X POST "http://localhost:3001/api/agents/register" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "scout", "role": "researcher"}'
```

**Create a Task:**
```bash
curl -X POST "http://localhost:3001/api/tasks" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"title": "Research competitors", "assigned_to": "scout", "priority": "medium"}'
```

**Poll Task Queue:**
```bash
curl "http://localhost:3001/api/tasks/queue?agent=scout" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Development

### Start Server
```bash
cd ~/workspace/mission-control
npx pnpm run dev
```

### Build for Production
```bash
npx pnpm build
npx pnpm start
```

### Run Tests
```bash
npx pnpm test              # Unit tests (282)
npx pnpm test:e2e          # E2E tests (295)
npx pnpm quality:gate      # All checks
```

## Documentation

- **Quickstart:** https://github.com/builderz-labs/mission-control/blob/main/docs/quickstart.md
- **Agent Setup:** https://github.com/builderz-labs/mission-control/blob/main/docs/agent-setup.md
- **Orchestration:** https://github.com/builderz-labs/mission-control/blob/main/docs/orchestration.md
- **Deployment:** https://github.com/builderz-labs/mission-control/blob/main/docs/deployment.md
- **Security Hardening:** https://github.com/builderz-labs/mission-control/blob/main/docs/SECURITY-HARDENING.md

## Notes

- **Database:** SQLite (WAL mode) in `.data/mission-control.db`
- **State Management:** Zustand 5
- **Real-time:** WebSocket + Server-Sent Events
- **Auth:** scrypt hashing, session tokens, RBAC (viewer/operator/admin)
- **Node Version:** v22+ required