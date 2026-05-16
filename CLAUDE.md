# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hermes Agent (v0.10.0) is "the self-improving AI agent" built by Nous Research. It's a Python-based LLM agent with a terminal interface, multi-platform messaging gateway, skill system, memory system, and research tooling (batch trajectory generation, RL environments).

**Tech stack:** Python 3.11+, `uv` package manager, pytest (with xdist), OpenAI/Anthropic SDKs, prompt_toolkit + Rich for CLI, Ink (React) for TUI, Docusaurus for docs.

## Development Setup

```bash
# Quick setup (recommended)
./setup-hermes.sh

# Manual dev setup
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

**Always activate the venv before running Python:**
```bash
source venv/bin/activate  # or source .venv/bin/activate
```

## Key Commands

### Running

```bash
hermes                    # Interactive CLI
hermes --tui              # React TUI (Ink-based)
hermes setup              # Full setup wizard
hermes model              # Switch model/provider
hermes tools              # Configure tools
hermes gateway start      # Start messaging gateway
hermes doctor             # Diagnose issues
hermes update             # Update to latest version
```

### Testing

**ALWAYS use the wrapper script** — do not call `pytest` directly. It enforces hermetic environment parity with CI:

```bash
scripts/run_tests.sh                                    # Full suite (~3000 tests)
scripts/run_tests.sh tests/agent/                       # One directory
scripts/run_tests.sh tests/tools/test_web_tools.py      # One file
scripts/run_tests.sh tests/agent/test_foo.py::test_bar  # One test
scripts/run_tests.sh -v --tb=long                       # With verbose output
```

The wrapper enforces: 4 xdist workers (matching CI), TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0, all credential env vars unset. Direct `pytest` with API keys set or `-n auto` on a 20-core machine will diverge from CI.

Integration and e2e tests are excluded by default (`--ignore=tests/integration --ignore=tests/e2e`).

### TUI (React/Ink) Development

```bash
cd ui-tui
npm install
npm run dev        # Watch mode
npm run build      # Full build
npm run type-check # tsc --noEmit
npm run lint       # eslint
npm test           # vitest
```

### Docker

```bash
docker build -t hermes-agent .
```

## Architecture Overview

See `AGENTS.md` for the full development guide. Key points:

### File Dependency Chain

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

### Main Entry Points

| Entry Point | File | Description |
|---|---|---|
| `hermes` CLI | `hermes_cli/main.py` | Main CLI entry — all `hermes` subcommands |
| Agent runner | `run_agent.py` | `AIAgent` class with core conversation loop |
| Interactive CLI | `cli.py` | `HermesCLI` class — TUI with prompt_toolkit |
| Gateway | `gateway/run.py` | Main gateway loop for messaging platforms |
| ACP Server | `acp_adapter/entry.py` | IDE integration (VS Code, Zed, JetBrains) |
| TUI gateway | `tui_gateway/entry.py` | stdio JSON-RPC for React TUI |
| MCP server | `mcp_serve.py` | MCP server entry |

### Core Components

- **AIAgent** (`run_agent.py`) — Core conversation loop. Synchronous agent loop with LLM API calls, tool execution, message history, context compression, prompt caching.
- **HermesCLI** (`cli.py`) — Interactive terminal orchestrator with prompt_toolkit, Rich formatting, skin engine, slash commands.
- **Tool System** (`tools/registry.py` + `tools/*.py`) — Self-registering tools. 40+ tools covering web, terminal, files, browser, MCP, delegation, skills, TTS, vision, cron, etc.
- **Terminal Backends** (`tools/environments/`) — Six execution environments: local, Docker, SSH, Modal, Daytona, Singularity.
- **Gateway** (`gateway/` + `gateway/platforms/`) — Messaging platforms: Telegram, Discord, Slack, WhatsApp, Signal, Email, Matrix, Feishu, WeCom, Weixin, DingTalk, SMS, Home Assistant, Mattermost, BlueBubbles.
- **Memory System** (`agent/memory_manager.py`, `hermes_state.py`) — SQLite session store with FTS5 full-text search, user profiles, Honcho integration.
- **Skills System** (`skills/`, `tools/skills_tool.py`) — Procedural memory with autonomous skill creation and self-improvement.
- **Cron Scheduler** (`cron/`) — Built-in cron with delivery to any platform.
- **TUI** (`ui-tui/` + `tui_gateway/`) — React Ink terminal UI with Python JSON-RPC backend.

### Configuration

User config lives at `~/.hermes/config.yaml` (settings) and `~/.hermes/.env` (API keys). Profiles support allows multiple isolated instances via `HERMES_HOME` environment variable.

## Adding New Tools

Requires changes in **2 files**:

1. Create `tools/your_tool.py` with `registry.register()` call
2. Add to `_HERMES_CORE_TOOLS` in `toolsets.py`

Auto-discovery: any `tools/*.py` with a top-level `registry.register()` call is imported automatically. All handlers MUST return a JSON string.

**State files**: Use `get_hermes_home()` from `hermes_constants` — never `Path.home() / ".hermes"`. This ensures profile compatibility.

## Adding Slash Commands

1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `HermesCLI.process_command()` in `cli.py`
3. If available in gateway, add handler in `gateway/run.py`

## Critical Policies

- **Prompt caching must not break**: Do not alter past context, change toolsets, reload memories, or rebuild system prompts mid-conversation.
- **NEVER hardcode `~/.hermes`**: Use `get_hermes_home()` for code paths, `display_hermes_home()` for user-facing messages. Hardcoding breaks profiles.
- **NEVER use `simple_term_menu`**: Rendering bugs in tmux/iTerm2. Use `curses` instead.
- **NEVER use `\033[K` in spinner code**: Leaks as literal `?[K` under prompt_toolkit. Use space-padding.
- **NEVER hardcode cross-tool references in schema descriptions**: Tools from other toolsets may be unavailable.
- **Tests must not write to `~/.hermes/`**: The `_isolate_hermes_home` fixture redirects to temp dir.

## Testing Guidelines

- Write behavioral tests, not change-detector tests (don't assert specific model names, config version literals, or enumeration counts)
- Assert invariants and relationships instead of hardcoded data snapshots
- Profile tests should mock `Path.home()` so profile paths resolve within temp dir
- Always run full suite before pushing

## Important Files Reference

| File | Purpose |
|---|---|
| `AGENTS.md` | Comprehensive development guide |
| `pyproject.toml` | Dependencies, entry points, pytest config |
| `.env.example` | Environment variable template |
| `hermes_cli/config.py` | DEFAULT_CONFIG, OPTIONAL_ENV_VARS, migration |
| `hermes_cli/commands.py` | Slash command registry |
| `tools/registry.py` | Central tool registry |
| `toolsets.py` | Toolset definitions |
| `hermes_constants.py` | Global constants, path helpers |
| `hermes_state.py` | SessionDB (SQLite + FTS5) |
| `tests/conftest.py` | Test fixtures, hermetic environment setup |
