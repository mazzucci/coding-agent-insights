#!/bin/bash
set -euo pipefail

# coding-agent-insights installer
# Auto-detects Cursor and/or Claude Code, installs hooks, and sets up Phoenix.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
err()   { echo -e "${RED}✗${NC} $*"; }
header(){ echo -e "\n${BOLD}$*${NC}"; }

# Track which agents were installed
INSTALLED_CURSOR=false
INSTALLED_CLAUDE_CODE=false

# ── Prerequisites ─────────────────────────────────────────────────────────────

header "Checking prerequisites…"

if ! command -v uv &>/dev/null; then
    warn "uv is required but not found."
    read -rp "  Install uv now? [Y/n]: " install_uv
    install_uv="${install_uv:-Y}"
    if [[ "$install_uv" =~ ^[Yy] ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv &>/dev/null; then
            err "uv installation succeeded but is not on PATH."
            echo "  Add ~/.local/bin to your PATH, then re-run this installer."
            exit 1
        fi
        info "uv installed — $(uv --version 2>/dev/null)"
    else
        err "uv is required. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
else
    info "uv $(uv --version 2>/dev/null || echo '(found)')"
fi

HAS_DOCKER=false
if command -v docker &>/dev/null; then
    HAS_DOCKER=true
    info "docker found"
else
    warn "docker not found (only needed for local Phoenix)"
fi

# ── Detect coding agents ─────────────────────────────────────────────────────

header "Detecting coding agents…"

HAS_CURSOR=false
HAS_CLAUDE_CODE=false

if [ -d "$HOME/.cursor" ]; then
    HAS_CURSOR=true
    info "Cursor detected (~/.cursor/)"
fi

if [ -d "$HOME/.claude" ]; then
    HAS_CLAUDE_CODE=true
    info "Claude Code detected (~/.claude/)"
fi

if [ "$HAS_CURSOR" = false ] && [ "$HAS_CLAUDE_CODE" = false ]; then
    warn "Neither Cursor nor Claude Code detected."
    echo ""
    echo "  Which agent(s) would you like to install hooks for?"
    echo ""
    echo "  1) Cursor"
    echo "  2) Claude Code"
    echo "  3) Both"
    echo ""
    read -rp "  Choose [1/2/3]: " agent_choice
    case "$agent_choice" in
        1) HAS_CURSOR=true ;;
        2) HAS_CLAUDE_CODE=true ;;
        3) HAS_CURSOR=true; HAS_CLAUDE_CODE=true ;;
        *) err "Invalid choice."; exit 1 ;;
    esac
fi

# If both are detected, ask which to install
if [ "$HAS_CURSOR" = true ] && [ "$HAS_CLAUDE_CODE" = true ]; then
    echo ""
    echo "  Both Cursor and Claude Code detected. Install hooks for:"
    echo ""
    echo "  1) Both     ${DIM}(recommended)${NC}"
    echo "  2) Cursor only"
    echo "  3) Claude Code only"
    echo ""
    read -rp "  Choose [1/2/3] (default: 1): " both_choice
    both_choice="${both_choice:-1}"
    case "$both_choice" in
        1) ;;  # keep both true
        2) HAS_CLAUDE_CODE=false ;;
        3) HAS_CURSOR=false ;;
        *) err "Invalid choice."; exit 1 ;;
    esac
fi

# ── Install Cursor hooks ─────────────────────────────────────────────────────

if [ "$HAS_CURSOR" = true ]; then
    header "Installing Cursor hooks…"

    CURSOR_HOOKS_DIR="$HOME/.cursor/hooks"
    CURSOR_HOOKS_JSON="$HOME/.cursor/hooks.json"
    CURSOR_ENV_FILE="$CURSOR_HOOKS_DIR/.coding-agent-insights.env"

    mkdir -p "$CURSOR_HOOKS_DIR"
    cp "$SCRIPT_DIR/hooks/trace-hook.sh" "$CURSOR_HOOKS_DIR/trace-hook.sh"
    chmod +x "$CURSOR_HOOKS_DIR/trace-hook.sh"
    cp "$SCRIPT_DIR/hooks/flush.py" "$CURSOR_HOOKS_DIR/flush.py"
    cp "$SCRIPT_DIR/hooks/core.py" "$CURSOR_HOOKS_DIR/core.py"
    mkdir -p "$CURSOR_HOOKS_DIR/adapters"
    cp "$SCRIPT_DIR/hooks/adapters/__init__.py" "$CURSOR_HOOKS_DIR/adapters/__init__.py"
    cp "$SCRIPT_DIR/hooks/adapters/cursor.py" "$CURSOR_HOOKS_DIR/adapters/cursor.py"
    cp "$SCRIPT_DIR/hooks/adapters/claude_code.py" "$CURSOR_HOOKS_DIR/adapters/claude_code.py"
    info "Copied hook scripts to $CURSOR_HOOKS_DIR"

    # Merge hooks.json
    python3 -c "
import json, sys, os

template_path = sys.argv[1]
target_path = sys.argv[2]

with open(template_path) as f:
    template = json.load(f)

target = {'version': 1, 'hooks': {}}
if os.path.exists(target_path):
    with open(target_path) as f:
        target = json.load(f)

for event, commands in template.get('hooks', {}).items():
    existing = target.setdefault('hooks', {}).setdefault(event, [])
    existing_cmds = {c.get('command') for c in existing}
    for cmd in commands:
        if cmd.get('command') not in existing_cmds:
            existing.append(cmd)

with open(target_path, 'w') as f:
    json.dump(target, f, indent=2)
    f.write('\n')
" "$SCRIPT_DIR/hooks/hooks.json" "$CURSOR_HOOKS_JSON"

    info "Merged hook entries into $CURSOR_HOOKS_JSON"
    INSTALLED_CURSOR=true
fi

# ── Install Claude Code hooks ────────────────────────────────────────────────

if [ "$HAS_CLAUDE_CODE" = true ]; then
    header "Installing Claude Code hooks…"

    CLAUDE_HOOKS_DIR="$HOME/.claude/hooks"
    CLAUDE_SETTINGS="$HOME/.claude/settings.json"
    CLAUDE_ENV_FILE="$CLAUDE_HOOKS_DIR/.coding-agent-insights.env"

    mkdir -p "$CLAUDE_HOOKS_DIR"
    cp "$SCRIPT_DIR/hooks/claude-code-trace-hook.sh" "$CLAUDE_HOOKS_DIR/claude-code-trace-hook.sh"
    chmod +x "$CLAUDE_HOOKS_DIR/claude-code-trace-hook.sh"
    cp "$SCRIPT_DIR/hooks/flush.py" "$CLAUDE_HOOKS_DIR/flush.py"
    cp "$SCRIPT_DIR/hooks/core.py" "$CLAUDE_HOOKS_DIR/core.py"
    mkdir -p "$CLAUDE_HOOKS_DIR/adapters"
    cp "$SCRIPT_DIR/hooks/adapters/__init__.py" "$CLAUDE_HOOKS_DIR/adapters/__init__.py"
    cp "$SCRIPT_DIR/hooks/adapters/cursor.py" "$CLAUDE_HOOKS_DIR/adapters/cursor.py"
    cp "$SCRIPT_DIR/hooks/adapters/claude_code.py" "$CLAUDE_HOOKS_DIR/adapters/claude_code.py"
    info "Copied hook scripts to $CLAUDE_HOOKS_DIR"

    # Merge Claude Code settings.json
    python3 -c "
import json, sys, os

target_path = sys.argv[1]
hook_command = sys.argv[2]

settings = {}
if os.path.exists(target_path):
    with open(target_path) as f:
        settings = json.load(f)

hooks = settings.setdefault('hooks', {})
stop_hooks = hooks.setdefault('Stop', [])

# Check if our hook is already registered
existing_cmds = {h.get('command', '') for h in stop_hooks if isinstance(h, dict)}
if hook_command not in existing_cmds:
    stop_hooks.append({'command': hook_command})

with open(target_path, 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
" "$CLAUDE_SETTINGS" "bash $CLAUDE_HOOKS_DIR/claude-code-trace-hook.sh"

    info "Registered Stop hook in $CLAUDE_SETTINGS"
    INSTALLED_CLAUDE_CODE=true
fi

# ── Phoenix setup ─────────────────────────────────────────────────────────────

header "Phoenix setup"
echo ""
echo "  coding-agent-insights sends traces to a Phoenix instance."
echo "  How would you like to set up Phoenix?"
echo ""
echo "  1) Local Docker  — spin up Phoenix v13.15.0 in a container"
echo "  2) Existing URL   — connect to a running Phoenix instance"
echo "  3) Skip           — I'll configure this later"
echo ""

PHOENIX_HOST="http://localhost:6006"
PHOENIX_SETUP="skip"

read -rp "  Choose [1/2/3] (default: 1): " phoenix_choice
phoenix_choice="${phoenix_choice:-1}"

case "$phoenix_choice" in
    1)
        PHOENIX_SETUP="docker"
        if [ "$HAS_DOCKER" = false ]; then
            err "Docker is required for local setup but not installed."
            echo "  Install Docker Desktop: https://www.docker.com/products/docker-desktop"
            echo "  Then re-run this installer, or choose option 2/3."
            exit 1
        fi
        PHOENIX_HOST="http://localhost:6006"
        ;;
    2)
        PHOENIX_SETUP="remote"
        read -rp "  Phoenix URL (e.g. https://phoenix.mycompany.com): " user_url
        if [ -z "$user_url" ]; then
            err "URL cannot be empty."
            exit 1
        fi
        PHOENIX_HOST="$user_url"
        ;;
    3)
        PHOENIX_SETUP="skip"
        warn "Skipping Phoenix setup. Set PHOENIX_HOST in the env file later."
        ;;
    *)
        err "Invalid choice. Please run the installer again."
        exit 1
        ;;
esac

# ── Project name ──────────────────────────────────────────────────────────────

PHOENIX_PROJECT="coding-agent-insights"
read -rp "  Phoenix project name (default: coding-agent-insights): " user_project
PHOENIX_PROJECT="${user_project:-coding-agent-insights}"

# ── Write env files ───────────────────────────────────────────────────────────

write_env_file() {
    local env_path="$1"
    local agent_type="$2"
    cat > "$env_path" <<EOF
# coding-agent-insights configuration
# Generated by install.sh — edit freely.
PHOENIX_HOST="$PHOENIX_HOST"
PHOENIX_PROJECT="$PHOENIX_PROJECT"
AGENT_TYPE="$agent_type"
# TRACES_DEBUG="true"
# TRACES_SKIP="field1,field2"
# TRACES_LOG="/tmp/coding-agent-insights.log"
EOF
    info "Settings saved to $env_path"
}

if [ "$INSTALLED_CURSOR" = true ]; then
    write_env_file "$CURSOR_ENV_FILE" "cursor"
fi

if [ "$INSTALLED_CLAUDE_CODE" = true ]; then
    write_env_file "$CLAUDE_ENV_FILE" "claude_code"
fi

# ── Start local Phoenix ──────────────────────────────────────────────────────

if [ "$PHOENIX_SETUP" = "docker" ]; then
    header "Starting Phoenix…"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
    echo ""

    for i in $(seq 1 15); do
        if curl -sf "$PHOENIX_HOST" >/dev/null 2>&1; then
            info "Phoenix is running at $PHOENIX_HOST"
            break
        fi
        if [ "$i" -eq 15 ]; then
            warn "Phoenix did not respond in time. Check: docker logs coding-agent-insights-phoenix"
        fi
        sleep 2
    done
fi

# ── Validate ──────────────────────────────────────────────────────────────────

if [ "$PHOENIX_SETUP" != "skip" ]; then
    if curl -sf "$PHOENIX_HOST" >/dev/null 2>&1; then
        info "Phoenix reachable at $PHOENIX_HOST"
    else
        warn "Could not reach Phoenix at $PHOENIX_HOST — check the URL or start it manually."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────

header "Installation complete!"
echo ""

if [ "$INSTALLED_CURSOR" = true ]; then
    echo "  Cursor:"
    echo "    Hook scripts:   $CURSOR_HOOKS_DIR/"
    echo "    Hooks config:   $CURSOR_HOOKS_JSON"
    echo "    Settings:       $CURSOR_ENV_FILE"
    echo ""
fi

if [ "$INSTALLED_CLAUDE_CODE" = true ]; then
    echo "  Claude Code:"
    echo "    Hook scripts:   $CLAUDE_HOOKS_DIR/"
    echo "    Settings file:  $CLAUDE_SETTINGS"
    echo "    Env config:     $CLAUDE_ENV_FILE"
    echo ""
fi

if [ "$PHOENIX_SETUP" = "docker" ]; then
    echo "  Phoenix UI:     $PHOENIX_HOST"
    echo ""
fi

if [ "$INSTALLED_CURSOR" = true ]; then
    echo "  Cursor will now trace agent sessions automatically."
fi
if [ "$INSTALLED_CLAUDE_CODE" = true ]; then
    echo "  Claude Code will now trace agent sessions on each Stop hook."
fi
echo ""
echo "  Traces flush to Phoenix automatically, or manually:"
if [ "$INSTALLED_CURSOR" = true ]; then
    echo "    uv run $CURSOR_HOOKS_DIR/flush.py --agent cursor"
fi
if [ "$INSTALLED_CLAUDE_CODE" = true ]; then
    echo "    uv run $CLAUDE_HOOKS_DIR/flush.py --agent claude_code --transcript <path>"
fi
echo ""
