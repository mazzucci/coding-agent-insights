#!/bin/bash
set -euo pipefail

# coding-agent-insights uninstaller
# Removes hook scripts, cleans hooks.json, and optionally stops Phoenix.

HOOKS_DIR="$HOME/.cursor/hooks"
HOOKS_JSON="$HOME/.cursor/hooks.json"
ENV_FILE="$HOOKS_DIR/.coding-agent-insights.env"
BUFFER="/tmp/cursor-traces.jsonl"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }

echo -e "\n${BOLD}Removing coding-agent-insights…${NC}\n"

# ── Remove hook scripts ───────────────────────────────────────────────────────

for f in trace-hook.sh flush.py; do
    if [ -f "$HOOKS_DIR/$f" ]; then
        rm "$HOOKS_DIR/$f"
        info "Removed $HOOKS_DIR/$f"
    fi
done

if [ -f "$ENV_FILE" ]; then
    rm "$ENV_FILE"
    info "Removed $ENV_FILE"
fi

# ── Clean hooks.json ──────────────────────────────────────────────────────────

if [ -f "$HOOKS_JSON" ]; then
    python3 -c "
import json, sys

path = sys.argv[1]
with open(path) as f:
    config = json.load(f)

hooks = config.get('hooks', {})
cleaned = {}
for event, commands in hooks.items():
    remaining = [c for c in commands if 'trace-hook.sh' not in c.get('command', '')]
    if remaining:
        cleaned[event] = remaining

config['hooks'] = cleaned

with open(path, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')
" "$HOOKS_JSON"
    info "Removed coding-agent-insights entries from $HOOKS_JSON"
fi

# ── Clear buffer ──────────────────────────────────────────────────────────────

if [ -f "$BUFFER" ]; then
    rm "$BUFFER"
    info "Removed buffer $BUFFER"
fi

if [ -f "/tmp/cursor-traces.log" ]; then
    rm "/tmp/cursor-traces.log"
    info "Removed debug log"
fi

# ── Phoenix container ─────────────────────────────────────────────────────────

if command -v docker &>/dev/null; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "coding-agent-insights-phoenix"; then
        echo ""
        read -rp "  Stop and remove the Phoenix container? [y/N]: " remove_phoenix
        if [[ "$remove_phoenix" =~ ^[Yy] ]]; then
            docker rm -f coding-agent-insights-phoenix >/dev/null 2>&1 || true
            info "Removed Phoenix container"

            read -rp "  Also remove Phoenix data volume? [y/N]: " remove_volume
            if [[ "$remove_volume" =~ ^[Yy] ]]; then
                docker volume rm coding-agent-insights_phoenix-data >/dev/null 2>&1 || true
                info "Removed Phoenix data volume"
            fi
        fi
    fi
fi

echo -e "\n${BOLD}coding-agent-insights has been uninstalled.${NC}\n"
