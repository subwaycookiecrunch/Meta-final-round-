#!/bin/bash
# Quick-start script for CodeReviewEnv v3
# Run from the Meta-project directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════╗"
echo "║  CodeReviewEnv v3 — Security Code Investigator   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated (Python $(python3 --version 2>&1 | cut -d' ' -f2))"
else
    echo "❌ No .venv found. Run: python3.12 -m venv .venv && source .venv/bin/activate && pip install openenv-core fastmcp fastapi uvicorn matplotlib"
    exit 1
fi

# Verify openenv
python3 -c "from openenv.core.env_server import MCPEnvironment; print('✅ OpenEnv installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ OpenEnv not installed. Run: pip install openenv-core fastmcp"
    exit 1
fi

# Check code snippets
if [ ! -f "data/code_snippets.json" ]; then
    echo "⚙️  Generating code snippets..."
    python3 scripts/generate_code_snippets.py
fi

echo ""
echo "Available commands:"
echo "  1) python3 demo.py                    # Run baseline comparison"
echo "  2) python3 -m uvicorn server.app:app  # Start HTTP server"
echo "  3) python3 train_grpo.py              # Train with GRPO (needs GPU)"
echo ""
echo "─────────────────────────────────────────────────────"

# Default: run demo
if [ "$1" = "demo" ] || [ -z "$1" ]; then
    echo "Running demo..."
    echo ""
    python3 demo.py
elif [ "$1" = "server" ]; then
    echo "Starting server on http://localhost:7860 ..."
    python3 -m uvicorn server.app:app --host 0.0.0.0 --port 7860
elif [ "$1" = "train" ]; then
    echo "Starting GRPO training..."
    python3 train_grpo.py
elif [ "$1" = "test" ]; then
    echo "Running tests..."
    python3 -c "
from openenv.core.env_server import CallToolAction, ListToolsAction
from code_review_env.server.environment import CodeReviewEnvironment
import re

env = CodeReviewEnvironment()
obs = env.reset(seed=42, difficulty='easy')
print('✅ Reset OK')

obs = env.step(ListToolsAction())
print(f'✅ 6 tools: {[t.name for t in obs.tools]}')

ctx = env.reset(seed=42).metadata['context']
files = re.findall(r'• (.+?)\s+\[', ctx)
obs = env.step(CallToolAction(tool_name='read_file', arguments={'file_path': files[0]}))
print(f'✅ read_file: got code for {files[0]}')

obs = env.step(CallToolAction(tool_name='search_code', arguments={'pattern': 'buffer'}))
print('✅ search_code OK')

obs = env.step(CallToolAction(tool_name='flag_vulnerable', arguments={'file_path': files[0], 'reasoning': 'test'}))
print('✅ flag_vulnerable OK')

obs = env.step(CallToolAction(tool_name='submit_report', arguments={'summary': 'test report', 'confidence': 'low'}))
print('✅ submit_report OK')

print()
print('🎉 All tests passed!')
"
fi
