#!/usr/bin/env bash
# ProjectTerra setup script
#
# Usage:
#   source scripts/setup.sh    (recommended - keeps venv active)
#   ./scripts/setup.sh         (works but you'll need to activate venv manually after)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== ProjectTerra Setup ==="

# Check Python version
python3 --version || { echo "Python 3.11+ required"; exit 1; }

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
fi

source .venv/bin/activate

# Load .env if it exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Platform-specific extras
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS - installing MLX support for Apple Silicon..."
    pip install -e ".[mlx]"
fi

# Optional: install Claude support
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY detected - installing Claude support..."
    pip install -e ".[claude]"
fi

# Create necessary directories
mkdir -p logs/sessions logs/benchmarks logs/research data/generated models/checkpoints models/base models/current configs

# Check for API keys (at least one required)
HAS_LLM=false
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Claude API: configured"
    HAS_LLM=true
fi

if [ -n "${GLM_API_KEY:-}" ]; then
    echo "GLM API: configured"
    HAS_LLM=true
fi

if [ "$HAS_LLM" = false ]; then
    echo ""
    echo "WARNING: No LLM API key configured."
    echo "Create a .env file first:"
    echo "  cp .env.example .env"
    echo "  # then edit .env and add at least one API key"
fi

echo ""
echo "=== Setup complete ==="
echo ""
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "NOTE: Run 'source .venv/bin/activate' to activate the virtual environment."
    echo ""
fi
echo "Commands:"
echo "  terra evolve         - Run one evolution session"
echo "  terra status         - Check evolution state"
echo "  terra generate-data  - Generate training data"
echo "  terra evaluate       - Run benchmarks"
echo "  terra download-base  - Download base model"
echo "  terra serve          - Start local model server"
