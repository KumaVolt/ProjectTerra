#!/usr/bin/env bash
# ProjectTerra launcher
# Usage: ./start.sh [command] [args...]
#
# Examples:
#   ./start.sh evolve
#   ./start.sh status
#   ./start.sh generate-data reasoning --samples 100
#   ./start.sh evaluate models/current
#   ./start.sh serve
#
# Without arguments, runs an evolution session.

set -euo pipefail
cd "$(dirname "$0")"

# First-time setup
if [ ! -d ".venv" ]; then
    echo "First run - running setup..."
    source scripts/setup.sh
    echo ""
fi

source .venv/bin/activate

# Load .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

terra "${@:-evolve}"
