#!/bin/bash

# PromptEvolver Setup Script
# This script sets up the development environment for PromptEvolver

set -euo pipefail  # Exit on error, unset variable, or failed pipe

echo "🚀 Setting up PromptEvolver..."

# Check if uv is installed, install if needed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate virtual environment (OS-specific)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "🔌 Activating virtual environment (Windows)..."
    source .venv/Scripts/activate
else
    echo "🔌 Activating virtual environment (Unix/macOS)..."
    source .venv/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
uv pip install -e .

# Ensure local env file exists for API keys
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "⚠️  .env.example not found; create a .env manually with OPENAI_API_KEY=..."
    fi
fi

# Check for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "⚠️  WARNING: OPENAI_API_KEY environment variable is not set"
    echo "   Please set it with: export OPENAI_API_KEY='your-api-key-here'"
else
    echo "✅ OPENAI_API_KEY is set"
fi

# Check for Node.js (required for promptfoo)
if ! command -v node &> /dev/null; then
    echo "⚠️  WARNING: Node.js is not installed"
    echo "   Please install Node.js to use promptfoo: https://nodejs.org/"
else
    echo "✅ Node.js is installed"
fi

# Check for Ollama (for local reasoning)
if ! command -v ollama &> /dev/null; then
    echo "⚠️  WARNING: Ollama is not installed"
    echo "   Install from https://ollama.com and start with: ollama serve"
else
    echo "✅ Ollama is installed"
    # Try to pull the reasoning model if missing
    if ! ollama list | grep -q "^phi4-mini-reasoning"; then
        echo "⬇️  Pulling model: phi4-mini-reasoning"
        ollama pull phi4-mini-reasoning || echo "⚠️  Could not pull model automatically. Pull it manually with: ollama pull phi4-mini-reasoning"
    else
        echo "✅ Model available: phi4-mini-reasoning"
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Open .env and add OPENAI_API_KEY=your-key (and any other credentials)."
echo "2. (Optional) Start Ollama if you want local reasoning: ollama serve" \
     " && ollama pull phi4-mini-reasoning"
echo "3. Run optimization with uv: uv run promptevolver.py"
echo "4. View results: uv run promptevolver.py --view-iteration 1"
echo ""
echo "For more options, run: uv run promptevolver.py --help"
