# PromptEvolver

Automatic prompt optimization using reasoning models and promptfoo.

## Overview

PromptEvolver is a Python CLI tool that automatically optimizes prompts by:
- Running promptfoo tests to evaluate prompt performance
- Using a **local Ollama reasoning model** to analyze test failures
- Generating improved prompts based on the analysis
- Repeating until optimal performance is achieved

**Key Innovation**: Uses a lightweight local reasoning model (user-specified) for intelligent evaluation, while promptfoo tests use OpenAI models. This hybrid approach provides fast, private reasoning with high-quality test execution.

## Quick Start

1. **Setup the environment:**
   ```bash
   ./setup.sh
   ```

2. **Add your API keys to `.env`:**
   The setup script creates `.env` if it does not exist. Open it in your editor and set values such as:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

3. **Start Ollama (optional – skip if using --use-openai-nano):**
   ```bash
   ollama serve
   # In another terminal:
   ollama pull qwen3:0.6b
   ```

4. **Run optimization (uv handles the virtualenv automatically):**
   ```bash
   uv run promptevolver.py
   # add --use-openai-nano to rely solely on OpenAI reasoning
   ```

5. **View results or compare iterations:**
   ```bash
   uv run promptevolver.py --view-iteration 1
   uv run promptevolver.py --compare
   ```

## What Happens During a Run

When you execute `uv run promptevolver.py`, the CLI orchestrates the following steps:

1. **Argument parsing & mode selection** – determines whether to run optimization or open the promptfoo viewer, and notes if you requested the OpenAI reasoning path with `--use-openai-nano`.
2. **Config + environment load** – reads the promptfoo YAML, extracts the initial prompt list, and loads any keys from the `.env` file next to `promptevolver.py`. The original prompts are cached so the file can be restored when the run ends.
3. **Reasoning model handshake** – either confirms the Ollama model is available or verifies `OPENAI_API_KEY` before switching to OpenAI `gpt-5-nano`. If neither path works, the run continues but skips automated analysis and prompt rewriting.
4. **Iteration loop** – for each round (up to `--iterations`):
   - writes the current candidate prompt back into the config;
   - calls `npx promptfoo@latest eval -o output/latest.json` (passing the `.env` file) and parses the resulting JSON to compute pass rate and average score;
   - prints the metrics and, when tests fail, asks the reasoning model to summarize the first few failures and return actionable suggestions.
5. **Prompt revision** – if suggestions exist and the 95 % pass-rate target is unmet, the reasoning model drafts a simpler replacement prompt. Guardrails ensure placeholders are kept, the text stays concise, and obviously bad candidates are discarded.
6. **Wrap-up & artifacts** – after the loop the tool highlights the best-performing prompt, prints a summary, saves `evolution_results.json`, and snapshots each variant to `iterations/iteration_N.yaml` for later inspection or viewer runs.
7. **Config reset** – finally rewrites the promptfoo config with the original baseline prompt so repeat runs always start from the same state.

## Features

- 🧠 **Local Reasoning**: Uses Ollama phi4-mini-reasoning for private, fast analysis (or OpenAI `gpt-5-nano` with `--use-openai-nano`)
- 📊 **Visual Feedback**: Clear progress indicators and result summaries  
- 🔄 **Smart Improvement**: Automatically generates better prompts with simplification bias
- 📁 **Iteration Tracking**: Saves each optimization round for comparison
- 🎯 **Target-driven**: Stops when 95% pass rate is achieved
- 🔍 **Built-in Viewer**: Integration with promptfoo's visualization UI
- 🛡️ **Anti-Complexity**: Prevents prompt over-optimization with length limits
- 📈 **Score Tracking**: Identifies best performing prompts across iterations

## Usage

```bash
# Run optimization with default settings
uv run promptevolver.py

# Custom configuration / iteration count
uv run promptevolver.py --config examples/customer_support.yaml --iterations 5 --model o4-mini

# Use OpenAI gpt-5-nano for reasoning instead of Ollama
uv run promptevolver.py --use-openai-nano

# View or compare iterations in the promptfoo UI
uv run promptevolver.py --view-iteration 2
uv run promptevolver.py --compare
```

## Configuration

PromptEvolver uses promptfoo YAML configuration files. See `promptfooconfig.yaml` for an example, or check the `examples/` directory for more configurations.

## Requirements

- Python 3.10+
- Node.js (for promptfoo CLI)
- **Ollama** (for local reasoning model, unless you use `--use-openai-nano`) – [ollama.com](https://ollama.com)
- OpenAI API key stored in `.env`
- uv package manager (installed by `setup.sh`)

## License

MIT License - see LICENSE file for details.
