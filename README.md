# CAD-RFT Evaluation

This repository contains tooling for evaluating text-to-CAD models with reinforcement fine-tuning signals. The current implementation focuses on the evaluation path only: generating CAD code from text prompts, compiling that code into geometry, rendering multi-view images, and combining those artefacts with an LLM judge to produce a scalar reward.

## Repository Layout

- `llm_judge/`: pluggable judge interfaces (OpenAI GPT responses or Gemini judges).
- `evaluate/`: orchestration logic for running models, compiling CAD code, and aggregating scores.
- `data/`: helper datasets used during evaluation experiments.

## Requirements

Install Python 3.10+ and create an isolated environment before pulling in dependencies. Using `venv` keeps the CadQuery/OCC binaries self-contained and avoids conflicts with system Anaconda installs:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Rendering with matplotlib runs fully off-screen and does not require an OpenGL context.
The prebuilt `cadquery` wheels pulled in by `pip install -r requirements.txt` ship with the
VTK-enabled OCC toolkit, resolving the `IVtkOCC_Shape` import errors seen in base Anaconda
installs. If `pip` reports dependency conflicts, ensure you are running a modern version
(`pip install --upgrade pip`) so it can select compatible NumPy releases (CadQuery 2.5+
requires NumPy 2.x, which recent matplotlib/trimesh wheels fully support).

## Model Interface

Generation models now derive from `evaluate.generator.BaseGenerator`. Implementations simply inherit from this base class and override `generate(prompt: str) -> str`. Once implemented, pass the fully qualified class path to the evaluator.

Example skeleton:

```python
# model/inference.py
from evaluate.generator import BaseGenerator

class EchoGenerator(BaseGenerator):
    def generate(self, prompt: str) -> str:
        return (
            "import cadquery as cq\n"
            "\n"
            "def build():\n"
            "    return cq.Workplane().box(1, 1, 1)\n"
        )
```

To use a custom generator class, instantiate `Evaluator` programmatically and pass your implementation (see `evaluate.__init__` for a quick-start example). The CLI focuses on the builtin OpenAI/Gemini/Hugging Face providers exposed via `--generator-backend`.

Builtin generators:
- `model.openai_llm_generator.OpenAIAPIGenerator` (OpenAI Responses API)
- `model.gemini_llm_generator.GeminiAPIGenerator` (Gemini API; pass any supported model name via `model_name`)
- `model.hf_llm_generator.HuggingFaceAPIGenerator` (Hugging Face OpenAI-compatible router; accepts `max_tokens`, default 16384)

Use the CLI flag `--generator-backend` (`openai`, `gemini`, or `huggingface`) to pick among them, and pass backend-specific options via `--generator-kwargs`.

If you are running a compatible service that exposes the OpenAI Chat Completions API shape, use `model.chat_endpoint_generator.ChatEndpointGenerator` from a short Python script and pass it into `Evaluator`.

## Running Evaluation

The CLI entrypoint lives under the `evaluate` package:

```bash
python -m evaluate \
  --input-file data/text2cad_rft_val.txt \
  --output-dir outputs/eval_run_001 \
  --generator-backend openai \
  --generator-kwargs '{"model_name": "gpt-5", "temperature": 1.0}' \
  --judge-backend openai \
  --judge-model gpt-5
```
Swap `--generator-backend` / `--judge-backend` to `gemini` or `huggingface` (and update the corresponding kwargs/model IDs) to target other providers.

Set the `OPENAI_API_KEY` environment variable so the LLM judge can invoke the OpenAI API.
When using the Gemini backend (`--judge-backend gemini`), configure authentication via
`GEMINI_API_KEY` (preferred) or the upstream `GOOGLE_API_KEY`. Set `--judge-model` to whichever
Gemini model string you want to query (e.g. `gemini-2.0-flash-exp`, `gemini-1.5-pro-latest`, etc.).
If you select the Hugging Face judge backend (`--judge-backend huggingface`, e.g. `--judge-model Qwen/Qwen3-32B:groq`),
set `HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN` so the router can authenticate.
The same API keys govern generation: to run Gemini for CAD generation, pass
`--generator-backend gemini --generator-kwargs '{"model_name": "gemini-2.0-flash"}'` (or any other Gemini model string) and ensure
`GEMINI_API_KEY`/`GOOGLE_API_KEY` is available. For Hugging Face generation models supply an HF token
(`HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`) that has access to the desired hosted model (e.g. `Qwen/Qwen3-32B:groq`) and set
`--generator-backend huggingface --generator-kwargs '{"model_name": "Qwen/Qwen3-32B", "max_tokens": 16384}'`.

Evaluation artefacts are stored under the chosen `output-dir`:

- `artefacts/sample_XXXXX/`: STL files and rendered PNGs per prompt.
- `results.jsonl`: per-sample metadata including judge reasoning.
- `metrics.json`: aggregate statistics (currently the average blended score).
- `report.html`: human-friendly table summarising prompts, generated code, compilation status, renderings, and judge feedback.
- `partial_state.jsonl`: checkpoint written incrementally (every 50 samples) so progress survives crashes; summarise it via `python scripts/aggregate_state.py outputs/<run>/partial_state.jsonl`.

## Reinforcement Fine-Tuning

The `train.rft` module now reuses the [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/install) RL loop. It plugs the existing CAD compilation + LLM judge reward into a single-step environment, letting the cookbook handle rollouts, advantage computation, and optimiser steps.

Setup (source install, matching the cookbook docs):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # pulls in tinker
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install -e ./tinker-cookbook
export TINKER_API_KEY=<your key>
```

Run training:

```bash
python -m train.rft \
  --prompt-file data/text2cad_rft_train.txt \
  --output-dir outputs/rft_run_001 \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --batch-size 8 \
  --group-size 4 \
  --judge-backend openai \
  --judge-model gpt-5
```

Key options:

- `--batch-size`: prompts per iteration (dataset groups).
- `--group-size`: rollouts per prompt (envs per group used to centre rewards).
- `--max-tokens`, `--temperature`: decoding parameters for rollouts.
- `--learning-rate`, `--lora-rank`: optimiser settings passed to the Tinker training client.

Artefacts under `--output-dir`:

- `rollout_artefacts/sample_*`: STL files, renders, and logs for each rollout the reward function scores.
- Cookbook logs/checkpoints under the same directory (metrics, optional logtree traces).

## Weights & Scoring

The final score is a convex combination of compilation success (1.0 for success, 0.0 otherwise) and the normalised judge score. You can control the blend with `--judge-weight` on the CLI.
