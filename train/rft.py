"""Reinforcement fine-tuning entrypoint built on the Tinker Cookbook RL loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

# Disable tokenizer parallelism to avoid warnings/deadlocks when forking for CAD runtime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tinker
import chz
from model.llm_shared import SYSTEM_MESSAGE
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from train.reward import RewardEvaluator
from utils.code_cleaning import sanitize_cad_code


LOGGER = logging.getLogger(__name__)



def _format_prompt(prompt: str, system_prompt: str) -> str:
    user_prompt = prompt.strip()
    return (
        f"{system_prompt.strip()}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        "Return only valid cadquery Python code implementing the request."
    )


def _sanitize_code(text: str) -> str:
    return sanitize_cad_code(text)


class CADBanditEnv(Env):
    """Single-step environment: prompt -> CAD code -> reward."""

    def __init__(
        self,
        *,
        prompt: str,
        tokenizer: Tokenizer,
        reward_evaluator: RewardEvaluator,
        sample_prefix: str,
        system_prompt: str,
        stop_condition: StopCondition,
        max_tokens: int,
    ) -> None:
        self._prompt = prompt
        self._tokenizer = tokenizer
        self._reward_evaluator = reward_evaluator
        self._sample_prefix = sample_prefix
        self._system_prompt = system_prompt
        self._stop_condition = stop_condition
        self._max_tokens = max_tokens
        rendered = _format_prompt(prompt, system_prompt)
        prompt_tokens = self._tokenizer.encode(rendered, add_special_tokens=True)
        self._initial_ob = tinker.ModelInput.from_ints(prompt_tokens)

    @property
    def stop_condition(self) -> StopCondition:
        return self._stop_condition

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return self._initial_ob, self._stop_condition

    async def step(self, action: list[int]) -> StepResult:
        cad_text = self._tokenizer.decode(action)
        cad_code = _sanitize_code(cad_text)
        reward_info = self._reward_evaluator.score(
            prompt=self._prompt,
            cad_code=cad_code,
            sample_id=self._sample_prefix,
        )
        
        # Apply progressive length penalty (cubic)
        # Penalty is negligible for short sequences, ramps up as it approaches soft_limit
        # Soft limit is 75% of max_tokens
        soft_limit = self._max_tokens * 0.75
        token_count = len(action)
        max_penalty = 0.5
        power = 7.0
        
        ratio = token_count / soft_limit
        length_penalty = max_penalty * (ratio ** power)
        
        final_reward = reward_info.reward - length_penalty
        final_reward = max(final_reward, -1.0)

        if token_count > soft_limit:
            LOGGER.warning(
                "Sample %s exceeded soft limit (%d > %d). Full output:\n%s",
                self._sample_prefix,
                token_count,
                int(soft_limit),
                cad_text,
            )

        LOGGER.info(
            "Sample %s: reward=%.3f (judge=%.2f, compiled=%s, len_pen=%.3f) tokens=%d/%d",
            self._sample_prefix,
            final_reward,
            reward_info.judge.score,
            "yes" if reward_info.compilation.success else "no",
            length_penalty,
            token_count,
            int(self._max_tokens),
        )

        metrics = {
            "judge/score": reward_info.judge.score,
            "compile/success": 1.0 if reward_info.compilation.success else 0.0,
            "token_count": float(token_count),
            "penalty/length": float(length_penalty),
        }
        return StepResult(
            reward=final_reward,
            episode_done=True,
            next_observation=self._initial_ob,
            next_stop_condition=self._stop_condition,
            metrics=metrics,
        )


class CADEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        *,
        prompt: str,
        tokenizer: Tokenizer,
        reward_evaluator: RewardEvaluator,
        sample_prefix: str,
        system_prompt: str,
        stop_condition: StopCondition,
        group_size: int,
        max_tokens: int,
    ) -> None:
        self._prompt = prompt
        self._tokenizer = tokenizer
        self._reward_evaluator = reward_evaluator
        self._sample_prefix = sample_prefix
        self._system_prompt = system_prompt
        self._stop_condition = stop_condition
        self._group_size = group_size
        self._max_tokens = max_tokens

    async def make_envs(self) -> Sequence[Env]:
        return [
            CADBanditEnv(
                prompt=self._prompt,
                tokenizer=self._tokenizer,
                reward_evaluator=self._reward_evaluator,
                sample_prefix=f"{self._sample_prefix}_rollout_{idx:02d}",
                system_prompt=self._system_prompt,
                stop_condition=self._stop_condition,
                max_tokens=self._max_tokens,
            )
            for idx in range(self._group_size)
        ]


class CADDataset(RLDataset):
    def __init__(
        self,
        *,
        prompts: list[str],
        batch_size: int,
        group_size: int,
        tokenizer: Tokenizer,
        reward_evaluator: RewardEvaluator,
        system_prompt: str,
        stop_condition: StopCondition,
        max_tokens: int,
    ) -> None:
        self._prompts = prompts
        self._batch_size = batch_size
        self._group_size = group_size
        self._tokenizer = tokenizer
        self._reward_evaluator = reward_evaluator
        self._system_prompt = system_prompt
        self._stop_condition = stop_condition
        self._max_tokens = max_tokens

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self._batch_size
        end = min(len(self._prompts), start + self._batch_size)
        builders: list[EnvGroupBuilder] = []
        for offset, prompt in enumerate(self._prompts[start:end]):
            prefix = f"batch_{index:05d}_prompt_{offset:03d}"
            builders.append(
                CADEnvGroupBuilder(
                    prompt=prompt,
                    tokenizer=self._tokenizer,
                    reward_evaluator=self._reward_evaluator,
                    sample_prefix=prefix,
                    system_prompt=self._system_prompt,
                    stop_condition=self._stop_condition,
                    group_size=self._group_size,
                    max_tokens=self._max_tokens,
                )
            )
        return builders

    def __len__(self) -> int:
        return len(self._prompts) // self._batch_size


@chz.chz
class CADDatasetBuilder(RLDatasetBuilder):
    prompt_file: Path
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    rollout_root: Path
    judge_backend: str
    judge_model_name: str
    judge_weight: float
    views: tuple[str, ...] = ("front", "side", "top", "iso")
    system_prompt: str = SYSTEM_MESSAGE
    stop_condition: StopCondition = chz.field(default_factory=tuple)
    max_prompts: int | None = None
    seed: int | None = None
    max_tokens: int = 4096

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        with self.prompt_file.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if self.max_prompts is not None:
            prompts = prompts[: self.max_prompts]
        rng = random.Random(self.seed)
        rng.shuffle(prompts)
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        reward_evaluator = RewardEvaluator(
            output_root=self.rollout_root,
            views=self.views,
            judge_backend=self.judge_backend,
            judge_model_name=self.judge_model_name,
            judge_weight=self.judge_weight,
        )
        dataset = CADDataset(
            prompts=prompts,
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            reward_evaluator=reward_evaluator,
            system_prompt=self.system_prompt,
            stop_condition=list(self.stop_condition),
            max_tokens=self.max_tokens,
        )
        return dataset, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAD reinforcement fine-tuning (Tinker Cookbook)")
    parser.add_argument("--prompt-file", type=Path, required=True, help="File containing one prompt per line")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for logs and artefacts")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name for Tinker")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts per training batch")
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt (envs per group)")
    parser.add_argument("--learning-rate", type=float, default=4.65e-4, help="Learning rate (Tinker LR rule for Qwen3-32B)")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for training client")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate per rollout")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for rollouts")
    parser.add_argument("--judge-backend", type=str, default="openai", help="Judge backend: openai|gemini|huggingface")
    parser.add_argument("--judge-model", type=str, default="gpt-5", help="Judge model name")
    parser.add_argument("--judge-weight", type=float, default=0.7, help="Blend weight for judge vs compile")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on prompts used for training")
    parser.add_argument("--seed", type=int, default=None, help="Seed for shuffling prompts")
    parser.add_argument("--resume-from", type=str, default=None, help="Tinker checkpoint URI to resume from (e.g. tinker://...)")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    return parser.parse_args()


async def _build_config(args: argparse.Namespace) -> rl_train.Config:
    log_path = args.output_dir
    log_path.mkdir(parents=True, exist_ok=True)

    dataset_builder = CADDatasetBuilder(
        prompt_file=args.prompt_file,
        batch_size=args.batch_size,
        group_size=args.group_size,
        model_name_for_tokenizer=args.model_name,
        rollout_root=log_path / "rollout_artefacts",
        judge_backend=args.judge_backend,
        judge_model_name=args.judge_model,
        judge_weight=args.judge_weight,
        max_prompts=args.max_prompts,
        seed=args.seed,
        stop_condition=("END_OF_CODE",),
        max_tokens=args.max_tokens,
    )

    cfg = rl_train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=str(log_path),
        lora_rank=args.lora_rank,
        loss_fn="importance_sampling",
        remove_constant_reward_groups=False,
        eval_every=0,  # disable default eval hooks
        save_every=10,
        load_checkpoint_path=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_run_name,
    )
    return cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger("tinker").setLevel(logging.INFO)
    args = _parse_args()
    cfg = asyncio.run(_build_config(args))
    asyncio.run(rl_train.main(cfg))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
