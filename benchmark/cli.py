from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from benchmark.core.config import BenchmarkConfig
from benchmark.core.logging import setup_logging
from benchmark.engine.inferencer import Inferencer
from benchmark.engine.scorer import Scorer


def _run_pipeline(cfg: BenchmarkConfig, stage: str) -> None:
    if stage in {"all", "infer"}:
        Inferencer(cfg).run()
    if stage in {"all", "score"}:
        Scorer(cfg).run()


def _resolve_models(args: argparse.Namespace, base_cfg: BenchmarkConfig) -> List[Optional[str]]:
    if args.models:
        return args.models
    if args.model_name:
        return [args.model_name]
    if args.provider == "openrouter":
        if base_cfg.defaults.openrouter_models:
            return base_cfg.defaults.openrouter_models
        raise ValueError("No --models given and defaults.openrouter_models is empty.")
    return [None]


def _apply_model_env_defaults(cfg: BenchmarkConfig) -> None:
    """
    Centralize all environment-variable fallbacks for model args here so that
    downstream components (VLMAdapter, OptionLetterRepairAgent) stay pure and
    never read os.environ themselves. Only fills in values that the user has
    not already set explicitly via CLI/YAML.
    """
    if cfg.model.architecture != "vlm":
        return

    args: Dict[str, Any] = cfg.model.args
    provider = str(args.get("provider", "openrouter")).lower()

    if provider == "openrouter":
        api_key = args.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print(f"\n{'=' * 70}")
            print("ERROR: OPENROUTER_API_KEY is not set.")
            print("Please obtain a key from https://openrouter.ai/keys, then:")
            print("  export OPENROUTER_API_KEY='your_api_key_here'")
            print(f"{'=' * 70}\n")
            sys.exit(1)
        args["api_key"] = api_key
        args.setdefault(
            "base_url",
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    elif provider == "local":
        args.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "empty"))
        if os.environ.get("OPENAI_BASE_URL"):
            args.setdefault("base_url", os.environ.get("OPENAI_BASE_URL"))

    # Independent repair extractor credentials (paper: GPT-5 as auxiliary).
    args.setdefault(
        "extractor_api_key",
        os.environ.get("EXTRACTOR_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )
    args.setdefault(
        "extractor_base_url",
        os.environ.get("EXTRACTOR_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )


def main() -> None:
    parser = argparse.ArgumentParser("bprbench")
    parser.add_argument("--log-level", type=str, default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Mode of operation")

    parser_run = subparsers.add_parser("run", help="Run benchmark evaluation")
    parser_run.add_argument("-c", "--config", type=str, help="Path to base YAML config (optional with --set)")
    parser_run.add_argument("--set", nargs="+", help="Override config keys, e.g. eval.batch_size=16")
    parser_run.add_argument("--stage", type=str, default="all", choices=["all", "infer", "score"])

    # Batch VLM evaluation channel
    parser_run.add_argument("--models", nargs="+", help="List of model names to evaluate sequentially")
    parser_run.add_argument("--model-name", type=str, help="Single model name (shorthand for --models NAME)")
    parser_run.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "local"],
        help="Provider for VLM endpoints",
    )
    parser_run.add_argument("--base-url", type=str, help="Override API base URL")
    parser_run.add_argument("--api-key", type=str, help="Override API key")
    parser_run.add_argument("--image-dir", type=str, help="Override image directory")
    parser_run.add_argument("--questions-file", type=str, help="Override questions JSON file")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "run":
        if not args.config and not args.set and not args.models and not args.model_name and not args.provider:
            parser_run.error("Must provide either --config, --set, --models, --model-name, or --provider")

        base_cfg = BenchmarkConfig.load(args.config, overrides=args.set)
        target_models = _resolve_models(args, base_cfg)

        if args.image_dir:
            base_cfg.data.image_dir = Path(args.image_dir)
        if args.questions_file:
            base_cfg.data.questions_file = Path(args.questions_file)

        for model_name in target_models:
            # Use model_copy for a clean deep-copy + targeted update, instead of
            # mutating a hand-rolled clone. Args are still touched in place
            # because they are a free-form Dict[str, Any] that Pydantic does not
            # model field-by-field.
            cfg = base_cfg.model_copy(deep=True)

            if model_name:
                if args.provider:
                    cfg.model.architecture = "vlm"
                    cfg.model.args["provider"] = args.provider

                cfg.model.args["model"] = model_name

                # Only override when an explicit CLI value is given; never
                # blindly pop existing YAML/env config (previous behaviour
                # silently dropped user-configured credentials).
                if args.base_url:
                    cfg.model.args["base_url"] = args.base_url
                if args.api_key:
                    cfg.model.args["api_key"] = args.api_key

            # Single source of truth for env-var fallbacks: do it here, once,
            # so VLMAdapter / OptionLetterRepairAgent can stay pure.
            _apply_model_env_defaults(cfg)

            provider_name = cfg.model.args.get("provider", cfg.model.architecture)
            display_name = cfg.model.display_name or model_name or cfg.model.architecture

            if model_name:
                safe_name = model_name.replace("/", "_").replace(":", "_")
                # Respect the user's explicitly configured output_dir (from YAML
                # or --set) as the base, and append provider/model underneath
                # it. Previously this overrode the user's base path entirely,
                # forcing everything under ./data/results.
                cfg.data.output_dir = base_cfg.data.output_dir / provider_name / safe_name
                cfg.eval.force_run = True

            print(f"\n{'=' * 60}")
            print(f"Running Eval | model={display_name} | provider={provider_name}")
            print(f"Output: {cfg.data.output_dir}")
            print(f"{'=' * 60}\n")

            try:
                _run_pipeline(cfg, args.stage)
            except Exception as exc:
                print(f"Failed to run model {model_name or cfg.model.architecture}: {exc}")


if __name__ == "__main__":
    main()
