from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 从项目根目录加载 .env
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
from benchmark.core.config import BenchmarkConfig
from benchmark.core.logging import setup_logging
from benchmark.engine.inferencer import Inferencer
from benchmark.engine.scorer import Scorer

VLM_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "configs" / "vlm_template.yaml"


def _load_vlm_template_config() -> BenchmarkConfig:
    """Load VLM template with env var expansion and path resolution (same as BenchmarkConfig.load)."""
    return BenchmarkConfig.load(VLM_TEMPLATE_PATH)


def _run_inference(cfg: BenchmarkConfig, stage: str) -> None:
    if stage in {"all", "infer"}:
        Inferencer(cfg).run()

    if stage in {"all", "score"}:
        Scorer(cfg.data.output_dir, positive_label=cfg.eval.positive_label).run()


def handle_config_mode(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    cfg = BenchmarkConfig.load(Path(args.config).resolve())
    _run_inference(cfg, args.stage)


def _normalize_vlm_models(args: argparse.Namespace, cfg: BenchmarkConfig) -> List[str]:
    if args.models:
        return args.models
    if args.model_name:
        return [args.model_name]
    if args.provider == "openrouter":
        if cfg.defaults.openrouter_models:
            return cfg.defaults.openrouter_models
        raise ValueError("No default openrouter models configured in defaults.openrouter_models.")
    return ["local-model"]


def handle_vlm_mode(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    base_cfg = _load_vlm_template_config()
    target_models = _normalize_vlm_models(args, base_cfg)
    if args.image_dir:
        base_cfg.data.image_dir = Path(args.image_dir)
    if args.questions_file:
        base_cfg.data.questions_file = Path(args.questions_file)

    cwd = Path.cwd()
    for model_name in target_models:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        out_dir = Path("data/results") / args.provider / safe_name

        print(f"\n{'=' * 60}")
        print(f"Running VLM | provider={args.provider} | model={model_name}")
        print(f"Output: {out_dir}")
        print(f"{'=' * 60}\n")

        cfg_obj = base_cfg.model_copy(deep=True)
        cfg_obj.data.output_dir = out_dir
        cfg_obj.eval.force_run = True

        model_args = cfg_obj.model.args
        model_args["provider"] = args.provider
        model_args["model"] = model_name

        if args.base_url:
            model_args["base_url"] = args.base_url
        else:
            model_args.pop("base_url", None)
        if args.api_key:
            model_args["api_key"] = args.api_key
        else:
            model_args.pop("api_key", None)

        try:
            # cfg_obj.data.image_dir = (cwd / cfg_obj.data.image_dir).resolve()
            # cfg_obj.data.questions_file = (cwd / cfg_obj.data.questions_file).resolve()
            # cfg_obj.data.output_dir = (cwd / cfg_obj.data.output_dir).resolve()

            _run_inference(cfg_obj, args.stage)
        except Exception as exc:
            print(f"Failed to run model {model_name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser("benchmark")
    parser.add_argument("--log-level", type=str, default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Mode of operation")

    parser_run = subparsers.add_parser("run", help="Run benchmark with a static YAML config")
    parser_run.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser_run.add_argument("--stage", type=str, default="all", choices=["all", "infer", "score"])
    parser_run.set_defaults(func=handle_config_mode)

    parser_vlm = subparsers.add_parser("vlm", help="Run VLM benchmark with CLI arguments")
    parser_vlm.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "local"],
        default="openrouter",
    )
    parser_vlm.add_argument("--models", nargs="+", help="List of model names")
    parser_vlm.add_argument("--model-name", type=str, help="Single model name")
    parser_vlm.add_argument("--base-url", type=str, help="API Base URL")
    parser_vlm.add_argument("--api-key", type=str, help="API key")
    parser_vlm.add_argument("--image-dir", type=str, help="Override image directory")
    parser_vlm.add_argument("--questions-file", type=str, help="Override questions JSON file")
    parser_vlm.add_argument("--stage", type=str, default="all", choices=["all", "infer", "score"])
    parser_vlm.set_defaults(func=handle_vlm_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
