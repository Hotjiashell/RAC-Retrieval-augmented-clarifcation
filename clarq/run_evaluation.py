#!/usr/bin/env python3
"""Run Huawei ClarQ evaluation with RAC's passage-grounded policy prompt.

All flags except ``--huawei-eval-dir`` are forwarded to the Huawei evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_EVAL_DIR = WORKSPACE_ROOT / "huawei_dial" / "workspace" / "eval"


def _launcher_args(argv: Sequence[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--huawei-eval-dir",
        type=Path,
        default=Path(os.environ.get("HUAWEI_CLARQ_EVAL_DIR", DEFAULT_EVAL_DIR)),
        help="Path to huawei_dial/workspace/eval; remaining flags go to evaluate.py.",
    )
    args, forwarded = parser.parse_known_args(argv)
    return args.huawei_eval_dir.resolve(), forwarded


def _load_evaluator(eval_dir: Path) -> Any:
    source = eval_dir / "evaluate.py"
    if not source.is_file():
        raise FileNotFoundError(f"Huawei ClarQ evaluator not found: {source}")
    directory = str(eval_dir)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("huawei_clarq_evaluate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Huawei ClarQ evaluator from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_rac_adapter(evaluator: Any) -> None:
    # The evaluator directory is now on sys.path, which makes clarq_eval and
    # the sibling RAC adapter modules importable regardless of checkout path.
    from rac_policy import (
        INITIAL_PASSAGE_LIMIT,
        POLICY_NAME,
        POLICY_VERSION,
        PROMPT_VERSION,
        RACInitialPassageRetriever,
        RACPolicyClient,
    )
    from rac_runner import RACTraceRunner

    original_build_components = evaluator.build_components
    original_parse_args = evaluator.parse_args
    original_run_config = evaluator._run_config
    original_resume_signature = evaluator._resume_signature

    def parse_args(argv: Sequence[str] | None = None) -> Any:
        args = original_parse_args(argv)
        if args.policy_enable_thinking:
            raise ValueError(
                "RAC ClarQ requires thinking to be disabled; use --no-policy-enable-thinking "
                "or set POLICY_ENABLE_THINKING=false"
            )
        # Keep the explicit value in the shared evaluator configuration as well.
        args.policy_enable_thinking = False
        return args

    def build_components(args: Any) -> tuple[Any, list[tuple[str, Any]], Any]:
        runner, health_clients, retriever = original_build_components(args)
        initial_passage_retriever = RACInitialPassageRetriever(
            retriever,
            content_chars=runner.case_content_chars,
            passage_limit=INITIAL_PASSAGE_LIMIT,
        )
        policy_client = RACPolicyClient(runner.policy_client, initial_passage_retriever)
        runner.retriever = initial_passage_retriever
        runner.policy_client = policy_client
        # Return the unwrapped retriever for Huawei's preflight probe. It is not
        # part of a trajectory and must not become a sample's fixed evidence.
        return RACTraceRunner(runner, policy_client, initial_passage_retriever), health_clients, retriever

    def run_config(args: Any) -> dict[str, Any]:
        config = original_run_config(args)
        config["policy_adapter"] = {
            "name": POLICY_NAME,
            "version": POLICY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "protocol": "initial_and_search_refreshed_passages_with_huawei_native_tools",
            "training_used": False,
            "thinking_enabled": False,
            "initial_passage_retrieval": {
                "query": "original_user_question",
                "retrieval_count_per_trajectory": "one baseline plus each executed search_case",
                "passage_limit": INITIAL_PASSAGE_LIMIT,
            },
        }
        return config

    def resume_signature(config: dict[str, Any]) -> dict[str, Any]:
        signature = original_resume_signature(config)
        signature["policy_adapter"] = config.get("policy_adapter")
        return signature

    evaluator.parse_args = parse_args
    evaluator.build_components = build_components
    evaluator._run_config = run_config
    evaluator._resume_signature = resume_signature


def main(argv: Sequence[str] | None = None) -> int:
    eval_dir, forwarded = _launcher_args(list(argv) if argv is not None else sys.argv[1:])
    evaluator = _load_evaluator(eval_dir)
    _install_rac_adapter(evaluator)
    return int(evaluator.main(forwarded))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
