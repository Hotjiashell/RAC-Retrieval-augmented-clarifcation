from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


CLARQ_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = CLARQ_DIR.parents[1] / "huawei_dial" / "workspace" / "eval"
for path in (CLARQ_DIR, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clarq_eval.models import EvaluationSample  # noqa: E402
from clarq_eval.parsing import PolicyProtocolError, parse_policy_response  # noqa: E402
from clarq_eval.runner import EvaluationRunner, TOOLS  # noqa: E402
from rac_policy import RACInitialPassageRetriever, RACPolicyClient  # noqa: E402
from rac_runner import RACTraceRunner  # noqa: E402
from run_evaluation import DEFAULT_EVAL_DIR, _install_rac_adapter, _load_evaluator  # noqa: E402


def model_response(
    content: str | None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "tool_calls": tool_calls or []},
            }
        ]
    }


def native_tool_call(name: str, arguments: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "model_call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def case(index: int) -> dict[str, Any]:
    return {
        "case_id": f"case-{index}",
        "title": f"Passage {index}",
        "content": f"Passage content {index}",
    }


class FakePolicyService:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def policy_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("Policy response sequence exhausted")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, max_results))
        if query == "initial question":
            return [case(index) for index in range(1, 7)]
        return [
            {
                "case_id": "target",
                "title": "Target case",
                "content": "Target answer",
            }
        ]


class FakeSimulator:
    def answer(self, sample: EvaluationSample, question: str) -> str:
        self.last_question = question
        return "Model X"


class FakeSuccessJudge:
    def judge(self, sample: EvaluationSample, cases: list[dict[str, Any]]) -> dict[str, Any]:
        raise AssertionError("Target should be found in the final retrieval")


SAMPLE = EvaluationSample(
    sample_id="sample-1",
    domain="electronics",
    target_case_id="target",
    initial_question="initial question",
    core_intent="hidden intent",
    known_info=("The model is Model X.",),
    target_case_title="Target case",
    target_case_content="Target answer",
)


class RACPolicyTests(unittest.TestCase):
    def test_initial_top_five_passages_refresh_after_search_case(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    None,
                    tool_calls=[native_tool_call("clarify_user", {"question": "Which model is affected?"})],
                    finish_reason="tool_calls",
                ),
                model_response(
                    None,
                    tool_calls=[native_tool_call("search_case", {"query": "device issue Model X"})],
                    finish_reason="tool_calls",
                ),
                model_response("Complete", finish_reason="stop"),
            ]
        )
        retriever = FakeRetriever()
        initial_passages = RACInitialPassageRetriever(retriever, content_chars=1_000)
        policy = RACPolicyClient(service, initial_passages)
        core_runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=FakeSimulator(),
            retriever=initial_passages,
            success_judge=FakeSuccessJudge(),
            max_turns=4,
            max_searches=2,
            top_k=5,
            success_top_k=1,
        )

        result = RACTraceRunner(core_runner, policy, initial_passages).run(SAMPLE)

        self.assertEqual(["initial question", "device issue Model X"], [query for query, _ in retriever.calls])
        self.assertEqual(
            ["clarify_user", "search_case", "complete"],
            [event["action"]["type"] for event in result["events"]],
        )
        self.assertTrue(result["success_judgment"]["success"])
        rac_metadata = result["rac_policy"]
        self.assertEqual(1, rac_metadata["initial_passage_retrieval_count"])
        self.assertEqual(2, rac_metadata["passage_evidence_retrieval_count"])
        self.assertEqual(5, len(rac_metadata["initial_passages"]))
        self.assertEqual(["Passage 1", "Passage 5"], [
            rac_metadata["initial_passages"][0]["title"],
            rac_metadata["initial_passages"][-1]["title"],
        ])
        self.assertEqual("search_case", rac_metadata["current_evidence_source"])
        self.assertEqual("Target case", rac_metadata["current_passages"][0]["title"])
        self.assertTrue(all(not call["kwargs"]["enable_thinking"] for call in service.calls))
        for call in service.calls[:2]:
            prompt = call["messages"][1]["content"]
            self.assertIn("Passage 1", prompt)
            self.assertIn("Passage 5", prompt)
            self.assertNotIn("Passage 6", prompt)
            self.assertNotIn("hidden intent", prompt)
        refreshed_prompt = service.calls[2]["messages"][1]["content"]
        self.assertIn("Target case", refreshed_prompt)
        self.assertNotIn("Passage 1", refreshed_prompt)
        self.assertEqual("initial_question", result["rac_policy"]["decisions"][0]["evidence_source"])
        self.assertEqual("search_case", result["rac_policy"]["decisions"][2]["evidence_source"])
        self.assertEqual(5, result["rac_policy"]["decisions"][0]["passage_count"])

    def test_tool_call_with_text_is_rejected(self) -> None:
        service = FakePolicyService(
            [
                model_response(
                    "I will clarify this.",
                    tool_calls=[native_tool_call("clarify_user", {"question": "Which model?"})],
                    finish_reason="tool_calls",
                )
            ]
        )
        retriever = RACInitialPassageRetriever(FakeRetriever(), content_chars=1_000)
        retriever.begin_sample(SAMPLE)
        retriever.search(SAMPLE.initial_question, 5)
        policy = RACPolicyClient(service, retriever)

        with self.assertRaisesRegex(PolicyProtocolError, "must not include assistant text"):
            policy.policy_chat(
                [{"role": "user", "content": SAMPLE.initial_question}],
                tools=TOOLS,
                enable_thinking=True,
            )

    def test_launcher_records_adapter_and_rejects_thinking(self) -> None:
        evaluator = _load_evaluator(EVAL_DIR)
        _install_rac_adapter(evaluator)
        with tempfile.TemporaryDirectory() as directory:
            args = evaluator.parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "http://policy.example/v1",
                    "--policy-model",
                    "policy-model",
                    "--simulator-base-url",
                    "http://simulator.example/v1",
                    "--simulator-model",
                    "simulator-model",
                    "--skip-judge",
                ]
            )
            config = evaluator._run_config(args)
            self.assertEqual("rac-clarq", config["policy_adapter"]["name"])
            self.assertFalse(config["policy_adapter"]["thinking_enabled"])
            self.assertEqual(5, config["policy_adapter"]["initial_passage_retrieval"]["passage_limit"])
            with self.assertRaisesRegex(ValueError, "requires thinking to be disabled"):
                evaluator.parse_args(
                    [
                        "--output-dir",
                        directory,
                        "--policy-base-url",
                        "http://policy.example/v1",
                        "--policy-model",
                        "policy-model",
                        "--simulator-base-url",
                        "http://simulator.example/v1",
                        "--simulator-model",
                        "simulator-model",
                        "--skip-judge",
                        "--policy-enable-thinking",
                    ]
                )
        native_config = dict(config)
        native_config.pop("policy_adapter")
        self.assertNotEqual(evaluator._resume_signature(config), evaluator._resume_signature(native_config))

    def test_default_evaluator_path_points_to_huawei_workspace(self) -> None:
        self.assertEqual(EVAL_DIR.resolve(), DEFAULT_EVAL_DIR.resolve())


if __name__ == "__main__":
    unittest.main()
