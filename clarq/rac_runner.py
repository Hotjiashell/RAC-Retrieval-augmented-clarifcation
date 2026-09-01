"""Attach RAC evidence metadata to Huawei ClarQ trajectories."""

from __future__ import annotations

from typing import Any

from rac_policy import (
    INITIAL_PASSAGE_LIMIT,
    POLICY_NAME,
    POLICY_VERSION,
    PROMPT_VERSION,
    RACInitialPassageRetriever,
    RACPolicyClient,
)


class RACTraceRunner:
    """Delegate all evaluation behavior while managing fixed RAC passages."""

    def __init__(
        self,
        runner: Any,
        policy_client: RACPolicyClient,
        initial_passage_retriever: RACInitialPassageRetriever,
    ):
        self._runner = runner
        self._policy_client = policy_client
        self._initial_passage_retriever = initial_passage_retriever

    @property
    def user_simulator(self) -> Any:
        return self._runner.user_simulator

    def run(self, sample: Any) -> dict[str, Any]:
        self._initial_passage_retriever.begin_sample(sample)
        self._policy_client.begin_sample(sample)
        try:
            result = self._runner.run(sample)
            evidence = self._initial_passage_retriever.audit_snapshot()
            result["rac_policy"] = {
                "name": POLICY_NAME,
                "version": POLICY_VERSION,
                "prompt_version": PROMPT_VERSION,
                "training_used": False,
                "thinking_enabled": False,
                "initial_passage_limit": INITIAL_PASSAGE_LIMIT,
                "initial_passage_retrieval_count": sum(
                    1
                    for entry in evidence["history"]
                    if entry["source"] == "initial_question"
                ),
                "passage_evidence_retrieval_count": evidence["retrieval_count"],
                "initial_passages": evidence["initial_passages"],
                "current_passages": evidence["current_passages"],
                "current_evidence_source": evidence["current_source"],
                "passage_evidence_history": evidence["history"],
                "decisions": self._policy_client.finish_sample(),
            }
            return result
        except Exception:
            # Huawei's evaluator records this as an infrastructure failure.
            self._policy_client.finish_sample()
            raise
        finally:
            self._initial_passage_retriever.finish_sample()
