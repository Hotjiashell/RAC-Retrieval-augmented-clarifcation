"""RAC clarification policy adapted to Huawei ClarQ's native action protocol.

RAC's original implementation trains a generator with passage-grounded
clarification examples. This module uses that inference-time idea directly:
the policy receives Top-5 passages from the current retrieval state and may ask
a clarification only when it is grounded in that evidence.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping, Sequence

from clarq_eval.clients import response_content
from clarq_eval.parsing import PolicyProtocolError, parse_policy_response
from clarq_eval.runner import serialize_search_results


POLICY_NAME = "rac-clarq"
POLICY_VERSION = "1.2"
PROMPT_VERSION = "2026-09-01-native-tools-complete-user-language"
INITIAL_PASSAGE_LIMIT = 5


RAC_SYSTEM_PROMPT = """You are a retrieval-augmented clarification policy for a conversational case-retrieval agent.

You receive the original user question, a conversation/tool trace, and the current set of passage evidence. The first evidence set was retrieved once from the original user question before this conversation started. Every later successful search_case refreshes the current evidence set with the new retrieval results. Passages are evidence for asking clarifying questions; they are not user-confirmed facts.

Use exactly one action at a time:
- Call clarify_user only when one missing user-known fact would materially change the relevant cases. Ask one concise, discriminative question. Every specific entity, option, condition, or distinction in that question must be supported by the current passages. Do not introduce assumptions or facts that are absent from them. Do not repeat an answered question. Use the same language as the user; for example, use Chinese when the user speaks Chinese.
- Call search_case when the request is specific enough. Its query may use only the original request and confirmed user replies, never an inferred fact from a passage. Use the same language as the user; for example, use Chinese when the user speaks Chinese.
- Call Complete only after at least one search_case result is available and the latest cases are sufficient. Do not answer the technical question yourself.

Make exactly one native tool call using the supplied tool definitions. This includes Complete: call `Complete` with an empty arguments object when the latest retrieved cases are sufficient. Leave assistant text empty. Do not write a tool-call JSON object or terminal action text in assistant content."""


def _compact_text(value: Any, *, default: str = "", limit: int = 4_000) -> str:
    if not isinstance(value, str):
        return default
    return " ".join(value.split())[:limit]


def _tool_summary(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    calls: list[dict[str, Any]] = []
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append(
            {
                "name": _compact_text(function.get("name"), limit=80),
                "arguments": arguments,
            }
        )
    return calls


def _trace_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only state the policy is allowed to see from Huawei's trace."""

    trace: list[dict[str, Any]] = []
    for message in messages:
        role = _compact_text(message.get("role"), limit=40)
        if role == "system":
            # RAC_SYSTEM_PROMPT replaces the evaluator's generic system prompt.
            continue
        entry: dict[str, Any] = {"role": role or "unknown"}
        content = message.get("content")
        if content is not None:
            entry["content"] = str(content)
        tool_calls = _tool_summary(message.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
        tool_name = _compact_text(message.get("name"), limit=80)
        if tool_name:
            entry["tool_name"] = tool_name
        trace.append(entry)
    return trace


def _normalized_model_response(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize Qwen chat and raw-completion responses for the shared parser."""

    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise PolicyProtocolError("RAC policy response has no choices[0]") from error
    if not isinstance(choice, Mapping):
        raise PolicyProtocolError("RAC policy response choices[0] must be an object")
    if isinstance(choice.get("message"), Mapping):
        return response
    return {
        "choices": [
            {
                "finish_reason": choice.get("finish_reason"),
                "message": {"content": response_content(response), "tool_calls": []},
            }
        ]
    }


def _normalize_tool_action(name: Any, arguments: Any) -> tuple[str, dict[str, str]]:
    action = _compact_text(name, limit=80)
    if not isinstance(arguments, Mapping):
        raise PolicyProtocolError(f"RAC {action} arguments must be a JSON object")

    if action.lower() == "complete":
        if arguments:
            raise PolicyProtocolError("RAC Complete must use empty action.arguments")
        return "Complete", {}
    if action not in {"clarify_user", "search_case"}:
        raise PolicyProtocolError(f"Unsupported RAC action: {action!r}")

    field = "question" if action == "clarify_user" else "query"
    limit = 1_000 if field == "question" else 2_000
    value = _compact_text(arguments.get(field), limit=limit)
    if not value:
        raise PolicyProtocolError(f"RAC {action} requires a non-empty arguments.{field}")
    return action, {field: value}


class RACInitialPassageRetriever:
    """Capture initial evidence and refresh it after each executed search.

    ``EvaluationRunner`` already makes the baseline retrieval before its first
    policy call. The wrapper records its first five cases as the initial
    evidence set per worker thread. Each later retriever call made by an
    executed ``search_case`` replaces the policy's current evidence with the
    first five new cases. No extra retrieval is introduced by the wrapper.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        content_chars: int,
        passage_limit: int = INITIAL_PASSAGE_LIMIT,
    ):
        if passage_limit <= 0:
            raise ValueError("passage_limit must be positive")
        self._retriever = retriever
        self._content_chars = content_chars
        self._passage_limit = passage_limit
        self._local = threading.local()

    def begin_sample(self, sample: Any) -> None:
        self._local.initial_question = str(getattr(sample, "initial_question", ""))
        self._local.initial_passages = None
        self._local.current_passages = None
        self._local.evidence_history = []

    def finish_sample(self) -> None:
        for attribute in (
            "initial_question",
            "initial_passages",
            "current_passages",
            "evidence_history",
        ):
            if hasattr(self._local, attribute):
                delattr(self._local, attribute)

    def search(self, query: str, max_results: int | None = None) -> list[Any]:
        results = list(self._retriever.search(query, max_results))
        initial_question = getattr(self._local, "initial_question", None)
        if initial_question is not None:
            passages = serialize_search_results(results[: self._passage_limit], self._content_chars)
            initial_passages = getattr(self._local, "initial_passages", None)
            source = "initial_question" if initial_passages is None else "search_case"
            if initial_passages is None:
                if query != initial_question:
                    raise RuntimeError(
                        "RAC evidence must begin with the original-question baseline retrieval"
                    )
                self._local.initial_passages = passages
            self._local.current_passages = passages
            self._local.evidence_history.append(
                {
                    "source": source,
                    "query": query,
                    "passage_count": len(passages),
                    "passages": passages,
                }
            )
        return results

    def snapshot(self) -> tuple[list[dict[str, Any]], int, str]:
        passages = getattr(self._local, "current_passages", None)
        if passages is None:
            raise RuntimeError(
                "RAC passage evidence is unavailable: the original-question baseline retrieval "
                "must run before the first policy call"
            )
        history = getattr(self._local, "evidence_history", [])
        source = str(history[-1].get("source") if history else "initial_question")
        return [dict(passage) for passage in passages], len(history), source

    def audit_snapshot(self) -> dict[str, Any]:
        current_passages, retrieval_count, current_source = self.snapshot()
        initial_passages = getattr(self._local, "initial_passages", None)
        history = getattr(self._local, "evidence_history", [])
        return {
            "initial_passages": [dict(passage) for passage in initial_passages or []],
            "current_passages": current_passages,
            "current_source": current_source,
            "retrieval_count": retrieval_count,
            "history": [
                {
                    "source": entry["source"],
                    "query": entry["query"],
                    "passage_count": entry["passage_count"],
                    "passages": [dict(passage) for passage in entry["passages"]],
                }
                for entry in history
            ],
        }


class RACPolicyClient:
    """Render RAC's fixed-passage prompt through Huawei's API policy client."""

    def __init__(self, client: Any, initial_passage_retriever: RACInitialPassageRetriever):
        self.client = client
        self._initial_passage_retriever = initial_passage_retriever
        self._call_lock = threading.Lock()
        self._next_call_number = 0
        self._local = threading.local()

    def begin_sample(self, sample: Any) -> None:
        self._local.sample_id = str(getattr(sample, "sample_id", ""))
        self._local.decisions = []

    def finish_sample(self) -> list[dict[str, Any]]:
        decisions = getattr(self._local, "decisions", [])
        self._local.decisions = []
        return [dict(item) for item in decisions]

    def policy_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        model_mode: str = "qwen3_5",
        tokenizer_path: str | None = None,
        enable_thinking: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if not tools:
            raise ValueError("RAC ClarQ requires Huawei's native action tool definitions")

        passages, retrieval_count, evidence_source = self._initial_passage_retriever.snapshot()
        prompt_messages = [
            {"role": "system", "content": RAC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._prompt_content(messages, passages, retrieval_count, evidence_source),
            },
        ]
        response = self.client.policy_chat(
            prompt_messages,
            tools=tools,
            model_mode=model_mode,
            tokenizer_path=tokenizer_path,
            # RAC ClarQ never exposes a thinking trace. Keep this hard-coded so
            # a caller cannot accidentally re-enable it through evaluator flags.
            enable_thinking=False,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        parsed = parse_policy_response(_normalized_model_response(response))
        if len(parsed.tool_calls) != 1:
            raise PolicyProtocolError("RAC policy response must contain exactly one native tool call")
        if parsed.cleaned_content:
            raise PolicyProtocolError("RAC tool-call responses must not include assistant text")
        call = parsed.tool_calls[0]
        action, arguments = _normalize_tool_action(call.name, call.arguments)
        self._record_decision(action, arguments, len(passages), evidence_source)
        return self._tool_response(action, arguments)

    def _prompt_content(
        self,
        messages: Sequence[Mapping[str, Any]],
        passages: list[dict[str, Any]],
        retrieval_count: int,
        evidence_source: str,
    ) -> str:
        trace = _trace_from_messages(messages)
        initial_question = next(
            (
                str(message.get("content") or "")
                for message in messages
                if message.get("role") == "user"
            ),
            "",
        )
        evidence = [
            {
                "rank": passage.get("rank"),
                "title": passage.get("title"),
                "content": passage.get("content"),
            }
            for passage in passages
        ]
        return (
            "Original user question:\n"
            + initial_question
            + "\n\nCurrent passage evidence ("
            + f"source={evidence_source}; {len(evidence)} of top-{INITIAL_PASSAGE_LIMIT} passages retained; "
            + f"retrieval_count={retrieval_count}):\n"
            + json.dumps(evidence, ensure_ascii=False)
            + "\n\nConversation and tool trace:\n"
            + json.dumps(trace, ensure_ascii=False)
        )

    def _tool_response(self, action: str, arguments: dict[str, str]) -> dict[str, Any]:
        with self._call_lock:
            self._next_call_number += 1
            call_id = f"rac_call_{self._next_call_number}"
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": action, "arguments": arguments},
                            }
                        ],
                    },
                }
            ]
        }

    def _record_decision(
        self,
        action: str,
        arguments: dict[str, str],
        passage_count: int,
        evidence_source: str,
    ) -> None:
        decisions = getattr(self._local, "decisions", None)
        if decisions is None:
            decisions = []
            self._local.decisions = decisions
        decisions.append(
            {
                "turn": len(decisions) + 1,
                "passage_count": passage_count,
                "evidence_source": evidence_source,
                "action": {"name": action, "arguments": dict(arguments)},
            }
        )
