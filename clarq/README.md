# RAC ClarQ Policy Evaluation

This integration applies RAC as an inference-time clarification strategy to
Huawei ClarQ. It uses Huawei's existing ClarQ test data, user simulator,
retriever, trajectory loop, Success Judge, and reports. RAC's SFT, noisy-data
generation, and DPO training scripts are not used.

For each trajectory, Huawei's `EvaluationRunner` performs its normal baseline
retrieval before the first policy call. The RAC adapter keeps its first five
results as the initial passage evidence. Every executed later `search_case`
refreshes the next policy turn's evidence with the first five newly retrieved
cases. Therefore the adapter introduces no extra retrieval: it consumes the
same baseline and agent searches that Huawei already executes.

The policy uses the evaluator's OpenAI-compatible API client. `qwen3_5` sends
native tools to `/chat/completions`; `qwen3` renders the tokenizer template and
sends it to `/completions`. In both cases RAC enforces
`enable_thinking=false`. `--policy-enable-thinking` is rejected.

## Setup

Install the Huawei evaluator requirements in the environment used to execute
this adapter, then configure the external services:

```bash
python3 -m pip install -r ../../huawei_dial/workspace/eval/requirements.txt
cp config.example.env .env
```

Set the policy API endpoint/model, the Huawei user simulator endpoint/model,
and the existing Elasticsearch/embedding settings in `.env`. The policy must
support Huawei's native `clarify_user`, `search_case`, and `Complete` function
calls. `Complete` must be a native tool call with an empty arguments object.

## Run

Run a small integration check first:

```bash
./run_evaluation.sh --check-only
./run_evaluation.sh --limit 20 --workers 4 --output-dir outputs/rac-smoke
```

All Huawei evaluator flags are accepted. The default data is
`huawei_dial/workspace/ClarQ/profile_split/test`, and outputs retain Huawei's
standard `trajectories.jsonl`, metrics, and report files. Each successful
trajectory adds `rac_policy` metadata, including initial/current passages, the
evidence refresh history, and normalized actions for auditability.
