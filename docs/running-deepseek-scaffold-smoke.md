# Running DeepSeek Scaffold Smoke

This document explains how to run the bounded `DeepSeek` scaffold smoke path inside the project Docker container.

This is **not** real `DeepSeek-V2-Lite` pretrained inference.

It is a bring-up checkpoint for the current scaffold path:

- structured scaffold loading
- prompt prefill
- iterative greedy decode
- contiguous vs paged MLA generation parity
- DeepSeek-style MLA projection aliases in the bounded loader path
- bounded MoE scaffold loading and execution

## Recommended Command

From the host machine, run:

```bash
scripts/docker/run-deepseek-scaffold-smoke.sh
```

The wrapper defaults to:

- Docker container: `vattn-$USER`
- smoke mode: `compare`
- parity enforcement: `--require-match`

So the command will exit non-zero if contiguous and paged scaffold generation diverge.

## Alternate Modes

To run only the contiguous scaffold path:

```bash
scripts/docker/run-deepseek-scaffold-smoke.sh contiguous
```

To run only the paged scaffold path:

```bash
scripts/docker/run-deepseek-scaffold-smoke.sh paged
```

To exercise the most realistic current synthetic HF-style surface:

```bash
scripts/docker/run-deepseek-scaffold-smoke.sh compare \
  --checkpoint-layout hf_dir \
  --query-mode q_lora \
  --mlp-mode moe
```

To keep the emitted synthetic checkpoint directory on disk for separate inspection:

```bash
scripts/docker/run-deepseek-scaffold-smoke.sh compare \
  --checkpoint-layout hf_dir \
  --query-mode q_lora \
  --mlp-mode moe \
  --output-dir /workspace/tmp/deepseek-v2-lite-scaffold
```

Then inspect the emitted directory directly inside the container:

```bash
docker exec -w /workspace vattn-$USER \
  python scripts/inspect_deepseek_checkpoint.py /workspace/tmp/deepseek-v2-lite-scaffold
```

## Expected Output

The script prints a JSON summary including:

- prompt token IDs
- generated token IDs
- final logits shape
- cache token counts

In `compare` mode it also prints:

- whether generated tokens match
- whether final logits match
- whether cache token counts match
- or a `blocked` status plus the runtime error if the real paged path cannot execute

## What This Validates

This smoke path is currently meant to validate Phase `7f` scaffold bring-up work:

- the scaffold can run prefill + decode in-container
- the paged MLA wrapper path can produce the same greedy-generation result as the contiguous path

It does **not** validate:

- real `DeepSeek-V2-Lite` pretrained weight loading
- full real DeepSeek MoE semantics
- full production inference quality

## Interpreting A Blocked Compare Run

If `compare` mode exits non-zero and reports a blocked status, that means the harness reached a real runtime limitation in the current paged MLA path.

That is still useful:

- contiguous scaffold generation is working
- the runtime now fails at a concrete wrapper / kernel compatibility boundary

At that point, the next work should focus on the paged MLA runtime path rather than the scaffold harness itself.
