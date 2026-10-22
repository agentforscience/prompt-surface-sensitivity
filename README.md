# Surface-Level Prompt Sensitivity (whitespace, casing, punctuation, JSON)

This project tests whether semantically neutral prompt edits change LLM behaviour. The edits are trailing whitespace, ALL-CAPS/lowercase, removed or altered punctuation, collapsed newlines, and JSON wrapping. Effects are measured on classification accuracy, output length, IFEval instruction compliance, and extraction F1. Ten local instruct models were run with vLLM: Qwen2.5 0.5B → 14B (bf16 and 4-bit AWQ), Llama-3.2-1B, Llama-3.1-8B and Gemma-2-9B. Each got 24,046 prompts with greedy decoding, and every effect is calibrated against the inference stack's own run-to-run noise. Gemma-9B has no IFEval results (its run was unfinished at the deadline).

## Key findings (full details in [REPORT.md](REPORT.md))
- **Format sensitivity is large for small models and plateaus at ~4 pp for 7–14B.** Classification accuracy spread across 10 meaning-preserving variants averages 14–21 pp at ≤1.2B, 5–7 pp at 1.5–3B, and 3.9–4.9 pp at 7–14B. The permutation null is ~1–3 pp. The hypothesised "3–8%" fits only the ≥1.5B models.
- **ALL-CAPS is the most harmful edit for small models** (16 of 48 significant accuracy changes, up to −27 pp) and is about neutral for ≥7B classification. It still lowers IFEval compliance at every size measured: −6 pp at 0.5B, −3 to −4.4 pp at 7–14B (significant at 7B and 14B). It shrinks only slightly with scale, so it isn't negligible at ≥7B.
- **Trailing whitespace does not lengthen outputs** (length ratio 0.94–1.07, n.s.). Llama-3/Gemma-2 chat templates strip it entirely. The "+15–25%" hypothesis is refuted.
- **JSON-wrapped prompts do not give +10–15% extraction accuracy.** JSON input mostly *hurts* ≤1.5B models (up to −13.5 F1) and is mixed at 7–14B (SQuAD up to +8.5; CoNLL down to −17.4). JSON *output* is task-dependent: +25 F1 on NER for Qwen-3B/7B, −7 to −14 F1 on SQuAD.
- **Sensitivity falls with scale, but not log-linearly.** Spread changes by −9.5 pp per 10× params (p = 0.004), mostly below 1.5B; Qwen-7B and Qwen-14B have the same spread (3.9 pp). Beyond ALL-CAPS, which edits help or hurt is idiosyncratic: r ≈ 0.2 between bf16 and AWQ copies of the same model, against r = 0.996 between replicate runs.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate
uv sync                      # or: uv pip install -r requirements.txt
export HF_HOME=/data/transformers   # models/datasets from the HF cache (HF_HUB_OFFLINE=1 is set by the scripts)
# CoNLL-2003 test parquet (not in the HF cache):
mkdir -p datasets/conll2003 && curl -L -o datasets/conll2003/test.parquet \
  https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/test/0.parquet

python src/build_prompts.py                       # -> data/prompts.jsonl (24,046 prompts)
python src/generate.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --name qwen1.5b-awq --gpu-gb 4.4
#   ...or queue all models on shared GPUs:  python src/gpu_queue.py logs/jobs.txt
python src/score.py qwen1.5b-awq                  # -> results/scored/qwen1.5b-awq.pkl
python src/analyze.py                             # -> results/analysis/*.csv, figures/*.png (~3 min)
python src/make_tables.py                         # -> results/report_tables.md
```
On a free GPU, one model takes ~4–20 min. NLTK `punkt_tab` (for the IFEval checkers) is read from `~/nltk_data`.

## Files
| path | content |
|---|---|
| `planning.md` | motivation, hypothesis decomposition, pre-registered analysis plan |
| `REPORT.md` | full results, statistics, discussion, limitations |
| `src/build_prompts.py` | datasets, templates, and the surface-variant transforms |
| `src/generate.py` | vLLM greedy generation (resumable; holds GPU memory against startup races) |
| `src/gpu_queue.py` | scheduler for shared GPUs (single-GPU or tensor-parallel alternatives) |
| `src/score.py` | label parsing, IFEval checks (`src/ifeval_lib/`, from lm-eval-harness), SQuAD/CoNLL F1 |
| `src/analyze.py` | McNemar/Holm, bootstrap, permutation nulls, scale fits, noise floor, figures |
| `src/make_tables.py` | renders the REPORT tables from the CSVs (`results/report_tables.md`) |
| `data/prompts.jsonl`, `data/prompt_examples.md` | every prompt used, plus rendered examples |
| `results/gen/` | raw generations (one JSONL per model, plus a chat-template example) |
| `results/analysis/` | all result tables (CSV); `results/analysis_stdout.txt` is the printed summary |
| `figures/` | heatmap, spread-by-task, scale, length, CAPS compliance, extraction plots |
| `logs/` | generation logs and queue history |
