# Surface-Level Prompt Sensitivity: How Whitespace, Casing, and Punctuation Alter LLM Outputs

## 1. Executive Summary

**Question.** When a prompt's meaning is held fixed, do edits to whitespace, casing, punctuation, or structure (JSON wrapping) change what an instruction-tuned LLM outputs, and does this sensitivity shrink with model size?

**Approach.** Ten open-weight instruct models were run locally with vLLM: Qwen2.5 0.5B → 14B (bf16 and/or AWQ-4bit), Llama-3.2-1B, Llama-3.1-8B and Gemma-2-9B. Decoding was greedy over 24,046 prompts per model:
- 4 classification tasks × 10 surface variants;
- 541 IFEval prompts × 6 variants;
- 2 extraction tasks under a 2×2 plain/JSON design.

That is about 235k generations in total. A replicate run and a built-in placebo condition measure the noise floor of the inference stack itself. *Gemma-9B did not finish its IFEval generations before the deadline, so Exp 2/3 cover 9 models.*

**Key findings.**
1. **Surface edits matter a lot for small models and modestly (3–5 pp) for 7–14B models.** Accuracy spread across 10 meaning-preserving variants (max − min) has a median of 5.8 pp over 40 model×task cells (IQR 2.4–11.9; range 1.0–28). It exceeds a permutation null in 27/40 cells. Models ≥7B average 3.9–4.9 pp, inside the hypothesised 3–8% band. ≤1.2B models average 14–21 pp, far above it. At n = 300–500, item sampling alone produces a 0.7–5.4 pp spread (mean 2.7 pp), so much of the ≥7B spread outside AG News is at noise level.
2. **ALL-CAPS is the dominant harmful edit for small models only.** It accounts for 16 of the 48 Holm-significant accuracy changes, with up to −27 pp (Qwen-0.5B-AWQ on SST-2). For ≥7B models its classification effect is about zero (−0.7 to +1.2 pp). Trailing whitespace is the least influential edit (mean |Δ| 0.5 pp).
3. **Trailing whitespace does not lengthen outputs.** Length ratios are 0.94–1.07, with no consistent direction and none significant. Hypothesis H2 (+15–25%) is refuted. Llama-3.x and Gemma-2 chat templates strip trailing whitespace entirely (`| trim`), so for them the edit is a no-op by construction.
4. **ALL-CAPS lowers IFEval compliance at every size tested, up to 14B.** The drop is −5.7 to −6.7 pp at 0.5B and −3.1 to −4.4 pp at 1.5–14.7B. It is significant at 7B (−4.4 pp, p = 0.02) and at 14B (−3.7 pp, p = 0.04), but not for Llama-8B (−3.1 pp). The effect shrinks slightly with scale (+1.6 pp per decade, p = 0.03), but it is not negligible at 7–14B. Lowercasing also hurts (−0.7 to −4.3 pp), so the cause is non-canonical casing rather than "shouting".
5. **JSON-structured prompts do not give +10–15% extraction accuracy across sizes.** Wrapping the *input* in JSON hurts most ≤1.5B models (−2 to −13.5 F1). It has mixed effects at 7–14B: SQuAD +8.5 (Llama-8B), +5.5 (Qwen-7B), +2.3 (Gemma), −0.5 (Qwen-14B); CoNLL −17.4 (Llama-8B), −4.5 (Qwen-14B). Requesting JSON *output* has large, task-dependent effects: +25 F1 on CoNLL NER for Qwen-3B/7B, but −7 to −14 F1 on SQuAD for the same models.
6. **Sensitivity decreases with scale, but plateaus.** Across 10 models, spread falls by −9.5 pp per decade of parameters (p = 0.004; Spearman ρ = −0.87) and the prediction flip rate by −9.2 pp per decade (p = 0.001). Most of the drop happens between 0.5B and 1.5B; Qwen-7B and Qwen-14B have the *same* mean spread (3.9 pp). The shape is a steep drop followed by a floor, not a clean log-linear decline.
7. **Which edits help or hurt is idiosyncratic.** Excluding uppercase, edit effects correlate only weakly between the bf16 and 4-bit AWQ versions of *the same model* (r = 0.28 at 0.5B, 0.19 at 1.5B), against r = 0.996 between two runs of the same model. Across models the correlations range from −0.90 to +0.77 with no family structure. For Llama-1B, ALL-CAPS *raised* classification accuracy by +7 pp.

**Practical implication.** For ≤3B models, prompt-format choices are as large a source of variance as many reported model or method differences; evaluate over several formats. Larger models (7–14B) are robust to most single edits on classification, but not to casing on instruction-following. Avoid ALL-CAPS templates at any size. Don't assume JSON wrapping helps: choose and validate the output format per task.

## 2. Research Question & Motivation

**Hypothesis under test.**
- (H1) Semantically identical formatting changes produce 3–8% accuracy variance on classification.
- (H2) Trailing whitespace increases output length by 15–25%.
- (H3) ALL-CAPS instructions reduce compliance by 5–12% in small models but have negligible effect on ≥7B models.
- (H4) JSON-structured prompts improve extraction accuracy by 10–15% across all sizes.
- (H5) Format sensitivity decreases log-linearly with model scale.

**Why it matters.** Practitioners and benchmark authors make "harmless" formatting choices all the time: template casing, separators, a stray trailing newline, wrapping inputs in JSON. If those choices shift accuracy by several points, comparisons between prompts or models are partly comparisons between formats.

**Gap.** No literature review was pre-gathered for this workspace; the resource-finder stage was skipped, so the context below is from background knowledge. Sclar et al. (2023, *FormatSpread*) showed large accuracy spreads over joint format choices in few-shot *completion* prompts and found that scale does not remove them. Mizrahi et al. (2024) argued for multi-prompt evaluation. Prior work mostly pools many format choices into one spread, uses completion-style rather than chat-template prompts, and does not measure verbosity or instruction compliance. This study attributes effects to individual atomic edits on chat-formatted prompts, measures four outcome types, and calibrates every effect against the inference stack's own run-to-run noise.

## 3. Experimental Setup

### Models (all local, vLLM 0.32.0; no API keys were available on this host)
| name | checkpoint | params | weights | Exp 2/3 (IFEval) |
|---|---|---|---|---|
| qwen0.5b | Qwen/Qwen2.5-0.5B-Instruct | 0.49B | bf16 | ✓ |
| qwen0.5b-awq | Qwen/Qwen2.5-0.5B-Instruct-AWQ | 0.49B | AWQ int4 | ✓ |
| llama1b | unsloth/Llama-3.2-1B-Instruct (ungated mirror) | 1.24B | bf16 | ✓ |
| qwen1.5b | Qwen/Qwen2.5-1.5B-Instruct | 1.54B | bf16 | ✓ |
| qwen1.5b-awq | Qwen/Qwen2.5-1.5B-Instruct-AWQ | 1.54B | AWQ int4 | ✓ |
| qwen3b-awq | Qwen/Qwen2.5-3B-Instruct-AWQ | 3.09B | AWQ int4 | ✓ |
| qwen7b-awq | Qwen/Qwen2.5-7B-Instruct-AWQ | 7.62B | AWQ int4 | ✓ |
| llama8b-awq | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (TP=2) | 8.03B | AWQ int4 | ✓ |
| gemma9b-awq | hugging-quants/gemma-2-9b-it-AWQ-INT4 | 9.24B | AWQ int4 | pending (88% of prompts done; IFEval chunk unfinished) |
| qwen14b-awq | Qwen/Qwen2.5-14B-Instruct-AWQ | 14.7B | AWQ int4 | ✓ |

The 4-bit Qwen ladder (0.5B → 14B) is the cleanest scale comparison because it holds family and quantization fixed. The bf16 0.5B/1.5B runs act as a quantization control.

**Decoding:** temperature 0 (greedy), seed 42. max_tokens is 10 (classification), 1024 (IFEval), 64 (SQuAD), 256 (CoNLL). max_model_len is 2048–2560 (the longest prompt plus generation is 1,605 tokens). Each model's default chat template and default system prompt are used, with a single user turn. Hardware: shared NVIDIA RTX A6000s (48 GB) with 4–12 GB slices per job; enforce_eager, prefix caching on.

### Tasks and conditions (`src/build_prompts.py`; rendered examples in `data/prompt_examples.md`)
**Exp 1, classification (H1/H5).** SST-2 validation (n=500), AG News test (500), Emotion test (500), and BoolQ validation (300; passages ≤200 words). The template is `instruction + "Field: {input}" + answer-format line + "Label:"`. There are 10 variants. Each edits the *template text only*; the data are inserted verbatim.

| variant | edit |
|---|---|
| base | clean template |
| trailing_space / trailing_newline | append `" "` / `"\n"` to the user message |
| double_space | every space in the template doubled |
| single_line | newlines in the template → spaces |
| lowercase / uppercase | template lower- / UPPER-cased |
| no_punct | `.,:;?!` removed from the template |
| exclaim | `.` → `!` |
| sep_dash | `:` → ` -` (field separators) |

The output is parsed as the first label word in the response (prefix match; e.g. "sport" → sports). Unparseable outputs count as wrong; parse rates are 87–100%.

**Exp 2/3, IFEval (H2/H3).** All 541 IFEval prompts under `base, trailing_space, trailing_newline, trailing_2newline, uppercase, lowercase`. Compliance uses the official IFEval checkers (vendored from lm-evaluation-harness into `src/ifeval_lib/`), checked against the original prompt's instruction arguments. Length is counted in generated tokens.

**Exp 4, extraction (H4).** SQuAD v1.1 validation (n=400; EM/F1 with the official normalisation) and CoNLL-2003 test (n=300 sentences with ≥1 entity; micro-F1 over (type, text) pairs). The design is 2×2: *input format* ∈ {labelled plain text, JSON object `{"task","passage","question","output_format"}`} × *output format* ∈ {plain answer, JSON answer}. The semantic content is identical across cells. A single lenient parser is used for every cell (JSON including fenced, string-valued, duplicate-key, or entity-list forms; plain `TYPE: text` lines), so differences reflect the model's answers.

### Statistics (`src/analyze.py`)
All comparisons are paired.
- Variant vs base: exact McNemar test with Holm correction over the 9 variants within each model×task, plus paired bootstrap 95% CIs (2,000 resamples).
- Spread: bootstrap CI, plus a permutation null (each item's outcomes shuffled across variants, 1,000 permutations), giving the spread expected with no format effect.
- Lengths: Wilcoxon signed-rank test; bootstrap CI on the ratio of means.
- Extraction: paired bootstrap and Wilcoxon on per-item F1 differences.
- Scale: OLS of per-model sensitivity on log10(params), plus Spearman ρ.
- Noise floor: (a) a full replicate run of qwen1.5b-awq; (b) placebos. For Llama and Gemma the trailing-whitespace variants are byte-identical after templating, so any difference is pure inference noise.

## 4. Results

Raw generations: `results/gen/*.jsonl`. Scored rows: `results/scored/*.pkl`. Tables: `results/analysis/*.csv`, rendered by `src/make_tables.py` into `results/report_tables.md`. Printed analysis: `results/analysis_stdout.txt`.

### 4.1 Exp 1: classification accuracy under surface edits (H1)

Mean accuracy change versus base (pp), averaged over the 4 tasks (`figures/cls_delta_heatmap.png`):

| variant | qwen0.5b | qwen0.5b-awq | llama1b | qwen1.5b | qwen1.5b-awq | qwen3b-awq | qwen7b-awq | llama8b-awq | gemma9b-awq | qwen14b-awq |
|---|---|---|---|---|---|---|---|---|---|---|
| trailing_space | -1.6 | -0.6 | +0.0 (placebo) | -0.4 | +0.2 | +0.1 | -0.1 | +0.1 (placebo) | +0.0 (placebo) | -0.2 |
| trailing_newline | -0.8 | -3.7 | +0.0 (placebo) | -0.2 | -0.5 | +0.1 | +0.4 | +0.1 (placebo) | +0.0 (placebo) | +0.1 |
| double_space | -4.0 | -5.8 | -5.3 | +0.5 | +0.5 | +1.1 | -0.1 | +1.1 | -0.6 | -0.7 |
| single_line | -0.1 | -5.1 | -3.7 | -1.1 | -0.7 | -0.0 | +0.4 | +0.1 | +0.4 | +0.8 |
| lowercase | +0.7 | +1.0 | +2.2 | -0.1 | -0.1 | +0.1 | +0.3 | -0.3 | +0.4 | +1.2 |
| uppercase | **-8.8** | **-17.8** | **+7.2** | **-3.6** | **-2.4** | **-4.3** | **-0.7** | **+0.5** | **-0.2** | **+1.2** |
| no_punct | -2.5 | -5.8 | +2.1 | +1.9 | -1.3 | -0.5 | +0.8 | -1.0 | -2.7 | +1.0 |
| exclaim | +3.2 | +0.6 | -0.1 | -0.2 | +0.1 | -0.7 | +0.5 | -0.3 | +0.0 | +0.5 |
| sep_dash | -1.6 | -2.4 | +1.8 | -0.5 | -0.3 | -0.1 | +0.6 | -0.1 | -0.3 | -0.3 |

Spread (max − min accuracy over the 10 variants), in pp, per model × task. The figure in parentheses is the permutation-null mean spread; ✱ marks perm-p < 0.05 (`figures/cls_spread_by_task.png`):

| model | SST-2 | AG News | Emotion | BoolQ | mean |
|---|---|---|---|---|---|
| qwen0.5b | 11.0 (2.2)✱ | 24.0 (4.0)✱ | 14.4 (4.1)✱ | 6.7 (4.3)✱ | 14.0 |
| qwen0.5b-awq | 28.0 (3.6)✱ | 22.4 (4.4)✱ | 26.4 (4.4)✱ | 8.3 (5.3)✱ | 21.3 |
| llama1b | 14.8 (2.7)✱ | 20.0 (3.1)✱ | 7.2 (3.6)✱ | 25.7 (5.4)✱ | 16.9 |
| qwen1.5b | 2.2 (1.3)✱ | 20.4 (3.2)✱ | 2.4 (2.9) | 2.3 (2.8) | 6.8 |
| qwen1.5b-awq | 1.2 (1.3) | 11.2 (2.8)✱ | 6.0 (3.1)✱ | 3.3 (3.0) | 5.4 |
| qwen3b-awq | 3.8 (1.5)✱ | 10.6 (2.4)✱ | 8.8 (2.9)✱ | 1.3 (2.3) | 6.1 |
| qwen7b-awq | 1.8 (1.2) | 9.8 (2.3)✱ | 2.0 (2.1) | 2.0 (1.7) | 3.9 |
| llama8b-awq | 1.0 (1.3) | 10.6 (3.1)✱ | 4.2 (2.5)✱ | 2.7 (2.3) | 4.6 |
| gemma9b-awq | 1.0 (0.7) | 14.2 (2.4)✱ | 2.8 (2.0) | 1.7 (1.3) | 4.9 |
| qwen14b-awq | 2.6 (1.2)✱ | 3.8 (2.0)✱ | 5.6 (2.5)✱ | 3.7 (2.4)✱ | 3.9 |

- Over the 40 cells: median spread 5.8 pp (IQR 2.4–11.9), 23% inside 3–8 pp, 42% above 8 pp, and 27/40 above the permutation null.
- Of 360 variant-vs-base tests, 48 are Holm-significant: uppercase 16, no_punct 9, single_line 6, double_space 4, lowercase 4, trailing_newline 4, exclaim 2, sep_dash 2, trailing_space 1. Mean prediction flip rate vs base: uppercase 17.6%, no_punct 14.7%, double_space 9.8%, trailing_space 1.6%.
- AG News is the most format-sensitive task (spread ≥9.8 pp for every model except Qwen-14B). Much of its swing comes from models answering an off-list label ("politics" for World, "sport", "news"), and how often they do that depends on the format.

**Verdict H1:** *Partially supported.* Format-induced accuracy differences are real and usually significant. The 3–8 pp figure describes 1.5B–14B models on average, but ≤1.2B models show 7–28 pp. For ≥7B models, spreads on SST-2, Emotion and BoolQ are mostly within ~2 pp of the permutation null. Their 3.9–4.9 pp averages are driven by AG News.

### 4.2 Exp 2: trailing whitespace and output length (H2)

Ratio of mean response length (tokens) with the variant vs base on IFEval, with 95% bootstrap CIs; ✱ = Wilcoxon p < 0.05 (`figures/length_change.png`):

| model | trailing_space | trailing_newline | trailing_2newline | (uppercase) |
|---|---|---|---|---|
| qwen0.5b | 0.95 [0.90, 1.00] | 0.94 [0.88, 1.00] | 0.95 [0.89, 1.01] | 1.03 |
| qwen0.5b-awq | 1.00 [0.95, 1.04] | 0.95 [0.90, 1.02] | 0.99 [0.93, 1.05] | 1.00 |
| llama1b (placebo: template trims) | 0.99 [0.96, 1.03] | 0.99 [0.96, 1.03] | 0.98 [0.95, 1.02] | 1.05✱ |
| qwen1.5b | 1.02 [0.96, 1.08] | 0.98 [0.92, 1.04] | 1.07 [1.01, 1.14] | 1.10✱ |
| qwen1.5b-awq | 0.96 [0.91, 1.00] | 0.97 [0.91, 1.03] | 0.99 [0.93, 1.06] | 1.01 |
| qwen3b-awq | 0.98 [0.93, 1.04] | 1.02 [0.96, 1.09] | 0.98 [0.92, 1.05] | 1.01✱ |
| qwen7b-awq | 1.00 [0.97, 1.02] | 1.01 [0.97, 1.05] | 1.02 [0.98, 1.06] | 1.05✱ |
| llama8b-awq (placebo: template trims) | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] | 0.99 [0.98, 1.00] | 1.03✱ |
| qwen14b-awq | 0.98 [0.95, 1.00] | 0.99 [0.96, 1.03] | 1.00 [0.96, 1.03] | 1.02✱ |

None of the 21 Qwen trailing-whitespace comparisons is significant by Wilcoxon (smallest p = 0.06). The ratios scatter on both sides of 1 within 0.94–1.07, comparable to the placebo (0.98–1.00) and replicate (0.985) ranges.

**Verdict H2:** *Refuted.* There is no +15–25% length increase from trailing whitespace; the largest point estimate is +7%. Incidentally, ALL-CAPS prompts make answers slightly *longer* (+1–10%; significant for 6 of 9 models).

### 4.3 Exp 3: ALL-CAPS and instruction compliance (H3)

IFEval prompt-level strict accuracy, change versus base, in pp with 95% CI; bold = McNemar p < 0.05 (uncorrected) (`figures/caps_compliance.png`):

| model | base | uppercase Δ | lowercase Δ | uppercase Δ excl. change_case prompts |
|---|---|---|---|---|
| qwen0.5b | 27.2 | **-5.7 [-9.8, -1.8]** | -3.9 | -7.2 |
| qwen0.5b-awq | 27.4 | **-6.7 [-10.7, -2.6]** | -4.3 | -8.3 |
| llama1b | 47.9 | **-4.4 [-8.1, -0.4]** | -0.7 | -5.9 |
| qwen1.5b | 41.4 | -3.9 [-7.9, -0.2] | -3.3 | -5.9 |
| qwen1.5b-awq | 37.5 | -3.7 [-7.4, 0.2] | -3.3 | -4.4 |
| qwen3b-awq | 58.2 | -3.3 [-7.6, 0.9] | -3.5 | -4.6 |
| qwen7b-awq | 71.7 | **-4.4 [-8.1, -0.9]** | -2.8 | -4.2 |
| llama8b-awq | 74.1 | -3.1 [-6.7, 0.6] | -2.2 | -2.6 |
| qwen14b-awq | 77.8 | **-3.7 [-7.2, -0.4]** | -2.0 | -5.0 |

Scale trend across these 9 models: +1.6 pp per decade (p = 0.028; Spearman ρ = 0.79). The trend is not significant within the Qwen-AWQ ladder alone (n = 5, p = 0.23). Noise reference: byte-identical inputs moved strict accuracy by +1.5 to +2.8 pp (Llama-1B placebo), −0.6 to −1.1 pp (Llama-8B placebo), and −0.7 pp (Qwen-1.5B replicate).

**Verdict H3:** *Partially supported.* ALL-CAPS consistently reduces compliance: all 9 point estimates are negative (5 significant, including Qwen-14B), and excluding prompts whose instructions concern letter case leaves them similar or larger. The effect shrinks only slightly with scale. Only the 0.5B models reach the predicted 5–12 pp; at 7–14B it is −3 to −4.4 pp, which is not negligible (significant at 7B and 14B), though close to the ~1–3 pp noise floor. Lowercase hurts too, though somewhat less.

### 4.4 Exp 4: JSON-structured prompts and extraction (H4)

F1 (×100) for each input/output format cell (`figures/extraction_formats.png`):

| model | SQuAD p/p | SQuAD json-in | SQuAD json-out | SQuAD json/json | CoNLL p/p | CoNLL json-in | CoNLL json-out | CoNLL json/json |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b | 59.7 | 57.2 | 70.9 | 64.6 | 17.1 | 11.9 | 16.1 | 10.2 |
| qwen0.5b-awq | 49.9 | 41.0 | 59.9 | 41.9 | 10.7 | 8.8 | 15.1 | 12.2 |
| llama1b | 61.6 | 53.7 | 69.1 | 65.7 | 21.5 | 6.1 | 21.6 | 9.9 |
| qwen1.5b | 82.4 | 81.8 | 84.5 | 81.8 | 26.6 | 28.2 | 38.0 | 32.5 |
| qwen1.5b-awq | 81.4 | 81.3 | 81.8 | 78.4 | 26.5 | 28.3 | 25.9 | 26.6 |
| qwen3b-awq | 80.6 | 79.8 | 73.2 | 70.0 | 1.4 | 6.4 | 27.1 | 27.1 |
| qwen7b-awq | 82.2 | 85.1 | 68.0 | 76.1 | 27.8 | 24.7 | 52.6 | 48.8 |
| llama8b-awq | 64.8 | 75.4 | 58.8 | 65.2 | 40.3 | 13.2 | 52.6 | 45.0 |
| gemma9b-awq | 81.4 | 81.6 | 79.3 | 83.6 | 51.4 | 50.9 | 50.9 | 53.1 |
| qwen14b-awq | 87.2 | 87.4 | 88.1 | 87.0 | 44.8 | 36.8 | 51.2 | 50.3 |

(p/p = plain input and plain output; json-in = JSON input, plain output; json-out = plain input, JSON output.)

The JSON-input columns give the main effect averaged over output formats, with 95% CI; bold = CI > 0. The JSON-output columns are the effect with plain input:

| model | JSON-input SQuAD | JSON-input CoNLL | JSON-output SQuAD | JSON-output CoNLL |
|---|---|---|---|---|
| qwen0.5b | -4.4 [-6.7, -2.1] | -5.5 [-7.4, -3.7] | +11.2 | -1.1 |
| qwen0.5b-awq | -13.5 [-16.5, -10.4] | -2.4 [-3.5, -1.2] | +10.1 | +4.4 |
| llama1b | -5.7 [-8.5, -2.9] | -13.5 [-15.7, -11.2] | +7.5 | +0.1 |
| qwen1.5b | -1.6 [-3.3, -0.1] | -2.0 [-4.1, 0.0] | +2.2 | +11.4 |
| qwen1.5b-awq | -1.7 [-3.7, 0.1] | +1.2 [-0.4, 2.8] | +0.5 | -0.6 |
| qwen3b-awq | -2.0 [-4.1, -0.0] | **+2.5 [0.7, 4.4]** | -7.4 | +25.7 |
| qwen7b-awq | **+5.5 [3.8, 7.4]** | -3.4 [-5.4, -1.4] | -14.2 | +24.8 |
| llama8b-awq | **+8.5 [6.2, 10.9]** | -17.4 [-19.8, -14.8] | -5.9 | +12.3 |
| gemma9b-awq | **+2.3 [1.0, 3.5]** | +0.8 [-1.0, 2.5] | -2.1 | -0.6 |
| qwen14b-awq | -0.5 [-1.5, 0.7] | -4.5 [-6.5, -2.5] | +0.9 | +6.4 |

Requested JSON was produced at 86–100% validity.

**Verdict H4:** *Refuted as stated.* No model gains 10–15 points from a JSON-structured prompt on both tasks. Gains appear only for SQuAD at 7–9B (+2 to +8.5), and the same models often lose on CoNLL (Llama-8B −17.4). Small models are mostly hurt. JSON *output* formatting has large effects whose sign depends on the task and model: structured output helps multi-entity NER and can badly constrain short-span QA. The strongest single case is Qwen-3B, which answered "NONE" to 97% of CoNLL sentences under the plain-list output instruction (F1 1.4) but reached F1 27.1 when asked for JSON.

### 4.5 Exp 5: sensitivity vs scale (H5)

OLS of per-model sensitivity metrics on log10(params) (`figures/scale_sensitivity.png`, `results/analysis/scale_fits.csv`):

| metric | subset (n) | slope per 10× params | p | Spearman ρ |
|---|---|---|---|---|
| classification spread | all (10) | −9.5 pp | 0.004 | −0.87 |
| classification spread | Qwen-AWQ ladder 0.5→14B (5) | −10.6 pp | 0.084 | −0.80 |
| prediction flip rate | all (10) | −9.2 pp | 0.001 | −0.93 |
| prediction flip rate | Qwen-AWQ ladder (5) | −10.2 pp | 0.051 | −0.90 |
| mean abs. accuracy Δ | all (10) | −2.2 pp | 0.012 | −0.84 |
| CAPS compliance Δ | all with IFEval (9) | +1.6 pp | 0.028 | +0.79 |
| trailing-ws length abs(log ratio) | Qwen (7) | −2.1% | 0.061 | −0.80 |
| JSON-input F1 Δ | all (10) | +4.1 pp | 0.083 | +0.60 |

Qwen-AWQ ladder mean spreads: 21.3 (0.5B) → 5.4 (1.5B) → 6.1 (3B) → 3.9 (7B) → 3.9 (14B) pp. Flip rates: 19.7 → 7.2 → 5.3 → 3.7 → 3.8%.

**Verdict H5:** *Directionally supported for classification and casing; the functional form is not log-linear.* Sensitivity falls with size (rank correlations −0.84 to −0.93), but the decline is concentrated below ~1.5B, and Qwen-7B and Qwen-14B are indistinguishable. A floor of about 4 pp spread and 4% flips remains, part of which is item-level noise (the null spread for these models is 0.7–3.1 pp). Part of the decrease is also a ceiling effect: base accuracy rises from 52–63% to 75–81%.

### 4.6 Noise floor and idiosyncrasy

| check | identical outputs | outcome difference |
|---|---|---|
| Qwen-1.5B-AWQ replicate run, classification | 99.7% | max cell abs(Δacc) 0.4 pp; variant-effect correlation r = 0.996 |
| Qwen-1.5B-AWQ replicate run, IFEval | 76% | base length ratio 0.985; strict acc. −0.7 pp |
| Llama-1B placebo (template trims), classification / IFEval | 99.6% / 23–31% | max abs(Δacc) 0.4 pp / length 0.98–0.99, strict +1.5 to +2.8 pp |
| Llama-8B placebo, classification / IFEval | 99.8–99.9% / 75–76% | max abs(Δacc) 0.4 pp / length 0.99–1.00, strict −0.6 to −1.1 pp |
| Gemma-9B placebo, classification | 99.7% | 0.0 pp |

Greedy decoding in vLLM is not bitwise deterministic across batch compositions. Short classification outputs are almost unaffected, but long generations diverge often, and that alone moves IFEval compliance by up to ~3 pp. IFEval effects of ≲3 pp therefore cannot be separated from inference noise. The classification and extraction effects reported above are far above this floor.

Cross-model correlation of per-cell edit effects, excluding uppercase (`results/analysis/effect_corr_*.csv`): bf16 vs AWQ of the same model gives r = 0.28 (0.5B) and 0.19 (1.5B), against r = 0.996 for replicate runs. Other pairs range from −0.90 to +0.77 (e.g. Llama-8B vs Gemma-9B +0.77; Gemma-9B vs Qwen-1.5B −0.90). The ≥7B effects are small, though, so these correlations are noisy. Apart from the ALL-CAPS penalty in small Qwen models, *which* formats help or hurt behaves like model-specific noise, and even 4-bit quantization of the same weights reshuffles it.

## 5. Analysis & Discussion

- **What is robust:**
  - ALL-CAPS templates are the most harmful common formatting choice. They hurt classification in small Qwen models and IFEval compliance in every model measured (0.5–14.7B). Llama-1B, where they *help* classification, illustrates the idiosyncrasy.
  - Small models (≤1.2B) are highly format-sensitive: 7–28 pp spreads and 12–20% of predictions flipped.
  - Trailing whitespace is nearly harmless and does not change verbosity.
  - Larger models (7–14B) are robust to most single edits on classification (flip rates ≈3–5%).
- **What the hypothesis got wrong:** The magnitudes in H2 and H4, and the claim that ≥7B models are immune to casing effects on compliance. The length hypothesis may come from raw completion-mode prompting, where a trailing space changes BPE tokenization at the generation boundary. In chat templates the whitespace sits *before* the end-of-turn token, and two of the three families strip it anyway.
- **Mechanisms visible in errors:**
  - *Label-vocabulary drift:* under some formats, models answer with off-list labels ("politics", "sport", "news"), which drives the AG News spreads.
  - *Degenerate abstention:* Qwen-3B answered "NONE" to 97% of CoNLL sentences under the plain-list output instruction but extracted entities when asked for JSON.
  - *Over-structuring:* with JSON output on SQuAD, Qwen-3B/7B return longer, paraphrased spans inside `{"answer": ...}`, which lowers F1.
  - *JSON input confusing NER:* Llama-1B and Llama-8B lose 13–17 F1 on CoNLL when the sentence is a JSON string value.
- **Quantization interacts with format sensitivity:** 4-bit AWQ at 0.5B raised the mean spread from 14 to 21 pp. At 1.5B there was no difference (6.8 vs 5.4 pp).
- **Alternative explanations considered:**
  - *Parser artifacts:* one lenient parser is shared across conditions; parse rates are reported; failures were inspected and two parser gaps were fixed before final scoring ("sport" synonym; string-valued and list-of-objects JSON).
  - *Item noise:* addressed with paired tests and permutation nulls.
  - *Inference nondeterminism:* measured directly (§4.6).
  - *Ceiling and floor effects:* base accuracy rises with scale, which partly confounds H5.

## 6. Limitations

- **Scale range and completeness.** The largest model is 14.7B (4-bit). Gemma-9B lacks IFEval results (Exp 2/3) because its final generation chunk had not finished at the reporting deadline. The ≥7B part of H3 rests on Qwen-7B, Llama-8B and Qwen-14B. The work ran on GPUs shared with other users' jobs, which hold 40–45 GB per card, and most of the wall-clock time went to waiting for memory.
- **Model families.** Qwen2.5 (7 runs), Llama-3.x (2) and Gemma-2 (1); no API models (no keys on the host). Hosted APIs may strip or normalise whitespace server-side.
- **Quantization.** All models ≥3B are AWQ-4bit. The 0.5B/1.5B bf16 controls show that quantization changes *which* edits matter.
- **Single base template per task, single edits.** Effects are relative to one base template; a different anchor could change the deltas. Interactions between edits are not studied.
- **Sample sizes.** n = 300–500 per classification cell gives ~±4 pp CIs on deltas. IFEval (n = 541) effects near 3 pp sit at the inference-noise floor.
- **Multiple comparisons.** Holm correction is applied within each classification model×task family. IFEval and extraction p-values are uncorrected; read them alongside the CIs.
- **Contamination.** All benchmarks are public and probably in training data. This affects absolute accuracy more than within-item format contrasts.

## 7. Conclusions & Next Steps

Surface formatting changes LLM behaviour well beyond noise, but not in the uniform way the hypothesis describes. The effect is very large below ~1.5B parameters and plateaus at a small but nonzero level (about 4 pp spread) from 7B to 14B. ALL-CAPS dominates for small models and still costs 3–4 pp of instruction compliance at 7–14B. Most other edits have model-specific signs. Of the five sub-hypotheses, H1 and H3 are partially supported, H5 is directionally supported but not log-linear (it plateaus), and H2 (trailing whitespace → longer outputs) and H4 (JSON prompts → +10–15% extraction) are refuted.

Recommended follow-ups:
1. Finish IFEval for Gemma-9B and add ≥30B models to test whether the casing compliance penalty disappears.
2. Repeat Exp 2 in raw completion mode, where trailing whitespace changes tokenization at the generation boundary.
3. Randomise the base template (FormatSpread-style) to separate the effects of edits from the choice of anchor.
4. Test API models, including whether providers normalise whitespace.
5. Use batch-invariant or deterministic inference kernels to remove the ~3 pp IFEval noise floor.

## References

- Sclar, M., Choi, Y., Tsvetkov, Y., Suhr, A. (2023). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design* (FormatSpread). arXiv:2310.11324.
- Mizrahi, M., et al. (2024). *State of What Art? A Call for Multi-Prompt LLM Evaluation*. TACL.
- Zhou, J., et al. (2023). *Instruction-Following Evaluation for Large Language Models* (IFEval). arXiv:2311.07911. Checker code from EleutherAI lm-evaluation-harness (Apache-2.0).
- Datasets: SST-2 (Socher et al., 2013), AG News (Zhang et al., 2015), Emotion (Saravia et al., 2018), BoolQ (Clark et al., 2019), SQuAD v1.1 (Rajpurkar et al., 2016), CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003; tner/conll2003 parquet).
- Models: Qwen2.5 (Qwen Team, 2024); Llama 3.1/3.2 (Meta, 2024); Gemma 2 (Google, 2024). Inference: vLLM 0.32.0 (Kwon et al., 2023).

## Appendix: Reproducibility

- Environment: Python 3.12.12, vllm 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0 (`pyproject.toml`, `requirements.txt`).
- Cost: no API spend. GPU time was about 4–50 min per model for 24k prompts on 4–12 GB slices (contended).
- Seeds: dataset sampling uses `random_state=42`; vLLM `seed=42`; bootstrap and permutation use `numpy.default_rng(42)`. Greedy decoding is still subject to batch-dependent numerics (§4.6).
- Shared-GPU engineering (`src/generate.py`, `src/gpu_queue.py`): jobs reserve GPU memory as soon as it is free and hand it to vLLM just before vLLM measures free memory. For TP=1 this is an in-process tensor; for TP=2, per-GPU holder processes are released by forked workers. Without this, launches lost the memory race to other jobs about 90% of the time.
- Exact prompts: `data/prompts.jsonl` (24,046 records incl. metadata); examples: `data/prompt_examples.md`; one chat-templated rendering per model: `results/gen/<model>_template_example.txt`.
