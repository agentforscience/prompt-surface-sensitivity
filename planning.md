# Planning: Surface-Level Prompt Sensitivity

## Motivation & Novelty Assessment

### Why This Research Matters
Practitioners routinely edit prompts in ways they consider meaning-preserving: an accidental trailing space, a template written in ALL CAPS "for emphasis", lowercased instructions, stripped punctuation, or wrapping the input in JSON. If such edits move accuracy by several points, then (a) benchmark comparisons between models or prompts are partly noise, and (b) production prompts are fragile in ways that are hard to debug. Knowing *which* surface edits matter, *how much*, and *whether scale fixes it* tells practitioners where robustness checks are needed.

### Gap in Existing Work
(No literature review was pre-gathered for this workspace; the resource-finder stage was skipped. Summary from background knowledge.) Sclar et al. 2023 ("Quantifying LMs' Sensitivity to Spurious Features in Prompt Design", FormatSpread) showed accuracy spreads of up to 76 points across separator, casing, and spacing choices in few-shot *completion* prompts, and found that larger models are not reliably more robust. Mizrahi et al. 2024 (multi-prompt evaluation) and the PromptBench / "prompt brittleness" papers report similar effects for paraphrases. Gaps: (1) most work studies few-shot completion-style prompts, not chat templates as deployed; (2) effects are usually pooled into one "format spread", not attributed to individual edit types (whitespace vs. casing vs. punctuation); (3) claims about *output length* (trailing whitespace) and *instruction compliance* (ALL CAPS) are folklore rather than measured; (4) scale trends are seldom tested inside one model family where size is the main thing that changes.

### Our Novel Contribution
A controlled study inside one model family (Qwen2.5-Instruct, 0.5B to 14B), with cross-family checks (Llama-3.x) where GPU memory allows. Each atomic surface edit is applied on its own, to a chat-template prompt, and its effect is measured on (i) classification accuracy, (ii) response length, (iii) IFEval instruction compliance, and (iv) extraction accuracy (plain vs. JSON-structured prompts). Each numeric claim in the hypothesis gets an explicit pass/fail test.

### Experiment Justification
- **Exp 1 (classification × 10 surface variants)**: tests H1, the 3–8% accuracy spread. Four tasks (SST-2, AG News, Emotion, BoolQ) keep it from being a single-dataset artifact.
- **Exp 2 (trailing whitespace → length)**: tests H2, +15–25% output length. Uses open-ended IFEval prompts, with greedy decoding so the comparison is paired.
- **Exp 3 (ALL CAPS → compliance)**: tests H3, a 5–12% compliance drop for small models and ~0 for ≥7B. IFEval gives programmatic compliance checks.
- **Exp 4 (JSON-structured prompts → extraction)**: tests H4, +10–15% extraction accuracy. A 2×2 design (input plain/JSON × output plain/JSON) on SQuAD (span extraction) and CoNLL-2003 (entity extraction) separates "JSON-formatted prompt" from "JSON-formatted answer".
- **Exp 5 (scale analysis)**: tests H5, that sensitivity falls log-linearly with size. Regress per-model sensitivity metrics from Exps 1–4 on log10(params).

## Research Question
Do semantically neutral surface edits to a prompt (whitespace, casing, punctuation, structural JSON wrapping) change LLM accuracy, output length, and instruction compliance, and does this sensitivity shrink with model scale?

## Hypothesis Decomposition
- H1: Across surface variants, accuracy on classification varies by 3–8 points (max − min, per model × task).
- H2: A trailing space or newline in the user message increases mean response length by 15–25% relative to the clean prompt.
- H3: Converting the prompt to ALL CAPS lowers IFEval compliance by 5–12 points for models <7B and by <~2 points for ≥7B.
- H4: A JSON-structured prompt raises extraction accuracy (SQuAD F1, CoNLL entity F1) by 10–15 points at every model size.
- H5: Sensitivity (accuracy spread / std across variants) decreases linearly in log(params).

## Proposed Methodology

### Approach
Run open-weight instruct models locally with vLLM. There are no API keys on this host, and the GPUs are shared, with about 5–10 GB free per slice. Decoding is greedy (temperature 0), so every difference between conditions comes from the prompt; each item is its own control (paired design). Models: Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B (AWQ), 14B (AWQ), plus Llama-3.2-1B and Llama-3.1-8B (AWQ) as a second family, depending on GPU availability.

### Experimental Steps
1. Build datasets: SST-2 val (n=500), AG News test (500), Emotion test (500), BoolQ val (300, passages truncated); IFEval (541); SQuAD v1.1 val (400); CoNLL-2003 test (300 sentences with ≥1 entity). Fixed seed 42 for sampling.
2. Classification prompt template = instruction + input + "Answer:" cue. Variants are applied to the *template text only*; the data are never altered: `base, trailing_space, trailing_newline, double_space, single_line (newlines→spaces), lowercase, uppercase, no_punct, exclaim (periods→!), sep_dash (": "→" - ")`. Parse the first label word in the output (max 10 tokens); unparseable counts as wrong and is reported separately.
3. IFEval conditions: `base, trailing_space, trailing_newline, trailing_2newline, uppercase, lowercase`; max_tokens 1024. Length is measured in tokens and words; compliance uses the official IFEval checkers (prompt-level and instruction-level, strict and loose).
4. Extraction 2×2: input {plain text, JSON object with task/context/question keys} × output {plain answer, JSON answer}. Metrics: SQuAD EM/F1 (official normalisation) and CoNLL micro-F1 over (type, string) entity pairs.
5. Scale regression: per-model metrics vs log10(params).

### Baselines
The `base` (clean, conventionally formatted) prompt is the baseline for every variant. For classification, majority-class accuracy is also reported.

### Evaluation Metrics
Accuracy; spread (max − min across variants) and std; length ratio (variant/base, mean of per-item ratios plus ratio of means); IFEval prompt/instruction-level strict accuracy; SQuAD EM/F1; NER micro-F1; parse-failure rate.

### Statistical Analysis Plan
- Per variant vs base: McNemar exact test (paired correct/incorrect) for accuracy; Wilcoxon signed-rank for lengths; paired bootstrap 95% CIs (2,000 resamples) for differences.
- Spread significance: a permutation-style null. Is the observed spread larger than what resampling items gives? Report a bootstrap CI of the spread and compare it with the spread between two random halves of base items.
- Holm–Bonferroni correction within each family of tests (model × task).
- Scale: OLS of sensitivity on log10(params) with slope CI, plus Spearman ρ. With ≤7 models, power is low; state this.
- α = 0.05.

## Expected Outcomes
H1 is supported if the median spread falls in 3–8 points and the spreads exceed the resampling noise. H2 is supported if trailing-whitespace length ratios fall in 1.15–1.25. H3 is supported if small models show 5–12 point drops and ≥7B show |Δ| < 2. H4 is supported if the JSON-input conditions gain ≥10 F1 at all sizes. H5 is supported if the slope is negative and significant. Any of these may be refuted; we report what happens.

## Timeline and Milestones
Setup 20 min → code + smoke test on 0.5B 40 min → model runs (GPU-bound, run in background, biggest model first when memory allows) 2–3 h → analysis 45 min → report 30 min.

## Potential Challenges
- GPU contention (≤6 GB free right now): run small models first, queue 7B/14B AWQ until a slice frees, and use a CPU llama.cpp GGUF fallback if necessary. If 14B cannot run, report the scale analysis on the models that did.
- Parsing failures can masquerade as accuracy changes, so parse rates are reported separately.
- Uppercasing IFEval prompts can change what the "change_case" instructions mean; robustness analysis excludes those instructions.

## Success Criteria
All five sub-hypotheses receive a quantitative verdict with CIs on at least 5 models spanning ≥1 order of magnitude in size, and all raw generations are saved for audit.

## Deviations from Plan (recorded during execution)
- **Models:** No API keys on the host, so all models were run locally. GPUs were shared, with mostly 4–6 GB slices free. Completed: Qwen2.5 0.5B/1.5B (bf16 + AWQ), 3B/7B/14B-AWQ, Llama-3.2-1B, Llama-3.1-8B-AWQ (TP=2), and Gemma-2-9B-AWQ. Gemma-9B is missing IFEval results (final chunk unfinished at the deadline). Memory-reservation logic was added to win GPU memory races against other jobs. The 0.5B/1.5B AWQ checkpoints were downloaded so the Qwen ladder has a single quantization level.
- **Added noise-floor controls** (not in the original plan): a full replicate run of qwen1.5b-awq, and the observation that Llama/Gemma chat templates trim trailing whitespace. That makes those variants exact placebos, which measure vLLM's batch-dependent nondeterminism.
- **Added a permutation null for the spread statistic**, because at n=300–500 a max−min over 10 variants is ~3 pp even with no true effect.
- **Parser fixes before final scoring**, applied identically to all conditions: "sport" → sports; CoNLL JSON with string values, duplicate keys, or `{"entities":[{type,text}]}` lists.
