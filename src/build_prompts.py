"""Build every prompt used in the study (model-agnostic) -> data/prompts.jsonl.

Each record is a single-turn chat user message plus the metadata needed to
score it. The surface variants change only the *template text* (instructions,
field labels, separators). The task data (review text, passage, question, ...)
is never altered, so the semantic content is identical across variants.

Experiments
  cls  : 4 classification tasks x 10 surface variants             (H1, H5)
  ifev : IFEval prompts x {base, trailing ws x3, upper, lower}     (H2, H3)
  ext  : SQuAD + CoNLL-2003 x 2x2 {plain/JSON input} x {plain/JSON output} (H4)
"""
import glob
import json
import os
import random
import re

import pandas as pd

SEED = 42
HUB = os.environ.get("HF_HUB_DIR", "/data/transformers/hub")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "prompts.jsonl")

N_CLS = 500        # items per single-input classification task
N_BOOLQ = 300      # BoolQ passages are long -> fewer items
N_SQUAD = 400
N_CONLL = 300


def hub_file(pattern):
    files = glob.glob(os.path.join(HUB, pattern))
    assert files, pattern
    return files[0]


# --------------------------------------------------------------------------
# Surface transforms. A template is a list of parts: str literals (template
# text, transformable) and ("slot", value) tuples (data, never transformed).
# --------------------------------------------------------------------------
def _no_punct(s):
    return re.sub(r"[.,:;?!]", "", s)


TEXT_TRANSFORMS = {
    "base": lambda s: s,
    "double_space": lambda s: s.replace(" ", "  "),
    "single_line": lambda s: re.sub(r"\n+", " ", s),
    "lowercase": str.lower,
    "uppercase": str.upper,
    "no_punct": _no_punct,
    "exclaim": lambda s: s.replace(".", "!"),
    "sep_dash": lambda s: s.replace(":", " -"),
}
# Variants applied to the final rendered string (after slot filling).
SUFFIX_VARIANTS = {"trailing_space": " ", "trailing_newline": "\n"}
CLS_VARIANTS = ["base", "trailing_space", "trailing_newline", "double_space",
                "single_line", "lowercase", "uppercase", "no_punct", "exclaim",
                "sep_dash"]


def render(parts, variant):
    tf = TEXT_TRANSFORMS.get(variant, TEXT_TRANSFORMS["base"])
    out = "".join(p[1] if isinstance(p, tuple) else tf(p) for p in parts)
    return out + SUFFIX_VARIANTS.get(variant, "")


# --------------------------------------------------------------------------
# Classification tasks
# --------------------------------------------------------------------------
def cls_tasks(rng):
    tasks = {}
    d = pd.read_parquet(hub_file("datasets--stanfordnlp--sst2/snapshots/*/data/validation-*.parquet"))
    d = d.sample(n=N_CLS, random_state=SEED)
    labels = ["negative", "positive"]
    tasks["sst2"] = dict(labels=labels, items=[
        dict(slots=[r.sentence.strip()], gold=labels[r.label]) for r in d.itertuples()],
        template=["Classify the sentiment of the following movie review as positive or negative.\n\nReview: ",
                  0, "\n\nAnswer with one word, either positive or negative.\nSentiment:"])

    d = pd.read_parquet(hub_file("datasets--fancyzhx--ag_news/snapshots/*/data/test-*.parquet"))
    d = d.sample(n=N_CLS, random_state=SEED)
    labels = ["world", "sports", "business", "technology"]
    tasks["agnews"] = dict(labels=labels, items=[
        dict(slots=[r.text.strip()], gold=labels[r.label]) for r in d.itertuples()],
        template=["Classify the topic of the following news article. The possible topics are world, sports, business, and technology.\n\nArticle: ",
                  0, "\n\nAnswer with one word from the list of topics.\nTopic:"])

    d = pd.read_parquet(hub_file("datasets--dair-ai--emotion/snapshots/*/split/test-*.parquet"))
    d = d.sample(n=N_CLS, random_state=SEED)
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    tasks["emotion"] = dict(labels=labels, items=[
        dict(slots=[r.text.strip()], gold=labels[r.label]) for r in d.itertuples()],
        template=["Classify the emotion expressed in the following text. The possible emotions are sadness, joy, love, anger, fear, and surprise.\n\nText: ",
                  0, "\n\nAnswer with one word from the list of emotions.\nEmotion:"])

    d = pd.read_parquet(hub_file("datasets--google--boolq/snapshots/*/data/validation-*.parquet"))
    d = d[d.passage.str.split().str.len() <= 200].sample(n=N_BOOLQ, random_state=SEED)
    labels = ["no", "yes"]
    tasks["boolq"] = dict(labels=labels, items=[
        dict(slots=[r.passage.strip(), r.question.strip().rstrip("?") + "?"],
             gold="yes" if r.answer else "no") for r in d.itertuples()],
        template=["Read the passage and answer the question with yes or no.\n\nPassage: ", 0,
                  "\n\nQuestion: ", 1, "\n\nAnswer with one word, either yes or no.\nAnswer:"])
    return tasks


def build_cls(rng):
    recs = []
    for task, spec in cls_tasks(rng).items():
        for i, it in enumerate(spec["items"]):
            parts = [p if isinstance(p, str) else ("slot", it["slots"][p]) for p in spec["template"]]
            for v in CLS_VARIANTS:
                recs.append(dict(exp="cls", task=task, item=i, variant=v, gold=it["gold"],
                                 labels=spec["labels"], prompt=render(parts, v), max_tokens=10))
    return recs


# --------------------------------------------------------------------------
# IFEval: trailing whitespace (length) and casing (compliance)
# --------------------------------------------------------------------------
IFEVAL_VARIANTS = {
    "base": lambda s: s,
    "trailing_space": lambda s: s + " ",
    "trailing_newline": lambda s: s + "\n",
    "trailing_2newline": lambda s: s + "\n\n",
    "uppercase": str.upper,
    "lowercase": str.lower,
}


def build_ifeval():
    recs = []
    with open(hub_file("datasets--google--IFEval/snapshots/*/ifeval_input_data.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        for v, fn in IFEVAL_VARIANTS.items():
            recs.append(dict(exp="ifev", task="ifeval", item=r["key"], variant=v,
                             prompt=fn(r["prompt"]), base_prompt=r["prompt"],
                             instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"],
                             max_tokens=1024))
    return recs


# --------------------------------------------------------------------------
# Extraction: 2x2 (input format x output format)
# --------------------------------------------------------------------------
SQUAD_TASK = "Read the passage and answer the question by extracting the shortest span of text from the passage that answers it."
SQUAD_OUT = {"plain": "Respond with only the extracted answer span and nothing else.",
             "json": 'Respond with only a JSON object of the form {"answer": "<extracted span>"}.'}
CONLL_TASK = ("Extract all named entities from the sentence. Entity types: PER (person), "
              "ORG (organization), LOC (location), MISC (other names such as nationalities, events, or products).")
CONLL_OUT = {"plain": "List each entity on its own line in the format TYPE: entity text. If there are no entities, write NONE.",
             "json": 'Respond with only a JSON object of the form {"PER": [...], "ORG": [...], "LOC": [...], "MISC": [...]} listing the entity texts of each type (use empty lists when there are none).'}
CONLL_TAGS = {0: "O", 1: "B-ORG", 2: "B-MISC", 3: "B-PER", 4: "I-PER", 5: "B-LOC", 6: "I-ORG", 7: "I-MISC", 8: "I-LOC"}


def conll_entities(tokens, tags):
    ents, cur = [], None
    for tok, t in zip(tokens, tags):
        tag = CONLL_TAGS[int(t)]
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
            if cur:
                ents.append(cur)
            cur = [tag[2:], [tok]]
        elif tag.startswith("I-"):
            cur[1].append(tok)
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    return [[t, " ".join(w)] for t, w in ents]


def ext_prompt(task_text, fields, out_text, in_fmt):
    """Same semantic content, either as labelled plain text or as a JSON object."""
    if in_fmt == "plain":
        body = "\n\n".join(f"{k.capitalize()}: {v}" for k, v in fields.items())
        return f"{task_text}\n\n{body}\n\n{out_text}"
    obj = {"task": task_text, **fields, "output_format": out_text}
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_ext():
    recs = []
    d = pd.read_parquet(hub_file("datasets--rajpurkar--squad/snapshots/*/plain_text/validation-*.parquet"))
    d = d.sample(n=N_SQUAD, random_state=SEED)
    for i, r in enumerate(d.itertuples()):
        for fi in ("plain", "json"):
            for fo in ("plain", "json"):
                recs.append(dict(exp="ext", task="squad", item=i, variant=f"in_{fi}__out_{fo}",
                                 in_fmt=fi, out_fmt=fo, gold=list(r.answers["text"]),
                                 prompt=ext_prompt(SQUAD_TASK, {"passage": r.context, "question": r.question},
                                                   SQUAD_OUT[fo], fi), max_tokens=64))
    d = pd.read_parquet(os.path.join(ROOT, "datasets", "conll2003", "test.parquet"))
    d["ents"] = [conll_entities(t, g) for t, g in zip(d.tokens, d.tags)]
    d = d[(d.ents.str.len() > 0) & (d.tokens.str.len() >= 6) & (d.tokens.str.len() <= 40)]
    d = d.sample(n=N_CONLL, random_state=SEED)
    for i, r in enumerate(d.itertuples()):
        sent = " ".join(r.tokens)
        for fi in ("plain", "json"):
            for fo in ("plain", "json"):
                recs.append(dict(exp="ext", task="conll", item=i, variant=f"in_{fi}__out_{fo}",
                                 in_fmt=fi, out_fmt=fo, gold=r.ents,
                                 prompt=ext_prompt(CONLL_TASK, {"sentence": sent}, CONLL_OUT[fo], fi),
                                 max_tokens=256))
    return recs


def main():
    rng = random.Random(SEED)
    recs = build_cls(rng) + build_ifeval() + build_ext()
    for k, r in enumerate(recs):
        r["uid"] = k
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(recs)
    print(df.groupby(["exp", "task"]).size())
    print("total", len(recs))
    # Save a few rendered examples for documentation
    with open(os.path.join(ROOT, "data", "prompt_examples.md"), "w") as f:
        for (exp, task), g in df.groupby(["exp", "task"]):
            f.write(f"# {exp} / {task}\n\n")
            for _, r in g[g.item == g.item.iloc[0]].iterrows():
                f.write(f"## variant = {r.variant}\n```\n{r.prompt!r}\n```\n\n")


if __name__ == "__main__":
    main()
