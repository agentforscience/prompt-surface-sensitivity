"""Score all generations -> results/scored/<name>.parquet (one row per prompt).

Columns: model, exp, task, item, variant, plus per-experiment metrics:
  cls : correct (bool), parsed (bool; a label word was found)
  ifev: n_tokens, n_words, prompt_strict, prompt_loose, inst_strict (list), inst_ids
  ext : em, f1 (SQuAD) / f1, tp, fp, fn (CoNLL), json_ok (valid JSON when JSON requested)
The *same* lenient answer parser is used for every condition, so differences reflect
the model's answers rather than our parser's preferences.
"""
import collections
import json
import os
import re
import string
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NLTK refuses group-writable data dirs (e.g. /data/nltk_data); use a private one.
os.environ.setdefault("NLTK_DATA", os.path.expanduser("~/nltk_data"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ifeval_lib import utils as ifu  # noqa: E402

SYNONYMS = {"sadness": ["sad"], "technology": ["tech", "sci"], "joy": ["happ"], "sports": ["sport"]}


# ---------------------------------------------------------------- classification
def parse_label(text, labels):
    """Return the label whose (prefix) match appears earliest in the output, else None."""
    t = text.lower()
    best = None
    for lab in labels:
        for form in [lab] + SYNONYMS.get(lab, []):
            m = re.search(r"\b" + re.escape(form), t)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), lab)
    return best[1] if best else None


# ---------------------------------------------------------------- extraction helpers
def extract_json(text, pairs=False):
    """Find the first parseable JSON object in text (handles ```json fences).
    pairs=True returns the top-level (key, value) list so duplicate keys are kept."""
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        if pairs:
                            return json.loads(t[start:j + 1], object_pairs_hook=lambda kv: kv)
                        return json.loads(t[start:j + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def squad_norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_f1(pred, gold):
    p, g = squad_norm(pred).split(), squad_norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def squad_answer(text):
    obj = extract_json(text)
    if isinstance(obj, dict) and "answer" in obj:
        return str(obj["answer"]), True
    ans = text.strip().split("\n")[0]
    ans = re.sub(r'^(answer|extracted answer|span)\s*[:\-]\s*', "", ans, flags=re.I).strip().strip('"')
    return ans, False


ETYPES = ["PER", "ORG", "LOC", "MISC"]


def conll_entities_pred(text):
    obj = extract_json(text, pairs=True)
    ents = []
    if isinstance(obj, list):
        for k, v in obj:
            if isinstance(v, list) and v and all(isinstance(e, list) for e in v):
                # {"entities": [{"type": "PER", "text": "..."}, ...]} (pairs-hook gives lists of pairs)
                for e in v:
                    e = dict(e)
                    t = str(e.get("type", e.get("label", ""))).upper()[:4].rstrip("_")
                    t = {"PERS": "PER", "LOCA": "LOC", "ORGA": "ORG"}.get(t, t)
                    txt = e.get("text", e.get("entity", e.get("name", "")))
                    if t in ETYPES and isinstance(txt, str) and txt.strip():
                        ents.append((t, txt))
                continue
            if isinstance(v, (str, int, float)):
                v = [v]  # lenient: a bare string counts as a one-entity list
            k2 = k.upper()[:4].rstrip("_")
            k2 = {"PERS": "PER", "LOCA": "LOC", "ORGA": "ORG"}.get(k2, k2)
            if k2 in ETYPES and isinstance(v, list):
                ents += [(k2, str(e)) for e in v if isinstance(e, (str, int, float)) and str(e).strip()]
        return ents, True
    for line in text.split("\n"):
        m = re.match(r"^\s*[-*\d.]*\s*\**\s*(PER|ORG|LOC|MISC)\w*\**\s*[:\-]\s*(.+)$", line.strip(), flags=re.I)
        if m:
            ents.append((m.group(1).upper(), m.group(2).strip().strip('"').strip()))
    return ents, False


def ent_key(t, s):
    return (t, " ".join(s.lower().split()))


# ---------------------------------------------------------------- main
def score_model(name):
    prompts = {}
    for l in open(os.path.join(ROOT, "data", "prompts.jsonl")):
        p = json.loads(l)
        prompts[p["uid"]] = p
    rows = []
    for l in open(os.path.join(ROOT, "results", "gen", f"{name}.jsonl")):
        g = json.loads(l)
        p = prompts[g["uid"]]
        r = dict(model=name, uid=g["uid"], exp=p["exp"], task=p["task"], item=p["item"],
                 variant=p["variant"], n_tokens=g["n_tokens"], n_words=len(g["text"].split()),
                 finish=g["finish"], n_prompt_tokens=g["n_prompt_tokens"], text=g["text"])
        if p["exp"] == "cls":
            lab = parse_label(g["text"], p["labels"])
            r.update(parsed=lab is not None, correct=bool(lab == p["gold"]), pred=lab)
        elif p["exp"] == "ifev":
            inp = ifu.InputExample(key=p["item"], instruction_id_list=p["instruction_id_list"],
                                   prompt=p["base_prompt"], kwargs=p["kwargs"])
            s = ifu.test_instruction_following_strict(inp, g["text"])
            lo = ifu.test_instruction_following_loose(inp, g["text"])
            r.update(prompt_strict=s.follow_all_instructions, prompt_loose=lo.follow_all_instructions,
                     inst_strict=s.follow_instruction_list, inst_loose=lo.follow_instruction_list,
                     inst_ids=p["instruction_id_list"])
        else:
            r.update(in_fmt=p["in_fmt"], out_fmt=p["out_fmt"])
            if p["task"] == "squad":
                ans, is_json = squad_answer(g["text"])
                r.update(em=max(float(squad_norm(ans) == squad_norm(a)) for a in p["gold"]),
                         f1=max(squad_f1(ans, a) for a in p["gold"]), json_ok=is_json)
            else:
                pred, is_json = conll_entities_pred(g["text"])
                P = collections.Counter(ent_key(t, s) for t, s in pred)
                G = collections.Counter(ent_key(t, s) for t, s in p["gold"])
                tp = sum((P & G).values())
                fp, fn = sum(P.values()) - tp, sum(G.values()) - tp
                r.update(tp=tp, fp=fp, fn=fn, f1=2 * tp / max(1, 2 * tp + fp + fn), json_ok=is_json)
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results", "scored"), exist_ok=True)
    df.to_pickle(os.path.join(ROOT, "results", "scored", f"{name}.pkl"))
    return df


if __name__ == "__main__":
    for n in sys.argv[1:]:
        df = score_model(n)
        print(n, len(df))
        c = df[df.exp == "cls"]
        if len(c):
            print(c.pivot_table(index="variant", columns="task", values="correct").round(3))
            print("parse rate", c.parsed.mean().round(3))
        i = df[df.exp == "ifev"]
        if len(i):
            print(i.groupby("variant")[["n_tokens", "prompt_strict", "prompt_loose"]].mean().round(3))
        e = df[df.exp == "ext"]
        if len(e):
            print(e.pivot_table(index="variant", columns="task", values=["f1", "json_ok"]).round(3))
