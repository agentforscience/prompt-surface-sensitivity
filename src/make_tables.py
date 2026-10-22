"""Render the REPORT.md result tables from results/analysis/*.csv -> results/report_tables.md."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "results", "analysis")
ORDER = ["qwen0.5b", "qwen0.5b-awq", "llama1b", "qwen1.5b", "qwen1.5b-awq", "qwen3b-awq", "qwen7b-awq",
         "llama8b-awq", "gemma9b-awq", "qwen14b-awq"]
TRIMS = {"llama1b", "llama8b-awq", "gemma9b-awq"}
VARS = ["trailing_space", "trailing_newline", "double_space", "single_line", "lowercase", "uppercase",
        "no_punct", "exclaim", "sep_dash"]


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    D = pd.read_csv(os.path.join(A, "cls_variant_deltas.csv"))
    S = pd.read_csv(os.path.join(A, "cls_spread.csv"))
    L = pd.read_csv(os.path.join(A, "ifeval_length.csv"))
    C = pd.read_csv(os.path.join(A, "ifeval_compliance.csv"))
    X = pd.read_csv(os.path.join(A, "ext_cells.csv"))
    R = pd.read_csv(os.path.join(A, "ext_effects.csv"))
    F = pd.read_csv(os.path.join(A, "scale_fits.csv"))
    models = [m for m in ORDER if m in set(D.model)]
    parts = []

    H = D.groupby(["variant", "model"]).delta.mean().unstack() * 100
    rows = []
    for v in VARS:
        r = [v]
        for m in models:
            val = f"{H.loc[v, m]:+.1f}"
            if v.startswith("trailing") and m in TRIMS:
                val += " (placebo)"
            r.append(f"**{val}**" if v == "uppercase" else val)
        rows.append(r)
    parts.append("### mean accuracy delta (pp)\n" + md(rows, ["variant"] + models))

    rows = []
    for m in models:
        g = S[S.model == m].set_index("task")
        cells = [f"{g.loc[t, 'spread'] * 100:.1f} ({g.loc[t, 'null_spread_mean'] * 100:.1f}){'✱' if g.loc[t, 'perm_p'] < .05 else ''}"
                 for t in ["sst2", "agnews", "emotion", "boolq"]]
        rows.append([m] + cells + [f"{g.spread.mean() * 100:.1f}"])
    parts.append("### spread\n" + md(rows, ["model", "SST-2", "AG News", "Emotion", "BoolQ", "mean"]))
    parts.append(f"cells={len(S)} median={S.spread.median() * 100:.1f} IQR={S.spread.quantile(.25) * 100:.1f}-"
                 f"{S.spread.quantile(.75) * 100:.1f} in3-8={((S.spread >= .03) & (S.spread <= .08)).mean():.2f} "
                 f">8={(S.spread > .08).mean():.2f} perm_sig={(S.perm_p < .05).sum()}/{len(S)} "
                 f"holm_sig={(D.p_holm < .05).sum()}/{len(D)}\n"
                 + str(D[D.p_holm < .05].groupby("variant").size().sort_values(ascending=False).to_dict())
                 + "\nflip: " + str((D.groupby("variant").flip_rate.mean() * 100).round(1).to_dict())
                 + "\nabsdelta: " + str((D.groupby("variant").delta.apply(lambda x: x.abs().mean()) * 100).round(1).to_dict()))

    rows = []
    for m in [m for m in models if m in set(L.model)]:
        g = L[L.model == m].set_index("variant")
        r = [m + (" (placebo: template trims)" if m in TRIMS else "")]
        for v in ["trailing_space", "trailing_newline", "trailing_2newline"]:
            r.append(f"{g.loc[v, 'ratio']:.2f} [{g.loc[v, 'ci_lo']:.2f}, {g.loc[v, 'ci_hi']:.2f}]"
                     + ("✱" if g.loc[v, "wilcoxon_p"] < .05 else ""))
        r.append(f"{g.loc['uppercase', 'ratio']:.2f}" + ("✱" if g.loc["uppercase", "wilcoxon_p"] < .05 else ""))
        rows.append(r)
    parts.append("### length\n" + md(rows, ["model", "trailing_space", "trailing_newline", "trailing_2newline", "(uppercase)"]))

    rows = []
    for m in [m for m in models if m in set(C.model)]:
        g = C[(C.model == m)]
        up = g[(g.variant == "uppercase") & (g.metric == "strict")].iloc[0]
        lo = g[(g.variant == "lowercase") & (g.metric == "strict")].iloc[0]
        nc = g[(g.variant == "uppercase") & (g.metric == "strict_no_case")].iloc[0]
        b = f"{up.delta * 100:+.1f} [{up.ci_lo * 100:.1f}, {up.ci_hi * 100:.1f}]"
        rows.append([m, f"{up.base * 100:.1f}", f"**{b}**" if up.p < .05 else b, f"{lo.delta * 100:+.1f}",
                     f"{nc.delta * 100:+.1f}"])
    parts.append("### caps compliance\n" + md(rows, ["model", "base", "uppercase Δ", "lowercase Δ", "uppercase Δ excl. change_case"]))

    X["cell"] = X.in_fmt + "/" + X.out_fmt
    P = X.pivot_table(index="model", columns=["task", "cell"], values="f1") * 100
    rows = []
    for m in models:
        r = [m]
        for t in ["squad", "conll"]:
            for c in ["plain/plain", "json/plain", "plain/json", "json/json"]:
                r.append(f"{P.loc[m, (t, c)]:.1f}")
        rows.append(r)
    parts.append("### extraction cells\n" + md(rows, ["model", "SQuAD p/p", "SQuAD json-in", "SQuAD json-out", "SQuAD json/json",
                                                    "CoNLL p/p", "CoNLL json-in", "CoNLL json-out", "CoNLL json/json"]))
    rows = []
    for m in models:
        r = [m]
        for t in ["squad", "conll"]:
            e = R[(R.model == m) & (R.task == t) & (R.effect == "json_input_main")].iloc[0]
            s = f"{e.delta * 100:+.1f} [{e.ci_lo * 100:.1f}, {e.ci_hi * 100:.1f}]"
            r.append(f"**{s}**" if e.ci_lo > 0 else s)
        for t in ["squad", "conll"]:
            e = R[(R.model == m) & (R.task == t) & (R.effect == "json_output|plain_in")].iloc[0]
            r.append(f"{e.delta * 100:+.1f}")
        rows.append(r)
    parts.append("### json effects\n" + md(rows, ["model", "JSON-input SQuAD", "JSON-input CoNLL", "JSON-output SQuAD", "JSON-output CoNLL"]))
    parts.append("### scale fits\n" + F.round(3).to_markdown(index=False) if hasattr(F, "to_markdown") else F.round(3).to_string())
    out = "\n\n".join(parts)
    open(os.path.join(ROOT, "results", "report_tables.md"), "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
