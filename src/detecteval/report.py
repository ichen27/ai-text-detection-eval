"""Turn results/scores/*.jsonl into results/summary.json, results/report.md and figures.

    python -m detecteval.report
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config, metrics  # noqa: E402

MODELS = ["clef-flash", "clef", "clef-omni"]
QUESTION_LABELS = {"ai_yes": "Yes/no: AI-generated?", "author": "Choice: human or AI",
                   "pct_ai": "Score: what % is AI?"}
GROUP_LABELS = {
    "raid_human": "Human: RAID (abstracts, books, news, poetry)",
    "essay_native": "Human: native-speaker student essays",
    "essay_nonnative": "Human: non-native TOEFL essays",
    "toefl_gpt4_polished": "Mixed: TOEFL essays polished by GPT-4",
    "raid_2023": "AI: 2023 chat models (RAID)",
    "raid_paraphrased": "AI: 2023 models, paraphrased (RAID)",
    "fresh_2026": "AI: 2026 models",
    "humanized_2026": "AI: 2026 models, humanized",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUMAN_COLOR, AI_COLOR = "#2a78d6", "#eb6834"
MODEL_COLORS = {"clef-flash": "#2a78d6", "clef": "#eb6834", "clef-omni": "#1baf7a"}
TARGET_FPR = 0.05


def load() -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    with open(config.DATA_DIR / "eval_set.jsonl") as handle:
        texts = {row["id"]: row for row in map(json.loads, handle)}
    scores: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        path = config.RESULTS_DIR / "scores" / f"{model}.jsonl"
        if path.exists():
            with open(path) as handle:
                scores[model] = {row["id"]: row["scores"] for row in map(json.loads, handle) if row["id"] in texts}
    return texts, scores


def arrays(texts: dict, model_scores: dict, question: str, **where) -> np.ndarray:
    return np.array([model_scores[i][question] for i, text in texts.items()
                     if i in model_scores and all(text[key] == value for key, value in where.items())])


def summarise(texts: dict, scores: dict) -> dict:
    summary: dict = {"n_texts": len(texts), "target_fpr": TARGET_FPR, "models": {}}
    for model, model_scores in scores.items():
        summary["models"][model] = {}
        for question in QUESTION_LABELS:
            human = arrays(texts, model_scores, question, label="human")
            ai = arrays(texts, model_scores, question, label="ai")
            threshold = metrics.threshold_for_fpr(human, TARGET_FPR)
            both = np.concatenate([human, ai])
            labels = np.concatenate([np.zeros(human.size), np.ones(ai.size)])
            groups = {}
            for group in GROUP_LABELS:
                values = arrays(texts, model_scores, question, group=group)
                if values.size:
                    groups[group] = {"n": int(values.size), "mean_score": float(values.mean()),
                                     "flagged_at_0.5": metrics.flag_rate(values, 0.5),
                                     "flagged_at_fpr_threshold": metrics.flag_rate(values, threshold)}
            summary["models"][model][question] = {
                "n_human": int(human.size), "n_ai": int(ai.size),
                "auroc": metrics.auroc(human, ai),
                "auroc_ci95": metrics.bootstrap_ci(metrics.auroc, human, ai),
                "tpr_at_5pct_fpr": metrics.tpr_at_fpr(human, ai, 0.05),
                "tpr_at_5pct_fpr_ci95": metrics.bootstrap_ci(
                    lambda h, a: metrics.tpr_at_fpr(h, a, 0.05), human, ai),
                "tpr_at_1pct_fpr": metrics.tpr_at_fpr(human, ai, 0.01),
                "threshold_at_5pct_fpr": threshold,
                "fpr_at_0.5": metrics.flag_rate(human, 0.5), "tpr_at_0.5": metrics.flag_rate(ai, 0.5),
                "ece": metrics.expected_calibration_error(both, labels),
                "calibration": metrics.calibration(both, labels),
                "groups": groups,
            }
    return summary


def polish_effect(texts: dict, scores: dict) -> dict:
    """Paired change in score when the same TOEFL essay is polished by GPT-4."""
    out = {}
    for model, model_scores in scores.items():
        out[model] = {}
        for question in QUESTION_LABELS:
            diffs = []
            for i in texts:
                if i.startswith("toefl_gpt4_polished-"):
                    original = i.replace("toefl_gpt4_polished-", "essay_nonnative-")
                    if i in model_scores and original in model_scores:
                        diffs.append(model_scores[i][question] - model_scores[original][question])
            if diffs:
                out[model][question] = {"n_pairs": len(diffs), "mean_change": float(np.mean(diffs)),
                                        "share_increased": float(np.mean(np.array(diffs) > 0))}
    return out


def variant_check(variant: str) -> dict:
    """Compare standard scores with a check variant ("joint" or "repeat") on the texts both cover."""
    out = {}
    for model in MODELS:
        base_path = config.RESULTS_DIR / "scores" / f"{model}.jsonl"
        variant_path = config.RESULTS_DIR / "scores" / f"{model}.{variant}.jsonl"
        if not (base_path.exists() and variant_path.exists()):
            continue
        with open(variant_path) as handle:
            other = {row["id"]: row["scores"] for row in map(json.loads, handle)}
        with open(base_path) as handle:
            base = {row["id"]: row["scores"] for row in map(json.loads, handle) if row["id"] in other}
        if not base:
            continue
        out[model] = {"n": len(base)}
        for question in QUESTION_LABELS:
            diffs = np.array([abs(base[i][question] - other[i][question]) for i in base])
            out[model][question] = {"mean_abs_diff": float(diffs.mean()), "max_abs_diff": float(diffs.max())}
    return out


def by_writer(texts: dict, scores: dict, summary: dict, question: str = "pct_ai") -> dict:
    """Share of each AI writer's texts flagged at the 5%-FPR threshold, per model."""
    out: dict = {}
    for model, model_scores in scores.items():
        threshold = summary["models"][model][question]["threshold_at_5pct_fpr"]
        for i, text in texts.items():
            if text["label"] == "ai" and i in model_scores:
                cell = out.setdefault(f"{text['group']} / {text['generator']}", {}).setdefault(model, [0, 0])
                cell[0] += model_scores[i][question] > threshold
                cell[1] += 1
    return {writer: {model: {"flagged": int(c[0]), "n": c[1]} for model, c in cells.items()}
            for writer, cells in sorted(out.items())}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def figure_distributions(texts: dict, scores: dict, path) -> None:
    models = [m for m in MODELS if m in scores]
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.6 * len(models) + 0.9), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(0, 1, 21)
    for r, model in enumerate(models):
        for c, (question, label) in enumerate(QUESTION_LABELS.items()):
            ax = axes[r][c]
            style(ax)
            human = arrays(texts, scores[model], question, label="human")
            ai = arrays(texts, scores[model], question, label="ai")
            for values, color, name in ((human, HUMAN_COLOR, "Human-written"), (ai, AI_COLOR, "AI-generated")):
                weights = np.full(values.size, 100 / values.size)
                ax.hist(values, bins=bins, weights=weights, color=color, alpha=0.55, label=name,
                        edgecolor=SURFACE, linewidth=1)
            ax.set_xlim(0, 1)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            auc = metrics.auroc(human, ai)
            ax.set_title(f"{model} · {label}", fontsize=10, color=INK, loc="left")
            ax.text(0.98, 0.92, f"AUROC {auc:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color=MUTED)
            if c == 0:
                ax.set_ylabel("% of texts", fontsize=9, color=MUTED)
            if r == len(models) - 1:
                ax.set_xlabel("Model's AI score (0 = human, 1 = AI)", fontsize=9, color=MUTED)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("How each model scores human and AI text, by question", x=0.012, ha="left", fontsize=13, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_groups(summary: dict, question: str, path) -> None:
    models = list(summary["models"])
    groups = [g for g in GROUP_LABELS if g in summary["models"][models[0]][question]["groups"]]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(groups) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for y, group in enumerate(groups):
        ax.axhline(y, color=GRID, linewidth=0.6, zorder=0)
        for model in models:
            value = 100 * summary["models"][model][question]["groups"][group]["flagged_at_fpr_threshold"]
            ax.scatter(value, y, s=70, color=MODEL_COLORS[model], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=model if y == 0 else None)
    ax.set_yticks(range(len(groups)), [GROUP_LABELS[g] for g in groups], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xlabel("% of texts flagged as AI", fontsize=9, color=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    ax.set_title(f"Share flagged as AI by group, threshold set to flag {TARGET_FPR:.0%} of all human texts\n"
                 f"Question: {QUESTION_LABELS[question]}", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def markdown(summary: dict, polish: dict, repeat: dict, joint: dict, writers: dict) -> str:
    lines = ["# Results", "", f"Texts scored: {summary['n_texts']}. Scores run 0 (human) to 1 (AI).", "",
             "## Overall separation (human vs AI, mixed texts excluded)", "",
             "| Model | Question | AUROC (95% CI) | Detected at 5% FPR (95% CI) | Detected at 1% FPR | "
             "Humans flagged at 0.5 | AI flagged at 0.5 | ECE |", "|---|---|---|---|---|---|---|---|"]
    for model, by_question in summary["models"].items():
        for question, s in by_question.items():
            lines.append(
                f"| {model} | {QUESTION_LABELS[question]} | {s['auroc']:.3f} ({s['auroc_ci95'][0]:.2f}-"
                f"{s['auroc_ci95'][1]:.2f}) | {s['tpr_at_5pct_fpr']:.0%} ({s['tpr_at_5pct_fpr_ci95'][0]:.0%}-"
                f"{s['tpr_at_5pct_fpr_ci95'][1]:.0%}) | {s['tpr_at_1pct_fpr']:.0%} | {s['fpr_at_0.5']:.0%} | "
                f"{s['tpr_at_0.5']:.0%} | {s['ece']:.2f} |")
    for question, label in QUESTION_LABELS.items():
        lines += ["", f"## By group: {label}", "",
                  "Share flagged as AI at the threshold that flags 5% of all human texts (and at a fixed 0.5).", "",
                  "| Group | n | " + " | ".join(summary["models"]) + " |",
                  "|---|---|" + "---|" * len(summary["models"])]
        first = next(iter(summary["models"].values()))[question]["groups"]
        for group in first:
            cells = []
            for model in summary["models"]:
                g = summary["models"][model][question]["groups"][group]
                cells.append(f"{g['flagged_at_fpr_threshold']:.0%} ({g['flagged_at_0.5']:.0%}), "
                             f"mean {g['mean_score']:.2f}")
            lines.append(f"| {GROUP_LABELS[group]} | {first[group]['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Calibration of the percentage question", "",
              "If the score were a calibrated probability, the share of AI texts in each bin would match the bin.", "",
              "| Score bin | " + " | ".join(f"{m}: n, actual AI share" for m in summary["models"]) + " |",
              "|---|" + "---|" * len(summary["models"])]
    for b in range(5):
        cells = []
        for model in summary["models"]:
            row = summary["models"][model]["pct_ai"]["calibration"][b]
            cells.append(f"{row['n']}, {row['share_ai']:.0%}" if row["n"] else "0, n/a")
        low = summary["models"][next(iter(summary["models"]))]["pct_ai"]["calibration"][b]
        lines.append(f"| {low['low']:.1f}-{low['high']:.1f} | " + " | ".join(cells) + " |")
    if polish:
        lines += ["", "## Effect of GPT-4 polishing on the same TOEFL essays", "",
                  "| Model | Question | Pairs | Mean score change | Share of essays whose score rose |",
                  "|---|---|---|---|---|"]
        for model, by_question in polish.items():
            for question, p in by_question.items():
                lines.append(f"| {model} | {QUESTION_LABELS[question]} | {p['n_pairs']} | {p['mean_change']:+.3f} | "
                             f"{p['share_increased']:.0%} |")
    if writers:
        lines += ["", "## Detection by writer (percentage question, 5% FPR threshold)", "",
                  "Flagged / total. Counts per writer are small, so treat differences between writers as rough.", "",
                  "| Group / writer | " + " | ".join(summary["models"]) + " |",
                  "|---|" + "---|" * len(summary["models"])]
        for writer, cells in writers.items():
            lines.append(f"| {writer} | " + " | ".join(
                f"{cells[m]['flagged']}/{cells[m]['n']}" if m in cells else "" for m in summary["models"]) + " |")
    for title, note, check in (
            ("Run-to-run variation", "The same question asked twice, in separate requests.", repeat),
            ("Do questions in one request affect each other?",
             "Each question asked alone versus all three in one request.", joint)):
        if check:
            lines += ["", f"## {title}", "", note + " Cells show mean (max) absolute difference in score.", "",
                      "| Model | Texts | " + " | ".join(QUESTION_LABELS.values()) + " |", "|---|---|---|---|---|"]
            for model, c in check.items():
                lines.append(f"| {model} | {c['n']} | " + " | ".join(
                    f"{c[q]['mean_abs_diff']:.3f} ({c[q]['max_abs_diff']:.3f})" for q in QUESTION_LABELS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    texts, scores = load()
    if not scores:
        raise SystemExit("no score files found; run python -m detecteval.score first")
    summary = summarise(texts, scores)
    polish, repeat, joint = polish_effect(texts, scores), variant_check("repeat"), variant_check("joint")
    summary["polish_effect"], summary["repeat_check"], summary["joint_check"] = polish, repeat, joint
    writers = by_writer(texts, scores, summary)
    summary["by_writer_pct_ai"] = writers
    figures = config.RESULTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (config.RESULTS_DIR / "report.md").write_text(markdown(summary, polish, repeat, joint, writers))
    figure_distributions(texts, scores, figures / "score_distributions.png")
    for question in QUESTION_LABELS:
        figure_groups(summary, question, figures / f"flagged_by_group_{question}.png")
    print((config.RESULTS_DIR / "report.md").read_text())


if __name__ == "__main__":
    main()
