"""Markdown and figures for ``compare``."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .compare import COMBINATIONS, DETECTORS, GROUP_LABELS  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One fixed colour per detector, in every figure.
COLORS = {"clef": "#eb6834", "binoculars": "#2a78d6", "classifier-raid": "#eda100",
          "classifier-mirrors": "#1baf7a", "all-three": "#6250d6"}
SHORT_NAMES = {"clef": "Clef 27B (asked, no training)", "binoculars": "Binoculars (0.5B pair)",
               "classifier-raid": "Classifier: RAID only", "classifier-mirrors": "Classifier: RAID + mirrors",
               "all-three": "All three combined", "fastdetect": "Fast-DetectGPT (0.5B pair)",
               "clef-flash": "Clef-flash 9B", "clef-omni": "Clef-omni", "clef+binoculars": "Clef + Binoculars",
               "clef+classifier": "Clef + classifier (mirrors)", "local-two": "Binoculars + classifier (mirrors)"}
ORDER = ["clef", "clef-flash", "clef-omni", "binoculars", "fastdetect", "classifier-raid", "classifier-mirrors",
         "clef+binoculars", "clef+classifier", "local-two", "all-three"]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def interval(pair) -> str:
    return f"{pair[0]:.0%}-{pair[1]:.0%}"


def overall_table(entry: dict) -> list[str]:
    lines = ["| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | "
             "AI caught, bar set for 1% | Humans actually flagged |", "|---|---|---|---|---|---|"]
    for name in ORDER:
        d = entry["detectors"].get(name)
        if d:
            lines.append(
                f"| {SHORT_NAMES[name]} | {d['auroc']:.3f} ({d['auroc_ci95'][0]:.2f}-{d['auroc_ci95'][1]:.2f}) | "
                f"{pct(d['tpr_5pct'])} ({interval(d['tpr_5pct_ci95'])}) | {pct(d['fpr_5pct'])} "
                f"({interval(d['fpr_5pct_ci95'])}) | {pct(d['tpr_1pct'])} | {pct(d['fpr_1pct'])} |")
    return lines


def group_table(entry: dict, names: list[str]) -> list[str]:
    names = [name for name in names if name in entry["detectors"]]
    lines = ["| Group | Test texts | " + " | ".join(SHORT_NAMES[name] for name in names) + " |",
             "|---|---|" + "---|" * len(names)]
    for group, label in GROUP_LABELS.items():
        cells = [entry["detectors"][name]["groups"].get(group) for name in names]
        if any(cells):
            n = next(cell["n"] for cell in cells if cell)
            shares = " | ".join(pct(cell["share"]) if cell else "" for cell in cells)
            lines.append(f"| {label} | {n} | {shares} |")
    return lines


def ladder_table(slices: dict, names: list[str], rungs: list[int]) -> list[str]:
    lines = ["| Detector | " + " | ".join(f"{rung} words" for rung in rungs) + " |", "|---|" + "---|" * len(rungs)]
    for name in names:
        cells = []
        for rung in rungs:
            d = slices.get(f"long@{rung}", {}).get("detectors", {}).get(name)
            cells.append(f"{pct(d['tpr_5pct'])} ({pct(d['fpr_5pct'])})" if d else "")
        if any(cells):
            lines.append(f"| {SHORT_NAMES[name]} | " + " | ".join(cells) + " |")
    return lines


def generator_table(entry: dict) -> list[str]:
    lines = ["| Detector | Generators seen in training | Generators never seen in training |", "|---|---|---|"]
    for name in ORDER:
        d = entry["detectors"].get(name)
        if d and d["fresh_2026_by_generator"]["seen in training"]["n"]:
            seen, unseen = (d["fresh_2026_by_generator"][k] for k in ("seen in training", "never seen in training"))
            lines.append(f"| {SHORT_NAMES[name]} | {pct(seen['share'])} of {seen['n']} | "
                         f"{pct(unseen['share'])} of {unseen['n']} |")
    return lines


def markdown(summary: dict) -> str:
    slices, local = summary["slices"], summary["local_slices"]
    main = ["clef", "binoculars", "classifier-raid", "classifier-mirrors", "all-three"]
    lines = [
        "# Detector comparison", "",
        "Each text is in a calibration half or a test half. The bar for flagging a text, and the weights of the "
        "combined detectors, were set on the calibration half. Every number below is measured on the test half, "
        "so \"humans actually flagged\" can differ from the 5% or 1% the bar was set for. Detectors are compared "
        "only on texts that all of them have scored.", "",
        "Detectors:", ""]
    lines += [f"- **{SHORT_NAMES[name]}**: {DETECTORS[name]}" for name in ORDER if name in summary["detectors"]]
    lines += [f"- **{SHORT_NAMES[name]}**: logistic regression over {', '.join(parts)}"
              for name, (_, parts) in COMBINATIONS.items()]

    lines += ["", "## 1. Does longer text help? The same documents cut to several lengths", "",
              "AI text caught when the bar is set to flag 5% of humans (humans actually flagged in brackets).", ""]
    rungs = [rung for rung in (75, 150, 300) if f"long@{rung}" in slices]
    lines += ladder_table(slices, main, rungs)
    entry = slices["long@300"]
    lines += ["", f"{entry['n_texts']} of the long set's {entry['n_in_set']} documents have been scored by every "
              "detector, including Clef. " + (
                  "The rest are 2026-model texts that Clef has not scored yet, so this table covers human text and "
                  "2023-era machine text only." if entry["n_texts"] < entry["n_in_set"] else "")]
    lines += ["", "The detectors that run locally have scored every document, including the 2026-model texts:", ""]
    lines += ladder_table(local, ["binoculars", "classifier-raid", "classifier-mirrors", "local-two"],
                          [75, 150, 300])
    same = local.get("long@300 (documents that reach 500 words)")
    if same and "long@500" in local:
        lines += ["", f"Going from 300 to 500 words, on the {same['n_texts']} documents long enough for both "
                  "(mostly human text and 2026-model text):", "", "| Detector | 300 words | 500 words |",
                  "|---|---|---|"]
        for name in ["binoculars", "classifier-raid", "classifier-mirrors", "local-two"]:
            before, after = same["detectors"].get(name), local["long@500"]["detectors"].get(name)
            if before and after:
                lines.append(f"| {SHORT_NAMES[name]} | {pct(before['tpr_5pct'])} ({pct(before['fpr_5pct'])}) | "
                             f"{pct(after['tpr_5pct'])} ({pct(after['fpr_5pct'])}) |")

    lines += ["", "## 2. All detectors at 300 words", ""] + overall_table(slices["long@300"])
    lines += ["", "Share of each group flagged as AI, bar set for 5% of humans:", ""]
    lines += group_table(slices["long@300"], main)
    lines += ["", "Local detectors on every 300-word document, including 2026-model texts:", ""]
    lines += overall_table(local["long@300"]) + [""]
    lines += group_table(local["long@300"], ["binoculars", "classifier-raid", "classifier-mirrors", "local-two"])

    lines += ["", "## 3. Human writing the models cannot have seen", "",
              "Forum posts from the same sites: 2019 (possibly in training data) against posts published "
              "after the Clef models were released. An AUROC of 0.5 means the detector scores the two "
              "alike; above 0.5 means recent posts look more AI-like to it.", "",
              "| Detector | 2019 posts | Recent posts | AUROC, recent over 2019 (95% CI) |", "|---|---|---|---|"]
    for name in ORDER:
        row = summary["seen_vs_unseen_humans"].get(name)
        if row:
            lines.append(f"| {SHORT_NAMES[name]} | {row['n_2019']} | {row['n_recent']} | "
                         f"{row['auroc_recent_over_2019']:.2f} ({row['auroc_ci95'][0]:.2f}-"
                         f"{row['auroc_ci95'][1]:.2f}) |")

    lines += ["", "## 4. The first test set (150 words or fewer)", ""] + overall_table(slices["short@150"])
    lines += ["", "Share of each group flagged as AI, bar set for 5% of humans:", ""]
    lines += group_table(slices["short@150"], main)

    lines += ["", "## 5. Current models the classifier was and was not trained on", "",
              "Share of 2026-model texts flagged, bar set for 5% of humans. \"Seen\" are the four generators that "
              "wrote the training mirrors; the other four never appear in training.", "",
              "Short set:", ""] + generator_table(slices["short@150"])
    lines += ["", "Long set at 300 words (local detectors):", ""] + generator_table(local["long@300"])
    if summary["polish_effect"]:
        lines += ["", "## 6. Human text polished by a 2026 model", "",
                  "Share of pairs whose score rose after polishing (300 words).", "",
                  "| Detector | Pairs | Score rose |", "|---|---|---|"]
        lines += [f"| {SHORT_NAMES[name]} | {row['n_pairs']} | {pct(row['share_increased'])} |"
                  for name in ORDER if (row := summary["polish_effect"].get(name))]
    return "\n".join(lines) + "\n"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def figure_ladder(summary: dict, path) -> None:
    rungs = [rung for rung in (75, 150, 300) if f"long@{rung}" in summary["slices"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ends = []
    for name, color in COLORS.items():
        points = [(index, 100 * d["tpr_5pct"]) for index, rung in enumerate(rungs)
                  if (d := summary["slices"][f"long@{rung}"]["detectors"].get(name))]
        if len(points) < 2:
            continue
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ends.append((ys[-1], xs[-1], name, color))
    # Direct labels at the line ends, nudged apart so they do not collide.
    ends.sort()
    placed: list[float] = []
    for value, x, name, color in ends:
        y = value if not placed or value - placed[-1] >= 5 else placed[-1] + 5
        placed.append(y)
        ax.plot([x + 0.03, x + 0.09], [value, y], color=color, linewidth=1)
        ax.text(x + 0.11, y, f"{SHORT_NAMES[name]}  {value:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(rungs)), [f"{rung} words" for rung in rungs])
    ax.set_xlim(-0.15, len(rungs) - 1 + 1.25)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of AI texts caught", fontsize=9, color=MUTED)
    entry = summary["slices"]["long@300"]
    scope = ("human and 2023-model texts" if entry["n_texts"] < entry.get("n_in_set", entry["n_texts"])
             else "all long-set texts")
    ax.set_title(f"AI text caught as the same documents get longer ({scope})\nBar set to flag 5% of human texts, "
                 "measured on held-out texts", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figure_groups(summary: dict, path, key: str = "long@300") -> None:
    entry = summary["slices"][key]
    names = [name for name in COLORS if name in entry["detectors"]]
    groups = [group for group in GROUP_LABELS if any(group in entry["detectors"][name]["groups"] for name in names)]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(groups) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for y, group in enumerate(groups):
        ax.axhline(y, color=GRID, linewidth=0.6, zorder=0)
        for name in names:
            cell = entry["detectors"][name]["groups"].get(group)
            if cell:
                ax.scatter(100 * cell["share"], y, s=70, color=COLORS[name], edgecolor=SURFACE, linewidth=1.5,
                           zorder=3, label=SHORT_NAMES[name] if y == 0 else None)
    ax.set_yticks(range(len(groups)), [GROUP_LABELS[group] for group in groups], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xlabel("% of texts flagged as AI", fontsize=9, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    length = "300-word texts" if key == "long@300" else "texts of 150 words or fewer"
    ax.set_title(f"Share flagged as AI by group, {length}\nBar set to flag 5% of human texts, measured on held-out "
                 "texts", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figures(summary: dict) -> None:
    directory = config.RESULTS_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    if "long@300" in summary["slices"]:
        figure_ladder(summary, directory / "length_ladder.png")
        figure_groups(summary, directory / "detectors_by_group_300_words.png")
    if "short@150" in summary["slices"]:
        figure_groups(summary, directory / "detectors_by_group_short.png", "short@150")
