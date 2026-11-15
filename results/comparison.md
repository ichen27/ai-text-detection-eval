# Detector comparison

Each text is in a calibration half or a test half. The bar for flagging a text, and the weights of the combined detectors, were set on the calibration half. Every number below is measured on the test half, so "humans actually flagged" can differ from the 5% or 1% the bar was set for. Detectors are compared only on texts that all of them have scored.

Detectors:

- **Clef 27B (asked, no training)**: Clef 27B, percentage question (no training)
- **Clef-flash 9B**: Clef-flash 9B, human-or-AI question (no training)
- **Clef-omni**: Clef-omni, human-or-AI question (no training)
- **Binoculars (0.5B pair)**: Binoculars, 0.5B model pair (no training)
- **Fast-DetectGPT (0.5B pair)**: Fast-DetectGPT, 0.5B model pair (no training)
- **Classifier: RAID only**: Classifier trained on RAID (2023 models)
- **Classifier: RAID + mirrors**: Classifier trained on RAID + current-model mirrors
- **Clef + Binoculars**: logistic regression over clef, binoculars
- **Clef + classifier (mirrors)**: logistic regression over clef, classifier-mirrors
- **All three combined**: logistic regression over clef, binoculars, classifier-mirrors
- **Binoculars + classifier (mirrors)**: logistic regression over binoculars, classifier-mirrors

## 1. Does longer text help? The same documents cut to several lengths

AI text caught when the bar is set to flag 5% of humans (humans actually flagged in brackets).

| Detector | 75 words | 150 words | 300 words |
|---|---|---|---|
| Clef 27B (asked, no training) | 49% (7%) | 59% (4%) | 76% (5%) |
| Binoculars (0.5B pair) | 51% (2%) | 69% (5%) | 73% (4%) |
| Classifier: RAID only | 60% (5%) | 69% (6%) | 69% (3%) |
| Classifier: RAID + mirrors | 78% (6%) | 85% (5%) | 91% (3%) |
| All three combined | 86% (4%) | 93% (3%) | 95% (5%) |

All 1003 documents of the long set are included.

Going from 300 to 500 words, on the 409 documents long enough for both (human text and 2026-model text; almost no 2023-model text is that long):

| Detector | 300 words | 500 words |
|---|---|---|
| Clef 27B (asked, no training) | 63% (5%) | 74% (6%) |
| Binoculars (0.5B pair) | 53% (0%) | 54% (4%) |
| Classifier: RAID only | 47% (5%) | 49% (5%) |
| Classifier: RAID + mirrors | 91% (3%) | 95% (2%) |
| All three combined | 93% (4%) | 92% (3%) |

## 2. All detectors at 300 words

| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | AI caught, bar set for 1% | Humans actually flagged |
|---|---|---|---|---|---|
| Clef 27B (asked, no training) | 0.937 (0.91-0.96) | 76% (69%-81%) | 5% (3%-8%) | 54% | 0% |
| Clef-flash 9B | 0.817 (0.78-0.85) | 47% (40%-54%) | 7% (5%-11%) | 37% | 5% |
| Clef-omni | 0.779 (0.74-0.82) | 40% (34%-47%) | 8% (6%-12%) | 30% | 4% |
| Binoculars (0.5B pair) | 0.918 (0.89-0.94) | 73% (66%-78%) | 4% (2%-7%) | 50% | 0% |
| Fast-DetectGPT (0.5B pair) | 0.917 (0.89-0.94) | 70% (64%-76%) | 4% (2%-7%) | 45% | 0% |
| Classifier: RAID only | 0.935 (0.91-0.96) | 69% (63%-75%) | 3% (2%-6%) | 62% | 0% |
| Classifier: RAID + mirrors | 0.988 (0.98-0.99) | 91% (86%-94%) | 3% (1%-5%) | 55% | 0% |
| Clef + Binoculars | 0.977 (0.96-0.99) | 89% (83%-92%) | 5% (3%-8%) | 73% | 1% |
| Clef + classifier (mirrors) | 0.985 (0.98-0.99) | 88% (83%-92%) | 3% (2%-6%) | 78% | 1% |
| Binoculars + classifier (mirrors) | 0.987 (0.98-0.99) | 92% (87%-95%) | 6% (4%-10%) | 83% | 1% |
| All three combined | 0.992 (0.99-1.00) | 95% (91%-97%) | 5% (3%-8%) | 87% | 1% |

Share of each group flagged as AI, bar set for 5% of humans:

| Group | Test texts | Clef 27B (asked, no training) | Binoculars (0.5B pair) | Classifier: RAID only | Classifier: RAID + mirrors | All three combined |
|---|---|---|---|---|---|---|
| Human: RAID documents | 106 | 4% | 5% | 0% | 0% | 3% |
| Human: native-speaker student essays | 40 | 0% | 0% | 0% | 0% | 0% |
| Human: forum posts from 2019 | 65 | 3% | 5% | 9% | 2% | 2% |
| Human: forum posts after the models' release | 52 | 12% | 4% | 4% | 12% | 15% |
| Mixed: human text polished by a 2026 model | 27 | 22% | 7% | 48% | 89% | 59% |
| AI: 2023 chat models (RAID) | 79 | 85% | 78% | 89% | 91% | 95% |
| AI: 2023 models, paraphrased (RAID) | 31 | 71% | 94% | 77% | 84% | 97% |
| AI: 2026 models | 64 | 77% | 66% | 50% | 91% | 95% |
| AI: 2026 models, humanized | 19 | 42% | 37% | 42% | 100% | 89% |

## 3. Human writing the models cannot have seen

Forum posts from the same sites: 2019 (possibly in training data) against posts published after the Clef models were released. An AUROC of 0.5 means the detector scores the two alike; above 0.5 means recent posts look more AI-like to it.

| Detector | 2019 posts | Recent posts | AUROC, recent over 2019 (95% CI) |
|---|---|---|---|
| Clef 27B (asked, no training) | 120 | 116 | 0.71 (0.64-0.77) |
| Clef-flash 9B | 120 | 116 | 0.56 (0.49-0.64) |
| Clef-omni | 120 | 116 | 0.59 (0.52-0.66) |
| Binoculars (0.5B pair) | 120 | 116 | 0.45 (0.37-0.53) |
| Fast-DetectGPT (0.5B pair) | 120 | 116 | 0.45 (0.37-0.52) |
| Classifier: RAID only | 120 | 116 | 0.47 (0.39-0.54) |
| Classifier: RAID + mirrors | 120 | 116 | 0.61 (0.53-0.67) |

## 4. The first test set (150 words or fewer)

| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | AI caught, bar set for 1% | Humans actually flagged |
|---|---|---|---|---|---|
| Clef 27B (asked, no training) | 0.885 (0.85-0.92) | 65% (58%-72%) | 5% (2%-9%) | 40% | 1% |
| Clef-flash 9B | 0.692 (0.64-0.74) | 38% (32%-46%) | 6% (4%-11%) | 33% | 2% |
| Clef-omni | 0.731 (0.68-0.78) | 41% (34%-49%) | 8% (5%-13%) | 31% | 3% |
| Binoculars (0.5B pair) | 0.814 (0.77-0.86) | 45% (38%-53%) | 2% (1%-6%) | 35% | 1% |
| Fast-DetectGPT (0.5B pair) | 0.818 (0.77-0.86) | 51% (44%-59%) | 3% (1%-7%) | 40% | 0% |
| Classifier: RAID only | 0.876 (0.84-0.91) | 61% (54%-68%) | 5% (2%-9%) | 33% | 0% |
| Classifier: RAID + mirrors | 0.984 (0.97-0.99) | 92% (87%-95%) | 8% (4%-12%) | 85% | 2% |
| Clef + Binoculars | 0.939 (0.91-0.96) | 69% (62%-75%) | 3% (2%-7%) | 58% | 1% |
| Clef + classifier (mirrors) | 0.983 (0.97-0.99) | 92% (86%-95%) | 6% (3%-10%) | 79% | 2% |
| Binoculars + classifier (mirrors) | 0.977 (0.96-0.99) | 81% (75%-86%) | 2% (1%-5%) | 69% | 0% |
| All three combined | 0.985 (0.97-0.99) | 95% (91%-97%) | 7% (4%-12%) | 71% | 1% |

Share of each group flagged as AI, bar set for 5% of humans:

| Group | Test texts | Clef 27B (asked, no training) | Binoculars (0.5B pair) | Classifier: RAID only | Classifier: RAID + mirrors | All three combined |
|---|---|---|---|---|---|---|
| Human: RAID documents | 78 | 9% | 0% | 0% | 10% | 8% |
| Human: native-speaker student essays | 48 | 2% | 0% | 0% | 2% | 0% |
| Human: non-native TOEFL essays | 47 | 0% | 9% | 17% | 9% | 13% |
| Mixed: TOEFL essays polished by GPT-4 | 48 | 65% | 2% | 52% | 83% | 83% |
| AI: 2023 chat models (RAID) | 80 | 75% | 61% | 88% | 96% | 96% |
| AI: 2023 models, paraphrased (RAID) | 27 | 30% | 63% | 63% | 93% | 96% |
| AI: 2026 models | 49 | 69% | 18% | 29% | 84% | 94% |
| AI: 2026 models, humanized | 21 | 62% | 24% | 33% | 95% | 90% |

## 5. Current models the classifier was and was not trained on

Share of 2026-model texts flagged, bar set for 5% of humans. "Seen" are the four generators that wrote the training mirrors; the other four never appear in training.

Short set:

| Detector | Generators seen in training | Generators never seen in training |
|---|---|---|
| Clef 27B (asked, no training) | 75% of 24 | 64% of 25 |
| Clef-flash 9B | 29% of 24 | 24% of 25 |
| Clef-omni | 29% of 24 | 16% of 25 |
| Binoculars (0.5B pair) | 25% of 24 | 12% of 25 |
| Fast-DetectGPT (0.5B pair) | 42% of 24 | 20% of 25 |
| Classifier: RAID only | 46% of 24 | 12% of 25 |
| Classifier: RAID + mirrors | 92% of 24 | 76% of 25 |
| Clef + Binoculars | 71% of 24 | 40% of 25 |
| Clef + classifier (mirrors) | 96% of 24 | 96% of 25 |
| Binoculars + classifier (mirrors) | 83% of 24 | 44% of 25 |
| All three combined | 96% of 24 | 92% of 25 |

Long set at 300 words:

| Detector | Generators seen in training | Generators never seen in training |
|---|---|---|
| Clef 27B (asked, no training) | 83% of 30 | 71% of 34 |
| Clef-flash 9B | 47% of 30 | 35% of 34 |
| Clef-omni | 40% of 30 | 24% of 34 |
| Binoculars (0.5B pair) | 60% of 30 | 71% of 34 |
| Fast-DetectGPT (0.5B pair) | 57% of 30 | 71% of 34 |
| Classifier: RAID only | 63% of 30 | 38% of 34 |
| Classifier: RAID + mirrors | 93% of 30 | 88% of 34 |
| Clef + Binoculars | 87% of 30 | 88% of 34 |
| Clef + classifier (mirrors) | 93% of 30 | 91% of 34 |
| Binoculars + classifier (mirrors) | 90% of 30 | 85% of 34 |
| All three combined | 93% of 30 | 97% of 34 |

## 6. Human text polished by a 2026 model

Share of pairs whose score rose after polishing (300 words).

| Detector | Pairs | Score rose |
|---|---|---|
| Clef 27B (asked, no training) | 56 | 96% |
| Clef-flash 9B | 56 | 73% |
| Clef-omni | 56 | 66% |
| Binoculars (0.5B pair) | 56 | 70% |
| Fast-DetectGPT (0.5B pair) | 56 | 70% |
| Classifier: RAID only | 56 | 96% |
| Classifier: RAID + mirrors | 56 | 100% |
