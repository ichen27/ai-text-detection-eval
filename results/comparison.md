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
| Clef 27B (asked, no training) | 49% (7%) | 59% (4%) | 81% (5%) |
| Binoculars (0.5B pair) | 63% (2%) | 78% (5%) | 83% (4%) |
| Classifier: RAID only | 74% (5%) | 84% (6%) | 85% (3%) |
| Classifier: RAID + mirrors | 79% (4%) | 89% (6%) | 95% (3%) |
| All three combined | 92% (5%) | 95% (4%) | 98% (5%) |

752 of the long set's 830 documents have been scored by every detector, including Clef. The rest are 2026-model texts that Clef has not scored yet, so this table covers human text and 2023-era machine text only.

The detectors that run locally have scored every document, including the 2026-model texts:

| Detector | 75 words | 150 words | 300 words |
|---|---|---|---|
| Binoculars (0.5B pair) | 58% (2%) | 74% (5%) | 78% (4%) |
| Classifier: RAID only | 70% (5%) | 78% (6%) | 78% (3%) |
| Classifier: RAID + mirrors | 78% (4%) | 91% (6%) | 95% (3%) |
| Binoculars + classifier (mirrors) | 84% (5%) | 91% (5%) | 94% (8%) |

Going from 300 to 500 words, on the 297 documents long enough for both (mostly human text and 2026-model text):

| Detector | 300 words | 500 words |
|---|---|---|
| Binoculars (0.5B pair) | 56% (0%) | 61% (4%) |
| Classifier: RAID only | 58% (5%) | 58% (5%) |
| Classifier: RAID + mirrors | 94% (3%) | 89% (2%) |
| Binoculars + classifier (mirrors) | 89% (2%) | 83% (4%) |

## 2. All detectors at 300 words

| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | AI caught, bar set for 1% | Humans actually flagged |
|---|---|---|---|---|---|
| Clef 27B (asked, no training) | 0.940 (0.91-0.97) | 81% (73%-87%) | 5% (3%-8%) | 59% | 0% |
| Binoculars (0.5B pair) | 0.966 (0.95-0.98) | 83% (75%-89%) | 4% (2%-7%) | 59% | 0% |
| Fast-DetectGPT (0.5B pair) | 0.963 (0.94-0.98) | 80% (72%-86%) | 4% (2%-7%) | 52% | 0% |
| Classifier: RAID only | 0.979 (0.96-0.99) | 85% (78%-91%) | 3% (2%-6%) | 78% | 0% |
| Classifier: RAID + mirrors | 0.991 (0.98-1.00) | 95% (89%-97%) | 3% (2%-6%) | 71% | 0% |
| Clef + Binoculars | 0.992 (0.99-1.00) | 96% (91%-99%) | 2% (1%-4%) | 85% | 1% |
| Clef + classifier (mirrors) | 0.985 (0.97-0.99) | 89% (82%-94%) | 3% (2%-6%) | 76% | 1% |
| Binoculars + classifier (mirrors) | 0.991 (0.98-1.00) | 96% (91%-99%) | 9% (6%-13%) | 90% | 2% |
| All three combined | 0.995 (0.99-1.00) | 98% (94%-99%) | 5% (3%-9%) | 93% | 1% |

Share of each group flagged as AI, bar set for 5% of humans:

| Group | Test texts | Clef 27B (asked, no training) | Binoculars (0.5B pair) | Classifier: RAID only | Classifier: RAID + mirrors | All three combined |
|---|---|---|---|---|---|---|
| Human: RAID documents | 106 | 4% | 5% | 0% | 2% | 5% |
| Human: native-speaker student essays | 40 | 0% | 0% | 0% | 0% | 0% |
| Human: forum posts from 2019 | 65 | 3% | 5% | 9% | 5% | 3% |
| Human: forum posts after the models' release | 52 | 12% | 4% | 4% | 8% | 13% |
| AI: 2023 chat models (RAID) | 79 | 85% | 78% | 89% | 94% | 97% |
| AI: 2023 models, paraphrased (RAID) | 31 | 71% | 94% | 77% | 97% | 100% |

Local detectors on every 300-word document, including 2026-model texts:

| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | AI caught, bar set for 1% | Humans actually flagged |
|---|---|---|---|---|---|
| Binoculars (0.5B pair) | 0.945 (0.92-0.97) | 78% (71%-84%) | 4% (2%-7%) | 56% | 0% |
| Fast-DetectGPT (0.5B pair) | 0.942 (0.92-0.96) | 76% (69%-82%) | 4% (2%-7%) | 51% | 0% |
| Classifier: RAID only | 0.959 (0.94-0.98) | 78% (71%-84%) | 3% (2%-6%) | 70% | 0% |
| Classifier: RAID + mirrors | 0.992 (0.98-1.00) | 95% (90%-97%) | 3% (2%-6%) | 68% | 0% |
| Binoculars + classifier (mirrors) | 0.988 (0.98-0.99) | 94% (89%-97%) | 8% (5%-12%) | 89% | 2% |

| Group | Test texts | Binoculars (0.5B pair) | Classifier: RAID only | Classifier: RAID + mirrors | Binoculars + classifier (mirrors) |
|---|---|---|---|---|---|
| Human: RAID documents | 106 | 5% | 0% | 2% | 9% |
| Human: native-speaker student essays | 40 | 0% | 0% | 0% | 0% |
| Human: forum posts from 2019 | 65 | 5% | 9% | 5% | 8% |
| Human: forum posts after the models' release | 52 | 4% | 4% | 8% | 12% |
| AI: 2023 chat models (RAID) | 79 | 78% | 89% | 94% | 95% |
| AI: 2023 models, paraphrased (RAID) | 31 | 94% | 77% | 97% | 100% |
| AI: 2026 models | 38 | 66% | 58% | 95% | 87% |

## 3. Human writing the models cannot have seen

Forum posts from the same sites: 2019 (possibly in training data) against posts published after the Clef models were released. An AUROC of 0.5 means the detector scores the two alike; above 0.5 means recent posts look more AI-like to it.

| Detector | 2019 posts | Recent posts | AUROC, recent over 2019 (95% CI) |
|---|---|---|---|
| Clef 27B (asked, no training) | 120 | 116 | 0.71 (0.64-0.77) |
| Clef-flash 9B | 120 | 116 | 0.56 (0.49-0.64) |
| Binoculars (0.5B pair) | 120 | 116 | 0.45 (0.37-0.53) |
| Fast-DetectGPT (0.5B pair) | 120 | 116 | 0.45 (0.37-0.52) |
| Classifier: RAID only | 120 | 116 | 0.47 (0.39-0.54) |
| Classifier: RAID + mirrors | 120 | 116 | 0.57 (0.50-0.64) |

## 4. The first test set (150 words or fewer)

| Detector | AUROC (95% CI) | AI caught, bar set for 5% of humans (95% CI) | Humans actually flagged | AI caught, bar set for 1% | Humans actually flagged |
|---|---|---|---|---|---|
| Clef 27B (asked, no training) | 0.885 (0.85-0.92) | 65% (58%-72%) | 5% (2%-9%) | 40% | 1% |
| Clef-flash 9B | 0.692 (0.64-0.74) | 38% (32%-46%) | 6% (4%-11%) | 33% | 2% |
| Clef-omni | 0.731 (0.68-0.78) | 41% (34%-49%) | 8% (5%-13%) | 31% | 3% |
| Binoculars (0.5B pair) | 0.814 (0.77-0.86) | 45% (38%-53%) | 2% (1%-6%) | 35% | 1% |
| Fast-DetectGPT (0.5B pair) | 0.818 (0.77-0.86) | 51% (44%-59%) | 3% (1%-7%) | 40% | 0% |
| Classifier: RAID only | 0.876 (0.84-0.91) | 61% (54%-68%) | 5% (2%-9%) | 33% | 0% |
| Classifier: RAID + mirrors | 0.965 (0.95-0.98) | 86% (80%-90%) | 6% (4%-11%) | 72% | 2% |
| Clef + Binoculars | 0.939 (0.91-0.96) | 69% (62%-75%) | 3% (2%-7%) | 58% | 1% |
| Clef + classifier (mirrors) | 0.960 (0.94-0.98) | 80% (73%-85%) | 5% (2%-9%) | 68% | 0% |
| Binoculars + classifier (mirrors) | 0.944 (0.92-0.96) | 72% (65%-78%) | 3% (1%-7%) | 59% | 2% |
| All three combined | 0.967 (0.95-0.98) | 83% (77%-88%) | 6% (4%-11%) | 63% | 1% |

Share of each group flagged as AI, bar set for 5% of humans:

| Group | Test texts | Clef 27B (asked, no training) | Binoculars (0.5B pair) | Classifier: RAID only | Classifier: RAID + mirrors | All three combined |
|---|---|---|---|---|---|---|
| Human: RAID documents | 78 | 9% | 0% | 0% | 3% | 5% |
| Human: native-speaker student essays | 48 | 2% | 0% | 0% | 0% | 0% |
| Human: non-native TOEFL essays | 47 | 0% | 9% | 17% | 19% | 15% |
| Mixed: TOEFL essays polished by GPT-4 | 48 | 65% | 2% | 52% | 92% | 79% |
| AI: 2023 chat models (RAID) | 80 | 75% | 61% | 88% | 95% | 88% |
| AI: 2023 models, paraphrased (RAID) | 27 | 30% | 63% | 63% | 81% | 74% |
| AI: 2026 models | 49 | 69% | 18% | 29% | 73% | 84% |
| AI: 2026 models, humanized | 21 | 62% | 24% | 33% | 86% | 76% |

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
| Classifier: RAID + mirrors | 88% of 24 | 60% of 25 |
| Clef + Binoculars | 71% of 24 | 40% of 25 |
| Clef + classifier (mirrors) | 92% of 24 | 80% of 25 |
| Binoculars + classifier (mirrors) | 62% of 24 | 28% of 25 |
| All three combined | 92% of 24 | 76% of 25 |

Long set at 300 words (local detectors):

| Detector | Generators seen in training | Generators never seen in training |
|---|---|---|
| Binoculars (0.5B pair) | 68% of 19 | 63% of 19 |
| Fast-DetectGPT (0.5B pair) | 68% of 19 | 63% of 19 |
| Classifier: RAID only | 74% of 19 | 42% of 19 |
| Classifier: RAID + mirrors | 95% of 19 | 95% of 19 |
| Binoculars + classifier (mirrors) | 95% of 19 | 79% of 19 |
