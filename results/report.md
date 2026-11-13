# Results

Texts scored: 836. Scores run 0 (human) to 1 (AI).

## Overall separation (human vs AI, mixed texts excluded)

| Model | Question | AUROC (95% CI) | Detected at 5% FPR (95% CI) | Detected at 1% FPR | Humans flagged at 0.5 | AI flagged at 0.5 | ECE |
|---|---|---|---|---|---|---|---|
| clef-flash | Yes/no: AI-generated? | 0.688 (0.65-0.72) | 38% (31%-43%) | 22% | 14% | 48% | 0.17 |
| clef-flash | Choice: human or AI | 0.726 (0.69-0.76) | 41% (36%-48%) | 35% | 38% | 67% | 0.15 |
| clef-flash | Score: what % is AI? | 0.648 (0.61-0.69) | 31% (24%-38%) | 18% | 26% | 49% | 0.13 |
| clef | Yes/no: AI-generated? | 0.884 (0.86-0.91) | 62% (52%-69%) | 38% | 3% | 51% | 0.24 |
| clef | Choice: human or AI | 0.808 (0.78-0.84) | 48% (43%-54%) | 25% | 2% | 39% | 0.29 |
| clef | Score: what % is AI? | 0.886 (0.86-0.91) | 67% (58%-73%) | 46% | 8% | 71% | 0.15 |
| clef-omni | Yes/no: AI-generated? | 0.721 (0.68-0.76) | 24% (11%-30%) | 4% | 22% | 53% | 0.12 |
| clef-omni | Choice: human or AI | 0.767 (0.73-0.80) | 43% (35%-51%) | 21% | 6% | 46% | 0.22 |
| clef-omni | Score: what % is AI? | 0.701 (0.66-0.74) | 13% (9%-24%) | 5% | 24% | 48% | 0.12 |

## By group: Yes/no: AI-generated?

Share flagged as AI at the threshold that flags 5% of all human texts (and at a fixed 0.5).

| Group | n | clef-flash | clef | clef-omni |
|---|---|---|---|---|
| Human: RAID (abstracts, books, news, poetry) | 160 | 4% (15%), mean 0.22 | 5% (2%), mean 0.08 | 2% (13%), mean 0.25 |
| Human: native-speaker student essays | 88 | 9% (20%), mean 0.35 | 2% (2%), mean 0.06 | 14% (53%), mean 0.53 |
| Human: non-native TOEFL essays | 91 | 1% (7%), mean 0.25 | 7% (3%), mean 0.15 | 0% (9%), mean 0.28 |
| Mixed: TOEFL essays polished by GPT-4 | 90 | 9% (36%), mean 0.41 | 54% (41%), mean 0.42 | 3% (36%), mean 0.43 |
| AI: 2023 chat models (RAID) | 186 | 48% (60%), mean 0.56 | 72% (63%), mean 0.57 | 33% (69%), mean 0.59 |
| AI: 2023 models, paraphrased (RAID) | 60 | 30% (40%), mean 0.42 | 53% (42%), mean 0.41 | 13% (38%), mean 0.49 |
| AI: 2026 models | 113 | 32% (43%), mean 0.43 | 57% (45%), mean 0.43 | 18% (41%), mean 0.45 |
| AI: 2026 models, humanized | 48 | 21% (25%), mean 0.35 | 46% (29%), mean 0.37 | 15% (35%), mean 0.43 |

## By group: Choice: human or AI

Share flagged as AI at the threshold that flags 5% of all human texts (and at a fixed 0.5).

| Group | n | clef-flash | clef | clef-omni |
|---|---|---|---|---|
| Human: RAID (abstracts, books, news, poetry) | 160 | 4% (26%), mean 0.31 | 6% (2%), mean 0.08 | 2% (4%), mean 0.16 |
| Human: native-speaker student essays | 88 | 10% (57%), mean 0.54 | 7% (3%), mean 0.09 | 12% (15%), mean 0.25 |
| Human: non-native TOEFL essays | 91 | 1% (40%), mean 0.45 | 1% (0%), mean 0.09 | 1% (1%), mean 0.11 |
| Mixed: TOEFL essays polished by GPT-4 | 90 | 19% (74%), mean 0.67 | 21% (7%), mean 0.20 | 4% (4%), mean 0.18 |
| AI: 2023 chat models (RAID) | 186 | 54% (77%), mean 0.73 | 57% (45%), mean 0.45 | 63% (66%), mean 0.59 |
| AI: 2023 models, paraphrased (RAID) | 60 | 33% (63%), mean 0.61 | 38% (32%), mean 0.31 | 32% (32%), mean 0.38 |
| AI: 2026 models | 113 | 34% (58%), mean 0.58 | 47% (37%), mean 0.40 | 27% (31%), mean 0.35 |
| AI: 2026 models, humanized | 48 | 21% (50%), mean 0.50 | 31% (27%), mean 0.30 | 21% (23%), mean 0.28 |

## By group: Score: what % is AI?

Share flagged as AI at the threshold that flags 5% of all human texts (and at a fixed 0.5).

| Group | n | clef-flash | clef | clef-omni |
|---|---|---|---|---|
| Human: RAID (abstracts, books, news, poetry) | 160 | 6% (18%), mean 0.30 | 6% (9%), mean 0.17 | 2% (14%), mean 0.31 |
| Human: native-speaker student essays | 88 | 6% (43%), mean 0.47 | 3% (5%), mean 0.12 | 15% (61%), mean 0.52 |
| Human: non-native TOEFL essays | 91 | 1% (24%), mean 0.40 | 3% (9%), mean 0.29 | 0% (5%), mean 0.30 |
| Mixed: TOEFL essays polished by GPT-4 | 90 | 7% (58%), mean 0.50 | 58% (66%), mean 0.55 | 0% (14%), mean 0.37 |
| AI: 2023 chat models (RAID) | 186 | 38% (57%), mean 0.51 | 73% (76%), mean 0.60 | 19% (61%), mean 0.51 |
| AI: 2023 models, paraphrased (RAID) | 60 | 23% (40%), mean 0.44 | 43% (48%), mean 0.45 | 7% (43%), mean 0.47 |
| AI: 2026 models | 113 | 31% (47%), mean 0.45 | 71% (75%), mean 0.59 | 11% (36%), mean 0.44 |
| AI: 2026 models, humanized | 48 | 17% (31%), mean 0.41 | 62% (67%), mean 0.55 | 4% (29%), mean 0.44 |

## Calibration of the percentage question

If the score were a calibrated probability, the share of AI texts in each bin would match the bin.

| Score bin | clef-flash: n, actual AI share | clef: n, actual AI share | clef-omni: n, actual AI share |
|---|---|---|---|
| 0.0-0.2 | 136, 44% | 257, 18% | 72, 15% |
| 0.2-0.4 | 208, 45% | 123, 37% | 258, 45% |
| 0.4-0.6 | 227, 48% | 118, 64% | 295, 63% |
| 0.6-0.8 | 173, 83% | 216, 96% | 121, 79% |
| 0.8-1.0 | 2, 100% | 32, 100% | 0, n/a |

## Effect of GPT-4 polishing on the same TOEFL essays

| Model | Question | Pairs | Mean score change | Share of essays whose score rose |
|---|---|---|---|---|
| clef-flash | Yes/no: AI-generated? | 90 | +0.160 | 86% |
| clef-flash | Choice: human or AI | 90 | +0.217 | 89% |
| clef-flash | Score: what % is AI? | 90 | +0.096 | 88% |
| clef | Yes/no: AI-generated? | 90 | +0.272 | 94% |
| clef | Choice: human or AI | 90 | +0.100 | 89% |
| clef | Score: what % is AI? | 90 | +0.252 | 98% |
| clef-omni | Yes/no: AI-generated? | 90 | +0.151 | 98% |
| clef-omni | Choice: human or AI | 90 | +0.073 | 83% |
| clef-omni | Score: what % is AI? | 90 | +0.069 | 87% |

## Detection by writer (percentage question, 5% FPR threshold)

Flagged / total. Counts per writer are small, so treat differences between writers as rough.

| Group / writer | clef-flash | clef | clef-omni |
|---|---|---|---|
| fresh_2026 / deepseek-v4.1-flash-1017 | 3/15 | 5/15 | 0/15 |
| fresh_2026 / gemma-4-26b-a4b-it | 4/14 | 13/14 | 1/14 |
| fresh_2026 / glm-5.3 | 3/12 | 6/12 | 0/12 |
| fresh_2026 / gpt-oss-120b | 7/16 | 13/16 | 3/16 |
| fresh_2026 / kimi-k2.6 | 3/14 | 8/14 | 0/14 |
| fresh_2026 / llama-4-scout-17b-16e-instruct | 6/16 | 14/16 | 5/16 |
| fresh_2026 / mistral-small-3.1-24b-instruct | 8/16 | 13/16 | 3/16 |
| fresh_2026 / qwen3.8-27b | 1/10 | 8/10 | 0/10 |
| humanized_2026 / deepseek-v4.1-flash-1017 | 0/6 | 2/6 | 0/6 |
| humanized_2026 / gemma-4-26b-a4b-it | 0/6 | 3/6 | 0/6 |
| humanized_2026 / glm-5.3 | 0/6 | 3/6 | 0/6 |
| humanized_2026 / gpt-oss-120b | 2/6 | 3/6 | 0/6 |
| humanized_2026 / kimi-k2.6 | 1/6 | 6/6 | 0/6 |
| humanized_2026 / llama-4-scout-17b-16e-instruct | 1/6 | 4/6 | 1/6 |
| humanized_2026 / mistral-small-3.1-24b-instruct | 3/6 | 5/6 | 1/6 |
| humanized_2026 / qwen3.8-27b | 1/6 | 4/6 | 0/6 |
| raid_2023 / chatgpt | 7/27 | 19/27 | 1/27 |
| raid_2023 / cohere-chat | 5/27 | 13/27 | 3/27 |
| raid_2023 / gpt4 | 9/27 | 19/27 | 2/27 |
| raid_2023 / llama-chat | 16/36 | 29/36 | 7/36 |
| raid_2023 / mistral-chat | 18/36 | 33/36 | 14/36 |
| raid_2023 / mpt-chat | 16/33 | 23/33 | 8/33 |
| raid_paraphrased / chatgpt | 1/15 | 3/15 | 0/15 |
| raid_paraphrased / gpt4 | 5/15 | 5/15 | 0/15 |
| raid_paraphrased / llama-chat | 4/15 | 9/15 | 2/15 |
| raid_paraphrased / mistral-chat | 4/15 | 9/15 | 2/15 |

## Run-to-run variation

The same question asked twice, in separate requests. Cells show mean (max) absolute difference in score.

| Model | Texts | Yes/no: AI-generated? | Choice: human or AI | Score: what % is AI? |
|---|---|---|---|---|
| clef-flash | 100 | 0.003 (0.109) | 0.008 (0.216) | 0.004 (0.083) |
| clef | 100 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| clef-omni | 100 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |

## Do questions in one request affect each other?

Each question asked alone versus all three in one request. Cells show mean (max) absolute difference in score.

| Model | Texts | Yes/no: AI-generated? | Choice: human or AI | Score: what % is AI? |
|---|---|---|---|---|
| clef-flash | 836 | 0.062 (0.337) | 0.092 (0.527) | 0.067 (0.339) |
| clef | 836 | 0.048 (0.571) | 0.139 (0.843) | 0.057 (0.421) |
| clef-omni | 836 | 0.124 (0.277) | 0.277 (0.689) | 0.190 (0.462) |
