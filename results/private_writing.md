# Private writing

How often each detector flags documents that were never published. The flagging bar is the one set to flag 5% of human texts on the calibration half of the long set; these documents had no part in setting it. A detector that treats this writing like other human writing should flag about 5% of it.

Long documents are cut into consecutive passages of up to 500 words, so passages from one document are not independent. Cells read: passages flagged of passages scored (share; 95% interval treating passages as independent), then documents with at least one flagged passage.

| Group | Documents | Passages |
|---|---|---|
| Personal journal, 2024 | 1 | 66 |
| Dated before ChatGPT (30 Nov 2022) | 32 | 60 |
| School work, undated (2019 to mid-2023) | 59 | 82 |

The texts, titles and per-document scores are not in this repository. A blank cell means the detector was not run on that group.

## At 75 words

| Detector | Public human texts (test half) | Personal journal, 2024 | Dated before ChatGPT (30 Nov 2022) | School work, undated (2019 to mid-2023) |
|---|---|---|---|---|
| Clef 27B, asked | 19 of 263 (7%; 5%-11%) |  | 19 of 60 (32%; 21%-44%); 15 of 32 documents | 24 of 82 (29%; 21%-40%); 20 of 59 documents |
| Binoculars (0.5B pair) | 6 of 263 (2%; 1%-5%) | 2 of 66 (3%; 1%-10%); 1 of 1 documents | 5 of 60 (8%; 4%-18%); 5 of 32 documents | 4 of 82 (5%; 2%-12%); 4 of 59 documents |
| Classifier, RAID only | 12 of 263 (5%; 3%-8%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 10 of 60 (17%; 9%-28%); 10 of 32 documents | 11 of 82 (13%; 8%-22%); 10 of 59 documents |
| Classifier, RAID + mirrors | 15 of 263 (6%; 3%-9%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 8 of 60 (13%; 7%-24%); 7 of 32 documents | 4 of 82 (5%; 2%-12%); 4 of 59 documents |
| Binoculars + classifier (mirrors) | 8 of 263 (3%; 2%-6%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 10 of 60 (17%; 9%-28%); 9 of 32 documents | 6 of 82 (7%; 3%-15%); 6 of 59 documents |
| Clef + Binoculars | 17 of 263 (6%; 4%-10%) |  | 35 of 60 (58%; 46%-70%); 22 of 32 documents | 32 of 82 (39%; 29%-50%); 25 of 59 documents |
| All three combined | 10 of 263 (4%; 2%-7%) |  | 19 of 60 (32%; 21%-44%); 14 of 32 documents | 18 of 82 (22%; 14%-32%); 16 of 59 documents |

## At 150 words

| Detector | Public human texts (test half) | Personal journal, 2024 | Dated before ChatGPT (30 Nov 2022) | School work, undated (2019 to mid-2023) |
|---|---|---|---|---|
| Clef 27B, asked | 11 of 263 (4%; 2%-7%) |  | 16 of 60 (27%; 17%-39%); 11 of 32 documents | 25 of 82 (30%; 22%-41%); 22 of 59 documents |
| Binoculars (0.5B pair) | 13 of 263 (5%; 3%-8%) | 2 of 66 (3%; 1%-10%); 1 of 1 documents | 7 of 60 (12%; 6%-22%); 7 of 32 documents | 7 of 82 (9%; 4%-17%); 7 of 59 documents |
| Classifier, RAID only | 15 of 263 (6%; 3%-9%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 14 of 60 (23%; 14%-35%); 13 of 32 documents | 11 of 82 (13%; 8%-22%); 10 of 59 documents |
| Classifier, RAID + mirrors | 12 of 263 (5%; 3%-8%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 6 of 60 (10%; 5%-20%); 5 of 32 documents | 4 of 82 (5%; 2%-12%); 4 of 59 documents |
| Binoculars + classifier (mirrors) | 13 of 263 (5%; 3%-8%) | 0 of 66 (0%; 0%-6%); 0 of 1 documents | 9 of 60 (15%; 8%-26%); 9 of 32 documents | 6 of 82 (7%; 3%-15%); 6 of 59 documents |
| Clef + Binoculars | 10 of 263 (4%; 2%-7%) |  | 30 of 60 (50%; 38%-62%); 24 of 32 documents | 31 of 82 (38%; 28%-49%); 25 of 59 documents |
| All three combined | 9 of 263 (3%; 2%-6%) |  | 23 of 60 (38%; 27%-51%); 21 of 32 documents | 16 of 82 (20%; 12%-29%); 14 of 59 documents |

## At 300 words

| Detector | Public human texts (test half) | Personal journal, 2024 | Dated before ChatGPT (30 Nov 2022) | School work, undated (2019 to mid-2023) |
|---|---|---|---|---|
| Clef 27B, asked | 12 of 263 (5%; 3%-8%) |  | 21 of 50 (42%; 29%-56%); 15 of 29 documents | 20 of 65 (31%; 21%-43%); 16 of 48 documents |
| Binoculars (0.5B pair) | 10 of 263 (4%; 2%-7%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 2 of 50 (4%; 1%-13%); 2 of 29 documents | 4 of 65 (6%; 2%-15%); 4 of 48 documents |
| Classifier, RAID only | 8 of 263 (3%; 2%-6%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 3 of 50 (6%; 2%-16%); 3 of 29 documents | 8 of 65 (12%; 6%-22%); 7 of 48 documents |
| Classifier, RAID + mirrors | 7 of 263 (3%; 1%-5%) | 2 of 65 (3%; 1%-11%); 1 of 1 documents | 4 of 50 (8%; 3%-19%); 4 of 29 documents | 4 of 65 (6%; 2%-15%); 4 of 48 documents |
| Binoculars + classifier (mirrors) | 17 of 263 (6%; 4%-10%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 11 of 50 (22%; 13%-35%); 11 of 29 documents | 7 of 65 (11%; 5%-21%); 7 of 48 documents |
| Clef + Binoculars | 12 of 263 (5%; 3%-8%) |  | 32 of 50 (64%; 50%-76%); 23 of 29 documents | 28 of 65 (43%; 32%-55%); 23 of 48 documents |
| All three combined | 12 of 263 (5%; 3%-8%) |  | 22 of 50 (44%; 31%-58%); 20 of 29 documents | 19 of 65 (29%; 20%-41%); 17 of 48 documents |

## At 500 words

| Detector | Public human texts (test half) | Personal journal, 2024 | Dated before ChatGPT (30 Nov 2022) | School work, undated (2019 to mid-2023) |
|---|---|---|---|---|
| Clef 27B, asked | 7 of 111 (6%; 3%-12%) |  | 13 of 37 (35%; 22%-51%); 12 of 25 documents | 12 of 43 (28%; 17%-43%); 9 of 29 documents |
| Binoculars (0.5B pair) | 4 of 111 (4%; 1%-9%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 3 of 37 (8%; 3%-21%); 3 of 25 documents | 5 of 43 (12%; 5%-24%); 5 of 29 documents |
| Classifier, RAID only | 6 of 111 (5%; 3%-11%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 6 of 37 (16%; 8%-31%); 6 of 25 documents | 9 of 43 (21%; 11%-35%); 9 of 29 documents |
| Classifier, RAID + mirrors | 2 of 111 (2%; 0%-6%) | 2 of 65 (3%; 1%-11%); 1 of 1 documents | 4 of 37 (11%; 4%-25%); 4 of 25 documents | 7 of 43 (16%; 8%-30%); 7 of 29 documents |
| Binoculars + classifier (mirrors) | 2 of 111 (2%; 0%-6%) | 1 of 65 (2%; 0%-8%); 1 of 1 documents | 7 of 37 (19%; 9%-34%); 7 of 25 documents | 13 of 43 (30%; 19%-45%); 12 of 29 documents |
| Clef + Binoculars | 9 of 111 (8%; 4%-15%) |  | 16 of 37 (43%; 29%-59%); 14 of 25 documents | 18 of 43 (42%; 28%-57%); 15 of 29 documents |
| All three combined | 3 of 111 (3%; 1%-8%) |  | 13 of 37 (35%; 22%-51%); 11 of 25 documents | 15 of 43 (35%; 22%-50%); 13 of 29 documents |

## If the bar were set on the private human group ("Dated before ChatGPT (30 Nov 2022)")

The bar is moved to flag 5% of that group's passages, and AI text caught is measured on the public test half. The bar is set on the same private passages it is judged on, so treat these as indicative. AUROC compares public AI text with each kind of human text; 1.0 is perfect separation.

| Length | Detector | AI caught, bar set on public humans | AI caught, bar set on private humans | AUROC against public humans | AUROC against private humans |
|---|---|---|---|---|---|
| 75 words | Clef 27B, asked | 49% | 36% | 0.83 | 0.58 |
| 75 words | Binoculars (0.5B pair) | 51% | 41% | 0.88 | 0.81 |
| 75 words | Classifier, RAID only | 60% | 34% | 0.88 | 0.77 |
| 75 words | Classifier, RAID + mirrors | 78% | 63% | 0.96 | 0.91 |
| 150 words | Clef 27B, asked | 59% | 41% | 0.89 | 0.66 |
| 150 words | Binoculars (0.5B pair) | 69% | 54% | 0.91 | 0.85 |
| 150 words | Classifier, RAID only | 69% | 53% | 0.92 | 0.81 |
| 150 words | Classifier, RAID + mirrors | 85% | 69% | 0.98 | 0.95 |
| 300 words | Clef 27B, asked | 76% | 65% | 0.94 | 0.77 |
| 300 words | Binoculars (0.5B pair) | 73% | 73% | 0.92 | 0.87 |
| 300 words | Classifier, RAID only | 69% | 65% | 0.93 | 0.83 |
| 300 words | Classifier, RAID + mirrors | 91% | 85% | 0.99 | 0.97 |
| 500 words | Clef 27B, asked | 74% | 64% | 0.94 | 0.81 |
| 500 words | Binoculars (0.5B pair) | 54% | 49% | 0.83 | 0.77 |
| 500 words | Classifier, RAID only | 49% | 38% | 0.88 | 0.70 |
| 500 words | Classifier, RAID + mirrors | 95% | 79% | 0.99 | 0.97 |
