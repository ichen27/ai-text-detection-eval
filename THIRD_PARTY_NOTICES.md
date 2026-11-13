# Third-party data and models

`data/eval_set.jsonl` and `data/long_set.jsonl` contain excerpts (at most 150 and 500 words each) from
two published datasets released under the MIT licence, and `data/long_set.jsonl` also contains Stack
Exchange posts under CC BY-SA 4.0. Sources, licences and the methods implemented here are listed below.

## RAID

Dugan, L., Hwang, A., Trhlik, F., Ludan, J. M., Zhu, A., Xu, H., Ippolito, D., and Callison-Burch, C.
"RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors." ACL 2024.
https://github.com/liamdugan/raid — https://huggingface.co/datasets/liamdugan/raid (MIT).

Used for: human-written texts, 2023-era model generations and paraphrased generations
(groups `raid_human`, `raid_2023`, `raid_paraphrased`).

## GPT detectors are biased against non-native English writers

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., and Zou, J. Patterns 4(7), 2023.
https://github.com/Weixin-Liang/ChatGPT-Detector-Bias (MIT, per the repository README).

Used for: TOEFL essays by non-native writers, the same essays polished by GPT-4, and US student essays
(groups `essay_nonnative`, `toefl_gpt4_polished`, `essay_native`). The student essays originate from the
Hewlett Foundation ASAP corpus; their anonymisation tokens were replaced with plain fillers here.

## Stack Exchange

Questions from Stack Exchange network sites, fetched through the public API, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Used for the human groups
`forum_recent` and `forum_2019` in `data/long_set.jsonl`. Each record carries the attribution the licence
asks for: the post's `link`, its `author` display name, its `title` and `created` date. The texts were
changed as follows: code blocks, quotations and tables were removed, HTML was stripped, whitespace was
collapsed and the text was cut to at most 500 words. Because these records are adapted CC BY-SA material,
those records remain under CC BY-SA 4.0 whatever licence the rest of the repository takes.

The group `polished_2026` contains model-edited versions of some of those posts and of some student
essays; the forum-derived ones are likewise adaptations under CC BY-SA 4.0.

Posts used only for training the classifier (2017-2018, from a separate list of sites) are fetched by
`fetch_stackexchange` and are not redistributed here.

## Methods implemented from published papers

No code was copied from these projects; the statistics are implemented from the papers' descriptions.

- Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., and
  Goldstein, T. "Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text." ICML 2024.
- Bao, G., Zhao, Y., Teng, Z., Yang, L., and Zhang, Y. "Fast-DetectGPT: Efficient Zero-Shot Detection of
  Machine-Generated Text via Conditional Probability Curvature." ICLR 2024.
- Emi, B., and Spero, M. "Technical Report on the Pangram AI-Generated Text Classifier." 2024. The idea of
  training on machine "mirrors" of human documents comes from this report.

## Models downloaded at run time

Not redistributed here; each is fetched from Hugging Face when the local detectors run.

- `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0), the model pair for Binoculars.
- `distilbert/distilroberta-base` (Apache-2.0), the encoder fine-tuned into the classifier.

## MIT licence text

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
