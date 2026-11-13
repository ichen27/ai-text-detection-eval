# Third-party data

`data/eval_set.jsonl` contains excerpts (at most 150 words each) from two published datasets, both
released under the MIT licence by their authors.

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
