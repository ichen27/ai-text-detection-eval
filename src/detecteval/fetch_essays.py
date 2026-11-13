"""Download the essay sets from Liang et al. (2023), "GPT detectors are biased against non-native
English writers" (https://github.com/Weixin-Liang/ChatGPT-Detector-Bias, MIT licence)."""
from __future__ import annotations

import urllib.request

from . import config

BASE = "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/Data_and_Results/Human_Data/"
SETS = ["TOEFL_real_91", "TOEFL_gpt4polished_91", "HewlettStudentEssay_real_88"]


def main() -> None:
    out_dir = config.raw_dir() / "liang2023"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        target = out_dir / f"{name}.json"
        if not target.exists():
            with urllib.request.urlopen(BASE + name + "/data.json", timeout=60) as response:
                target.write_bytes(response.read())
        print(target)


if __name__ == "__main__":
    main()
