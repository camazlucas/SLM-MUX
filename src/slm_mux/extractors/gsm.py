"""GSM8K answer extractor.

Ported from utils/gsm_utils.py.  Handles the ``#### <number>`` convention
used by GSM8K, with a fallback to the last standalone number in the response.
"""

import re

from slm_mux.extractors.base import AnswerExtractor


def extract_answer_gsm(text: str) -> str:
    """Extract the numeric answer from a GSM8K-style response.

    Looks for the last ``#### <number>`` marker.  If none is found, falls back
    to the last standalone number in the text.

    Returns:
        The extracted number as a string, or empty string on failure.
    """
    if text is None:
        return ""
    # Take the last occurrence of #### to be safe
    # [repro-gsm8k-8b] aceita separador de milhar e "$": o regex original lia "#### 1,000" como "1"
    num = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
    matches = list(re.finditer(r"####\s*\$?\s*(" + num + r")", text))
    if matches:
        return matches[-1].group(1)
    # Fallback: last standalone number
    nums = re.findall(num, text)
    return nums[-1] if nums else ""


def normalize_answer_gsm(ans: str) -> str:
    """Normalise a GSM8K answer by stripping whitespace, commas, and leading zeros."""
    if ans is None:
        return ""
    ans = ans.strip()
    if ans.startswith("####"):
        ans = ans[4:].strip()
    ans = ans.replace(",", "")
    # [repro-gsm8k-8b] "18.00" e "18" são a mesma resposta
    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = ans.split(".")[0]
    # Remove leading zeros (but keep a single zero)
    if re.fullmatch(r"0+", ans):
        return "0"
    ans = re.sub(r"^0+", "", ans)
    return ans


class GSMExtractor(AnswerExtractor):
    """Extractor for GSM8K ``#### <number>`` answers."""

    def extract(self, response: str, **kwargs) -> str:
        """Extract and normalise a numeric answer from the response."""
        raw = extract_answer_gsm(response)
        return normalize_answer_gsm(raw)

    @property
    def name(self) -> str:
        return "gsm"
