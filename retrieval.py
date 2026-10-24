"""Lightweight BM25 retrieval over the scheme documents.

Pure Python, no embedding model to download, works for Hindi (Devanagari) and
English keywords. Swap in a real vector store later without changing callers:
just keep the `SchemeIndex.search(query, k)` signature.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

# Hindi number words / common spoken forms mapped to the tokens used in the data.
SYNONYMS = {
    "kisaan": "kisan", "kheti": "farmer", "khet": "farmer",
    "beti": "daughter", "ladki": "daughter", "betiyan": "daughter",
    "bima": "insurance", "bimari": "health", "ilaaj": "health",
    "padhai": "education", "padhaai": "education",
    "ghar": "house", "makan": "house", "makaan": "house",
    "budhapa": "pension", "buzurg": "pension",
    "gaay": "farmer", "bhains": "farmer",
    "vidhwa": "widow", "viklang": "disabled",
    "rozgar": "employment", "naukri": "employment", "kaam": "employment",
    "dukan": "business", "karobar": "business",
}


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if t in SYNONYMS:
            out.append(SYNONYMS[t])
    return out


def _doc_text(s: dict) -> str:
    # Name and keywords are repeated so they weigh more than long prose.
    return " ".join(
        [
            s["name"], s["name"], s.get("name_hi", ""), s.get("name_hi", ""),
            s["category"], s["summary"], s["eligibility"], s["benefit"],
            s.get("keywords", ""), s.get("keywords", ""),
        ]
    )


class SchemeIndex:
    def __init__(self, schemes: list[dict], k1: float = 1.5, b: float = 0.75):
        self.schemes = schemes
        self.by_id = {s["id"]: s for s in schemes}
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(_doc_text(s))) for s in schemes]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "SchemeIndex":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = tokenize(query)
        if not q:
            return []
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in q:
                f = doc.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                score += self.idf.get(t, 0) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.schemes[i] for _, i in scored[:k]]
