"""Text sentiment with RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest): positive / neutral / negative."""
import numpy as np
from config import SENTIMENT_MODEL

_pipe = None
_LEGACY = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text-classification", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL,
                         top_k=None, device=-1, truncation=True, max_length=256)
    return _pipe


def analyze(texts):
    """-> [{'label': 'positive'|'neutral'|'negative', 'polarity': P(positive) - P(negative)}]"""
    if not texts:
        return []
    results = []
    for out in get_pipe()(list(texts), batch_size=16):
        if isinstance(out, dict):
            out = [out]
        p = {}
        for d in out:
            lab = d["label"].lower()
            p[_LEGACY.get(lab, lab)] = float(d["score"])
        results.append({"label": max(p, key=p.get), "polarity": round(p.get("positive", 0.0) - p.get("negative", 0.0), 3)})
    return results


def tag(items):
    """Add 'sentiment' and 'polarity' to every dict that has a 'text'."""
    res = analyze([i["text"] for i in items])
    for item, r in zip(items, res):
        item["sentiment"], item["polarity"] = r["label"], r["polarity"]


def summarize(segments):
    """Per speaker: overall sentiment (duration-weighted), average polarity, and counts."""
    by = {}
    for s in segments:
        by.setdefault(s["speaker"], []).append(s)
    out = {}
    for spk, segs in by.items():
        w = [max(0.2, s["end"] - s["start"]) for s in segs]
        mean = float(np.average([s["polarity"] for s in segs], weights=w))
        out[spk] = {
            "overall": "positive" if mean >= 0.15 else "negative" if mean <= -0.15 else "neutral",
            "mean_polarity": round(mean, 2),
            "counts": {k: sum(1 for s in segs if s["sentiment"] == k) for k in ("positive", "neutral", "negative")},
            "lines": len(segs),
        }
    return out
