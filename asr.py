"""Parakeet TDT 0.6B v3 transcription. Returns WORDS with timestamps, so speakers can be
assigned word by word instead of per chunk."""
import os, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from config import (SR, MODEL_ROOT, ASR_DIR, ASR_URL, ASR_FILES, ASR_WORKERS, BATCH)


def ensure_asr():
    model_dir = os.path.join(MODEL_ROOT, ASR_DIR)
    if not all(os.path.exists(os.path.join(model_dir, f)) for f in ASR_FILES):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        print("Downloading Parakeet TDT 0.6B v3 int8 (~0.5 GB)...", flush=True)
        tar_path = os.path.join(MODEL_ROOT, ASR_DIR + ".tar.bz2")
        urllib.request.urlretrieve(ASR_URL, tar_path)
        with tarfile.open(tar_path, "r:bz2") as t:
            t.extractall(MODEL_ROOT)
        os.remove(tar_path)
    return model_dir


class Parakeet:
    def __init__(self, model_dir, threads):
        import sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(model_dir, "encoder.int8.onnx"),
            decoder=os.path.join(model_dir, "decoder.int8.onnx"),
            joiner=os.path.join(model_dir, "joiner.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=threads, provider="cpu", decoding_method="greedy_search")

    def recognize(self, clips):
        """clips -> list of (text, tokens, timestamps)."""
        streams = []
        for c in clips:
            st = self.rec.create_stream()
            st.accept_waveform(SR, c)
            streams.append(st)
        self.rec.decode_streams(streams)
        out = []
        for st in streams:
            r = st.result
            out.append((r.text.strip(), list(r.tokens), list(r.timestamps)))
        return out

    def texts(self, clips):
        """Plain texts; clips shorter than 0.3 s give ''."""
        ok = [i for i, c in enumerate(clips) if len(c) >= int(0.3 * SR)]
        res = [""] * len(clips)
        if ok:
            for i, (text, _, _) in zip(ok, self.recognize([clips[i] for i in ok])):
                res[i] = text
        return res


# Recognizers are slow to build, so they are kept between runs (Streamlit reruns reuse them).
_cache = {}


def get_recognizer(model_dir, threads, slot):
    key = (model_dir, threads, slot)
    if key not in _cache:
        _cache[key] = Parakeet(model_dir, threads)
    return _cache[key]


def to_words(text, tokens, stamps, t0, t1):
    """Group sentencepiece tokens into words with start/end times (seconds, whole-file clock)."""
    if tokens and len(tokens) == len(stamps):
        words = []
        for tok, t in zip(tokens, stamps):
            if tok.startswith("\u2581") or not words:
                words.append({"text": tok.lstrip("\u2581"), "start": t0 + float(t)})
            else:
                words[-1]["text"] += tok
        words = [w for w in words if w["text"].strip()]
        for i, w in enumerate(words):
            nxt = words[i + 1]["start"] if i + 1 < len(words) else t1
            w["end"] = max(w["start"] + 0.05, min(nxt, w["start"] + 0.6))
        return words
    return [{"text": text, "start": t0, "end": t1}] if text else []   # no timestamps: whole chunk


def _worker(audio, spans, model_dir, threads, slot, on_batch):
    rec = get_recognizer(model_dir, threads, slot)
    words = []
    for i in range(0, len(spans), BATCH):
        batch = spans[i:i + BATCH]
        for (s, e), (text, toks, stamps) in zip(batch, rec.recognize([audio[s:e] for s, e in batch])):
            if text:
                words += to_words(text, toks, stamps, s / SR, e / SR)
        if on_batch:
            on_batch(len(batch))
    return words


def pick_workers(spans):
    """Short audio does not need 4 model copies - loading them would cost more than it saves."""
    speech_s = sum(e - s for s, e in spans) / SR
    return max(1, min(ASR_WORKERS, int(speech_s // 90) + 1))


def transcribe(audio, spans, model_dir, workers, threads, on_batch=None):
    if not spans:
        return []
    groups = [g for g in np.array_split(np.arange(len(spans)), workers) if len(g)]
    with ThreadPoolExecutor(len(groups)) as ex:
        parts = list(ex.map(
            lambda a: _worker(audio, [spans[i] for i in a[1]], model_dir, threads, a[0], on_batch),
            enumerate(groups)))
    words = [w for part in parts for w in part]
    words.sort(key=lambda w: w["start"])
    return words
