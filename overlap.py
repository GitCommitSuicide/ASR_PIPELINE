"""Overlapped speech: cut the time range out, split it into one stream per speaker with MossFormer2
(ClearVoice, trained on overlapped 2-speaker speech), name each stream by voice, transcribe it."""
import os, itertools
import numpy as np
import soundfile as sf
from config import SR
from asr import get_recognizer
from diarization import get_extractor, embed

_separator = None


def get_separator():
    global _separator
    if _separator is None:
        from clearvoice import ClearVoice
        _separator = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    return _separator


def separate(mix_path):
    out = get_separator()(input_path=mix_path, online_write=False)
    if isinstance(out, dict):
        out = out[next(iter(out))]
    return [np.squeeze(np.asarray(x, dtype=np.float32)) for x in list(out)[:2]]


def match_speakers(sources, speakers, refs, ext):
    """The separator's streams are unlabeled - decide which stream is which speaker by voice."""
    cands = [s for s in speakers if s in refs]
    fallback = (list(speakers) + ["UNKNOWN"] * len(sources))[:len(sources)]
    if len(cands) < len(sources) or any(len(x) < int(0.5 * SR) for x in sources):
        return fallback
    embs = [embed(ext, x) for x in sources]
    best, best_score = None, -9.0
    for perm in itertools.permutations(range(len(cands)), len(sources)):
        score = sum(float(embs[i] @ refs[cands[j]]) for i, j in enumerate(perm))
        if score > best_score:
            best, best_score = perm, score
    return [cands[j] for j in best]


def process_overlaps(audio, overlaps, refs, model_dir, out_dir):
    """Fills o['audio'] (the mix) and o['streams'] = [{'speaker', 'audio', 'text'}] for every overlap."""
    if not overlaps:
        return
    os.makedirs(out_dir, exist_ok=True)
    cpus = os.cpu_count() or 4
    asr = get_recognizer(model_dir, cpus, "overlap")
    ext = get_extractor(cpus)
    for i, o in enumerate(overlaps, 1):
        mix_path = os.path.join(out_dir, f"overlap_{i:03d}_mix.wav")
        sf.write(mix_path, audio[int(o["start"] * SR):int(o["end"] * SR)], SR)
        o["audio"], o["streams"] = mix_path, []
        try:
            sources = separate(mix_path)
        except Exception as ex:                       # e.g. clearvoice not installed
            o["error"] = str(ex)
            continue
        names = match_speakers(sources, o["speakers"], refs, ext)
        for src, name, text in zip(sources, names, asr.texts(sources)):
            path = os.path.join(out_dir, f"overlap_{i:03d}_{name}.wav")
            sf.write(path, src, SR)
            o["streams"].append({"speaker": name, "audio": path, "text": text})
