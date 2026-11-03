"""Who spoke when.

Pass 1  pyannote segmentation-3.0 (overlap-aware) + TitaNet + sherpa's clustering. It may over-split.
Pass 2  our own hierarchical (average-linkage) clustering of each pass-1 speaker's voice fingerprint,
        so the same person who was split in pieces is merged back, and tiny noise "speakers" disappear.
"""
import os
import numpy as np
from config import (SR, MODEL_ROOT, SEG_HF_REPO, SEG_FILE, EMB_HF_REPO, EMB_FILE, SHERPA_THRESHOLD,
                    MIN_SPEAKER_S, DIAR_THREADS, MIN_OVERLAP, MAX_GAP_S, MAX_SEGMENT_S)


def ensure_diar_models():
    from huggingface_hub import hf_hub_download
    seg_dir = os.path.join(MODEL_ROOT, "pyannote-segmentation-3-0")
    seg, emb = os.path.join(seg_dir, SEG_FILE), os.path.join(MODEL_ROOT, EMB_FILE)
    if not os.path.exists(seg):
        os.makedirs(seg_dir, exist_ok=True)
        hf_hub_download(SEG_HF_REPO, SEG_FILE, local_dir=seg_dir)
    if not os.path.exists(emb):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        hf_hub_download(EMB_HF_REPO, EMB_FILE, local_dir=MODEL_ROOT)
    return seg, emb


# ---------- voice fingerprints ----------
_extractors = {}


def get_extractor(threads):
    if threads not in _extractors:
        import sherpa_onnx
        _, emb = ensure_diar_models()
        _extractors[threads] = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads, provider="cpu"))
    return _extractors[threads]


def embed(ext, clip):
    st = ext.create_stream()
    st.accept_waveform(SR, clip)
    st.input_finished()
    e = np.array(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def label_refs(audio, turns, ext, clips_per_label=4, max_clip_s=5.0):
    """One unit-length voice fingerprint per label, from that label's longest turns."""
    refs = {}
    for lab in {t[2] for t in turns}:
        mine = sorted((t for t in turns if t[2] == lab), key=lambda t: t[1] - t[0], reverse=True)[:clips_per_label]
        embs, w = [], []
        for s, e, _ in mine:
            if e - s >= 0.8:
                embs.append(embed(ext, audio[int(s * SR):int(min(e, s + max_clip_s) * SR)]))
                w.append(min(e - s, max_clip_s))
        if embs:
            m = np.average(embs, axis=0, weights=w)
            refs[lab] = m / (np.linalg.norm(m) + 1e-9)
    return refs


# ---------- pass 1 ----------
def sherpa_segments(audio, on_progress=None):
    import sherpa_onnx
    seg, emb = ensure_diar_models()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
            num_threads=DIAR_THREADS, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=DIAR_THREADS, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=SHERPA_THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    if not cfg.validate():
        raise RuntimeError("Diarization config invalid - check the model files in " + MODEL_ROOT)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def cb(done, total):
        if on_progress:
            on_progress(done, total)
        return 0

    result = sd.process(audio, callback=cb).sort_by_start_time()
    return [(r.start, r.end, f"RAW_{r.speaker}") for r in result]


# ---------- pass 2 ----------
def recluster(turns, refs, num_speakers, threshold):
    """Merge pass-1 speakers whose voices are alike. Returns (turns, {name: fingerprint})."""
    from scipy.cluster.hierarchy import linkage, fcluster
    dur = {}
    for s, e, lab in turns:
        dur[lab] = dur.get(lab, 0.0) + (e - s)
    keys = sorted(refs)
    if len(keys) >= 2:
        Z = linkage(np.stack([refs[k] for k in keys]), method="average", metric="cosine")
        if num_speakers:
            cl = fcluster(Z, t=min(int(num_speakers), len(keys)), criterion="maxclust")
        else:
            cl = fcluster(Z, t=threshold, criterion="distance")
    else:
        cl = np.ones(len(keys), dtype=int)
    groups = {}
    for k, c in zip(keys, cl):
        groups.setdefault(int(c), []).append(k)

    def centroid(members):
        m = np.average([refs[k] for k in members], axis=0, weights=[dur[k] for k in members])
        return m / (np.linalg.norm(m) + 1e-9)

    cents = {c: centroid(m) for c, m in groups.items()}
    if not num_speakers:                       # fold tiny noise "speakers" into the closest real one
        for c in sorted(list(groups), key=lambda c: sum(dur[k] for k in groups[c])):
            if len(groups) > 1 and sum(dur[k] for k in groups[c]) < MIN_SPEAKER_S:
                tgt = max((o for o in groups if o != c), key=lambda o: float(cents[o] @ cents[c]))
                groups[tgt] += groups.pop(c)
                cents.pop(c)
                cents[tgt] = centroid(groups[tgt])

    lab2c = {k: c for c, members in groups.items() for k in members}
    anchored = [(s, e, lab2c[l]) for s, e, l in turns if l in lab2c]
    if not anchored:
        return [(s, e, "SPEAKER_00") for s, e, _ in turns], {}
    a_mid = np.array([(s + e) / 2 for s, e, _ in anchored])
    a_c = [c for _, _, c in anchored]

    mapped = []
    for s, e, l in sorted(turns):
        c = lab2c.get(l)
        if c is None:                          # label with no usable voice sample: take the nearest turn's speaker
            c = a_c[int(np.argmin(np.abs((s + e) / 2 - a_mid)))]
        mapped.append((s, e, c))

    order = []                                 # name speakers by first appearance
    for _, _, c in mapped:
        if c not in order:
            order.append(c)
    name = {c: f"SPEAKER_{i:02d}" for i, c in enumerate(order)}
    return ([(s, e, name[c]) for s, e, c in mapped],
            {name[c]: cents[c] for c in order if c in cents})


def merge_adjacent(turns, gap=0.3):
    out, last = [], {}
    for s, e, l in sorted(turns):
        i = last.get(l)
        if i is not None and s <= out[i][1] + gap:
            out[i] = (out[i][0], max(out[i][1], e), l)
        else:
            out.append((s, e, l))
            last[l] = len(out) - 1
    return sorted(out)


def find_overlaps(turns):
    """Time ranges where two different speakers' turns intersect (pyannote marks these natively)."""
    found = []
    for i, (s1, e1, a) in enumerate(turns):          # turns sorted by start
        for s2, e2, b in turns[i + 1:]:
            if s2 >= e1:
                break
            if a != b and min(e1, e2) - max(s1, s2) >= MIN_OVERLAP:
                found.append({"start": max(s1, s2), "end": min(e1, e2), "speakers": sorted({a, b})})
    found.sort(key=lambda o: o["start"])
    merged = []
    for o in found:
        if merged and o["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], o["end"])
            merged[-1]["speakers"] = sorted(set(merged[-1]["speakers"] + o["speakers"]))
        else:
            merged.append(dict(o))
    return merged


def diarize(audio, settings, on_progress=None):
    """-> {"turns": [(start, end, 'SPEAKER_00')], "overlaps": [...], "refs": {speaker: fingerprint}}"""
    raw = sherpa_segments(audio, on_progress)
    if not raw:
        return {"turns": [], "overlaps": [], "refs": {}}
    ext = get_extractor(DIAR_THREADS)
    refs = label_refs(audio, raw, ext)
    turns, refs = recluster(raw, refs, settings.num_speakers, settings.merge_threshold)
    turns = merge_adjacent(turns)
    return {"turns": turns, "overlaps": find_overlaps(turns), "refs": refs}


# ---------- words -> speaker-pure transcript lines ----------
def assign_words(words, turns):
    if not turns:
        for w in words:
            w["speaker"] = "SPEAKER"
        return
    t_s = np.array([t[0] for t in turns])
    t_e = np.array([t[1] for t in turns])
    names = [t[2] for t in turns]
    for w in words:
        ov = np.minimum(w["end"], t_e) - np.maximum(w["start"], t_s)
        i = int(np.argmax(ov))
        if ov[i] <= 0:                                   # word fell in a gap: nearest turn
            i = int(np.argmin(np.abs((w["start"] + w["end"]) / 2 - (t_s + t_e) / 2)))
        w["speaker"] = names[i]


def build_segments(words):
    """Consecutive words of one speaker -> one line. A speaker change always starts a new line."""
    segs = []
    for w in words:
        last = segs[-1] if segs else None
        if (last and last["speaker"] == w["speaker"] and w["start"] - last["end"] <= MAX_GAP_S
                and w["end"] - last["start"] <= MAX_SEGMENT_S):
            last["text"] += " " + w["text"]
            last["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "text": w["text"], "speaker": w["speaker"]})
    return segs
