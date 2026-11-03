# pip install soundfile numpy torch silero-vad python-dotenv sherpa-onnx huggingface_hub clearvoice
#
# Steps (everything heavy runs in its own thread, at the same time):
#   1. ffmpeg converts any wav/mp3 to 16 kHz mono
#   2. Silero VAD finds the speech parts                       (feeds steps 3 and 4)
#   3. Parakeet TDT 0.6B v3 (int8)        -> transcript          [thread pool, ASR_WORKERS models]
#   4. SenseVoice (int8)                  -> tone/emotion per chunk  [1 thread]
#   5. pyannote segmentation-3.0 + TitaNet + hierarchical clustering -> who spoke when [2 threads]
#   6. after 3-5: where 2 speakers talk at once, cut that time range out, split it into one
#      stream per speaker with MossFormer2 (trained on overlapped speech), and transcribe each stream
import os, sys, re, time, itertools, subprocess, tarfile, tempfile, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import torch

load_dotenv()

# ---------- settings ----------
MODEL_ROOT = r"D:\content"

# ASR: NVIDIA Parakeet TDT 0.6B v3 (int8). Much more accurate than the 110M model.
ASR_DIR   = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
ASR_URL   = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + ASR_DIR + ".tar.bz2"
ASR_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]

# Tone / emotion: SenseVoiceSmall int8 (tags every chunk as happy / sad / angry / neutral ...)
TONE_HF_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
TONE_DIR     = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
TONE_FILES   = ["model.int8.onnx", "tokens.txt"]

# Diarization: pyannote segmentation-3.0 (overlap-aware) + TitaNet-Large embeddings + hierarchical clustering
SEG_HF_REPO, SEG_FILE = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"
EMB_HF_REPO, EMB_FILE = "csukuangfj/speaker-embedding-models", "nemo_en_titanet_large.onnx"
# swap EMB_FILE for "nemo_en_titanet_small.onnx" if you want diarization even faster

DIAR_THRESHOLD = 0.5   # auto speaker count: higher = fewer speakers, lower = more speakers
DIAR_THREADS   = 2     # threads for the diarization job
EMO_THREADS    = 1     # threads for the tone job

SR          = 16000
ASR_WORKERS = 4        # parallel Parakeet copies (each ~0.7 GB RAM) - lower this if memory is tight
BATCH       = 8        # chunks decoded together
MAX_CHUNK   = 15       # longest speech chunk in seconds
MIN_OVERLAP = 0.5      # ignore overlaps shorter than this (seconds)


def ts(s):
    return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:05.2f}"


def timed(fn, *args):
    """Run fn and also return how long it took (used for the threaded jobs)."""
    t0 = time.time()
    out = fn(*args)
    return out, time.time() - t0


# ---------- 1. audio ----------
def load_audio(path):
    wav = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), wav],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    audio, _ = sf.read(wav, dtype="float32")
    if os.path.exists(wav):
        os.remove(wav)
    return audio


# ---------- 2. VAD ----------
_vad_model = None

def get_vad_model():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad(onnx=True)
    return _vad_model


def run_vad(audio, vad_model=None):
    from silero_vad import get_speech_timestamps
    model = vad_model if vad_model is not None else get_vad_model()
    spans = get_speech_timestamps(
        torch.from_numpy(audio), model, sampling_rate=SR,
        min_silence_duration_ms=300, max_speech_duration_s=MAX_CHUNK, speech_pad_ms=100)
    return [(s["start"], s["end"]) for s in spans]   # in samples


# ---------- 3. transcription: Parakeet TDT 0.6B v3 ----------
def ensure_asr():
    """Download + unpack the Parakeet model the first time."""
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

    def decode(self, clips):
        streams = []
        for c in clips:
            st = self.rec.create_stream()
            st.accept_waveform(SR, c)
            streams.append(st)
        self.rec.decode_streams(streams)
        return [st.result.text.strip() for st in streams]

    def decode_safe(self, clips):
        """Like decode, but very short clips just give an empty text."""
        ok = [i for i, c in enumerate(clips) if len(c) >= int(0.3 * SR)]
        texts = [""] * len(clips)
        for i, t in zip(ok, self.decode([clips[i] for i in ok]) if ok else []):
            texts[i] = t
        return texts


def asr_threads(diarize_now, tone):
    """Threads per Parakeet worker, leaving cores free for the diarization / tone jobs."""
    reserved = (DIAR_THREADS if diarize_now else 0) + (EMO_THREADS if tone else 0)
    cpus = max(ASR_WORKERS, (os.cpu_count() or 4) - reserved)
    return max(1, cpus // ASR_WORKERS)


def asr_worker(audio, spans, model_dir, threads, on_batch=None):
    rec = Parakeet(model_dir, threads)
    out = []
    for i in range(0, len(spans), BATCH):
        batch = spans[i:i + BATCH]
        texts = rec.decode([audio[s:e] for s, e in batch])
        for (s, e), text in zip(batch, texts):
            if text:
                out.append({"start": s / SR, "end": e / SR, "text": text})
        if on_batch:
            on_batch(len(batch))
    return out


def transcribe(audio, spans, model_dir, threads, on_batch=None):
    if not spans:
        return []
    groups = [g for g in np.array_split(np.arange(len(spans)), ASR_WORKERS) if len(g)]
    with ThreadPoolExecutor(len(groups)) as ex:
        parts = list(ex.map(
            lambda g: asr_worker(audio, [spans[i] for i in g], model_dir, threads, on_batch), groups))
    return [seg for part in parts for seg in part]      # still in time order


# ---------- 4. tone / emotion: SenseVoice ----------
def ensure_tone():
    d = os.path.join(MODEL_ROOT, TONE_DIR)
    if not all(os.path.exists(os.path.join(d, f)) for f in TONE_FILES):
        os.makedirs(d, exist_ok=True)
        from huggingface_hub import hf_hub_download
        for f in TONE_FILES:
            hf_hub_download(TONE_HF_REPO, f, local_dir=d)
    return d


def _clean_emotion(r):
    """SenseVoice puts a tag like <|HAPPY|> in the emotion field (or in the text on older builds)."""
    emo = getattr(r, "emotion", "") or ""
    if not emo:
        m = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED|EMO_UNKNOWN)\|>", r.text or "", re.I)
        emo = m.group(1) if m else ""
    emo = re.sub(r"[<|>]", "", emo).strip().lower()
    return "unknown" if emo in ("", "emo_unknown") else emo


class SenseVoiceTone:
    def __init__(self, model_dir, threads):
        import sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=os.path.join(model_dir, "model.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=threads, provider="cpu", language="auto", use_itn=False)

    def decode(self, clips):
        streams = []
        for c in clips:
            st = self.rec.create_stream()
            st.accept_waveform(SR, c)
            streams.append(st)
        self.rec.decode_streams(streams)
        return [_clean_emotion(st.result) for st in streams]


def detect_tone(audio, spans, tone_dir, threads, on_batch=None):
    """Returns {chunk start time: emotion} for every VAD chunk."""
    rec, out = SenseVoiceTone(tone_dir, threads), {}
    for i in range(0, len(spans), 16):
        batch = spans[i:i + 16]
        for (s, e), emo in zip(batch, rec.decode([audio[s:e] for s, e in batch])):
            out[round(s / SR, 3)] = emo
        if on_batch:
            on_batch(len(batch))
    return out


def tone_summary(segments):
    """Per speaker: dominant tone + seconds spent in each tone."""
    sec = {}
    for s in segments:
        b = sec.setdefault(s["speaker"], {})
        b[s["emotion"]] = b.get(s["emotion"], 0) + (s["end"] - s["start"])
    out = {}
    for spk, b in sec.items():
        known = {k: v for k, v in b.items() if k != "unknown"} or b
        out[spk] = {"dominant": max(known, key=known.get),
                    "seconds": {k: round(v, 1) for k, v in sorted(b.items(), key=lambda kv: -kv[1])}}
    return out


# ---------- 5. diarization: pyannote segmentation-3.0 + TitaNet + hierarchical clustering ----------
def ensure_diar_models():
    from huggingface_hub import hf_hub_download
    seg_dir = os.path.join(MODEL_ROOT, "pyannote-segmentation-3-0")
    seg = os.path.join(seg_dir, SEG_FILE)
    emb = os.path.join(MODEL_ROOT, EMB_FILE)
    if not os.path.exists(seg):
        os.makedirs(seg_dir, exist_ok=True)
        hf_hub_download(SEG_HF_REPO, SEG_FILE, local_dir=seg_dir)
    if not os.path.exists(emb):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        hf_hub_download(EMB_HF_REPO, EMB_FILE, local_dir=MODEL_ROOT)
    return seg, emb


def find_overlaps(turns):
    """pyannote segmentation can output two speakers at the same moment.
    Wherever two different speakers' turns intersect, that is overlapping speech."""
    found = []
    for i, (s1, e1, a) in enumerate(turns):          # turns are sorted by start time
        for s2, e2, b in turns[i + 1:]:
            if s2 >= e1:
                break
            if a != b and min(e1, e2) - max(s1, s2) >= MIN_OVERLAP:
                found.append({"start": max(s1, s2), "end": min(e1, e2), "speakers": sorted({a, b})})
    found.sort(key=lambda o: o["start"])
    merged = []
    for o in found:                                   # glue neighbouring overlaps together
        if merged and o["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], o["end"])
            merged[-1]["speakers"] = sorted(set(merged[-1]["speakers"] + o["speakers"]))
        else:
            merged.append(dict(o))
    return merged


def diarize(audio, num_speakers=None, on_progress=None):
    """Returns (turns, overlaps). turns = [(start_s, end_s, 'SPEAKER_00'), ...]"""
    import sherpa_onnx
    seg_path, emb_path = ensure_diar_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_path),
            num_threads=DIAR_THREADS, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_path, num_threads=DIAR_THREADS, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=int(num_speakers) if num_speakers else -1,
            threshold=DIAR_THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise RuntimeError("Diarization config invalid - check the model files in " + MODEL_ROOT)
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def cb(done, total):                              # called from this worker thread
        if on_progress:
            on_progress(done, total)
        return 0                                      # 0 = keep going

    result = sd.process(audio, callback=cb).sort_by_start_time()
    turns = [(r.start, r.end, f"SPEAKER_{r.speaker:02d}") for r in result]
    return turns, find_overlaps(turns)


def assign_speakers(segments, turns):
    """Give each transcript segment the speaker whose turn overlaps it the most."""
    if not turns:
        return
    for seg in segments:
        best, best_ov = turns[0][2], -1
        for s, e, spk in turns:
            ov = min(seg["end"], e) - max(seg["start"], s)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best_ov <= 0:                              # nothing overlapped -> nearest turn
            mid = (seg["start"] + seg["end"]) / 2
            best = min(turns, key=lambda t: abs(mid - (t[0] + t[1]) / 2))[2]
        seg["speaker"] = best



# ---------- 6. overlapped speech: cut by timestamp -> separate -> transcribe ----------
_separator = None

def get_separator():
    """MossFormer2 speech separation (ClearVoice) - trained on 2-speaker overlapped speech, 16 kHz."""
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


def make_extractor(threads):
    import sherpa_onnx
    _, emb_path = ensure_diar_models()
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_path, num_threads=threads, provider="cpu"))


def embed(ext, clip):
    st = ext.create_stream()
    st.accept_waveform(SR, clip)
    st.input_finished()
    e = np.array(ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def speaker_refs(audio, turns, ext):
    """One 'voice fingerprint' per speaker, from their longest turns."""
    refs = {}
    for spk in {t[2] for t in turns}:
        mine = sorted([t for t in turns if t[2] == spk], key=lambda t: t[1] - t[0], reverse=True)[:3]
        embs = [embed(ext, audio[int(s * SR):int(min(e, s + 8) * SR)]) for s, e, _ in mine if e - s >= 1.0]
        if embs:
            m = np.mean(embs, axis=0)
            refs[spk] = m / (np.linalg.norm(m) + 1e-9)
    return refs


def match_speakers(sources, speakers, refs, ext):
    """The separator's streams are unlabeled - decide which stream is which speaker by voice."""
    cands = [s for s in speakers if s in refs]
    fallback = (list(speakers) + ["UNKNOWN"] * len(sources))[:len(sources)]
    if len(cands) < len(sources) or any(len(x) < int(0.5 * SR) for x in sources):
        return fallback
    embs = [embed(ext, x) for x in sources]
    best, best_score = None, -9
    for perm in itertools.permutations(range(len(cands)), len(sources)):
        score = sum(float(embs[i] @ refs[cands[j]]) for i, j in enumerate(perm))
        if score > best_score:
            best, best_score = perm, score
    return [cands[j] for j in best]


def process_overlaps(audio, overlaps, turns, model_dir, out_dir, tone_dir=None):
    """For every time range where 2+ speakers talk at once: save the mix, split it into one stream
    per speaker, name each stream by voice, then transcribe (and tone-tag) each stream."""
    if not overlaps:
        return
    os.makedirs(out_dir, exist_ok=True)
    cpus = os.cpu_count() or 4
    asr = Parakeet(model_dir, cpus)
    tone = SenseVoiceTone(tone_dir, cpus) if tone_dir else None
    ext = make_extractor(cpus)
    refs = speaker_refs(audio, turns, ext)
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
        texts = asr.decode_safe(sources)
        tones = tone.decode(sources) if tone else [None] * len(sources)
        for src, name, text, emo in zip(sources, names, texts, tones):
            p = os.path.join(out_dir, f"overlap_{i:03d}_{name}.wav")
            sf.write(p, src, SR)
            o["streams"].append({"speaker": name, "audio": p, "text": text, "emotion": emo})


def _apply_diarization(result, audio, turns, overlaps):
    """Put speakers, tone summary and separated overlaps into the result. Returns seconds spent."""
    T, t0 = result["timings"], time.time()
    assign_speakers(result["segments"], turns)
    T["Speaker Alignment"] = time.time() - t0
    t1 = time.time()
    process_overlaps(audio, overlaps, turns, result["model_dir"], result["out_dir"], result.get("tone_dir"))
    T["Overlap Separation"] = time.time() - t1
    result["overlaps"] = overlaps
    result["speakers"] = sorted({s["speaker"] for s in result["segments"]})
    result["diarized"] = True
    if result.get("tone_dir"):
        result["tone"] = tone_summary(result["segments"])
    return time.time() - t0


# ---------- main step A: transcript + tone (+ diarization in parallel if asked) ----------
def run_transcription(path, out_dir="output", log=print, progress_cb=None,
                      diarize_now=False, num_speakers=None, tone=True):
    T, t_all = {}, time.time()
    os.makedirs(out_dir, exist_ok=True)
    say = progress_cb or (lambda p, m: None)

    say(0.03, "Loading models...")
    t0 = time.time()
    model_dir = ensure_asr()
    tone_dir = ensure_tone() if tone else None
    vad_model = get_vad_model()
    T["Model Preparation"] = time.time() - t0

    say(0.10, "Converting and loading audio...")
    t0 = time.time()
    audio = load_audio(path)
    T["Audio Loading"] = time.time() - t0
    T["Loading"] = T["Model Preparation"] + T["Audio Loading"]
    audio_path = os.path.join(out_dir, "audio_16k.wav")      # kept for a later diarization run
    sf.write(audio_path, audio, SR)
    log(f"Audio: {len(audio) / SR / 60:.1f} min")

    # worker threads only update this dict; the main thread talks to Streamlit
    lock = threading.Lock()
    prog = {"asr": 0, "emo": 0, "total": 1, "dia": 0.0}

    def on_asr(n):
        with lock:
            prog["asr"] += n

    def on_emo(n):
        with lock:
            prog["emo"] += n

    def on_dia(done, total):
        with lock:
            prog["dia"] = done / max(1, total)

    with ThreadPoolExecutor(3) as ex:
        # diarization does not need VAD, so it starts right away and overlaps with VAD + ASR
        f_dia = ex.submit(timed, diarize, audio, num_speakers, on_dia) if diarize_now else None

        say(0.15, "Detecting speech with Silero VAD...")
        t0 = time.time()
        spans = run_vad(audio, vad_model)
        T["VAD"] = time.time() - t0
        log(f"VAD: {len(spans)} chunks in {T['VAD']:.2f}s")
        prog["total"] = max(1, len(spans))

        f_asr = ex.submit(timed, transcribe, audio, spans, model_dir,
                          asr_threads(diarize_now, tone), on_asr)
        f_emo = ex.submit(timed, detect_tone, audio, spans, tone_dir, EMO_THREADS, on_emo) if tone else None
        futures = [f for f in (f_asr, f_emo, f_dia) if f]
        while not all(f.done() for f in futures):
            time.sleep(0.3)
            with lock:
                a, e, d = prog["asr"] / prog["total"], prog["emo"] / prog["total"], prog["dia"]
            parts = [("Transcribing", a)] + ([("Tone", e)] if tone else []) + ([("Diarizing", d)] if diarize_now else [])
            say(0.20 + 0.70 * sum(v for _, v in parts) / len(parts),
                "  |  ".join(f"{n} {v * 100:.0f}%" for n, v in parts))

        segments, T["Transcription"] = f_asr.result()
        emo_map = {}
        if f_emo:
            emo_map, T["Tone"] = f_emo.result()
        if f_dia:
            (turns, overlaps), T["Diarization"] = f_dia.result()

    for s in segments:
        s["speaker"] = "SPEAKER"          # placeholder until speakers are known
        s["emotion"] = emo_map.get(round(s["start"], 3), "unknown") if tone else None

    result = {
        "segments": segments, "overlaps": [], "diarized": False, "speakers": [],
        "tone": tone_summary(segments) if tone else {},
        "timings": T, "audio_duration_s": len(audio) / SR,
        "audio_path": audio_path, "model_dir": model_dir, "tone_dir": tone_dir, "out_dir": out_dir,
    }
    if diarize_now:
        say(0.93, "Aligning speakers, separating overlapped speech...")
        _apply_diarization(result, audio, turns, overlaps)

    T["TOTAL"] = time.time() - t_all      # wall-clock: the jobs overlap, so it is not their sum
    say(1.0, f"Done in {T['TOTAL']:.1f}s")
    return result


# ---------- main step B: diarization later (only when it was not run with the transcript) ----------
def run_diarization(result, num_speakers=None, progress_cb=None):
    say = progress_cb or (lambda p, m: None)
    T = result["timings"]
    audio, _ = sf.read(result["audio_path"], dtype="float32")

    lock, frac = threading.Lock(), [0.0]
    def on_dia(done, total):
        with lock:
            frac[0] = done / max(1, total)

    t0 = time.time()
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(diarize, audio, num_speakers, on_dia)
        while not fut.done():
            time.sleep(0.3)
            with lock:
                f = frac[0]
            say(0.05 + 0.80 * f, f"Diarizing (pyannote segmentation) {f * 100:.0f}%")
        turns, overlaps = fut.result()
    T["Diarization"] = time.time() - t0

    say(0.90, "Aligning speakers, separating overlapped speech...")
    extra = _apply_diarization(result, audio, turns, overlaps)
    T["TOTAL"] += T["Diarization"] + extra
    say(1.0, "Diarization complete")
    return result


# ---------- convenience wrapper ----------
def run_pipeline(path, num_speakers=None, out_dir="output", diarization=True, tone=True,
                 log=print, progress_cb=None):
    return run_transcription(path, out_dir, log, progress_cb,
                             diarize_now=diarization, num_speakers=num_speakers, tone=tone)


def to_text(result):
    lines = []
    for s in result["segments"]:
        who = f"{s['speaker']}: " if result.get("diarized") else ""
        mood = f" ({s['emotion']})" if s.get("emotion") and s["emotion"] != "unknown" else ""
        lines.append(f"[{ts(s['start'])} -> {ts(s['end'])}] {who}{mood.strip()} {s['text']}".replace("  ", " "))
    if result.get("tone"):
        lines.append("\n===== TONE PER SPEAKER =====")
        for spk, t in result["tone"].items():
            detail = ", ".join(f"{k} {v}s" for k, v in t["seconds"].items())
            lines.append(f"{spk}: mostly {t['dominant']}  ({detail})")
    if result.get("overlaps"):
        lines.append("\n===== OVERLAPPING SPEECH (separated) =====")
        for o in result["overlaps"]:
            lines.append(f"[{ts(o['start'])} -> {ts(o['end'])}] {' + '.join(o['speakers'])}")
            for st in o.get("streams", []):
                mood = f" ({st['emotion']})" if st.get("emotion") and st["emotion"] != "unknown" else ""
                lines.append(f"    {st['speaker']}{mood}: {st['text']}")
            if o.get("error"):
                lines.append(f"    (could not separate: {o['error']})")
    return "\n".join(lines)


if __name__ == "__main__":
    # python pipeline.py audio.mp3            -> transcript + tone + speakers + overlaps
    # python pipeline.py audio.mp3 plain      -> transcript + tone only (fastest)
    res = run_pipeline(sys.argv[1], diarization=not (len(sys.argv) > 2 and sys.argv[2] == "plain"))
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(to_text(res))
    for k, v in res["timings"].items():
        print(f"{k:<25}{v:8.2f} sec")
    print("Saved: transcript.txt")