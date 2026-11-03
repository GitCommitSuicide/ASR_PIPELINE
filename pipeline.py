"""Orchestrator.  Flow:

  audio -> VAD -+-> Parakeet TDT v3 (thread pool)  -> words with timestamps -+
                |                                                             +-> speaker-pure lines
                +-> pyannote seg + hierarchical clustering (own thread) ------+     |
                                                                                    +-> RoBERTa sentiment
  overlapped time ranges -> MossFormer2 split -> transcribe each stream ------------+

  python pipeline.py audio.mp3 [plain]       (plain = no speakers / overlaps, fastest)
"""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from config import SR, Settings, DIAR_THREADS
from audio import load_audio, run_vad, ts
import asr, diarization, overlap, sentiment


def timed(fn, *args):
    t0 = time.time()
    return fn(*args), time.time() - t0


def asr_threads(workers, diarizing):
    """Threads per Parakeet worker, leaving cores for the diarization thread."""
    cpus = max(workers, (os.cpu_count() or 4) - (DIAR_THREADS if diarizing else 0))
    return max(1, cpus // workers)


def run(path, settings=None, out_dir="output", log=print, progress_cb=None):
    st = settings or Settings()
    say = progress_cb or (lambda p, m: None)
    T, t_all = {}, time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ---- load ----
    say(0.03, "Loading models and audio...")
    t0 = time.time()
    model_dir = asr.ensure_asr()
    audio = load_audio(path)
    T["Loading"] = time.time() - t0
    log(f"Audio: {len(audio) / SR / 60:.1f} min")

    # ---- VAD (alone, so it is not slowed by the other jobs) ----
    say(0.10, "Detecting speech...")
    t0 = time.time()
    spans = run_vad(audio)
    T["VAD"] = time.time() - t0
    log(f"VAD: {len(spans)} chunks in {T['VAD']:.2f}s")

    # ---- ASR and diarization at the same time ----
    lock = threading.Lock()
    prog = {"asr": 0, "dia": 0.0}

    def on_asr(n):
        with lock:
            prog["asr"] += n

    def on_dia(done, total):
        with lock:
            prog["dia"] = done / max(1, total)

    workers = asr.pick_workers(spans)
    with ThreadPoolExecutor(2) as ex:
        f_asr = ex.submit(timed, asr.transcribe, audio, spans, model_dir, workers,
                          asr_threads(workers, st.diarize), on_asr)
        f_dia = ex.submit(timed, diarization.diarize, audio, st, on_dia) if st.diarize else None
        futures = [f for f in (f_asr, f_dia) if f]
        while not all(f.done() for f in futures):
            time.sleep(0.3)
            with lock:
                a, d = prog["asr"] / max(1, len(spans)), prog["dia"]
            msg = f"Transcribing {a * 100:.0f}%" + (f"  |  Speakers {d * 100:.0f}%" if st.diarize else "")
            say(0.15 + 0.65 * ((a + d) / 2 if st.diarize else a), msg)
        words, T["Transcription"] = f_asr.result()
        D = None
        if f_dia:
            D, T["Diarization"] = f_dia.result()

    # ---- words -> speaker-pure lines ----
    say(0.82, "Building transcript...")
    diarization.assign_words(words, D["turns"] if D else None)
    segments = diarization.build_segments(words)
    speakers = sorted({s["speaker"] for s in segments})

    # ---- overlapped speech ----
    overlaps = D["overlaps"] if D else []
    if D and st.separate_overlaps and overlaps:
        say(0.86, f"Separating {len(overlaps)} overlapped moments...")
        t0 = time.time()
        overlap.process_overlaps(audio, overlaps, D["refs"], model_dir, out_dir)
        T["Overlap Separation"] = time.time() - t0

    # ---- sentiment ----
    summary = {}
    if st.sentiment and segments:
        say(0.94, "Analyzing sentiment (RoBERTa)...")
        t0 = time.time()
        sentiment.tag(segments)
        streams = [x for o in overlaps for x in o.get("streams", []) if x["text"]]
        sentiment.tag(streams)
        summary = sentiment.summarize(segments)
        T["Sentiment"] = time.time() - t0

    T["TOTAL"] = time.time() - t_all          # wall clock: ASR and diarization overlap
    say(1.0, f"Done in {T['TOTAL']:.1f}s")
    return {
        "segments": segments, "speakers": speakers, "diarized": bool(D),
        "overlaps": overlaps, "sentiment": summary, "timings": T,
        "audio_duration_s": len(audio) / SR,
    }


def to_text(result):
    lines = []
    for s in result["segments"]:
        who = f"{s['speaker']}: " if result["diarized"] else ""
        tag = f"[{s['sentiment']}] " if s.get("sentiment") else ""
        lines.append(f"[{ts(s['start'])} -> {ts(s['end'])}] {who}{tag}{s['text']}")
    if result["sentiment"]:
        lines.append("\n===== SENTIMENT PER SPEAKER =====")
        for spk, v in result["sentiment"].items():
            c = v["counts"]
            lines.append(f"{spk}: {v['overall']} (avg {v['mean_polarity']:+.2f}; "
                         f"{c['positive']} positive, {c['neutral']} neutral, {c['negative']} negative)")
    if result["overlaps"]:
        lines.append("\n===== OVERLAPPING SPEECH (separated by speaker) =====")
        for o in result["overlaps"]:
            lines.append(f"[{ts(o['start'])} -> {ts(o['end'])}] {' + '.join(o['speakers'])}")
            for x in o.get("streams", []):
                tag = f"[{x['sentiment']}] " if x.get("sentiment") else ""
                lines.append(f"    {x['speaker']}: {tag}{x['text']}")
            if o.get("error"):
                lines.append(f"    (could not separate: {o['error']})")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run(sys.argv[1], Settings(diarize=not (len(sys.argv) > 2 and sys.argv[2] == "plain")))
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(to_text(res))
    for k, v in res["timings"].items():
        print(f"{k:<22}{v:8.2f} sec")
    print("Saved: transcript.txt")