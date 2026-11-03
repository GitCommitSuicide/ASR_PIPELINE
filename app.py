# run:  streamlit run app.py
import os, json, tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from config import Settings
from audio import ts
from pipeline import run, to_text

load_dotenv()

st.set_page_config(page_title="Transcription", layout="wide")
st.title("Transcription, Speakers, Sentiment and Overlap Separation")
st.markdown(
    "Parakeet TDT 0.6B v3 for the words, pyannote segmentation with hierarchical clustering for the speakers, "
    "RoBERTa for sentiment, MossFormer2 to split overlapped speech into one stream per speaker.")

SENT_COLOR = {"positive": "#2e9e5b", "negative": "#d6453d", "neutral": "#8a8f98"}

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    diarize = st.checkbox("Detect speakers", value=True)
    separate = st.checkbox("Separate overlapped speech", value=True, disabled=not diarize)
    sentiment_on = st.checkbox("Sentiment (RoBERTa)", value=True)
    auto = st.checkbox("Auto-detect number of speakers", value=True, disabled=not diarize)
    n_speakers = st.number_input("Number of speakers", 1, 10, 3, disabled=(not diarize) or auto)
    merge_threshold = st.slider(
        "Speaker merge threshold", 0.20, 0.80, 0.45, 0.05, disabled=(not diarize) or (not auto),
        help="Auto mode only. Two detected voices closer than this are treated as the same person.\n\n"
             "Too many speakers: slide higher.\nDifferent people merged together: slide lower.")

file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])
if file:
    st.audio(file)


def make_progress():
    bar = st.progress(0.0, text="Starting...")
    def cb(pct, text):
        try:
            bar.progress(min(max(float(pct), 0.0), 1.0), text=text)
        except Exception:
            pass
    return cb


if file and st.button("Start", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
        f.write(file.getbuffer())
        path = f.name
    try:
        settings = Settings(diarize=diarize, separate_overlaps=separate, sentiment=sentiment_on,
                            num_speakers=None if auto else int(n_speakers), merge_threshold=merge_threshold)
        st.session_state["result"] = run(path, settings, progress_cb=make_progress())
        st.rerun()
    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------- results ----------
res = st.session_state.get("result")
if res:
    T, diarized = res["timings"], res["diarized"]
    st.divider()
    st.header("Results")

    keys = [("Loading", "Loading"), ("VAD", "VAD"), ("Transcription", "Transcription"),
            ("Speakers", "Diarization"), ("Overlap split", "Overlap Separation"),
            ("Sentiment", "Sentiment"), ("Total", "TOTAL")]
    shown = [(label, k) for label, k in keys if k in T]
    for col, (label, k) in zip(st.columns(len(shown)), shown):
        col.metric(label, f"{T[k]:.2f}s")
    if diarized:
        st.caption(f"{len(res['speakers'])} speaker(s) detected. Transcription and speaker detection ran in parallel, "
                   "so Total is less than the sum.")

    names = ["Transcript"] + (["By Speaker"] if diarized else []) + (["Sentiment"] if res["sentiment"] else []) \
        + (["Overlapping Speech"] if diarized else []) + ["Timings"]
    tabs = dict(zip(names, st.tabs(names)))

    def sent_tag(item):
        if not item.get("sentiment"):
            return ""
        return (f"<span style='color:{SENT_COLOR[item['sentiment']]};font-size:0.85em;'>"
                f"{item['sentiment']} {item['polarity']:+.2f}</span>")

    with tabs["Transcript"]:
        for s in res["segments"]:
            who = f"<span style='font-weight:bold;color:#4dabf7;'>{s['speaker']}</span> " if diarized else ""
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:rgba(255,255,255,0.05);border-radius:8px;'>"
                f"<span style='color:#888;font-family:monospace;'>[{ts(s['start'])} - {ts(s['end'])}]</span> "
                f"{who}{s['text']} {sent_tag(s)}</div>", unsafe_allow_html=True)

    if diarized:
        with tabs["By Speaker"]:
            for spk in res["speakers"]:
                segs = [s for s in res["segments"] if s["speaker"] == spk]
                with st.expander(f"{spk} ({len(segs)} lines)", expanded=True):
                    for s in segs:
                        st.markdown(f"`{ts(s['start'])} - {ts(s['end'])}` {s['text']} {sent_tag(s)}",
                                    unsafe_allow_html=True)

    if res["sentiment"]:
        with tabs["Sentiment"]:
            rows = [{"Speaker": spk, "Overall": v["overall"], "Average polarity": v["mean_polarity"],
                     "Positive": v["counts"]["positive"], "Neutral": v["counts"]["neutral"],
                     "Negative": v["counts"]["negative"], "Lines": v["lines"]}
                    for spk, v in res["sentiment"].items()]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Speaker")[["Positive", "Neutral", "Negative"]])
            st.caption("Polarity runs from -1 (negative) to +1 (positive).")

    if diarized:
        with tabs["Overlapping Speech"]:
            if not res["overlaps"]:
                st.info("No overlapping speech detected.")
            for i, o in enumerate(res["overlaps"], 1):
                st.markdown(f"**#{i}** `{ts(o['start'])} - {ts(o['end'])}`  Speakers: **{' + '.join(o['speakers'])}**")
                if os.path.exists(o.get("audio", "")):
                    st.caption("Original mix")
                    st.audio(o["audio"])
                if o.get("error"):
                    st.warning(f"Could not separate this part: {o['error']}  (pip install clearvoice)")
                for x in o.get("streams", []):
                    st.markdown(f"**{x['speaker']}**: {x['text'] or '(nothing intelligible)'} {sent_tag(x)}",
                                unsafe_allow_html=True)
                    if os.path.exists(x.get("audio", "")):
                        st.audio(x["audio"])
                st.divider()

    with tabs["Timings"]:
        rows = [{"Process": k, "Seconds": round(v, 2)} for k, v in T.items()]
        st.bar_chart(pd.DataFrame(rows).set_index("Process"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("Transcript (.txt)", to_text(res), "transcript.txt", "text/plain", use_container_width=True)
    d2.download_button("Data (.json)", json.dumps(res, indent=2, default=str), "transcript.json",
                       "application/json", use_container_width=True)