# pip install streamlit pandas python-dotenv
# run:  streamlit run app.py
import os, json, tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import pipeline as _pl
from pipeline import run_transcription, run_diarization, to_text, ts

load_dotenv()

st.set_page_config(page_title="Transcription", page_icon="🎙️", layout="wide")
st.title("🎙️ Transcription + Speakers + Tone + Overlap Separation")
st.markdown(
    "**Parakeet TDT 0.6B v3** transcript · **pyannote segmentation-3.0** + hierarchical clustering for speakers · "
    "**SenseVoice** for each person's tone · **MossFormer2** splits overlapped speech into one stream per speaker. "
    "Everything runs in parallel threads.")

EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐", "fearful": "😨",
         "disgusted": "🤢", "surprised": "😲", "unknown": "❔"}


def tone_badge(emo):
    return f" {EMOJI.get(emo, '❔')} _{emo}_" if emo and emo != "unknown" else ""


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    want_tone = st.checkbox("Detect tone / emotion (SenseVoice)", value=True,
                            help="Tags every speech chunk happy / sad / angry / neutral and summarises it per speaker.")
    want_diar = st.checkbox("Detect speakers + overlapping speech", value=True,
                            help="Runs in its own threads while the transcript is being made.")
    auto = st.checkbox("Auto-detect number of speakers", value=True, disabled=not want_diar)
    n_speakers = st.number_input("Number of speakers", 1, 10, 3, disabled=(not want_diar) or auto)

    if want_diar and auto:
        st.markdown("---")
        st.markdown("**🎚️ Sensitivity** (auto-detect only)")
        _pl.DIAR_THRESHOLD = st.slider(
            "Speaker separation threshold",
            min_value=0.10, max_value=0.80, value=0.50, step=0.05,
            help=(
                "Clustering threshold used to decide how many speakers there are.\n\n"
                "• **Higher** → fewer speakers (similar voices get merged)\n"
                "• **Lower** → more speakers (voices get split)\n\n"
                "If you see too many speakers, slide higher. If a speaker is missing, slide lower."
            ),
        )

file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])
if file:
    st.audio(file)


def make_progress(title):
    st.write(f"### {title}")
    bar = st.progress(0.0, text="Starting...")
    def cb(pct, text):
        try:
            bar.progress(min(max(float(pct), 0.0), 1.0), text=text)
        except Exception:
            pass
    return cb


# ---------- run ----------
if file and st.button("🚀 Start", type="primary", use_container_width=True):
    st.session_state.pop("result", None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
        f.write(file.getbuffer())
        path = f.name
    try:
        st.session_state["result"] = run_transcription(
            path,
            progress_cb=make_progress("⏳ Working"),
            diarize_now=want_diar,
            num_speakers=None if auto else int(n_speakers),
            tone=want_tone,
        )
        st.rerun()
    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------- results ----------
res = st.session_state.get("result")
if res:
    T = res["timings"]
    diarized = res["diarized"]
    st.divider()
    st.header("📊 Results")

    metrics = [("⏱️ Loading", "Loading"), ("🎙️ VAD", "VAD"), ("📝 Transcription", "Transcription")]
    if "Tone" in T:
        metrics.append(("🎭 Tone", "Tone"))
    if diarized:
        metrics.append(("👥 Diarization", "Diarization"))
        metrics.append(("🔀 Overlap split", "Overlap Separation"))
    metrics.append(("⚡ Total", "TOTAL"))
    for col, (label, key) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, f"{T.get(key, 0):.2f}s")
    st.caption("The jobs ran in parallel, so Total is less than the sum.")

    names = ["💬 Transcript"]
    if diarized:
        names.append("👤 By Speaker")
    if res.get("tone"):
        names.append("🎭 Tone")
    if diarized:
        names.append("🔀 Overlapping Speech")
    names.append("⏱️ Timings")
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["💬 Transcript"]:
        for s in res["segments"]:
            spk = f"<span style='font-weight:bold;color:#4dabf7;'>{s['speaker']}</span>: " if diarized else ""
            mood = EMOJI.get(s.get("emotion"), "") if s.get("emotion") not in (None, "unknown") else ""
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:rgba(255,255,255,0.05);border-radius:8px;'>"
                f"<span style='color:#888;font-family:monospace;'>[{ts(s['start'])} → {ts(s['end'])}]</span> "
                f"{spk}{mood} {s['text']}</div>", unsafe_allow_html=True)

    if diarized:
        with tabs["👤 By Speaker"]:
            for spk in res["speakers"]:
                segs = [s for s in res["segments"] if s["speaker"] == spk]
                with st.expander(f"**{spk}** ({len(segs)} turns)", expanded=True):
                    for s in segs:
                        st.markdown(f"`{ts(s['start'])} → {ts(s['end'])}`{tone_badge(s.get('emotion'))} {s['text']}")

    if res.get("tone"):
        with tabs["🎭 Tone"]:
            rows = [{"Speaker": spk, "Main tone": f"{EMOJI.get(t['dominant'], '❔')} {t['dominant']}",
                     "Breakdown (seconds)": ", ".join(f"{k} {v}" for k, v in t["seconds"].items())}
                    for spk, t in res["tone"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            chart = pd.DataFrame({spk: t["seconds"] for spk, t in res["tone"].items()}).fillna(0)
            st.bar_chart(chart)

    if diarized:
        with tabs["🔀 Overlapping Speech"]:
            if not res["overlaps"]:
                st.info("No overlapping speech detected.")
            for i, o in enumerate(res["overlaps"], 1):
                st.markdown(f"**#{i}** `{ts(o['start'])} → {ts(o['end'])}`  Speakers: **{' + '.join(o['speakers'])}**")
                if os.path.exists(o.get("audio", "")):
                    st.caption("Original mix")
                    st.audio(o["audio"])
                if o.get("error"):
                    st.warning(f"Could not separate this part: {o['error']}  (pip install clearvoice)")
                for stream in o.get("streams", []):
                    st.markdown(f"**{stream['speaker']}**{tone_badge(stream.get('emotion'))}: {stream['text'] or '_(nothing intelligible)_'}")
                    if os.path.exists(stream.get("audio", "")):
                        st.audio(stream["audio"])
                st.divider()

    with tabs["⏱️ Timings"]:
        rows = [{"Process": k, "Seconds": round(v, 2)} for k, v in T.items()]
        st.bar_chart(pd.DataFrame(rows).set_index("Process"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---------- speakers later (only if it wasn't ticked before the run) ----------
    if not diarized:
        st.divider()
        st.subheader("👥 Speakers + overlapping speech")
        if st.button("Run speaker detection now"):
            try:
                run_diarization(res, None if auto else int(n_speakers),
                                progress_cb=make_progress("⏳ Diarizing"))
                st.rerun()
            except Exception as e:
                st.error(f"Diarization failed: {e}")

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("📥 Transcript (.txt)", to_text(res), "transcript.txt", "text/plain", use_container_width=True)
    d2.download_button("📥 Data (.json)", json.dumps(res, indent=2), "transcript.json", "application/json", use_container_width=True)