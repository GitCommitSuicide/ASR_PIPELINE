"""Audio loading (ffmpeg) and voice activity detection (Silero)."""
import os, subprocess, tempfile
import soundfile as sf
import torch
from config import SR, MAX_CHUNK, VAD_MIN_SILENCE_MS


def ts(s):
    return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:05.2f}"


def load_audio(path):
    """Any wav/mp3 -> 16 kHz mono float32 numpy array."""
    wav = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), wav],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    audio, _ = sf.read(wav, dtype="float32")
    if os.path.exists(wav):
        os.remove(wav)
    return audio


_vad = None


def run_vad(audio):
    """Returns speech spans as (start_sample, end_sample)."""
    global _vad
    from silero_vad import load_silero_vad, get_speech_timestamps
    if _vad is None:
        _vad = load_silero_vad(onnx=True)
    prev = torch.get_num_threads()
    torch.set_num_threads(1)              # Silero is tiny; one thread avoids fighting the other jobs
    try:
        spans = get_speech_timestamps(
            torch.from_numpy(audio), _vad, sampling_rate=SR,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS, max_speech_duration_s=MAX_CHUNK,
            speech_pad_ms=150)
    finally:
        torch.set_num_threads(prev)
    return [(s["start"], s["end"]) for s in spans]
