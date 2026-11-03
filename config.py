"""All constants and the per-run Settings in one place."""
from dataclasses import dataclass
from typing import Optional

MODEL_ROOT = r"D:\content"
SR = 16000

# ---- ASR: NVIDIA Parakeet TDT 0.6B v3 (int8) via sherpa-onnx ----
ASR_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + ASR_DIR + ".tar.bz2"
ASR_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
ASR_WORKERS = 4        # max parallel Parakeet copies (~0.7 GB RAM each); fewer are used for short audio
BATCH = 8              # chunks decoded together

# ---- VAD (Silero) ----
MAX_CHUNK = 20         # longest speech chunk, seconds (longer chunks = more context = better ASR)
VAD_MIN_SILENCE_MS = 600

# ---- Diarization: pyannote segmentation-3.0 + TitaNet-Large ----
SEG_HF_REPO, SEG_FILE = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"
EMB_HF_REPO, EMB_FILE = "csukuangfj/speaker-embedding-models", "nemo_en_titanet_large.onnx"
SHERPA_THRESHOLD = 0.5  # pass 1 clustering. It is allowed to over-split; pass 2 merges.
MIN_SPEAKER_S = 3.0     # a "speaker" with less speech than this is noise and gets merged
DIAR_THREADS = 2

# ---- Overlapped speech ----
MIN_OVERLAP = 0.5       # seconds; shorter overlaps are ignored

# ---- Sentiment ----
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # RoBERTa: negative / neutral / positive

# ---- Turning words into transcript lines ----
MAX_GAP_S = 1.0         # same speaker, pause longer than this -> new line
MAX_SEGMENT_S = 20.0


@dataclass
class Settings:
    diarize: bool = True                 # speakers (+ overlaps)
    separate_overlaps: bool = True       # split overlapped speech into one stream per speaker
    sentiment: bool = True               # RoBERTa sentiment per line and per speaker
    num_speakers: Optional[int] = None   # None = auto-detect
    merge_threshold: float = 0.45        # auto mode: voice-distance below which two "speakers" are merged
                                         # (raise -> fewer speakers, lower -> more speakers)
