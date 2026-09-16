"""
Merge the 12 source ASR datasets into one TTS-ready dataset and push it
to the Hub.

Run inspect_datasets.py first and fill in SPEAKER_STRATEGY below based
on what you saw. Then:

    pip install datasets soundfile librosa huggingface_hub
    huggingface-cli login   # needs write access
    python prepare_dataset.py
"""

import re
import unicodedata

from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    Audio,
    DatasetDict,
)

# ---------------------------------------------------------------------
# CONFIG — edit this after reading inspect_datasets.py's output.
# ---------------------------------------------------------------------

OUTPUT_REPO_ID = "phonsobon/khmer-tts-training-data"
TARGET_SAMPLING_RATE = 16000  # facebook/mms-tts-khm and khmer-tts both use 16kHz

# Per-dataset settings. "include": False drops a dataset entirely (e.g.
# if inspect_datasets.py showed it's badly mixed-language or noisy).
# "speaker_tag": every row from this dataset gets this speaker_id UNLESS
# the dataset itself has a real speaker/speaker-like column (auto-detected)
# — used as a fallback single-speaker-per-source-dataset proxy.
SPEAKER_STRATEGY = {
    "phonsobon/ASR-kh-datasets-Gov-v1": {"include": True, "speaker_tag": "gov_v1"},
    "phonsobon/ASR-kh-datasets-Gov-v2": {"include": True, "speaker_tag": "gov_v2"},
    "phonsobon/ASR-Date-Khm-v1": {"include": True, "speaker_tag": "date_v1"},
    "phonsobon/ASR-Date-Khm": {"include": True, "speaker_tag": "date"},
    "phonsobon/ASR-kh-datasets-Gov": {"include": True, "speaker_tag": "gov"},
    "phonsobon/datasets-ASR-khmer-english-v1": {"include": True, "speaker_tag": "khen_v1"},
    "phonsobon/datasets-ASR-khmer-english": {"include": True, "speaker_tag": "khen"},
    "phonsobon/datasets-asr-generate": {"include": True, "speaker_tag": "generate"},
    "phonsobon/khmer-ASR-Datasets-v1": {"include": True, "speaker_tag": "khmer_v1"},
    "phonsobon/khmer-ASR-Datasets": {"include": True, "speaker_tag": "khmer"},
    "phonsobon/Datasets-ASR-New": {"include": True, "speaker_tag": "new"},
    "phonsobon/openslr42-khmer-male": {"include": True, "speaker_tag": "openslr42_male"},
}

# Set True to keep every row under one single global speaker (only do
# this once you've confirmed via inspect_datasets.py that the audio is
# actually one consistent voice — otherwise you'll train a blurry model).
FORCE_SINGLE_SPEAKER = False

MIN_DURATION_S = 0.5
MAX_DURATION_S = 15.0
MIN_TEXT_CHARS = 3

TEXT_CANDIDATES = ["text", "sentence", "transcription", "transcript", "label"]
AUDIO_CANDIDATES = ["audio", "audio_filepath", "wav"]
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "spk_id", "spk", "source", "narrator"]

# Khmer script + digits + common punctuation the MMS/VITS tokenizer
# will actually see. Anything else (stray control chars, garbage bytes)
# gets stripped rather than dropping the whole row, since a lot of these
# corpora mix in ASCII punctuation that's harmless.
KHMER_ALLOWED_RE = re.compile(
    r"[^\u1780-\u17FF\u19E0-\u19FF0-9a-zA-Z\s\.,!?។៕៖–\-'\"]"
)


def guess_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = KHMER_ALLOWED_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_and_normalize(name: str, cfg: dict):
    print(f"Loading {name} ...")
    ds = load_dataset(name, split="train")
    columns = ds.column_names

    audio_col = guess_column(columns, AUDIO_CANDIDATES)
    text_col = guess_column(columns, TEXT_CANDIDATES)
    speaker_col = guess_column(columns, SPEAKER_CANDIDATES)

    if audio_col is None or text_col is None:
        raise ValueError(
            f"{name}: could not find audio/text columns among {columns}. "
            "Add an explicit mapping for this dataset before continuing."
        )

    ds = ds.cast_column(audio_col, Audio(sampling_rate=TARGET_SAMPLING_RATE))

    def build_row(row):
        text = clean_text(str(row[text_col]))
        speaker_id = (
            str(row[speaker_col])
            if speaker_col is not None and row.get(speaker_col) is not None
            else cfg["speaker_tag"]
        )
        audio = row[audio_col]
        duration = len(audio["array"]) / audio["sampling_rate"] if audio["array"] is not None else 0.0
        return {
            "audio": audio,
            "text": text,
            "speaker_id": "single_speaker" if FORCE_SINGLE_SPEAKER else speaker_id,
            "duration": duration,
            "source_dataset": name,
        }

    ds = ds.map(build_row, remove_columns=columns, desc=f"normalizing {name}")

    before = len(ds)
    ds = ds.filter(
        lambda r: (
            len(r["text"]) >= MIN_TEXT_CHARS
            and MIN_DURATION_S <= r["duration"] <= MAX_DURATION_S
        ),
        desc=f"filtering {name}",
    )
    print(f"  {name}: {before} -> {len(ds)} rows after cleaning/duration filter")
    return ds


def main():
    parts = []
    for name, cfg in SPEAKER_STRATEGY.items():
        if not cfg.get("include", True):
            print(f"Skipping {name} (include=False)")
            continue
        try:
            parts.append(load_and_normalize(name, cfg))
        except Exception as e:
            print(f"!! Skipping {name} due to error: {e}")

    if not parts:
        raise RuntimeError("No datasets loaded successfully — nothing to merge.")

    merged = concatenate_datasets(parts)
    print(f"\nMerged total rows before dedup: {len(merged)}")

    # Drop exact duplicate (text, speaker_id) pairs — the -v1/v2/etc.
    # dataset variants likely overlap.
    seen = set()
    keep_idx = []
    for i, row in enumerate(merged):
        key = (row["text"], row["speaker_id"])
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    merged = merged.select(keep_idx)
    print(f"Rows after text-dedup: {len(merged)}")

    total_hours = sum(merged["duration"]) / 3600
    print(f"Total audio: {total_hours:.2f} hours")

    # 95/5 train/val split, shuffled.
    split = merged.train_test_split(test_size=0.05, seed=42)
    dsd = DatasetDict(train=split["train"], validation=split["test"])

    print(dsd)
    print(f"\nPushing to {OUTPUT_REPO_ID} ...")
    dsd.push_to_hub(OUTPUT_REPO_ID)
    print("Done.")


if __name__ == "__main__":
    main()
