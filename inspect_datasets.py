"""
Inspect the 12 source ASR datasets before building a TTS training set.

Run this FIRST. TTS quality depends on training on one consistent voice;
this prints enough per-dataset info (columns, audio stats, any
speaker-like column) for you to decide which dataset(s)/speaker(s) to
keep. It only loads a small sample of each dataset (via streaming) so it
runs in a couple of minutes on Colab, not the full multi-GB download.

    pip install datasets soundfile librosa
    python inspect_datasets.py
"""

import statistics
from datasets import load_dataset, Audio

DATASETS = [
    "phonsobon/ASR-kh-datasets-Gov-v1",
    "phonsobon/ASR-kh-datasets-Gov-v2",
    "phonsobon/ASR-Date-Khm-v1",
    "phonsobon/ASR-Date-Khm",
    "phonsobon/ASR-kh-datasets-Gov",
    "phonsobon/datasets-ASR-khmer-english-v1",
    "phonsobon/datasets-ASR-khmer-english",
    "phonsobon/datasets-asr-generate",
    "phonsobon/khmer-ASR-Datasets-v1",
    "phonsobon/khmer-ASR-Datasets",
    "phonsobon/Datasets-ASR-New",
    "phonsobon/openslr42-khmer-male",
]

# How many examples to actually pull per dataset for stats (streaming,
# so this doesn't download the whole dataset).
SAMPLE_SIZE = 200

# Column names we'll guess for text/speaker if not obviously named.
TEXT_CANDIDATES = ["text", "sentence", "transcription", "transcript", "label"]
AUDIO_CANDIDATES = ["audio", "audio_filepath", "wav"]
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "spk_id", "spk", "source", "narrator"]


def guess_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def inspect_one(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    try:
        ds = load_dataset(name, split="train", streaming=True)
    except Exception as e:
        print(f"  Could not load 'train' split streaming: {e}")
        try:
            ds = load_dataset(name, streaming=True)
            first_split = list(ds.keys())[0]
            print(f"  Available splits: {list(ds.keys())}. Using '{first_split}'.")
            ds = ds[first_split]
        except Exception as e2:
            print(f"  FAILED to load at all: {e2}")
            return

    columns = list(next(iter(ds)).keys())
    print(f"  Columns: {columns}")

    text_col = guess_column(columns, TEXT_CANDIDATES)
    audio_col = guess_column(columns, AUDIO_CANDIDATES)
    speaker_col = guess_column(columns, SPEAKER_CANDIDATES)
    print(f"  Guessed -> audio: {audio_col!r}, text: {text_col!r}, speaker: {speaker_col!r}")

    durations = []
    speakers = set()
    text_lens = []
    sample_texts = []
    n = 0
    for row in ds.take(SAMPLE_SIZE):
        n += 1
        if audio_col and row.get(audio_col) is not None:
            audio_val = row[audio_col]
            if isinstance(audio_val, dict):
                arr = audio_val.get("array")
                sr = audio_val.get("sampling_rate")
            else:
                # Newer `datasets` versions decode audio into an
                # AudioDecoder object instead of a plain dict.
                samples = audio_val.get_all_samples()
                arr = samples.data
                sr = samples.sample_rate
            if arr is not None and sr:
                num_samples = arr.shape[-1] if hasattr(arr, "shape") else len(arr)
                durations.append(num_samples / sr)
        if text_col and row.get(text_col):
            t = str(row[text_col])
            text_lens.append(len(t))
            if len(sample_texts) < 3:
                sample_texts.append(t[:80])
        if speaker_col and row.get(speaker_col) is not None:
            speakers.add(row[speaker_col])

    print(f"  Sampled {n} rows.")
    if durations:
        print(
            f"  Audio duration (s) over sample: "
            f"min={min(durations):.2f} max={max(durations):.2f} "
            f"mean={statistics.mean(durations):.2f}"
        )
    if text_lens:
        print(f"  Text length (chars): mean={statistics.mean(text_lens):.0f}")
    if sample_texts:
        print("  Sample text(s):")
        for t in sample_texts:
            print(f"    - {t}")
    if speaker_col:
        print(f"  Distinct speaker values in sample of {n}: {len(speakers)} -> {sorted(map(str, speakers))[:10]}")
    else:
        print("  No speaker-like column found — treat as unlabeled/mixed speakers.")


if __name__ == "__main__":
    for name in DATASETS:
        inspect_one(name)

    print(
        "\n\nNext step: look at the 'speaker' and 'sample text' output above.\n"
        "- If a speaker column exists with few distinct values and one\n"
        "  dominates, that's your best bet for a clean single-speaker set.\n"
        "- If several dataset NAMES look like they were each recorded by\n"
        "  one narrator (e.g. a single 'Gov' announcement reader), treating\n"
        "  each whole dataset as one speaker bucket is a reasonable proxy\n"
        "  even without an explicit speaker_id column.\n"
        "- Note any dataset that's mixed Khmer/English or has very noisy\n"
        "  text — you may want to exclude it in prepare_dataset.py.\n"
        "Update SPEAKER_STRATEGY at the top of prepare_dataset.py accordingly."
    )
