# Khmer TTS fine-tuning (VITS/MMS) — Colab T4, resumable

Goal: fine-tune a Khmer VITS/MMS text-to-speech model on your 12
`phonsobon/*` ASR datasets, training on a free Colab T4, with every
checkpoint pushed to the Hub and automatic resume after a crash or
Colab restart.

## Read this before running anything

**Target model.** `khmerttsopensource/khmer-tts` is itself a VITS model
fine-tuned from `facebook/mms-tts-khm`, trained only 2 epochs at batch
size 2, and it ships **generator weights only** (no discriminator). The
official fine-tuning recipe for VITS/MMS (`ylacombe/finetune-hf-vits`)
needs a checkpoint with *both* generator and discriminator to run its
GAN training loop.

So the practical starting point is: build a training-ready checkpoint
from `facebook/mms-tts-khm` (the same base `khmer-tts` came from) using
the recipe's own converter, then fine-tune that on your data. With far
more data and more epochs than `khmer-tts` had, this should end up
ahead of it anyway. `setup_train_checkpoint.md` covers the one command
this takes, and the alternative if you specifically want to start from
`khmer-tts`'s own weights.

**Your datasets are ASR data, not TTS data.** TTS wants *one consistent
voice*. Your 12 corpora were built for speech recognition and likely mix
many speakers/recording conditions. Training VITS on unlabeled
multi-speaker audio as if it's one speaker gives you a smeared,
inconsistent voice — worse than picking one clean speaker and training
on just that. **Run `inspect_datasets.py` first** and actually look at
the output before deciding:

- If any dataset (or a `speaker_id`/`source`-like column) isolates one
  narrator with reasonable volume (a few hours is enough — VITS needs
  far less data than ASR), filter to that one and get a clean
  single-speaker model. This is the recommended path and what
  `prepare_dataset.py` defaults to if it finds a usable speaker column.
- If nothing separates cleanly by speaker, you're choosing between a
  blurry single voice or not training at all — there's no multi-speaker
  option here since these datasets don't carry reliable speaker labels.

## Files

| File | Purpose |
|---|---|
| `inspect_datasets.py` | Loads each of the 12 datasets, prints row counts, columns, audio duration stats, and any speaker-like column + its cardinality. Run this first. |
| `prepare_dataset.py` | Merges/cleans the datasets into one HF dataset in the `audio` + `text` (+ optional `speaker_id`) shape the training recipe expects, and pushes it to your Hub. |
| `setup_train_checkpoint.md` | The one-time step to get a training-ready (generator+discriminator) checkpoint. |
| `training_config.json` | Config for `finetune-hf-vits`, tuned for a T4 (16GB) — small batch, grad accumulation, fp16, checkpoint-and-push every N steps. |
| `run_training_colab.sh` | The crash-resilient loop: launches training, and on any exit (OOM, Colab disconnect, runtime restart) re-downloads the latest checkpoint from the Hub and resumes, until training actually finishes. |
| `khmer_tts_colab.ipynb` | Colab notebook that runs the whole pipeline cell by cell. |

## Order of operations

1. Open `khmer_tts_colab.ipynb` in Colab, set a T4 runtime.
2. Cell: install deps, `huggingface-cli login` with a **write** token.
3. Run `inspect_datasets.py` → read the output → decide your speaker
   strategy (see above).
4. Edit the `SPEAKER_STRATEGY` block at the top of `prepare_dataset.py`
   if auto-detection didn't pick what you want, then run it. It pushes
   a merged dataset to `phonsobon/khmer-tts-training-data` (change the
   repo id at the top if you want a different name).
5. Follow `setup_train_checkpoint.md` once to get your training-ready
   base checkpoint.
6. Edit `training_config.json`: set `dataset_name` to your merged
   dataset repo, `model_name_or_path` to the checkpoint from step 5,
   `hub_model_id` to where you want checkpoints pushed.
7. Run `run_training_colab.sh` from a Colab cell (`!bash
   run_training_colab.sh`). It resumes automatically — if the runtime
   dies, just re-run the same cell after reconnecting; it will fetch
   the last pushed checkpoint from the Hub and continue.

## Why not just push a ready GAN training loop from scratch

VITS training needs two optimizers (generator/discriminator), a
monotonic-alignment loss, mel-spectrogram loss, KL loss, and feature
matching — hand-writing this correctly in one shot is a common source
of silently-bad models (it trains, loss goes down, audio still sounds
wrong). `finetune-hf-vits` is the maintained, widely-used implementation
of this exact recipe on top of `transformers`/`accelerate`, so we
build the resume/checkpoint-push/data-prep automation around it rather
than reinventing the training loop.
