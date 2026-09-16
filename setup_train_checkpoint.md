# One-time: build a training-ready checkpoint

`finetune-hf-vits`'s GAN training loop needs a checkpoint that has both
generator and discriminator weights. Plain inference checkpoints on the
Hub — including `facebook/mms-tts-khm` and `khmerttsopensource/khmer-tts`
— only ship the generator. The repo includes a converter that builds a
training-ready checkpoint by pairing an MMS generator with a freshly
initialized discriminator.

```bash
git clone https://github.com/ylacombe/finetune-hf-vits.git
cd finetune-hf-vits
pip install -r requirements.txt

# Build the Cython monotonic alignment search (required, not optional in practice)
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..

huggingface-cli login   # token needs write access

# Recommended: build from the official Khmer MMS checkpoint.
python convert_original_discriminator_checkpoint.py \
    --language_code khm \
    --pytorch_dump_folder_path ./khm-train-ready \
    --push_to_hub phonsobon/mms-tts-khm-train-ready
```

Use `phonsobon/mms-tts-khm-train-ready` (or the local
`./khm-train-ready` path) as `model_name_or_path` in
`training_config.json`.

## Why not convert `khmerttsopensource/khmer-tts` directly

The converter script downloads the *original* Facebook MMS checkpoint
for a language code and attaches a fresh discriminator to it — it isn't
built to take an arbitrary already-converted HF `VitsModel` (like
`khmer-tts`) as the generator source. You could adapt the script to load
`khmerttsopensource/khmer-tts`'s `model.safetensors` into the generator
half instead of downloading the original weights, but since:

- `khmer-tts` was trained only 2 epochs at batch size 2 (per its model
  card) — it's a light touch-up over the base, not a big improvement,
  and
- your merged dataset will likely be far larger than what fine-tuned
  `khmer-tts`,

starting from the official `facebook/mms-tts-khm` base with your data
and a proper multi-epoch GAN run should get you to a similar or better
place without the extra surgery. If you still want to start specifically
from `khmer-tts`'s weights after seeing your first results, say so and
we can adapt the converter to load its generator weights instead.

## License note

`facebook/mms-tts-khm` (and anything derived from it, including
`khmer-tts` and your fine-tune) is released under CC BY-NC 4.0 —
non-commercial. Keep that in mind for how DGWS/MPTC plans to use the
resulting model.
