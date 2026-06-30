# Soft Phonetic Bottlenecks for Spoken Language Recognition

This repository contains code and result files for the course paper:

Soft Phonetic Bottlenecks Improve the Privacy-Utility Trade-off in Spoken Language Recognition

## Project summary

This project compares acoustic, hard phonetic, and soft phonetic representations for spoken language recognition under a speaker-leakage diagnostic.

The main research question is:

Can phonetic bottleneck representations reduce speaker-identity leakage while preserving language-discriminative information?

## Representations

The experiments compare:

1. Wav2Vec2 acoustic embedding baseline
   - averaged high-layer Wav2Vec2/XLSR representations

2. Hard phone n-gram bottleneck
   - Allosaurus phone recognition
   - phone unigram, phone 1-2 gram, and phone 1-3 gram TF-IDF features

3. PPG soft phonetic bottleneck
   - phoneme posteriorgram features from facebook/wav2vec2-xlsr-53-espeak-cv-ft
   - utterance-level posterior statistics

4. Wav2Vec2 layer sweep diagnostic
   - layer-wise probing of language information and speaker leakage

## Dataset note

The raw TidyVoiceX/Tidy-X audio files are not included in this repository because of size and dataset access restrictions.

The repository includes pilot manifest files and result CSVs. To fully reproduce the experiments, place the raw dataset in the same structure used by the manifest paths, or regenerate the manifest with scripts/build_manifest.py.

## Main scripts

- scripts/build_manifest.py
- scripts/inspect_manifest.py
- scripts/make_pilot_subset.py
- scripts/make_speaker_pilot.py
- scripts/01_wav2vec_baseline.py
- scripts/02_phone_ngram_baseline.py
- scripts/04_ppg_soft_bottleneck.py
- scripts/05_wav2vec_layer_sweep.py
- scripts/06_make_final_results_v2.py

## Reproduction overview

From the project root, the main commands are:

1. Wav2Vec2 baseline

python scripts/01_wav2vec_baseline.py --language_manifest data/manifest_pilot.csv --speaker_manifest data/manifest_speaker_pilot.csv --audio_root /path/to/project --output_dir outputs --max_seconds 12 --overwrite

2. Phone n-gram baseline

python scripts/02_phone_ngram_baseline.py --language_manifest data/manifest_pilot.csv --speaker_manifest data/manifest_speaker_pilot.csv --audio_root /path/to/project --output_dir outputs --overwrite

3. PPG soft phonetic bottleneck

python scripts/04_ppg_soft_bottleneck.py --language_manifest data/manifest_pilot.csv --speaker_manifest data/manifest_speaker_pilot.csv --audio_root /path/to/project --output_dir outputs --max_seconds 12 --device cpu --overwrite

4. Wav2Vec2 layer sweep

python scripts/05_wav2vec_layer_sweep.py --language_manifest data/manifest_pilot.csv --speaker_manifest data/manifest_speaker_pilot.csv --audio_root /path/to/project --output_dir outputs --max_seconds 12 --device cpu --overwrite

5. Final results and figures

python scripts/06_make_final_results_v2.py

## Main result files

- outputs/final_paper_table_v2.csv
- outputs/final_main_table_v2.csv
- outputs/figure_final_tradeoff_v2.png
- outputs/figure_layer_language_with_ppg_v2.png
- outputs/figure_layer_leakage_with_ppg_v2.png

## Environment notes

The project was run on macOS using Python and Hugging Face Transformers.

The Allosaurus phone n-gram experiment may require a separate environment because Allosaurus depends on resampy, numba, and llvmlite.

The PPG experiment requires phonemizer and an espeak-ng backend.
