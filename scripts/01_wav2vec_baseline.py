from pathlib import Path
import argparse
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import torch

from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_path(path_value, audio_root):
    p = Path(str(path_value))
    if p.exists():
        return p
    candidate = audio_root / p
    if candidate.exists():
        return candidate
    return p


def normalized_leakage_score(acc, n_classes):
    chance = 1.0 / n_classes
    if acc <= chance:
        return 0.0
    return (acc - chance) / (1.0 - chance)


def load_audio_16k(path, max_seconds=12):
    """
    Load audio without librosa, to avoid numba/llvmlite issues.
    TidyVoiceX audio should already be 16 kHz WAV, but this function
    also handles resampling if needed.
    """
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo/multi-channel to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sr = 16000

    # Resample only if needed
    if sr != target_sr:
        common = gcd(sr, target_sr)
        up = target_sr // common
        down = sr // common
        audio = resample_poly(audio, up, down).astype("float32")
        sr = target_sr

    max_len = int(target_sr * max_seconds)
    if len(audio) > max_len:
        audio = audio[:max_len]

    return audio


def load_wav2vec_model(model_name, device):
    print("Loading Wav2Vec2 model:", model_name)
    print("Device:", device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    return feature_extractor, model


def extract_one_embedding(path, feature_extractor, model, device, max_seconds):
    audio = load_audio_16k(path, max_seconds=max_seconds)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(
            input_values,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states

        # For Wav2Vec2-large, use upper layers 17-24.
        # If model has fewer layers, use last hidden state.
        if len(hidden_states) >= 25:
            selected = hidden_states[17:25]
            hidden = torch.stack(selected, dim=0).mean(dim=0)
        else:
            hidden = outputs.last_hidden_state

        # Mean-pool over time to get one vector per utterance
        emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()

    return emb


def extract_embeddings_for_manifest(
    manifest_path,
    output_prefix,
    audio_root,
    feature_extractor,
    model,
    device,
    max_seconds=12,
    overwrite=False
):
    manifest_path = Path(manifest_path)
    output_prefix = Path(output_prefix)

    emb_path = output_prefix.with_suffix(".npy")
    index_path = output_prefix.with_suffix(".csv")

    if emb_path.exists() and index_path.exists() and not overwrite:
        print("\nLoading existing embeddings:")
        print(emb_path)
        X = np.load(emb_path)
        index_df = pd.read_csv(index_path)
        return X, index_df

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "speaker_id", "language_id", "split"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {manifest_path}: {missing}")

    embeddings = []
    rows = []

    print("\nExtracting embeddings from:", manifest_path)
    print("Rows:", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {manifest_path.name}"):
        path = resolve_path(row["path"], audio_root)

        if not path.exists():
            print("Missing file:", path)
            print("Original path from CSV:", row["path"])
            print("Audio root:", audio_root)
            continue

        try:
            emb = extract_one_embedding(
                path=path,
                feature_extractor=feature_extractor,
                model=model,
                device=device,
                max_seconds=max_seconds
            )
            embeddings.append(emb)
            rows.append(row.to_dict())

        except Exception as e:
            print("Error processing:", path)
            print("Error:", e)

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were extracted.")

    X = np.vstack(embeddings)
    index_df = pd.DataFrame(rows)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, X)
    index_df.to_csv(index_path, index=False)

    print("Saved embeddings:", emb_path)
    print("Saved index:", index_path)
    print("Shape:", X.shape)

    return X, index_df


def run_language_probe(X, index_df):
    """
    Train language classifier on split=train and test on split=dev.
    """
    train_mask = index_df["split"].astype(str).values == "train"
    dev_mask = index_df["split"].astype(str).values == "dev"

    if train_mask.sum() == 0 or dev_mask.sum() == 0:
        raise RuntimeError("Need both train and dev split for language probe.")

    X_train = X[train_mask]
    y_train = index_df.loc[train_mask, "language_id"].astype(str).values

    X_test = X[dev_mask]
    y_test = index_df.loc[dev_mask, "language_id"].astype(str).values

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("\n===== Wav2Vec2 Language Probe =====")
    print("Train samples:", len(y_train))
    print("Dev samples:", len(y_test))
    print("Languages:", len(set(y_train)))
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("\nClassification report:")
    print(classification_report(y_test, pred))

    return {
        "task": "language_recognition",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_classes": len(set(y_train)),
        "chance": 1.0 / len(set(y_train)),
        "normalized_leakage": np.nan
    }


def run_speaker_probe(X, index_df):
    """
    Train/test split within speaker-balanced manifest.
    """
    y = index_df["speaker_id"].astype(str).values

    counts = pd.Series(y).value_counts()
    keep_speakers = counts[counts >= 4].index
    mask = pd.Series(y).isin(keep_speakers).values

    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    n_classes = len(set(y))
    chance = 1.0 / n_classes
    norm_leakage = normalized_leakage_score(acc, n_classes)

    print("\n===== Wav2Vec2 Speaker Leakage Probe =====")
    print("Train samples:", len(y_train))
    print("Test samples:", len(y_test))
    print("Speakers:", n_classes)
    print("Chance:", round(chance, 4))
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Normalized speaker leakage:", round(norm_leakage, 4))

    return {
        "task": "speaker_leakage",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_classes": n_classes,
        "chance": chance,
        "normalized_leakage": norm_leakage
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language_manifest", type=str, default="data/manifest_pilot.csv")
    parser.add_argument("--speaker_manifest", type=str, default="data/manifest_speaker_pilot.csv")
    parser.add_argument("--audio_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--max_seconds", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor, model = load_wav2vec_model(
        model_name=args.model_name,
        device=device
    )

    # Language embeddings
    X_lang, lang_index = extract_embeddings_for_manifest(
        manifest_path=args.language_manifest,
        output_prefix=output_dir / "wav2vec_language_embeddings",
        audio_root=audio_root,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    language_result = run_language_probe(X_lang, lang_index)

    # Speaker embeddings
    X_spk, spk_index = extract_embeddings_for_manifest(
        manifest_path=args.speaker_manifest,
        output_prefix=output_dir / "wav2vec_speaker_embeddings",
        audio_root=audio_root,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    speaker_result = run_speaker_probe(X_spk, spk_index)

    results = pd.DataFrame([
        {
            "representation": "Wav2Vec2 acoustic embedding",
            "task": language_result["task"],
            "accuracy": language_result["accuracy"],
            "macro_f1": language_result["macro_f1"],
            "n_classes": language_result["n_classes"],
            "chance": language_result["chance"],
            "normalized_speaker_leakage": language_result["normalized_leakage"]
        },
        {
            "representation": "Wav2Vec2 acoustic embedding",
            "task": speaker_result["task"],
            "accuracy": speaker_result["accuracy"],
            "macro_f1": speaker_result["macro_f1"],
            "n_classes": speaker_result["n_classes"],
            "chance": speaker_result["chance"],
            "normalized_speaker_leakage": speaker_result["normalized_leakage"]
        }
    ])

    results_rounded = results.copy()
    for col in ["accuracy", "macro_f1", "chance", "normalized_speaker_leakage"]:
        results_rounded[col] = results_rounded[col].round(4)

    out_path = output_dir / "wav2vec_results.csv"
    results_rounded.to_csv(out_path, index=False)

    print("\n===== Saved Wav2Vec2 Results =====")
    print(results_rounded)
    print("\nSaved to:", out_path)


if __name__ == "__main__":
    main()
