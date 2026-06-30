from pathlib import Path
import argparse
import random
from math import gcd
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from transformers import AutoProcessor, AutoModelForCTC

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


def load_audio_16k(path, max_seconds=12):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sr = 16000

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


def normalized_leakage_score(acc, n_classes):
    chance = 1.0 / n_classes

    if acc <= chance:
        return 0.0

    return (acc - chance) / (1.0 - chance)


def choose_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def get_non_special_token_ids(processor, vocab_size):
    tokenizer = getattr(processor, "tokenizer", None)

    if tokenizer is None:
        return list(range(vocab_size))

    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    keep_ids = [
        i for i in range(vocab_size)
        if i not in special_ids
    ]

    if len(keep_ids) == 0:
        keep_ids = list(range(vocab_size))

    print("Vocabulary size:", vocab_size)
    print("Special token ids:", sorted(list(special_ids)))
    print("Using non-special token ids:", len(keep_ids))

    return keep_ids


def posterior_to_feature(probs, keep_ids):
    """
    Convert frame-level posteriorgram into one utterance-level vector.

    probs shape:
        time_frames x vocab_size

    Feature:
        mean posterior over time
        std posterior over time
        mean/std posterior entropy
        mean/std max posterior
    """
    eps = 1e-8

    phone_probs = probs[:, keep_ids]

    row_sums = phone_probs.sum(axis=1, keepdims=True)
    phone_probs = phone_probs / (row_sums + eps)

    mean_post = phone_probs.mean(axis=0)
    std_post = phone_probs.std(axis=0)

    entropy = -(phone_probs * np.log(phone_probs + eps)).sum(axis=1)
    max_prob = phone_probs.max(axis=1)

    extra = np.array([
        entropy.mean(),
        entropy.std(),
        max_prob.mean(),
        max_prob.std()
    ], dtype=np.float32)

    feature = np.concatenate([
        mean_post,
        std_post,
        extra
    ]).astype(np.float32)

    return feature


def extract_ppg_features_for_manifest(
    manifest_path,
    output_npy,
    output_csv,
    audio_root,
    processor,
    model,
    device,
    max_seconds=12,
    overwrite=False
):
    manifest_path = Path(manifest_path)
    output_npy = Path(output_npy)
    output_csv = Path(output_csv)

    if output_npy.exists() and output_csv.exists() and not overwrite:
        print("\nLoading existing PPG features:")
        print(output_npy)
        print(output_csv)

        X = np.load(output_npy)
        index_df = pd.read_csv(output_csv)

        return X, index_df

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "speaker_id", "language_id", "split"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in {manifest_path}: {missing}")

    features = []
    rows = []

    print("\nExtracting PPG soft bottleneck features from:", manifest_path)
    print("Rows:", len(df))

    keep_ids = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"PPG {manifest_path.name}"):
        path = resolve_path(row["path"], audio_root)

        if not path.exists():
            print("Missing file:", path)
            print("Original path from CSV:", row["path"])
            continue

        try:
            audio = load_audio_16k(path, max_seconds=max_seconds)

            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)

            logits = outputs.logits.squeeze(0).detach().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()

            if keep_ids is None:
                keep_ids = get_non_special_token_ids(processor, probs.shape[1])

            feat = posterior_to_feature(probs, keep_ids)

            features.append(feat)
            rows.append(row.to_dict())

        except Exception as e:
            print("Error processing:", path)
            print("Error type:", type(e).__name__)
            print("Error:", e)
            raise

    if len(features) == 0:
        raise RuntimeError("No PPG features were extracted.")

    X = np.vstack(features).astype(np.float32)
    index_df = pd.DataFrame(rows)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, X)
    index_df.to_csv(output_csv, index=False)

    print("Saved PPG features:", output_npy)
    print("Saved PPG index:", output_csv)
    print("Feature matrix shape:", X.shape)

    return X, index_df


def run_language_probe(X, index_df):
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
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("\n===== PPG Language Probe =====")
    print("Train samples:", len(y_train))
    print("Dev samples:", len(y_test))
    print("Languages:", len(set(y_train)))
    print("Chance:", round(1.0 / len(set(y_train)), 4))
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
    y = index_df["speaker_id"].astype(str).values

    counts = pd.Series(y).value_counts()
    keep_speakers = counts[counts >= 4].index
    mask = pd.Series(y).isin(keep_speakers).values

    X = X[mask]
    y = y[mask]

    if len(set(y)) < 2:
        raise RuntimeError("Need at least 2 speakers for speaker probe.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    n_classes = len(set(y))
    chance = 1.0 / n_classes
    norm_leakage = normalized_leakage_score(acc, n_classes)

    print("\n===== PPG Speaker Leakage Probe =====")
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
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--max_seconds", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    print("Using device:", device)
    print("Loading PPG model:", args.model_name)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name).to(device)
    model.eval()

    X_lang, lang_index = extract_ppg_features_for_manifest(
        manifest_path=args.language_manifest,
        output_npy=output_dir / "ppg_language_features.npy",
        output_csv=output_dir / "ppg_language_index.csv",
        audio_root=audio_root,
        processor=processor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    X_spk, spk_index = extract_ppg_features_for_manifest(
        manifest_path=args.speaker_manifest,
        output_npy=output_dir / "ppg_speaker_features.npy",
        output_csv=output_dir / "ppg_speaker_index.csv",
        audio_root=audio_root,
        processor=processor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    lang_result = run_language_probe(X_lang, lang_index)
    spk_result = run_speaker_probe(X_spk, spk_index)

    results = pd.DataFrame([
        {
            "representation": "PPG soft phonetic bottleneck",
            "task": lang_result["task"],
            "accuracy": lang_result["accuracy"],
            "macro_f1": lang_result["macro_f1"],
            "n_classes": lang_result["n_classes"],
            "chance": lang_result["chance"],
            "normalized_speaker_leakage": lang_result["normalized_leakage"]
        },
        {
            "representation": "PPG soft phonetic bottleneck",
            "task": spk_result["task"],
            "accuracy": spk_result["accuracy"],
            "macro_f1": spk_result["macro_f1"],
            "n_classes": spk_result["n_classes"],
            "chance": spk_result["chance"],
            "normalized_speaker_leakage": spk_result["normalized_leakage"]
        }
    ])

    results_rounded = results.copy()

    for col in ["accuracy", "macro_f1", "chance", "normalized_speaker_leakage"]:
        results_rounded[col] = results_rounded[col].round(4)

    out_path = output_dir / "ppg_soft_results.csv"
    results_rounded.to_csv(out_path, index=False)

    print("\n===== Saved PPG Soft Bottleneck Results =====")
    print(results_rounded)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
