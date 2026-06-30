from pathlib import Path
import argparse
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


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


def normalize_phone_string(phone_string):
    if phone_string is None:
        return ""

    phone_string = str(phone_string).strip()
    tokens = phone_string.split()
    return " ".join(tokens)


def load_allosaurus():
    print("Loading Allosaurus recognizer...")

    try:
        from allosaurus.app import read_recognizer
        recognizer = read_recognizer()
        print("Allosaurus loaded successfully.")
        return recognizer

    except Exception as e:
        print("Could not load Allosaurus.")
        print("Error type:", type(e).__name__)
        print("Error:", e)
        raise


def extract_phone_sequences_for_manifest(
    manifest_path,
    output_csv,
    audio_root,
    recognizer,
    overwrite=False
):
    manifest_path = Path(manifest_path)
    output_csv = Path(output_csv)

    if output_csv.exists() and not overwrite:
        print("\nLoading existing phone sequences:")
        print(output_csv)
        return pd.read_csv(output_csv)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "speaker_id", "language_id", "split"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in {manifest_path}: {missing}")

    rows = []

    print("\nExtracting phone sequences from:", manifest_path)
    print("Rows:", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Phone recognition {manifest_path.name}"):
        path = resolve_path(row["path"], audio_root)

        if not path.exists():
            print("Missing file:", path)
            print("Original path from CSV:", row["path"])
            continue

        try:
            phones = recognizer.recognize(str(path))
            phones = normalize_phone_string(phones)

            if len(phones) == 0:
                continue

            new_row = row.to_dict()
            new_row["phones"] = phones
            rows.append(new_row)

        except Exception as e:
            print("Error processing:", path)
            print("Error type:", type(e).__name__)
            print("Error:", e)
            raise

    if len(rows) == 0:
        raise RuntimeError("No phone sequences were extracted.")

    phone_df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    phone_df.to_csv(output_csv, index=False)

    print("Saved phone sequences:", output_csv)
    print("Rows saved:", len(phone_df))

    print("\nExample phone sequences:")
    print(phone_df[["path", "language_id", "speaker_id", "phones"]].head())

    return phone_df


def make_text_classifier(ngram_range=(1, 3), min_df=2):
    return make_pipeline(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=30000
        ),
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
    )


def fit_text_classifier_with_fallback(X_train, y_train, ngram_range=(1, 3)):
    """
    Try min_df=2 first. If the tiny dataset is too small, fall back to min_df=1.
    """
    last_error = None

    for min_df in [2, 1]:
        try:
            clf = make_text_classifier(ngram_range=ngram_range, min_df=min_df)
            clf.fit(X_train, y_train)
            return clf, min_df

        except ValueError as e:
            last_error = e
            print(f"Classifier failed with min_df={min_df}. Trying lower min_df...")

    raise RuntimeError(f"Text classifier failed. Last error: {last_error}")


def run_language_probe(phone_df, ngram_range=(1, 3)):
    train_mask = phone_df["split"].astype(str).values == "train"
    dev_mask = phone_df["split"].astype(str).values == "dev"

    if train_mask.sum() == 0 or dev_mask.sum() == 0:
        raise RuntimeError("Need both train and dev split for language probe.")

    X_train = phone_df.loc[train_mask, "phones"].astype(str).values
    y_train = phone_df.loc[train_mask, "language_id"].astype(str).values

    X_test = phone_df.loc[dev_mask, "phones"].astype(str).values
    y_test = phone_df.loc[dev_mask, "language_id"].astype(str).values

    clf, used_min_df = fit_text_classifier_with_fallback(
        X_train,
        y_train,
        ngram_range=ngram_range
    )

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("\n===== Phone n-gram Language Probe =====")
    print("n-gram range:", ngram_range)
    print("min_df:", used_min_df)
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


def run_speaker_probe(phone_df, ngram_range=(1, 3)):
    y = phone_df["speaker_id"].astype(str).values
    texts = phone_df["phones"].astype(str).values

    counts = pd.Series(y).value_counts()
    keep_speakers = counts[counts >= 4].index
    mask = pd.Series(y).isin(keep_speakers).values

    y = y[mask]
    texts = texts[mask]

    if len(set(y)) < 2:
        raise RuntimeError("Need at least 2 speakers for speaker probe.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    clf, used_min_df = fit_text_classifier_with_fallback(
        X_train,
        y_train,
        ngram_range=ngram_range
    )

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    n_classes = len(set(y))
    chance = 1.0 / n_classes
    norm_leakage = normalized_leakage_score(acc, n_classes)

    print("\n===== Phone n-gram Speaker Leakage Probe =====")
    print("n-gram range:", ngram_range)
    print("min_df:", used_min_df)
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


def run_ngram_ablation(language_phone_df, speaker_phone_df, output_csv):
    configs = [
        ("unigram", (1, 1)),
        ("unigram_bigram", (1, 2)),
        ("unigram_bigram_trigram", (1, 3))
    ]

    rows = []

    print("\n===== Running Phone n-gram Ablation =====")

    for name, ngram_range in configs:
        print("\n---", name, ngram_range, "---")

        lang_result = run_language_probe(language_phone_df, ngram_range=ngram_range)
        spk_result = run_speaker_probe(speaker_phone_df, ngram_range=ngram_range)

        rows.append({
            "phone_feature": name,
            "ngram_range": str(ngram_range),
            "language_accuracy": lang_result["accuracy"],
            "language_macro_f1": lang_result["macro_f1"],
            "speaker_accuracy": spk_result["accuracy"],
            "speaker_macro_f1": spk_result["macro_f1"],
            "speaker_chance": spk_result["chance"],
            "normalized_speaker_leakage": spk_result["normalized_leakage"]
        })

    ablation_df = pd.DataFrame(rows)

    rounded = ablation_df.copy()

    for col in [
        "language_accuracy",
        "language_macro_f1",
        "speaker_accuracy",
        "speaker_macro_f1",
        "speaker_chance",
        "normalized_speaker_leakage"
    ]:
        rounded[col] = rounded[col].round(4)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rounded.to_csv(output_csv, index=False)

    print("\n===== Saved Phone n-gram Ablation =====")
    print(rounded)
    print("Saved to:", output_csv)

    return rounded


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language_manifest", type=str, default="data/manifest_pilot.csv")
    parser.add_argument("--speaker_manifest", type=str, default="data/manifest_speaker_pilot.csv")
    parser.add_argument("--audio_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    audio_root = Path(args.audio_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recognizer = load_allosaurus()

    language_phone_df = extract_phone_sequences_for_manifest(
        manifest_path=args.language_manifest,
        output_csv=output_dir / "phone_sequences_language.csv",
        audio_root=audio_root,
        recognizer=recognizer,
        overwrite=args.overwrite
    )

    speaker_phone_df = extract_phone_sequences_for_manifest(
        manifest_path=args.speaker_manifest,
        output_csv=output_dir / "phone_sequences_speaker.csv",
        audio_root=audio_root,
        recognizer=recognizer,
        overwrite=args.overwrite
    )

    language_result = run_language_probe(language_phone_df, ngram_range=(1, 3))
    speaker_result = run_speaker_probe(speaker_phone_df, ngram_range=(1, 3))

    results = pd.DataFrame([
        {
            "representation": "Phone n-gram phonotactic vector",
            "task": language_result["task"],
            "accuracy": language_result["accuracy"],
            "macro_f1": language_result["macro_f1"],
            "n_classes": language_result["n_classes"],
            "chance": language_result["chance"],
            "normalized_speaker_leakage": language_result["normalized_leakage"]
        },
        {
            "representation": "Phone n-gram phonotactic vector",
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

    out_path = output_dir / "phone_ngram_results.csv"
    results_rounded.to_csv(out_path, index=False)

    print("\n===== Saved Phone n-gram Results =====")
    print(results_rounded)
    print("Saved to:", out_path)

    run_ngram_ablation(
        language_phone_df=language_phone_df,
        speaker_phone_df=speaker_phone_df,
        output_csv=output_dir / "phone_ngram_ablation.csv"
    )


if __name__ == "__main__":
    main()
