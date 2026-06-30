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
import matplotlib.pyplot as plt

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
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

    max_len = int(target_sr * max_seconds)
    if len(audio) > max_len:
        audio = audio[:max_len]

    return audio


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


def normalized_leakage_score(acc, n_classes):
    chance = 1.0 / n_classes
    if acc <= chance:
        return 0.0
    return (acc - chance) / (1.0 - chance)


def extract_layer_embeddings_for_manifest(
    manifest_path,
    output_npy,
    output_csv,
    audio_root,
    feature_extractor,
    model,
    device,
    max_seconds=12,
    overwrite=False
):
    manifest_path = Path(manifest_path)
    output_npy = Path(output_npy)
    output_csv = Path(output_csv)

    if output_npy.exists() and output_csv.exists() and not overwrite:
        print("\nLoading existing layer embeddings:")
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

    all_layer_embeddings = []
    rows = []

    print("\nExtracting Wav2Vec2 layer embeddings from:", manifest_path)
    print("Rows:", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Layer embeddings {manifest_path.name}"):
        path = resolve_path(row["path"], audio_root)

        if not path.exists():
            print("Missing file:", path)
            print("Original path from CSV:", row["path"])
            continue

        try:
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

            layer_vectors = []
            for hs in hidden_states:
                vec = hs.mean(dim=1).squeeze(0).detach().cpu().numpy()
                layer_vectors.append(vec)

            layer_vectors = np.stack(layer_vectors, axis=0).astype(np.float32)

            all_layer_embeddings.append(layer_vectors)
            rows.append(row.to_dict())

        except Exception as e:
            print("Error processing:", path)
            print("Error type:", type(e).__name__)
            print("Error:", e)
            raise

    if len(all_layer_embeddings) == 0:
        raise RuntimeError("No layer embeddings were extracted.")

    X = np.stack(all_layer_embeddings, axis=0).astype(np.float32)
    index_df = pd.DataFrame(rows)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, X)
    index_df.to_csv(output_csv, index=False)

    print("Saved layer embeddings:", output_npy)
    print("Saved index:", output_csv)
    print("Shape:", X.shape)
    print("Shape means: samples x layers x hidden_dim")

    return X, index_df


def run_language_probe_for_layer(X_layer, index_df):
    train_mask = index_df["split"].astype(str).values == "train"
    dev_mask = index_df["split"].astype(str).values == "dev"

    X_train = X_layer[train_mask]
    y_train = index_df.loc[train_mask, "language_id"].astype(str).values

    X_test = X_layer[dev_mask]
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

    return {
        "language_accuracy": acc,
        "language_macro_f1": macro_f1,
        "language_n_classes": len(set(y_train)),
        "language_chance": 1.0 / len(set(y_train))
    }


def run_speaker_probe_for_layer(X_layer, index_df):
    y = index_df["speaker_id"].astype(str).values

    counts = pd.Series(y).value_counts()
    keep_speakers = counts[counts >= 4].index
    mask = pd.Series(y).isin(keep_speakers).values

    X_layer = X_layer[mask]
    y = y[mask]

    if len(set(y)) < 2:
        raise RuntimeError("Need at least 2 speakers for speaker probe.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_layer,
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

    return {
        "speaker_accuracy": acc,
        "speaker_macro_f1": macro_f1,
        "speaker_n_classes": n_classes,
        "speaker_chance": chance,
        "normalized_speaker_leakage": norm_leakage
    }


def run_layer_sweep(X_lang, lang_index, X_spk, spk_index):
    n_layers = X_lang.shape[1]
    rows = []

    print("\n===== Running layer-wise probes =====")
    print("Number of hidden-state entries:", n_layers)
    print("Layer 0 is the convolutional / pre-transformer representation.")
    print("Layers 1..N are transformer layers.")

    for layer_idx in range(n_layers):
        print("\nLayer:", layer_idx)

        lang_result = run_language_probe_for_layer(
            X_layer=X_lang[:, layer_idx, :],
            index_df=lang_index
        )

        spk_result = run_speaker_probe_for_layer(
            X_layer=X_spk[:, layer_idx, :],
            index_df=spk_index
        )

        row = {
            "layer": layer_idx,
            **lang_result,
            **spk_result
        }

        print(
            "Language acc:",
            round(row["language_accuracy"], 4),
            "| Speaker acc:",
            round(row["speaker_accuracy"], 4),
            "| Norm leakage:",
            round(row["normalized_speaker_leakage"], 4)
        )

        rows.append(row)

    results = pd.DataFrame(rows)

    for col in results.columns:
        if col != "layer":
            results[col] = results[col].round(4)

    return results


def make_plots(results, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 4))
    plt.plot(results["layer"], results["language_accuracy"], marker="o")
    plt.axhline(results["language_chance"].iloc[0], linestyle="--")
    plt.xlabel("Wav2Vec2 layer")
    plt.ylabel("Language accuracy")
    plt.title("Language recognition across Wav2Vec2 layers")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_wav2vec_layer_sweep_language.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(results["layer"], results["normalized_speaker_leakage"], marker="o")
    plt.xlabel("Wav2Vec2 layer")
    plt.ylabel("Normalized speaker leakage")
    plt.title("Speaker leakage across Wav2Vec2 layers")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_wav2vec_layer_sweep_leakage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        results["language_accuracy"],
        results["normalized_speaker_leakage"],
        s=60
    )

    for _, row in results.iterrows():
        plt.annotate(
            str(int(row["layer"])),
            (row["language_accuracy"], row["normalized_speaker_leakage"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8
        )

    plt.xlabel("Language accuracy")
    plt.ylabel("Normalized speaker leakage")
    plt.title("Layer-wise language vs. speaker leakage trade-off")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_wav2vec_layer_tradeoff.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language_manifest", type=str, default="data/manifest_pilot.csv")
    parser.add_argument("--speaker_manifest", type=str, default="data/manifest_speaker_pilot.csv")
    parser.add_argument("--audio_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-xlsr-53")
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
    print("Loading Wav2Vec2 model:", args.model_name)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    model.eval()

    X_lang, lang_index = extract_layer_embeddings_for_manifest(
        manifest_path=args.language_manifest,
        output_npy=output_dir / "wav2vec_layer_language_embeddings.npy",
        output_csv=output_dir / "wav2vec_layer_language_index.csv",
        audio_root=audio_root,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    X_spk, spk_index = extract_layer_embeddings_for_manifest(
        manifest_path=args.speaker_manifest,
        output_npy=output_dir / "wav2vec_layer_speaker_embeddings.npy",
        output_csv=output_dir / "wav2vec_layer_speaker_index.csv",
        audio_root=audio_root,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        max_seconds=args.max_seconds,
        overwrite=args.overwrite
    )

    results = run_layer_sweep(
        X_lang=X_lang,
        lang_index=lang_index,
        X_spk=X_spk,
        spk_index=spk_index
    )

    out_path = output_dir / "wav2vec_layer_sweep_results.csv"
    results.to_csv(out_path, index=False)

    make_plots(results, output_dir)

    print("\n===== Saved Wav2Vec2 Layer Sweep Results =====")
    print(results)
    print("Saved to:", out_path)

    print("\nTop 5 layers by language accuracy:")
    print(
        results.sort_values("language_accuracy", ascending=False)
        [["layer", "language_accuracy", "language_macro_f1", "speaker_accuracy", "normalized_speaker_leakage"]]
        .head(5)
    )

    print("\nTop 5 layers by lowest speaker leakage:")
    print(
        results.sort_values("normalized_speaker_leakage", ascending=True)
        [["layer", "language_accuracy", "language_macro_f1", "speaker_accuracy", "normalized_speaker_leakage"]]
        .head(5)
    )


if __name__ == "__main__":
    main()
