from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw/TidyVoiceX_ASV")
OUTPUT_CSV = Path("data/manifest.csv")


def infer_split_and_flag(path: Path):
    """
    Infer train/dev split from the path name.
    """
    path_text = str(path).lower()

    if "train" in path_text:
        return "train", 1

    if "dev" in path_text or "valid" in path_text or "val" in path_text:
        return "dev", 2

    return "unknown", 0


def build_manifest(raw_dir: Path, output_csv: Path):
    wav_files = sorted(raw_dir.rglob("*.wav"))

    if len(wav_files) == 0:
        raise RuntimeError(f"No wav files found under: {raw_dir}")

    rows = []

    for wav_path in wav_files:
        # Expected structure:
        # data/raw/TidyVoiceX_ASV/TidyVoiceX_Train/id012833/fr/fr_30938087.wav
        #
        # wav_path.parent.name = fr
        # wav_path.parent.parent.name = id012833

        language_id = wav_path.parent.name
        speaker_id = wav_path.parent.parent.name
        split, flag = infer_split_and_flag(wav_path)

        rows.append({
            "path": str(wav_path),
            "speaker_id": speaker_id,
            "language_id": language_id,
            "split": split,
            "flag": flag,
            "filename": wav_path.name
        })

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return df


def print_summary(df: pd.DataFrame):
    print("\n===== Manifest Summary =====")
    print("Total utterances:", len(df))
    print("Total speakers:", df["speaker_id"].nunique())
    print("Total languages:", df["language_id"].nunique())

    print("\n===== Split Counts =====")
    print(df["split"].value_counts(dropna=False))

    print("\n===== Flag Counts =====")
    print(df["flag"].value_counts(dropna=False).sort_index())

    print("\n===== Top 20 Languages =====")
    print(df["language_id"].value_counts().head(20))

    print("\n===== Languages per Speaker =====")
    langs_per_speaker = df.groupby("speaker_id")["language_id"].nunique()
    print(langs_per_speaker.describe())

    print("\nSpeakers with at least 2 languages:", (langs_per_speaker >= 2).sum())
    print("Speakers with at least 3 languages:", (langs_per_speaker >= 3).sum())

    print("\n===== Example Rows =====")
    print(df.head(10))


def main():
    print("Scanning raw directory:", RAW_DIR)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

    df = build_manifest(RAW_DIR, OUTPUT_CSV)
    print_summary(df)

    print("\nSaved manifest to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
