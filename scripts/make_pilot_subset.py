from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/manifest_pilot.csv"
    )

    parser.add_argument(
        "--max_languages",
        type=int,
        default=5
    )

    parser.add_argument(
        "--train_per_language",
        type=int,
        default=80
    )

    parser.add_argument(
        "--dev_per_language",
        type=int,
        default=30
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "speaker_id", "language_id", "split", "flag"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only files that exist on your computer
    df["file_exists"] = df["path"].apply(lambda p: Path(str(p)).exists())
    missing_files = (~df["file_exists"]).sum()

    print("Original rows:", len(df))
    print("Existing files:", df["file_exists"].sum())
    print("Missing files:", missing_files)

    if missing_files > 0:
        print("\nWarning: Some files are missing. They will be removed.")
        print(df.loc[~df["file_exists"], "path"].head(10))

    df = df[df["file_exists"]].copy()
    df = df.drop(columns=["file_exists"])

    # Select languages that appear in both train and dev
    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()

    train_langs = set(train_df["language_id"].unique())
    dev_langs = set(dev_df["language_id"].unique())
    common_langs = train_langs & dev_langs

    if len(common_langs) < 2:
        raise RuntimeError(
            "Fewer than 2 languages appear in both train and dev. "
            "Check split names in manifest."
        )

    # Choose top languages by train size
    train_counts = train_df["language_id"].value_counts()
    top_langs = [
        lang for lang in train_counts.index
        if lang in common_langs
    ][:args.max_languages]

    print("\nSelected languages:")
    print(top_langs)

    sampled_parts = []

    for lang in top_langs:
        lang_train = train_df[train_df["language_id"] == lang]
        lang_dev = dev_df[dev_df["language_id"] == lang]

        sampled_train = lang_train.sample(
            min(len(lang_train), args.train_per_language),
            random_state=args.seed
        )

        sampled_dev = lang_dev.sample(
            min(len(lang_dev), args.dev_per_language),
            random_state=args.seed
        )

        sampled_parts.append(sampled_train)
        sampled_parts.append(sampled_dev)

    pilot_df = pd.concat(sampled_parts, ignore_index=True)

    # Sort for readability
    pilot_df = pilot_df.sort_values(
        by=["split", "language_id", "speaker_id", "filename"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(output_path, index=False)

    print("\n===== Pilot Manifest Summary =====")
    print("Saved to:", output_path)
    print("Rows:", len(pilot_df))
    print("Languages:", pilot_df["language_id"].nunique())
    print("Speakers:", pilot_df["speaker_id"].nunique())

    print("\nSplit counts:")
    print(pilot_df["split"].value_counts())

    print("\nLanguage counts:")
    print(pilot_df["language_id"].value_counts())

    print("\nLanguage by split:")
    print(pd.crosstab(pilot_df["language_id"], pilot_df["split"]))

    print("\nLanguages per speaker:")
    print(pilot_df.groupby("speaker_id")["language_id"].nunique().describe())

    print("\nSpeakers with >= 2 languages:")
    langs_per_speaker = pilot_df.groupby("speaker_id")["language_id"].nunique()
    print((langs_per_speaker >= 2).sum())

    print("\nExample rows:")
    print(pilot_df.head(10))


if __name__ == "__main__":
    main()
