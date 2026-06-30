from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output", type=str, default="data/manifest_speaker_pilot.csv")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_speakers", type=int, default=50)
    parser.add_argument("--utterances_per_speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["path", "speaker_id", "language_id", "split", "flag", "filename"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Original rows:", len(df))
    print("Original speakers:", df["speaker_id"].nunique())
    print("Original languages:", df["language_id"].nunique())

    sub = df[
        (df["split"].astype(str) == args.split) &
        (df["language_id"].astype(str) == args.language)
    ].copy()

    print("\nAfter filtering:")
    print("Split:", args.split)
    print("Language:", args.language)
    print("Rows:", len(sub))
    print("Speakers:", sub["speaker_id"].nunique())

    if len(sub) == 0:
        raise RuntimeError(f"No rows found for split={args.split}, language={args.language}")

    sub["file_exists"] = sub["path"].apply(lambda p: Path(str(p)).exists())
    print("\nFile existence:")
    print(sub["file_exists"].value_counts())

    sub = sub[sub["file_exists"]].drop(columns=["file_exists"]).copy()

    speaker_counts = sub["speaker_id"].value_counts()
    eligible_speakers = speaker_counts[
        speaker_counts >= args.utterances_per_speaker
    ].index.tolist()

    print("\nEligible speakers with at least",
          args.utterances_per_speaker,
          "utterances:",
          len(eligible_speakers))

    if len(eligible_speakers) < 2:
        raise RuntimeError(
            "Fewer than 2 eligible speakers. Try another language or lower utterances_per_speaker."
        )

    if len(eligible_speakers) > args.num_speakers:
        selected_speakers = (
            pd.Series(eligible_speakers)
            .sample(args.num_speakers, random_state=args.seed)
            .tolist()
        )
    else:
        selected_speakers = eligible_speakers

    sampled_parts = []

    for spk in selected_speakers:
        spk_df = sub[sub["speaker_id"] == spk].copy()
        sampled = spk_df.sample(
            n=args.utterances_per_speaker,
            random_state=args.seed
        )
        sampled_parts.append(sampled)

    speaker_pilot = pd.concat(sampled_parts, ignore_index=True)

    speaker_pilot = speaker_pilot.sort_values(
        by=["speaker_id", "filename"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_pilot.to_csv(output_path, index=False)

    print("\n===== Speaker Pilot Summary =====")
    print("Saved to:", output_path)
    print("Rows:", len(speaker_pilot))
    print("Speakers:", speaker_pilot["speaker_id"].nunique())
    print("Languages:", speaker_pilot["language_id"].unique())
    print("Split:", speaker_pilot["split"].unique())

    print("\nSpeaker counts:")
    print(speaker_pilot["speaker_id"].value_counts().describe())

    print("\nExample rows:")
    print(speaker_pilot.head(10))


if __name__ == "__main__":
    main()
