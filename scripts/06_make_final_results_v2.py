from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_task_row(df, task):
    return df[df["task"] == task].iloc[0]

def main():
    rows = []

    # 1. Wav2Vec2 baseline
    wav = pd.read_csv(OUTPUT_DIR / "wav2vec_results.csv")
    wav_lang = get_task_row(wav, "language_recognition")
    wav_spk = get_task_row(wav, "speaker_leakage")

    rows.append({
        "Representation": "Wav2Vec2 avg. layers 17–24",
        "Role": "Acoustic SSL baseline",
        "Language Accuracy": wav_lang["accuracy"],
        "Language Macro-F1": wav_lang["macro_f1"],
        "Speaker Accuracy": wav_spk["accuracy"],
        "Speaker Macro-F1": wav_spk["macro_f1"],
        "Speaker Chance": wav_spk["chance"],
        "Normalized Speaker Leakage": wav_spk["normalized_speaker_leakage"]
    })

    # 2. Phone n-gram ablation: unigram and 1-3 gram
    phone_ab = pd.read_csv(OUTPUT_DIR / "phone_ngram_ablation.csv")

    unigram = phone_ab[phone_ab["phone_feature"] == "unigram"].iloc[0]
    rows.append({
        "Representation": "Phone unigram",
        "Role": "Hard phonetic bottleneck",
        "Language Accuracy": unigram["language_accuracy"],
        "Language Macro-F1": unigram["language_macro_f1"],
        "Speaker Accuracy": unigram["speaker_accuracy"],
        "Speaker Macro-F1": unigram["speaker_macro_f1"],
        "Speaker Chance": unigram["speaker_chance"],
        "Normalized Speaker Leakage": unigram["normalized_speaker_leakage"]
    })

    trigram = phone_ab[phone_ab["phone_feature"] == "unigram_bigram_trigram"].iloc[0]
    rows.append({
        "Representation": "Phone 1–3 gram",
        "Role": "Hard phonotactic bottleneck",
        "Language Accuracy": trigram["language_accuracy"],
        "Language Macro-F1": trigram["language_macro_f1"],
        "Speaker Accuracy": trigram["speaker_accuracy"],
        "Speaker Macro-F1": trigram["speaker_macro_f1"],
        "Speaker Chance": trigram["speaker_chance"],
        "Normalized Speaker Leakage": trigram["normalized_speaker_leakage"]
    })

    # 3. PPG soft bottleneck
    ppg = pd.read_csv(OUTPUT_DIR / "ppg_soft_results.csv")
    ppg_lang = get_task_row(ppg, "language_recognition")
    ppg_spk = get_task_row(ppg, "speaker_leakage")

    rows.append({
        "Representation": "PPG soft phonetic bottleneck",
        "Role": "Soft phonetic bottleneck",
        "Language Accuracy": ppg_lang["accuracy"],
        "Language Macro-F1": ppg_lang["macro_f1"],
        "Speaker Accuracy": ppg_spk["accuracy"],
        "Speaker Macro-F1": ppg_spk["macro_f1"],
        "Speaker Chance": ppg_spk["chance"],
        "Normalized Speaker Leakage": ppg_spk["normalized_speaker_leakage"]
    })

    # 4. Best Wav2Vec2 layer from layer sweep
    layer = pd.read_csv(OUTPUT_DIR / "wav2vec_layer_sweep_results.csv")
    best_lang = layer.sort_values("language_accuracy", ascending=False).iloc[0]

    rows.append({
        "Representation": f"Wav2Vec2 layer {int(best_lang['layer'])}",
        "Role": "Best language layer diagnostic",
        "Language Accuracy": best_lang["language_accuracy"],
        "Language Macro-F1": best_lang["language_macro_f1"],
        "Speaker Accuracy": best_lang["speaker_accuracy"],
        "Speaker Macro-F1": best_lang["speaker_macro_f1"],
        "Speaker Chance": best_lang["speaker_chance"],
        "Normalized Speaker Leakage": best_lang["normalized_speaker_leakage"]
    })

    table = pd.DataFrame(rows)

    for col in [
        "Language Accuracy",
        "Language Macro-F1",
        "Speaker Accuracy",
        "Speaker Macro-F1",
        "Speaker Chance",
        "Normalized Speaker Leakage"
    ]:
        table[col] = table[col].astype(float).round(4)

    table.to_csv(OUTPUT_DIR / "final_main_table_v2.csv", index=False)

    # Compact paper table
    paper_table = table[[
        "Representation",
        "Language Accuracy",
        "Language Macro-F1",
        "Speaker Accuracy",
        "Normalized Speaker Leakage"
    ]].copy()

    paper_table.to_csv(OUTPUT_DIR / "final_paper_table_v2.csv", index=False)

    # Trade-off plot
    plt.figure(figsize=(7, 5))
    plt.scatter(
        table["Language Accuracy"],
        table["Normalized Speaker Leakage"],
        s=90
    )

    for _, row in table.iterrows():
        label = row["Representation"]
        plt.annotate(
            label,
            (row["Language Accuracy"], row["Normalized Speaker Leakage"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Language Accuracy")
    plt.ylabel("Normalized Speaker Leakage")
    plt.title("Language Recognition vs. Speaker Leakage")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_final_tradeoff_v2.png", dpi=300)
    plt.close()

    # Layer sweep with PPG reference line
    plt.figure(figsize=(8, 4))
    plt.plot(layer["layer"], layer["language_accuracy"], marker="o", label="Wav2Vec2 layer language acc.")
    plt.axhline(float(ppg_lang["accuracy"]), linestyle="--", label="PPG language acc.")
    plt.xlabel("Wav2Vec2 layer")
    plt.ylabel("Language Accuracy")
    plt.title("Language Accuracy Across Wav2Vec2 Layers")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_layer_language_with_ppg_v2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(layer["layer"], layer["normalized_speaker_leakage"], marker="o", label="Wav2Vec2 layer leakage")
    plt.axhline(float(ppg_spk["normalized_speaker_leakage"]), linestyle="--", label="PPG leakage")
    plt.xlabel("Wav2Vec2 layer")
    plt.ylabel("Normalized Speaker Leakage")
    plt.title("Speaker Leakage Across Wav2Vec2 Layers")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_layer_leakage_with_ppg_v2.png", dpi=300)
    plt.close()

    print("\n===== Final main table =====")
    print(table.to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_DIR / "final_main_table_v2.csv")
    print(OUTPUT_DIR / "final_paper_table_v2.csv")
    print(OUTPUT_DIR / "figure_final_tradeoff_v2.png")
    print(OUTPUT_DIR / "figure_layer_language_with_ppg_v2.png")
    print(OUTPUT_DIR / "figure_layer_leakage_with_ppg_v2.png")


if __name__ == "__main__":
    main()
