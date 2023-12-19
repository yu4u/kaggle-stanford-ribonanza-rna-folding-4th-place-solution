from pathlib import Path
import pandas as pd
from tqdm import tqdm


def get_bpp_paths(df):
    seq_ids = set(df["sequence_id"])
    bpp_paths = []

    for bpp_path in tqdm(Path("input/Ribonanza_bpp_files/extra_data").glob("**/*.txt")):
        seq_id = bpp_path.stem

        if seq_id in seq_ids:
            bpp_paths.append([seq_id, str(bpp_path)[-22:]])
            seq_ids.remove(seq_id)

    return bpp_paths


def main():
    # train bpp
    df = pd.read_parquet("input/train_data.parquet")
    bpp_paths = get_bpp_paths(df)
    pd.DataFrame(bpp_paths, columns=["sequence_id", "bpp_path"]).to_csv(
        "input/train_seq_id_to_bpp_path.csv", index=False)

    # test bpp
    df = pd.read_parquet("input/test_sequences.parquet")
    bpp_paths = get_bpp_paths(df)
    pd.DataFrame(bpp_paths, columns=["sequence_id", "bpp_path"]).to_csv(
        "input/test_seq_id_to_bpp_path.csv", index=False)


if __name__ == '__main__':
    main()
