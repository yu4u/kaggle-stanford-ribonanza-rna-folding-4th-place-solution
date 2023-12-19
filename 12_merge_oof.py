import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=str, default="oof.parquet")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_root = Path(__file__).parent.joinpath("input")
    oof_dir = data_root.joinpath("oof")
    df = pd.read_parquet(data_root.joinpath("val_sequences_structures_kfold.parquet"))
    preds = []

    for _, row in tqdm(df.iterrows()):
        seq_id = row["sequence_id"]
        oof_path = oof_dir.joinpath(f"{seq_id}.npy")
        oof = np.load(oof_path)
        preds.append(oof)

    concat_preds = np.concatenate(preds)
    submission = pd.DataFrame({
        "id": np.arange(0, len(concat_preds), 1),
        "reactivity_DMS_MaP": concat_preds[:, 1],
        "reactivity_2A3_MaP": concat_preds[:, 0]
    })
    submission.to_parquet(args.output, index=False)


if __name__ == '__main__':
    main()
