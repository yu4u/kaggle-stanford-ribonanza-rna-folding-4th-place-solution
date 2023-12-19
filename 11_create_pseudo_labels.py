import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sn", type=float, default=1.0)
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--output", type=str, default="pseudo_label.hdf5")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_root = Path(__file__).parent.joinpath("input")
    fold_root = data_root.joinpath("tattaka_pseudo_label", "test_prediction")
    num_fold = 0

    for fold_dir in fold_root.iterdir():
        if not fold_dir.is_dir():
            continue

        sns = []
        seq_ids = []

        for parquet_path in sorted(fold_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_path)
            sn = df["signal_to_noise"].values.reshape(2, -1).min(axis=0)
            sns.append(sn)
            seq_ids.append(df["sequence_id"].values[:len(df) // 2])

        sns = np.concatenate(sns, axis=0)
        seq_ids = np.concatenate(seq_ids, axis=0)

        if num_fold == 0:
            sns_all = sns
        else:
            sns_all += sns

        num_fold += 1

    sns_all /= num_fold
    seq_ids = seq_ids[sns_all >= args.sn]
    test_parquet_path = data_root.joinpath("test_sequences.parquet")
    df = pd.read_parquet(test_parquet_path)
    df = df.set_index("sequence_id")
    sub = pd.read_csv(args.submission)
    hdf5_path = data_root.joinpath(args.output)
    reacts = sub[["reactivity_2A3_MaP", "reactivity_DMS_MaP"]].values

    with h5py.File(hdf5_path, "w") as f:
        valid_seq_ids = []

        for seq_id in tqdm(seq_ids):
            start = df.loc[seq_id, "id_min"]
            end = df.loc[seq_id, "id_max"]
            seq = df.loc[seq_id, "sequence"]
            react = reacts[start:end + 1]

            if len(seq) > 207:
                continue

            k = "/".join([seq_id[:3], seq_id, "react"])
            f.create_dataset(k, data=react)
            k = "/".join([seq_id[:3], seq_id, "seq"])
            f.create_dataset(k, data=seq)
            valid_seq_ids.append(seq_id)

        f.create_dataset("seq_ids", data=valid_seq_ids)


if __name__ == '__main__':
    main()
