import argparse
from pathlib import Path
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    csv_paths = sorted(Path(args.input).glob("*.csv"))

    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)

        if i == 0:
            preds = df[["reactivity_DMS_MaP", "reactivity_2A3_MaP"]].values
        else:
            preds += df[["reactivity_DMS_MaP", "reactivity_2A3_MaP"]].values

    preds /= len(csv_paths)
    df[["reactivity_DMS_MaP", "reactivity_2A3_MaP"]] = preds
    df.to_parquet("submission.parquet", index=False)


if __name__ == '__main__':
    main()
