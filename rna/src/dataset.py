from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import h5py


class MyDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        assert mode in ["train", "val", "test"]
        data_root = Path(__file__).parents[2].joinpath("input")

        if mode in ["train", "val"]:
            parquet_path = data_root.joinpath("train_data.parquet")
            self.Lmax = 206

            if cfg.task.pseudo:
                self.Lmax = 207
        else:
            parquet_path = data_root.joinpath("test_sequences.parquet")
            self.Lmax = 457

        df = pd.read_parquet(parquet_path)
        self.cfg = cfg
        self.mode = mode
        fold_id = cfg.data.fold_id
        fold_num = cfg.data.fold_num

        if cfg.task.ngram == 1:
            self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        else:
            self.seq_map = {k: i for i, k in enumerate(itertools.product("ACGUX", repeat=cfg.task.ngram))}

        if mode == "test":
            self.seq = df["sequence"].values
            self.seq_id = df["sequence_id"].values
            self.bpp_dir = data_root.joinpath("Ribonanza_bpp_files", "extra_data")

            # TODO: merge code with train/val
            bpp_df = pd.read_csv(data_root.joinpath("test_seq_id_to_bpp_path.csv"))
            merged_df = pd.merge(df, bpp_df, on="sequence_id", how="left")
            self.bpp_paths = merged_df["bpp_path"].values
            return

        df_2A3 = df.loc[df.experiment_type == "2A3_MaP"]
        df_DMS = df.loc[df.experiment_type == "DMS_MaP"]

        split = list(KFold(n_splits=fold_num, random_state=42,
                           shuffle=True).split(df_2A3))[fold_id][0 if mode == 'train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        if not cfg.task.oof:
            if mode == "train":
                sn_th = cfg.task.sn_th
                m = (df_2A3["signal_to_noise"].values > sn_th) & (df_DMS["signal_to_noise"].values > sn_th)
            else:
                m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)

            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.seq_id = df_2A3['sequence_id'].values
        self.bpp_dir = data_root.joinpath("Ribonanza_bpp_files", "extra_data")
        bpp_df = pd.read_csv(data_root.joinpath("train_seq_id_to_bpp_path.csv"))
        merged_df = pd.merge(df_2A3, bpp_df, on="sequence_id", how="left")
        self.bpp_paths = merged_df["bpp_path"].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if "reactivity_0" in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if "reactivity_0" in c]].values

    def __len__(self):
        return len(self.seq)

    def encode_seq(self, seq):
        seq_len = len(seq)
        n = self.cfg.task.ngram
        seq = "X" * (n // 2) + seq + "X" * (n // 2)
        return [self.seq_map[tuple(seq[i:i + n])] for i in range(seq_len)]

    def __getitem__(self, idx):
        seq = self.seq[idx]

        if self.cfg.task.ngram == 1:
            seq = [self.seq_map[s] for s in seq]
        else:
            seq = self.encode_seq(seq)

        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)), constant_values=5 ** self.cfg.task.ngram - 1)

        if self.mode == "test":
            bpp_path = self.bpp_dir.joinpath(self.bpp_paths[idx])
            df = pd.read_csv(bpp_path, sep=" ", header=None)
            bpp_matrix = np.eye(self.Lmax, dtype=np.float32)

            for i, j, v in df.values:
                bpp_matrix[int(i) - 1, int(j) - 1] = v
                bpp_matrix[int(j) - 1, int(i) - 1] = v

            x = {'seq': torch.from_numpy(seq), 'mask': mask, "bpp": bpp_matrix}
            return x

        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]], -1))

        if self.cfg.task.pseudo:
            react = np.pad(react, ((0, self.Lmax - len(react)), (0, 0)), constant_values=np.nan)

        bpp_path = self.bpp_dir.joinpath(self.bpp_paths[idx])
        df = pd.read_csv(bpp_path, sep=" ", header=None)
        bpp_matrix = np.eye(self.Lmax, dtype=np.float32)

        for i, j, v in df.values:
            bpp_matrix[int(i) - 1, int(j) - 1] = v
            bpp_matrix[int(j) - 1, int(i) - 1] = v

        x = {'seq': torch.from_numpy(seq), 'mask': mask, "bpp": bpp_matrix}
        y = {'react': react, 'mask': mask}

        if self.cfg.task.oof:
            x["seq_id"] = self.seq_id[idx]

        return x, y


class PseudoDataset(Dataset):
    def __init__(self, cfg):
        data_root = Path(__file__).parents[2].joinpath("input")
        parquet_path = data_root.joinpath("test_sequences.parquet")
        self.Lmax = 207
        df = pd.read_parquet(parquet_path)
        self.cfg = cfg

        if cfg.task.ngram == 1:
            self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        else:
            self.seq_map = {k: i for i, k in enumerate(itertools.product("ACGUX", repeat=cfg.task.ngram))}

        self.bpp_dir = data_root.joinpath("Ribonanza_bpp_files", "extra_data")
        self.bpp_df = pd.read_csv(data_root.joinpath("test_seq_id_to_bpp_path.csv"), index_col=0)
        self.hdf5_path = data_root.joinpath("pseudo_label.hdf5")

        with h5py.File(self.hdf5_path, "r") as f:
            self.seq_ids = f["seq_ids"][:]

    def __len__(self):
        return len(self.seq_ids)

    def encode_seq(self, seq):
        seq_len = len(seq)
        n = self.cfg.task.ngram
        seq = "X" * (n // 2) + seq + "X" * (n // 2)
        return [self.seq_map[tuple(seq[i:i + n])] for i in range(seq_len)]

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx].decode("utf-8")

        with h5py.File(self.hdf5_path, "r") as f:
            k = "/".join([seq_id[:3], seq_id, "seq"])
            seq = f[k][()].decode("utf-8")
            k = "/".join([seq_id[:3], seq_id, "react"])
            react = f[k][:]
            react = np.pad(react, ((0, self.Lmax - len(react)), (0, 0)), constant_values=np.nan)

        if self.cfg.task.ngram == 1:
            seq = [self.seq_map[s] for s in seq]
        else:
            seq = self.encode_seq(seq)

        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)), constant_values=5 ** self.cfg.task.ngram - 1)

        bpp_path = self.bpp_dir.joinpath(self.bpp_df.loc[seq_id, "bpp_path"])
        df = pd.read_csv(bpp_path, sep=" ", header=None)
        bpp_matrix = np.eye(self.Lmax, dtype=np.float32)

        for i, j, v in df.values:
            bpp_matrix[int(i) - 1, int(j) - 1] = v
            bpp_matrix[int(j) - 1, int(i) - 1] = v

        x = {'seq': torch.from_numpy(seq), 'mask': mask, "bpp": bpp_matrix}
        y = {'react': react, 'mask': mask}
        return x, y


class PretrainDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        train_parquet_path = Path(__file__).parents[2].joinpath("input", "train_data.parquet")
        train_df = pd.read_parquet(train_parquet_path)
        train_df = train_df.loc[train_df.experiment_type == "2A3_MaP"]
        test_parquet_path = Path(__file__).parents[2].joinpath("input", "test_sequences.parquet")
        self.cfg = cfg
        self.mode = mode
        fold_id = cfg.data.fold_id
        fold_num = cfg.data.fold_num
        self.seq = train_df["sequence"].values
        self.seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.Lmax = 457
        split = list(KFold(n_splits=fold_num, random_state=42,
                           shuffle=True).split(self.seq))[fold_id][0 if mode == 'train' else 1]
        self.seq = self.seq[split]

        data_root = Path(__file__).parents[2].joinpath("input")
        self.bpp_dir = data_root.joinpath("Ribonanza_bpp_files", "extra_data")
        bpp_df = pd.read_csv(data_root.joinpath("train_seq_id_to_bpp_path.csv"))
        merged_df = pd.merge(train_df, bpp_df, on="sequence_id", how="left")
        self.bpp_paths = merged_df["bpp_path"].values

    def __len__(self):
        return len(self.seq)

    @staticmethod
    def mask_seq(seq, indices):
        seq = seq.copy()

        for i in indices:
            pattern = np.random.choice(["mask", "change", "none"], p=[0.8, 0.1, 0.1])
            if pattern == "mask":
                seq[i] = 4
            elif pattern == "change":
                candidates = [0, 1, 2, 3]
                candidates.remove(seq[i])
                seq[i] = np.random.choice(candidates)
            elif pattern == "none":
                pass
            else:
                raise ValueError(f"invalid pattern: {pattern}")

        return seq

    def __getitem__(self, idx):
        ori_seq = self.seq[idx]
        ori_seq = [self.seq_map[s] for s in ori_seq]
        ori_seq = np.array(ori_seq)
        indices = np.random.choice(len(ori_seq), int(len(ori_seq) * 0.15), replace=False)
        seq = self.mask_seq(ori_seq, indices)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        target_mask = torch.zeros(len(ori_seq), dtype=torch.bool)
        target_mask[indices] = True
        ori_seq[~target_mask] = -100
        seq = np.pad(seq, (0, self.Lmax - len(seq)))
        ori_seq = np.pad(ori_seq, (0, self.Lmax - len(ori_seq)), constant_values=-100)

        bpp_path = self.bpp_dir.joinpath(self.bpp_paths[idx])
        df = pd.read_csv(bpp_path, sep=" ", header=None)
        bpp_matrix = np.eye(self.Lmax, dtype=np.float32)

        for i, j, v in df.values:
            bpp_matrix[int(i) - 1, int(j) - 1] = v
            bpp_matrix[int(j) - 1, int(i) - 1] = v

        x = {"seq": torch.from_numpy(seq), "mask": mask, "bpp": bpp_matrix}
        y = {"seq": torch.from_numpy(ori_seq)}
        return x, y


def main():
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt
    cfg = OmegaConf.load("config.yaml")
    cfg.data.num_workers = 0
    dataset = PseudoDataset(cfg)
    # dataset = MyDataset(cfg, mode="train")

    for i in range(10):
        x, y = dataset[i]
        print("a")


if __name__ == '__main__':
    main()
