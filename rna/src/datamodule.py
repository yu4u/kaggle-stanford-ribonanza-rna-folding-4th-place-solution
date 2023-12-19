from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate

from .dataset import MyDataset, PretrainDataset, PseudoDataset


class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = Path(__file__).resolve().parents[2].joinpath("input")
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def get_dataset_class(self):
        if self.cfg.task.mode == "train":
            return MyDataset
        elif self.cfg.task.mode == "pretrain":
            return PretrainDataset
        else:
            raise NotImplementedError

    def setup(self, stage=None):
        dataset_class = self.get_dataset_class()
        self.train_dataset = dataset_class(self.cfg, "train")

        if self.cfg.task.pseudo:
            pseudo_dataset = PseudoDataset(self.cfg)
            self.train_dataset = ConcatDataset([self.train_dataset, pseudo_dataset])

        self.val_dataset = dataset_class(self.cfg, "val")

        if stage == "test":
            if self.cfg.task.oof:
                self.test_dataset = self.val_dataset
            else:
                self.test_dataset = dataset_class(self.cfg,  "test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
