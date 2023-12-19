import torch
import torch.nn as nn


def get_loss(cfg):
    if cfg.task.mode == "train":
        return MyLoss(cfg)
    elif cfg.task.mode == "pretrain":
        return PretrainLoss(cfg)
    else:
        raise ValueError(f"invalid mode: {cfg.task.mode}")


class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.l1_loss = nn.L1Loss(reduction="none")
        self.l2_loss = nn.MSELoss()

    def forward(self, y_pred, y_true, val=False):
        return_dict = dict()
        p = y_pred[y_true['mask'][:, :y_pred.shape[1]]]
        y = y_true['react'][y_true['mask']]

        if val:
            p = p.clip(0, 1)
            y = y.clip(0, 1)

        l1_loss = self.l1_loss(p, y)

        if not val:
            mask = ((p <= 0) & (y <= 0)) | ((p >= 1) & (y >= 1))
            l1_loss[mask] = 0

        l1_loss = l1_loss[~torch.isnan(l1_loss)]
        l1_loss = l1_loss.mean()
        return_dict["loss"] = l1_loss
        return return_dict


class PretrainLoss(nn.Module):
    def __init__(self, cfg):
        super(PretrainLoss, self).__init__()
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, y_pred, y_true):
        return_dict = dict()
        p = y_pred.transpose(1, 2)
        y = y_true["seq"][:, :y_pred.shape[1]]
        loss = self.loss(p, y)
        return_dict["loss"] = loss
        return return_dict


def main():
    pass


if __name__ == '__main__':
    main()
