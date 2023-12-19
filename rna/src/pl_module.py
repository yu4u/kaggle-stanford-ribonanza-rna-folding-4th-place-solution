from pathlib import Path
import numpy as np
import pandas as pd
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2


from .model import get_model_from_cfg
from .loss import get_loss


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.seq_lengths = None
        self.reacts = None
        self.cfg = cfg
        self.model = get_model_from_cfg(cfg, cfg.model.resume_path)

        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV2(self.model, decay=0.99)

        self.loss = get_loss(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y).items()}
        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        loss_dict = self.loss(output, y, val=True)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_start(self):
        self.reacts = []
        self.seq_lengths = []

    def test_step(self, batch, batch_idx):
        if self.cfg.task.oof:
            x, y = batch
        else:
            x = batch

        output = self.model(x)
        output = np.clip(output.float().cpu().numpy(), 0.0, 1.0)  # b, seq_len, 2
        seq_lengths = x["mask"].sum(1).cpu().numpy()

        if self.cfg.task.oof:
            seq_ids = x["seq_id"]
            output_dir = Path(__file__).resolve().parents[2].joinpath("input", "oof")
            output_dir.mkdir(exist_ok=True, parents=True)

            for i, seq_id in enumerate(seq_ids):
                seq_length = seq_lengths[i]
                np.save(output_dir.joinpath(f"{seq_id}.npy"), output[i][:seq_length])
        else:
            output = np.pad(output, ((0, 0), (0, 457 - output.shape[1]), (0, 0)))
            self.reacts.append(output)
            self.seq_lengths.append(seq_lengths)

    def on_test_epoch_end(self):
        if self.cfg.task.oof:
            return

        reacts = np.concatenate(self.reacts, 0)
        seq_lengths = np.concatenate(self.seq_lengths, 0)
        preds_processed = []

        for i, react in enumerate(reacts):
            preds_processed.append(react[:seq_lengths[i]])

        concat_preds = np.concatenate(preds_processed)
        submission = pd.DataFrame({
            "id": np.arange(0, len(concat_preds), 1),
            "reactivity_DMS_MaP": concat_preds[:, 1],
            "reactivity_2A3_MaP": concat_preds[:, 0]
        })
        filename = Path(self.cfg.model.resume_path).stem
        submission.to_csv(f"{filename}.csv", index=False)
        submission.head()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
                                                    warmup_lr=self.cfg.opt.lr / 10.0, **self.cfg.scheduler)
        lr_dict = dict(
            scheduler=scheduler,
            interval="epoch",  # same as default
            frequency=1,  # same as default
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
