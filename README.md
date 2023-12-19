# Stanford Ribonanza RNA Folding 4th Place Solution (yu4u's Part)

This is the implementation of the 4th place solution (yu4u's part) for [Stanford RiboNanza RNA Folding Competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) at Kaggle.
The overall solution is described in [this discussion](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460203).
The other implementations of our solution can be found in
[here](https://github.com/tattaka/stanford-ribonanza-rna-folding-public) (by @tattaka)
and [here](https://github.com/fuumin621/stanford-ribonanza-rna-folding-4th) (by @monnu).

## Preparation
- Download the competition dataset from [here](https://www.kaggle.com/c/stanford-ribonanza-rna-folding/data) and put them in `input` directory.
- Download the perquet files (`train_data.parquet` and `test_sequences.parquet`) from [here](https://www.kaggle.com/datasets/iafoss/stanford-ribonanza-rna-folding-converted) and put it in `input` directory.
- Build the docker image and run the container:

```shell
export UID=$(id -u)
docker compose up -d
docker compose exec dev /bin/bash
```

- Login to wandb.

## Training

### 1st Training
```shell
cd rna
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 task.sn_th=0.5 wandb.name=fold0 data.fold_id=0
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 task.sn_th=0.5 wandb.name=fold1 data.fold_id=1
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 task.sn_th=0.5 wandb.name=fold2 data.fold_id=2
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 task.sn_th=0.5 wandb.name=fold3 data.fold_id=3
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 task.sn_th=0.5 wandb.name=fold4 data.fold_id=4
```

### Finetune with Lower Learning Rate
```shell
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 trainer.max_epochs=32 wandb.name=ft_fold0 opt.lr=2e-4 scheduler.min_lr=0.0 task.sn_th=1.0 model.resume_path=[PATH_TO_FOLD0_CHECKPOINT] data.fold_id=0
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 trainer.max_epochs=32 wandb.name=ft_fold1 opt.lr=2e-4 scheduler.min_lr=0.0 task.sn_th=1.0 model.resume_path=[PATH_TO_FOLD1_CHECKPOINT] data.fold_id=1
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 trainer.max_epochs=32 wandb.name=ft_fold2 opt.lr=2e-4 scheduler.min_lr=0.0 task.sn_th=1.0 model.resume_path=[PATH_TO_FOLD2_CHECKPOINT] data.fold_id=2
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 trainer.max_epochs=32 wandb.name=ft_fold3 opt.lr=2e-4 scheduler.min_lr=0.0 task.sn_th=1.0 model.resume_path=[PATH_TO_FOLD3_CHECKPOINT] data.fold_id=3
python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp trainer.precision=16 data.num_workers=5 trainer.max_epochs=32 wandb.name=ft_fold4 opt.lr=2e-4 scheduler.min_lr=0.0 task.sn_th=1.0 model.resume_path=[PATH_TO_FOLD4_CHECKPOINT] data.fold_id=4
```

### Test for submission
Create submission files for each fold.

```shell
python 03_test.py trainer.accelerator=gpu trainer.devices=[0] data.batch_size=256 data.num_workers=10 trainer.precision=16 model.resume_path=[PATH_TO_CHECKPOINT]
```

Create oof files for each fold (for weighted ensemble).

```shell
python 03_test.py trainer.accelerator=gpu trainer.devices=[0] data.batch_size=256 data.num_workers=10 trainer.precision=16 model.resume_path=[PATH_TO_CHECKPOINT] data.fold_id=[FOLD_ID] task.oof=True
```

