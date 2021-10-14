import DenseFusionModule
import YCBDataModule
import pytorch_lightning as pl

# init DataModule
dataModule = YCBDataModule()

# init model
densefusion = DenseFusionModule()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer()
trainer.fit(densefusion, dataModule)