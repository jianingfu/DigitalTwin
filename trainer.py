import random
import torch
import os
from YCBDataModule import YCBDataModule
from LinemodDataModule import LinemodDataModule
from CustomDataModule import CustomDataModule
from DenseFusionModule import DenseFusionModule
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lib.randla_utils import randla_processing
from cfg.config import YCBConfig as Config

parser = argparse.ArgumentParser()
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model') # ckpt/last.ckpt
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 0, help='which epoch to start')
opt = parser.parse_args()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cfg = Config()
cfg.manualSeed = random.randint(1, 10000)
random.seed(cfg.manualSeed)
torch.manual_seed(cfg.manualSeed)

# :)
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    dataModule = YCBDataModule(cfg)

    if opt.resume_refinenet != '':
        cfg.refine_start = True
        cfg.decay_start = True
    else:
        cfg.refine_start = False

    # init model
    densefusion = DenseFusionModule(cfg)

    checkpoint_callback = ModelCheckpoint(dirpath='ckpt/', 
                            filename='df-{epoch:02d}-{val_dis:.5f}',
                            monitor="dis",
                            save_last=True,
                            save_top_k=1,
                            every_n_epochs=1)

    logger = TensorBoardLogger("tb_logs", name="dense_fusion")
    trainer = pl.Trainer(logger=logger, 
                            callbacks=[checkpoint_callback],
                            max_epochs=cfg.nepoch,
                            check_val_every_n_epoch=cfg.repeat_epoch,
                            accelerator="gpu",
                            devices=[1], 
                            strategy="ddp",
                            profiler="simple",
                            resume_from_checkpoint= opt.resume_posenet,
                            )
    if opt.resume_posenet:
        trainer.fit(densefusion, datamodule=dataModule, ckpt_path="ckpt/last.ckpt")
    else:
        trainer.fit(densefusion, datamodule=dataModule)
