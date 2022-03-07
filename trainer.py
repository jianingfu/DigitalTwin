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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = 'datasets/ycb/YCB_Video_Dataset', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 8, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.0015, help='pred_c regularization')
parser.add_argument('--w_rate', default=0.3, help='pred_c regularization decay rate')
parser.add_argument('--decay_margin', default=0.02, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.02, help='margin to start the training of iterative refinement')
parser.add_argument('--refine_epoch', default=15, help='epoch to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.01, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model') # ckpt/last.ckpt
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--visualize', type=bool, default = False, help='visualize using plotly')
parser.add_argument('--image_size', type=int, default=300, help="square side length of cropped image")
parser.add_argument('--use_normals', action="store_true", default=False, help="estimate normals and augment pointcloud")
parser.add_argument('--old_batch_mode', action="store_true", default=True, help="old batch mode, where batch size is 1 and gradients are accumulated")

opt = parser.parse_args()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# :)
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
        opt.image_size = 300
        # init DataModule
        dataModule = YCBDataModule(opt)
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 1
        opt.nepoch = opt.nepoch*opt.repeat_epoch
        # init DataModule
        dataModule = LinemodDataModule(opt)
    elif opt.dataset == 'custom':
        opt.dataset_root = 'datasets/custom/custom_preprocessed'
        opt.num_objects = 1
        opt.num_points = 500
        opt.outf = 'trained_models/custom'
        opt.log_dir = 'experiments/logs/custom'
        opt.repeat_epoch = 1
        # init DataModule
        dataModule = CustomDataModule(opt)
    else:
        print('Unknown dataset')

    if opt.resume_refinenet != '':
        opt.refine_start = True
        opt.decay_start = True

    # init model
    densefusion = DenseFusionModule(opt)

    checkpoint_callback = ModelCheckpoint(dirpath='ckpt/', 
                            filename='df-{epoch:02d}-{val_dis:.5f}',
                            monitor="dis",
                            save_last=True,
                            save_top_k=1,
                            every_n_epochs=1)

    logger = TensorBoardLogger("tb_logs", name="dense_fusion")
    trainer = pl.Trainer(logger=logger, 
                            callbacks=[checkpoint_callback],
                            max_epochs=opt.nepoch - opt.start_epoch,
                            check_val_every_n_epoch=opt.repeat_epoch,
                            gpus=1,
                            profiler="advanced",
                            resume_from_checkpoint= opt.resume_posenet,
                            )
    if opt.resume:
        trainer.fit(densefusion, datamodule=dataModule, ckpt_path="ckpt/last.ckpt")
    else:
        trainer.fit(densefusion, datamodule=dataModule)
