# Digital Twin
Digital Twin's Real-time 6DoF Object Pose Estimation in Pytorch Lightning \
reference: https://github.com/j96w/DenseFusion & https://github.com/adamchang2000/DenseFusion 

UC Berkeley OpenARK pose estimation codebase. 

## PyTorch Lightning
PyTorch Lightning is an open-source Python library that provides a high-level interface for PyTorch. \
For more information on installation and tutorial, visit https://www.pytorchlightning.ai/

## Set Up
```
conda install python=3.6
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge tensorboard
conda install -c anaconda scipy
conda install -c open3d-admin open3d

cd pointnet
python setup.py install
```
If the above instructions do not work for some reason, do it with `pip install`. \
The Linux server in the lab should be set up, just do `conda activate py36`. \

## Train
```
python trainer.py
```
change configurations, modify file `cfg/config.py`\
To resume training, add arguements following the instruction in `trainer.py`. Note `start_epoch` is not useful in lightning.

## Branches
**Main** - DenseFusion in Lightning Modules \
**6d_rot** - proposed Symmetry-aware 6DoF pose estimation algorithm \
**new_rot** - [deprecated] DenseFusion with a noval rotation representation, not producing great results \
**randlanet** - [deprecated] 6d_rot using RandLANet for semetic segmentation instead of PointNet \
**refactored** - refactored training configration for 6d_rot, checkout this branch for the lastest version \

## Lightning Module Notes
All the dataset and dataloaders are in `DataModule`, training module is called `DenseFusionModule`\
Lightning will handle checkpoint, logging, and profiling. 

checkpoints will be stored in folder `ckpt` \
profiling info will be printed after training is finished.

## Tensorboard
This repo currently use tesnorboard logging, logs will be stored at `tb_log`. \
if `visualiztion` is enabled in config, poinitcloud visualization will be added to tensorboard. \
To launch tensorboard, run
```
tensorboard --logdir tb_logs/dense_fusion/version_[NUMBER]
```

## Visualization
To visualized prediction point clouds on YCB dataset, run
```
python visualize_ycb_points.py
```
remember to tweak the input argumenets. The config file should be the version that the model trained on. \
Visualization should be stored in folder `visualization`. \
remember to renave it and organize the outputs as visualization will override the previous version.

## Evaluation
To produce numerical result, run
```
python eval_ycb.py
```
results will be printed to console. 
