# DigitalTwin
**[UNDER CONSTRUCTION]**
DenseFusion Lightning Module, reference: https://github.com/j96w/DenseFusion

OpenARK 3D pose estimation codebase.

## Set Up
```
pip install pytorch-lightning
cd pointnet
python setup.py install
```

## Train
```
python3 trainer.py
```

## Visualize Tensorboard Logs
```
tensorboard --logdir tb_logs/dense_fusion/version_0
```

Replace `0` with whichever version you are running.
