# Irregular Human Postures(IHP) dataset
This is an official repo of IHP dataset

## Installation
Please refer installation manual of [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

## Dataset
Download the dataset using the following link.

- IHP dataset

## HSENet
Train HSENet-U with the following code
```sh
python train_net.py --config configs/ihp_hsenet_R_50_FPN_3x.yaml --num-gpus 8 --resume
```
