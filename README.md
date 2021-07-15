# Irregular Human Postures(IHP) dataset
This is an official repo of IHP dataset

## Installation
Please refer installation manual of [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

## Dataset
Download the datasets using the links below.

- IHP dataset
https://www.dropbox.com/s/qqlh1k18qwff57f/IHP2021.tar.gz?dl=0
- MPHB dataset (only images)
http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/MPHB.html

## HSENet
Train HSENet-U with the following code
```sh
python train_net.py --config configs/ihp_hsenet_R_50_FPN_3x.yaml --num-gpus 8 --resume
```
