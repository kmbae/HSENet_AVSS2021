# Irregular Human Postures(IHP) dataset
This is an official repo of Irregular Human Postures (IHP) dataset dataset. IHP dataset is a irregular human posture dataset. This dataset consists of MPHB [1] dataset and street scene dataset. The IHP dataset uses subset of images from MPHB [1] dataset that is created using LSP [2] and MPII [3] dataset. The IHP dataset contains bounding box, segmentation and keypoint annotation using the .json file format as MS-COCO [4]. All annotations were created using COCO Annotator [6].

## Installation

Please refer installation manual of [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) [5]

## Dataset

Download the datasets using the links below.

- IHP dataset
https://www.dropbox.com/s/qqlh1k18qwff57f/IHP2021.tar.gz?dl=0

- MPHB dataset (only images)
http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/MPHB.html

After downloading the images from MPHB dataset, use ```mphb2ihp.ipynb``` to parse the images into IHP dataset style.

## HSENet

Train HSENet-U with the following code
```sh
python train_net.py --config configs/ihp_hsenet_R_50_FPN_3x.yaml --num-gpus 8 --resume
```

## References

[1] Y. Cai and X. Tan, “Weakly supervised human body detection under arbitrary poses,” in ICIP, 2016.

[2] S. Johnson and M. Everingham, “Clustered pose and nonlinear appearance models for human pose estimation.” in BMVC, 2010.

[3] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele, “2d human pose estimation: New benchmark and state of the art analysis,” in CVPR, 2014.

[4] Tsung Yi Lin, Michael Maire, Serge Belongie, JamesHays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick, “Microsoft COCO: Common ob-jects in context,” in ECCV, 2014.

[5] Y. Wu, A. Kirillov, F. Massa, Wan-Yen Lo and R. Girshick, "Detectron2," https://github.com/facebookresearch/detectron2, 2019.

[6] J. Brooks, “COCO Annotator,” https://github.com/jsbroks/coco-annotator/, 2019.
