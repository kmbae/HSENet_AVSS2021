#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets.builtin_meta import (
    COCO_PERSON_KEYPOINT_NAMES,
    COCO_PERSON_KEYPOINT_FLIP_MAP,
    KEYPOINT_CONNECTION_RULES
)

# HSENet
from hsenet.data.datasets import register_coco_instances
from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
from hsenet.config.config import add_posture_config
from hsenet.data import DatasetMapper, build_detection_train_loader#, build_detection_test_loader


dataset_root = 'datasets/'

meta = {
        "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
        "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        #"thing_classes": ['Walking', 'Crouch', 'Lying', 'Standing', 'Running', 'Sitting'],
        }

register_coco_instances("IHP_train", meta, dataset_root + "IHP/annotations/instances_mphbE_train2020.json", dataset_root + "IHP/train2020", dataset_source=0)
register_coco_instances("IHP_test", meta, dataset_root + "IHP/annotations/instances_mphbE_test2020.json", dataset_root + "IHP/test2020", dataset_source=0)
register_coco_instances("IHP-E_train", meta, dataset_root + "IHP-E/annotations/instances_mphbE_train2021.json", dataset_root + "IHP-E/train2021", dataset_source=0)
register_coco_instances("IHP-E_test", meta, dataset_root + "IHP-E/annotations/instances_mphbE_test2021.json", dataset_root + "IHP-E/test2021", dataset_source=0)
register_coco_instances("keypoints_coco_2017_train_mphbe", meta, dataset_root + "coco/annotations/person_keypoints_train2017.json", dataset_root + "coco/train2017", dataset_source=1)
register_coco_instances("keypoints_coco_2017_val_mphbe", meta, dataset_root + "coco/annotations/person_keypoints_val2017.json", dataset_root + "coco/val2017", dataset_source=1)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            #evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            evaluator_list.append(COCOEvaluator(dataset_name, ("bbox", "segm", "keypoints"), True, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    #@classmethod
    #def build_test_loader(cls, cfg, dataset_name):
    #    #return build_detection_test_loader(cfg, dataset_name, mapper=None)
    #    return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        #print(cfg)
        #return build_detection_train_loader(cfg, mapper=None)
        #tmp = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        #tmptmp = iter(tmp)
        #print(tmptmp.next())
        #raise Exception('Process ended')
        #return tmp
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
