from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch
#from multitask.config import add_dataset_category_config, add_multitask_config
#from vovnet import add_vovnet_config
#from multitask.config import add_dataset_category_config, add_multitask_config
import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer
import fvcore

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES

dataset_root = '/home/kimin/datasets/'

meta = {
        "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
        "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }


register_coco_instances("MPHBE2020_train", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_train2020.json", dataset_root + "MPHBE2020/train2020")
register_coco_instances("MPHBE2020_test", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_test2020.json", dataset_root + "MPHBE2020/test2020")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--img', default=None)
    args = args.parse_args()
    print("Command Line Args:", args)

    # Inference with a multitask model
    cfg = setup(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = 'checkpoints/mask_rcnn_R_50_FPN_3x/model_0269999.pth'
    predictor = DefaultPredictor(cfg)

    #im = cv2.imread('/home/kmbae/Dataset/coco/train2017/000000249958.jpg')#plane
    #im = cv2.imread('/home/kmbae/Dataset/coco/train2017/000000581749.jpg')#vegetables
    #im = cv2.imread('/home/kmbae/Dataset/coco/train2017/000000082833.jpg')#living room
    #im = cv2.imread('/home/kmbae/Dataset/coco/train2017/000000165852.jpg')#motorcycle
    #im = cv2.imread('/home/kmbae/Dataset/coco/train2017/000000165916.jpg')#teddy bear
    #im = cv2.imread('test.jpg')
    #im = cv2.imread('test2.jpg')
    #im = cv2.imread('Stanford-cooking-013.jpg')
    #im = cv2.imread('/home/kmbae/Dataset/Gwanak-coco/test2020/2-031-01460.jpg')
    im = cv2.imread(args.img)
    #import ipdb
    #ipdb.set_trace()
    #print(fvcore.nn.parameter_count_table(predictor.model))
    outputs = predictor(im)

    data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])

    v = Visualizer(im[:,:,::-1], data_meta, scale=1.2)
    #outputs["instances"].remove('pred_classes')
    #outputs["instances"].remove('pred_boxes')
    #outputs["instances"].remove('scores')
    #outputs["instances"].remove('pred_keypoints')
    #outputs["instances"].remove('pred_masks')
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = Image.fromarray(out.get_image())
    img.save('result1.jpg')
    img.show()

    #cv2.imshow('test', out.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
