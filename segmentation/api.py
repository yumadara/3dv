#from detectron2.utils.logger import setup_logger
#setup_logger()

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg

from segmentation.centernet.config import add_centernet_config


class CenterNetAPI:
    def __init__(self, config_path="segmentation/config/nuImages_CenterNet2_DLA_640_8x.yaml", 
                       weight_path="segmentation/centernet2_checkpoint.pth", conf=0.7):
        cfg = get_cfg()
        add_centernet_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
        if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
            cfg.MODEL.CENTERNET.INFERENCE_TH = conf
            cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
        cfg.MODEL.WEIGHTS = weight_path
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)
    
    def run(self, cv_image):
        # assumes input has BGR format
        out = self.predictor(cv_image)["instances"]
        pred_classes = out.pred_classes
        car_filter = pred_classes == 0 
        pred_masks = out.pred_masks[car_filter].detach().cpu().numpy()
        pred_boxes = out.pred_boxes.tensor[car_filter].detach().cpu().numpy()
        return pred_boxes, pred_masks
        