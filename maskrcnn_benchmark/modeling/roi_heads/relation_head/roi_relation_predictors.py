# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels
        num_classes = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
