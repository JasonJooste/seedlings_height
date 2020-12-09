import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.models.modifications import RoIHeadsWrapper
import types
NUM_CLASSES = 2


def make_vanilla_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    # Replace the model's RoIHead with our version
    roi_wapped = RoIHeadsWrapper(model.roi_heads)
    model.roi_heads = roi_wapped
    path = model_dir + "/RCNN-resnet-50.pt"
    torch.save(model, path)




