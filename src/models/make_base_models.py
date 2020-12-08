import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 2
def make_vanilla_model(model_dir):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    path = model_dir + "/RCNN-resnet-50.pt"
    torch.save(model, path)




