import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from src.models.modifications import RoIHeadsFinalLayer, FasterRCNNEndHeights
from torch import nn

NUM_CLASSES = 2

#
# def make_vanilla_model(model_dir, pretrained=True, trainable_backbone_layers=0):
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
#                                                                  trainable_backbone_layers=trainable_backbone_layers)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
#     # Replace the model's RoIHead with our version
#     roi_wapped = RoIHeadsWrapper(model.roi_heads)
#     model.roi_heads = roi_wapped
#     if pretrained:
#         path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained.pt"
#     else:
#         path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining.pt"
#     torch.save(model, path)

def make_final_layer_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    # Change the model class so the model uses our forward function
    model.__class__ = FasterRCNNEndHeights
    # Change the roi heads class so they use our forward function
    model.roi_heads.__class__ = RoIHeadsFinalLayer
    # Rebuild add extra weights to the fc layer in the roi head to handle the new features
    resolution = model.roi_heads.box_roi_pool.output_size[0]
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    # Extend the existing box head with weights for the height layer
    num_new_features = resolution ** 2
    existing_weights = model.roi_heads.box_head.fc6.weight
    new_weights =torch.randn(representation_size, num_new_features)
    extended_weights = torch.cat((existing_weights, new_weights), 1)
    model.roi_heads.box_head.fc6.weight = nn.Parameter(extended_weights)
    model.roi_heads.box_head.fc6.in_features = extended_weights.shape[1]
    model.roi_heads.box_roi_pool.featmap_names.append('heights')
    # Assign a new class prediction layer for two classes
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_final.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_final.pt"
    torch.save(model, path)


