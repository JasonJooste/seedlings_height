import torch
import torchvision
from torch.nn.init import xavier_normal_
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from src.models.modifications import RoIHeadsFinalLayer, FasterRCNNEndHeights, RoIHeadsVanilla, FasterRCNNStartHeights, \
    FasterRCNNVanilla
from torch import nn

NUM_CLASSES = 2
IMAGE_MEAN = [0.513498842716217, 0.5408999919891357, 0.5676814913749695, 0.003977019805461168]
IMAGE_STD = [0.21669578552246094, 0.22595597803592682, 0.2860477566719055, 0.007557219825685024]

#
def make_vanilla_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                   trainable_backbone_layers=trainable_backbone_layers)
    # This model has no new weights
    model.new_weights = nn.ParameterList()
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNVanilla
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining.pt"
    torch.save(model, path)

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
    new_weights = torch.zeros(representation_size, num_new_features)
    xavier_normal_(new_weights)
    extended_weights = torch.cat((existing_weights, new_weights), 1)
    model.roi_heads.box_head.fc6.weight = nn.Parameter(extended_weights)
    model.roi_heads.box_head.fc6.in_features = extended_weights.shape[1]
    model.roi_heads.box_roi_pool.featmap_names.append('heights')
    # Log the new weights that we added so they can be tracked
    model.existing_weights_shape = existing_weights.shape

    new_weights = torch.narrow(extended_weights, 1, 0, existing_weights.shape[1])
    # new_weights = torch.narrow(extended_weights, 1, existing_weights.shape[1], new_weights.shape[1])


    model.new_weights = nn.ParameterList([nn.Parameter(new_weights)])
    # Assign a new class prediction layer for two classes
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_final.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_final.pt"
    torch.save(model, path)

def make_first_layer_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    # Change the model class so the model uses our forward function
    model.__class__ = FasterRCNNStartHeights
    # Don't need to change the roi_heads. Everything is normal there
    model.roi_heads.__class__ = RoIHeadsVanilla
    # Rebuild first conv layer of the backbone to accept one extra layer
    new_shape = list(model.backbone.body.conv1.weight.shape)
    new_shape[1] = 1
    new_weights = torch.zeros(new_shape)
    xavier_normal_(new_weights)
    existing_weights = model.backbone.body.conv1.weight
    extended_weights = torch.cat((existing_weights, new_weights), 1)
    model.backbone.body.conv1.weight = nn.Parameter(extended_weights)
    model.existing_weights_shape = existing_weights.shape
    # Now change the transformation to accomodate 4d transformations
    model.transform.image_mean = IMAGE_MEAN
    model.transform.image_std = IMAGE_STD
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_first.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_first.pt"
    torch.save(model, path)

def make_normal_backbone_model(model_dir, pretrained=True, trainable_backbone_layers=0):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    # Change the model backbone
    model.backbone = torchvision.models.resnet50(pretrained=True, progress=True)
    # Assign the trainable layers (this is usually done during initialisation)
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_backbone_layers]
    for name, parameter in model.backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    # This model has no new weights
    model.new_weights = nn.ParameterList()
    # Assign model new class to override forward function so it can take height input (and ignore it)
    model.__class__ = FasterRCNNVanilla
    # Replace the model's RoIHead with our version
    model.roi_heads.__class__ = RoIHeadsVanilla
    # Assign a new class prediction layer for two classes
    representation_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(representation_size, NUM_CLASSES)
    if pretrained:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_pretrained_basic_backbone.pt"
    else:
        path = model_dir / f"RCNN-resnet-50_{trainable_backbone_layers}_layer_no_pretraining_basic_backbone.pt"
    torch.save(model, path)