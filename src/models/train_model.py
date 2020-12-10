import torch
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

import src.util.util as utils
import pandas as pd
from src.data.data_classes import SeedlingDataset
from sklearn.model_selection import train_test_split
import torch.optim.adam
from pycocotools.cocoeval import COCOeval
import sklearn.metrics
import numpy as np


def get_dataloader(params):
    # First split into train and validation
    all_train = pd.read_csv(params["train_file"])
    train, valid = train_test_split(all_train, test_size=params["valid_ratio"])
    train_dataset = SeedlingDataset(train)
    valid_dataset = SeedlingDataset(valid)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params["train_batch_size"],
        shuffle=True,
        num_workers=params["dataloader_num_workers"],
        collate_fn=utils.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=params["eval_batch_size"],
        shuffle=False,
        num_workers=params["dataloader_num_workers"],
        collate_fn=utils.collate_fn
    )
    return train_dataloader, valid_dataloader

#TODO: Pass extra params for optimiser
def get_optimiser(model_params, params):
    if params["optimiser"] == "ADAM":
        optimiser = torch.optim.Adam(model_params)
    return optimiser


# TODO: This could be more complicated later
def get_device(params):
    return torch.device('cuda')


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


########################################################################################################################
# Taken from coco_utils
def prepare_for_coco_detection(predictions):
    """
    Adds the image ID and converts to xywh format for the boxes
    :param predictions:
    :return:
    """
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = utils.convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        individual_boxes = [{
            "image_id": original_id,
            "category_id": labels[k],
            "bbox": box,
            "score": scores[k],
            } for k, box in enumerate(boxes)]
        coco_results.extend(individual_boxes)
    return coco_results


def convert_to_coco_ds(ds):
    images, targets, _ = ds[:]
    return convert_to_coco(images, targets)


def convert_to_coco(images, targets):
    coco_ds = COCO()
    assert len(images) == len(targets)
    ann_id = 0
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(images)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        #TODO - do we really need to send the entire image for getting the MAP? Can we just send an object that has a shape?
        img = images[img_idx]
        target = targets[img_idx]
        # Replacing their structure with mine
        image_id = target["image_id"].item()
        # image_id = torch.tensor([img_idx])
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = target["boxes"]
        # Convert to xywh
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = target['labels'].tolist()
        areas = target['area'].tolist()
        iscrowd = target['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
########################################################################################################################
@torch.no_grad()
def get_predictions(model, dataloader, params):
    device = get_device(params)
    predictions = {}
    losses = []
    #TODO: We need to think about batch size as well. We get a single loss for each batch
    for batch_images, batch_targets, _ in dataloader:
        batch_size = dataloader.batch_size
        images = list(img.to(device) for img in batch_images)
        batch_targets = ([{k: v.to(device) for k, v in t.items()} for t in batch_targets])
        batch_losses, batch_predictions = model(images, batch_targets)
        # Push the results back to cpu
        batch_outputs = [{k: v.to("cpu") for k, v in t.items()} for t in batch_predictions]
        # Map IDs to outputs
        res = {target["image_id"].item(): output for target, output in zip(batch_targets, batch_outputs)}
        predictions.update(res)
        # Push the losses back to the cpu
        batch_loss = sum(batch_losses.values())
        losses.append(batch_loss.item() * batch_size)
    # Now we average over the total number of values to get the average loss per sample
    av_loss = np.mean(losses) / len(dataloader.dataset)
    return predictions, av_loss


def get_MAP(coco_gt, coco_pred):
    """Return the MAP at IoU=05"""
    MAP_IND = 1
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[MAP_IND]


@utils.block_print
def get_train_MAP(predictions, images, targets):
    """ Get MAP and loss of a model on the training dataset. For this we already have predictions, losses and targets"""
    # Create a coco object for the true labels
    coco_gt = convert_to_coco(images, targets)
    # Create a coco object for the predictions
    predictions = prepare_for_coco_detection(predictions)
    if not predictions:
        # If we make no predictions then we would be dividing by 0. As convention let this be 0
        print("no preds")
        return 0
    coco_pred = coco_gt.loadRes(predictions)
    return get_MAP(coco_gt, coco_pred)


@utils.block_print
def evaluate_model_validation(model, dataloader, params):
    """
    Get the MAP and loss of a model on the validation dataset. This loads data from the dataloader.

    """
    # Make predictions for the validation set
    # model.eval()
    predictions, av_loss = get_predictions(model, dataloader, params)
    if not predictions:
        print("Nothing predicted")
        return av_loss, 0
    # Converts to COCO format - i.e. adds image labels and changes to xywh
    predictions = prepare_for_coco_detection(predictions)
    # We need to load the coco api with the ground truth labels
    coco_gt = convert_to_coco_ds(dataloader.dataset)
    # Reformats adds segmentation. This one has scores though
    coco_pred = coco_gt.loadRes(predictions)
    MAP = get_MAP(coco_gt, coco_pred)
    return av_loss, MAP


def fit(params):
    """
    Generic model fitting function
    """
    # Hack to get the RCNN to always return both the predictions and losses
    torch.jit.is_scripting = lambda: True
    device = get_device(params)
    train_dataloader, valid_dataloader = get_dataloader(params)
    model = torch.load(params["base_model_path"]).to(device)
    opt = get_optimiser(model.parameters(), params)
    for epoch in range(params["epochs"]):
        # Training
        model.train()
        epoch_losses = []
        epoch_maps = []
        for images, targets, image_ids in train_dataloader:
            batch_size = train_dataloader.batch_size
            # Send to the current device
            if device:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Get the losses
            assert len(targets) > 0, "Should at least have one bounding box per image (for now)"
            loss_dict, predictions = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            opt.step()
            opt.zero_grad()
            # Logging here
            # Push the results back to cpu
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in predictions]
            # Map IDs to outputs
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            MAP = get_train_MAP(res, images, targets)
            loss_value = losses.item()
            epoch_losses.append(loss_value * batch_size)
            epoch_maps.append(MAP)
        valid_loss, valid_MAP = evaluate_model_validation(model, valid_dataloader, params)
        # Get the average loss per sample
        train_loss = np.mean(epoch_losses) / len(train_dataloader.dataset)
        train_MAP = np.mean(epoch_maps)
        # Now validation
        print(f"EPOCH {epoch} valid loss: {valid_loss:.8f} | valid MAP: {valid_MAP:.3f} | train loss: {train_loss:.8f} | "
              f"train MAP: {train_MAP:.3f}")
        # TODO: Model saving

        # TODO: Early stopping

    return model
