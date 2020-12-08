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

def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets, _ = ds[img_idx]
        # Replacing their structure with mine
        image_id = targets["image_id"].item()
        # image_id = torch.tensor([img_idx])
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        # Convert to xywh
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
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

# TODO This might be inefficient with all of these data stored on the graphics card
def get_predictions(model, dataloader, params):
    device = get_device(params)
    image_ids = []
    targets = []
    predictions = {}
    for batch_images, batch_targets, batch_ids in dataloader:
        images = list(img.to(device) for img in batch_images)
        # TODO: I'd imagine the targets can stay in regular memory - though it's set up differently in the other
        #  file.. weird
        # targets.extend([{k: v.to(device) for k, v in t.items()} for t in batch_targets])
        targets.extend(batch_targets)
        batch_predictions = model(images)
        # Push the results back to cpu
        batch_outputs = [{k: v.to("cpu") for k, v in t.items()} for t in batch_predictions]
        # Map IDs to outputs
        res = {target["image_id"].item(): output for target, output in zip(batch_targets, batch_outputs)}
        predictions.update(res)
        image_ids.extend(batch_ids)
    return predictions, targets, image_ids


def evaluate_model(model, dataloader, epoch, params):
    # Make predictions for the validation set
    model.eval()
    predictions, targets, image_ids = get_predictions(model, dataloader, params)
    # Converts to COCO format - i.e. adds image labels and changes to xywh
    predictions = prepare_for_coco_detection(predictions)
    if not predictions:
        print("Nothing predicted")
        return
    # We need to load the coco api with the ground truth labels
    coco_gt = convert_to_coco_api(dataloader.dataset)
    # Reformats adds segmentation. This one has scores though
    coco_pred = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def fit(params):
    """
    Generic model fitting function
    """
    device = get_device(params)
    train_dataloader, valid_dataloader = get_dataloader(params)
    model = torch.load(params["base_model_path"]).to(device)
    opt = get_optimiser(model.parameters(), params)
    for epoch in range(params["epochs"]):
        # Training
        model.train()
        epoch_losses = []
        for images, targets, image_ids in train_dataloader:
            # Send to the current device
            # TODO: is images really a list of images? Shouldn't this just be a 4d tensor?
            if device:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Get the losses
            assert len(targets) > 0, "Should at least have one bounding box per image (for now)"
            # TODO: How to feed empty values here?
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            opt.step()
            opt.zero_grad()
            loss_value = losses.item()
            epoch_losses.append(loss_value)
        # Now validation
        print(f"EPOCH {epoch} loss: {np.mean(epoch_losses)}")
        # TODO: Return valid loss and valid MAP here
        evaluate_model(model, valid_dataloader, epoch, params)
        # Model saving

    return model
