import copy
import logging
import pathlib

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
import mlflow

MAP_IND = 1 # The index in all of the stats generated by COCOEval of the MAP @ IoU = 50 - our target
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.absolute()
logger = logging.getLogger(__name__)

def get_dataloader(params):
    # First split into train and validation
    train_file_path = base_dir.joinpath(params["train_file"])
    all_train = pd.read_csv(train_file_path)
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
    # TODO: Kind of a hack to deal with the two machines (my laptop and Rhodos)
    if torch.cuda.device_count() == 2:
        device = torch.device('cuda:1')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda')
    else:
        assert False, "There should be either one or two devices available"
    return device


@utils.block_print
def get_MAP(gt_anns, det_anns, img_size):
    """Return the MAP at IoU=05"""
    coco_gt = get_dataset(gt_anns, img_size)
    coco_det = get_dataset(det_anns, img_size)
    coco_eval = COCOeval(coco_gt, coco_det, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[MAP_IND]

@utils.block_print
def get_dataset(img_anns, img_size):
    """
    Create a COCO api object from a list of bounding boxes.
    The input takes the following format:
    {img_id_1: {"boxes": [N, 4], "scores": [N]}, img_id_M: {...}}

    That is a dictonary containing the annotations for M images, each of which have N separate bounding boxes stored as
    a tensor. Detections also have scores attached.
    """
    num_img = len(img_anns)
    images = []
    # Add the image annotations
    for i in range(num_img):
        img_dict = {'id': i, 'height': img_size[0], 'width': img_size[1]}
        images.append(img_dict)
    dataset = {'images': images, 'categories': [], 'annotations': []}
    ann_id = 1
    # Add the box annotations for each image
    for image in img_anns:
        boxes = img_anns[image]["boxes"].tolist()
        if "scores" in img_anns[image]:
            scores = img_anns[image]["scores"].tolist()
            assert len(scores) == len(boxes)
        else:
            scores = []
        for idx, box in enumerate(boxes):
            if not box:
                continue
            assert len(box) == 4
            # They need to be in WH format
            box[2] -= box[0]
            box[3] -= box[1]
            area = box[2] * box[3]
            ann = {"image_id": image, "bbox": box, "category_id": 1, "area": area,
                   "iscrowd": 0, "id": ann_id}
            if scores:
                ann["score"] = scores[idx]
            ann_id += 1
            cat = {"id": 1}
            dataset["annotations"].append(ann)
            dataset["categories"].append(cat)
    if not images:
        return
    coco_ds = COCO()
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def train_one_epoch(model, dataloader, opt, params):
    device = get_device(params)
    dets = {}
    gts = {}
    losses = []
    # Just a placeholder
    img_size = (256, 256)
    for batch_images, batch_targets, _ in dataloader:
        batch_size = len(batch_images)
        img_size = batch_images[0].shape
        # Save copies of the targets before they're pushed to the gpu
        batch_gts = {gt["image_id"].item(): copy.deepcopy(gt) for gt in batch_targets}
        gts.update(batch_gts)
        images = list(img.to(device) for img in batch_images)
        batch_targets = ([{k: v.to(device) for k, v in t.items()} for t in batch_targets])
        batch_losses, batch_predictions = model(images, batch_targets)
        batch_loss = sum(batch_losses.values())
        # Check if we're training here - if so we should update the gradients
        if opt:
            batch_loss.backward()
            opt.step()
            opt.zero_grad()
        # Push the results back to cpu
        batch_outputs = [{k: v.to("cpu") for k, v in t.items()} for t in batch_predictions]
        # Map IDs to outputs
        res = {target["image_id"].item(): output for target, output in zip(batch_targets, batch_outputs)}
        dets.update(res)
        # Push the losses back to the cpu
        losses.append(batch_loss.item() * batch_size)
    # Now we average over the total number of values to get the average loss per sample
    av_loss = np.sum(losses) / len(dataloader.dataset)
    # Sometimes its nice to comment out training to test the pipeline. This is for that case. Otherwise gts should always
    # be populated
    if gts:
        # An now we calculate the MAP
        MAP = get_MAP(gts, dets, img_size)
    else:
        MAP = 0
    return av_loss, MAP


def fit(params):
    """
    Generic model fitting function
    """
    # Hack to get the RCNN to always return both the predictions and losses
    torch.jit.is_scripting = lambda: True
    device = get_device(params)
    train_dataloader, valid_dataloader = get_dataloader(params)
    model_path = base_dir.joinpath(params["base_model_path"])
    model = torch.load(model_path).to(device)
    opt = get_optimiser(model.parameters(), params)
    # Model saving vars
    best_MAP = 0
    best_model = None
    best_model_epoch = 0
    # Early stopping vars
    worse_model_count = 0
    best_valid_loss = 0
    for epoch in range(params["epochs"]):
        train_av_loss, train_MAP = train_one_epoch(model, train_dataloader, opt, params)
        # The validation needs to stay in training mode to get the validation LOSS - which will be used for early stopping
        with torch.no_grad():
            valid_av_loss, valid_MAP = train_one_epoch(model, valid_dataloader, False, params)
        # Logging
        mlflow.log_metric("train-loss", train_av_loss, epoch)
        mlflow.log_metric("valid-loss", valid_av_loss, epoch)
        mlflow.log_metric("train-MAP", train_MAP, epoch)
        mlflow.log_metric("valid-MAP", valid_MAP, epoch)
        logging.log(logging.INFO, f"EPOCH {epoch} valid loss: {valid_av_loss:.8f} | valid MAP: {valid_MAP:.3f} | train loss: "
              f"{train_av_loss:.8f} | train MAP: {train_MAP:.3f}")
        # keep a copy of the best model
        if valid_MAP > best_MAP:
            #TODO: This would be much better with the state dict
            # e.g. best_model_state_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
            logger.log(logging.INFO, f"New best model in epoch {epoch} with valid MAP score of {valid_MAP}")
            model = model.to("cpu")
            best_model = copy.deepcopy(model)
            model = model.to(device)
            best_model_epoch = epoch
        # Now implement early stopping
        if valid_av_loss > best_valid_loss:
            worse_model_count += 1
            if worse_model_count >= params["patience"]:
                logger.log(logging.INFO, f"Stopped training early at epoch {epoch}")
                break
    mlflow.log_metric("best-epoch", best_model_epoch)
    return best_model
