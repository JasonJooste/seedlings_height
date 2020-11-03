import logging
import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from os import walk
from os import path


#TODO: Don't forget the second dataset!
class SeedlingDataset(Dataset):
    def __init__(self, im_dir, label_dir):
        super().__init__()
        self.im_dir = im_dir
        self.label_dir = label_dir
        # Get the list of files
        im_ids = self._get_ids(im_dir)
        #TODO: This is kind of a hack. There is probably a better way to organise all of the files into different groups without file names
        # Maybe some kind of csv file with all the data??
        label_ids = self._get_ids(label_dir)
        for ind in range(len(label_ids)):
            label_ids[ind] = label_ids[ind].replace("channels_cut-", "channels_buffer_removed_cut-")
        # If they don't agree we can throw some away (probably worth logging as well)
        shared_ids = list(set(im_ids) & set(label_ids))
        if len(shared_ids) != len(im_ids):
            ims_missing_label = list(set(im_ids) - set(label_ids))
            labels_missing_ind = list(set(label_ids) - set(im_ids))
            for im in ims_missing_label:
                logging.warning(f"Image {im}.tif is missing its corresponding label")
            for lab in labels_missing_ind:
                logging.warning(f"Label {lab}.xml is missing its corresponding image")
        self.ids = shared_ids
        # Read in a dictionary of bounding boxes
        self.img_boxes = self._get_boxes(shared_ids)

    def __getitem__(self, index: int):
        id = self.ids[index]
        records = self.img_boxes[self.img_boxes["id"] == id]
        # TODO: Not exactly sure what the next three lines actually do
        image = cv2.imread(f"{self.im_dir}/{id}.tif", cv2.IMREAD_COLOR)
        assert(image, "The image should exist")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #TODO: need to figure out a way to efficiently do image scaling
        # image /= 255.0
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        # TODO: Don't know what a crowd is
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        target = {}
        target["boxes"] = records[["xmin", "ymin", "xmax", "ymax"]].values
        target["area"] = (records["xmax"] - records["xmin"]) * (records["ymax"] - records["ymin"])
        target["labels"] = labels
        # TODO: What is this for?
        target["image_id"] = torch.tensor([index])
        target["iscrowd"] = iscrowd
        # TODO: Transformations here maybe?
        return image, target, id

    def _get_boxes(self, ids):
        # Get a dataframe with all bounding boxes
        # It's more efficient to convert into a dataframe at the end
        id_entries = []
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        truncateds = []
        for id in ids:
            filename = f"{self.label_dir}/{id}.xml"
            # TODO: Again a hack. I need to figure out a nicer system to manage file paths.
            #  Maybe just a csv file of logs and corresponding files?
            filename = filename.replace("_buffer_removed", "")
            # The xml parser reads in a tree structure
            root = ET.parse(filename).getroot()
            box_elems = root.findall("object")
            for ind, box_elem in enumerate(box_elems):
                id_entries.append(id)
                truncateds.append(box_elem.find("truncated").text == "1")
                box = box_elem.find("bndbox")
                xmins.append(int(box.find("xmin").text))
                xmaxs.append(int(box.find("xmax").text))
                ymins.append(int(box.find("ymin").text))
                ymaxs.append(int(box.find("ymax").text))
        # Create the final dataframe
        df = pd.DataFrame(zip(ids, xmins, xmaxs, ymins, ymaxs, truncateds), columns=["id", "xmin", "xmax", "ymin", "ymax", "truncated"])
        return df

    def _get_ids(self, directory : str):
        """
        Get a list of file ids within the directory
        :param directory: The directory to search in
        :return: A list of filenames stripped of extensions
        """
        # First check if the directory exists
        if not path.exists(directory):
            raise NotADirectoryError
        # A quick way to get all files in directory
        (_, _, img_paths) = next(walk(directory))
        # Strip the file extensions
        ids = [path.splitext(x)[0] for x in img_paths]
        return ids


DIR_INPUT = path.expanduser('~/Documents/Uni/Practical/seedlings/data')
im_dir = f"{DIR_INPUT}/tiled"
label_dir = f"{DIR_INPUT}/labels"
data = SeedlingDataset(im_dir,label_dir)
for i in range(100):
    (a,b,c) = data[i]
    print(b)




# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)