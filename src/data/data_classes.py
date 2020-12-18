import math
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import logging
import pandas as pd
import src.util.util as utils
import xml.etree.ElementTree as ET
import albumentations as A

COLOUR_MAX = 255.0
HEIGHT_MAX = 255.0
class SeedlingDataset(Dataset):
    def __init__(self, datafiles):
        super().__init__()
        if type(datafiles) is not pd.DataFrame:
            # This should be a csv file to read in results
            datafiles = pd.read_csv(datafiles)
        # datafiles structure:
        self.datafiles = datafiles
        self.img_boxes = self._read_boxes()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            tuple_list = [self[i] for i in range(start, stop, step)]
            unzipped = list(zip(* tuple_list))
            return unzipped
        elif isinstance(key, int):
            return self.get_value(key)
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))

    def get_value(self, index: int):
        id = self.datafiles["id"].iloc[index]
        records = self.img_boxes[self.img_boxes["id"] == id]
        # Read in the colour image
        img_path = self.datafiles.loc[self.datafiles["id"] == id, "im_filename", "height_filename"]
        assert len(img_path) == 1, "There should only be one match"
        img_path = img_path.iloc[0, 0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert image is not None, "The image should exist"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Scale to 0-1 space
        image = image / COLOUR_MAX
        image = torch.Tensor(image)
        # Rearrange image - pytorch standard models expect channel dimension first
        image = image.permute((2, 0, 1))
        # Read in the height map
        height_path = img_path.iloc[0, 1]
        height_image = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        height_image = height_image / HEIGHT_MAX
        height_image = torch.Tensor(height_image)
        # Concatenate the channels together
        total_image = torch.cat((image, height_image.unsqueeze(2)), 2)
        # Now the labels
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        # TODO: Put this into coco format straight away? Might be some overhead with uploading unncessary stuff to the GPU...
        target = {}
        target["image_id"] = torch.tensor(index)
        target["boxes"] = torch.Tensor(records[["xmin", "ymin", "xmax", "ymax"]].values)
        areas = (records["xmax"] - records["xmin"]) * (records["ymax"] - records["ymin"])
        target["area"] = torch.Tensor(areas.values)
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        return total_image, height_image, target, id

    def _read_boxes(self):
        # First get a list of boxes for each file
        file_boxes = [self._get_boxes(label_file) for label_file in self.datafiles["label_filename"]]
        df = pd.concat(file_boxes)
        return df

    # site_464_201710_030m_ortho_als11_3channels_cut - 256 - 0.TIF
    def _get_boxes(self, filename):
        # The xml parser reads in a tree structure
        this_id = utils.extract_base_id(filename)
        id_entries = []
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        truncateds = []
        root = ET.parse(filename).getroot()
        box_elems = root.findall("object")
        for ind, box_elem in enumerate(box_elems):
            id_entries.append(this_id)
            truncateds.append(box_elem.find("truncated").text == "1")
            box = box_elem.find("bndbox")
            xmins.append(int(box.find("xmin").text))
            xmaxs.append(int(box.find("xmax").text))
            ymins.append(int(box.find("ymin").text))
            ymaxs.append(int(box.find("ymax").text))
        # Create the final dataframe
        df = pd.DataFrame(zip(id_entries, xmins, xmaxs, ymins, ymaxs, truncateds),
                      columns=["id", "xmin", "xmax", "ymin", "ymax", "truncated"])
        return df

    def __len__(self) -> int:
        return len(self.datafiles)