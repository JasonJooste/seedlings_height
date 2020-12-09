from os import path
from src.data.data_classes import SeedlingDataset
import src.util.util as utils
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
"""
This is a script to generate a csv file with image and label file names for a train/test split"""
TEST_SIZE = 0.2
DIR_INPUT = path.abspath('../../data')

im_dir = f"{DIR_INPUT}/processed"
label_dir = f"{DIR_INPUT}/raw/labels"
# Find all image and label files in directories
im_ids = pd.DataFrame(utils.get_filenames(im_dir, ".tif"), columns=["im_filename"])
label_ids = pd.DataFrame(utils.get_filenames(label_dir, ".xml"), columns=["label_filename"])
# Transform the names so they match up with each other
height_ids = im_ids[im_ids["im_filename"].str.contains("CHM10cm")]
height_ids = height_ids.rename({"im_filename": "height_filename"}, axis=1)
colour_ids = im_ids[im_ids["im_filename"].str.contains("030m")]
height_ids["id"] = height_ids["height_filename"].map(utils.extract_base_id)
colour_ids["id"] = colour_ids["im_filename"].map(utils.extract_base_id)
label_ids["id"] = label_ids["label_filename"].map(utils.extract_base_id)
# Get the files that are shared across all
#TODO: This is strange. Height and colour don't have the same number of images and the smaller one (height) isn't even a subset.
shared_ids = height_ids.merge(colour_ids, on="id", how="inner").merge(label_ids, on="id", how="inner")
union_ids = height_ids.merge(colour_ids, on="id", how="outer").merge(label_ids, on="id", how="outer")
# Warn for ids that are not shared
missing_labels = union_ids["label_filename"].isna().to_numpy().nonzero()[0]
for missing_label_ind in missing_labels:
    missing_sample = union_ids["id"][missing_label_ind]
    logging.warning(f"No label file present for sample {missing_sample}")
missing_height = union_ids["height_filename"].isna().to_numpy().nonzero()[0]
for missing_height_ind in missing_labels:
    missing_sample = union_ids["id"][missing_height_ind]
    logging.warning(f"No height file present for sample {missing_sample}")
missing_ims = union_ids["im_filename"].isna().to_numpy().nonzero()[0]
for missing_im_ind in missing_labels:
    missing_sample = union_ids["id"][missing_im_ind]
    logging.warning(f"No image file present for sample {missing_sample}")
# Add full paths and extensions to filenames
shared_ids["label_filename"] = f"{label_dir}/" + shared_ids['label_filename'] + ".xml"
shared_ids["im_filename"] = f"{im_dir}/" + shared_ids['im_filename'] + ".tif"
shared_ids["height_filename"] = f"{im_dir}/" + shared_ids['height_filename'] + ".tif"
# Split into train test
train, test = train_test_split(shared_ids, test_size=TEST_SIZE)
train.to_csv(f"{DIR_INPUT}/site_464_201710_30_train.csv")
test.to_csv(f"{DIR_INPUT}/site_464_201710_30_test.csv")