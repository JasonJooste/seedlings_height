import logging

import os

import mlflow
import torch
from torch.utils.data import DataLoader

import src.util.util as utils
import sys
import yaml
import itertools
from random import shuffle
import pandas as pd
from src.data.data_classes import SeedlingDataset
from src.models.make_base_models import make_vanilla_model
from src.models.train_model import fit
from src.models.train_model import train_one_epoch
from datetime import datetime
import copy
import pathlib

#TODO: Not sure if it is good practice to leave this here to be executed at import.
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.absolute()
logger = logging.getLogger(__name__)

def get_existing_config(this_config, existing_configs, config_filenames):
    assert len(existing_configs) == len(config_filenames)
    # Each config file has one extra entry: Model_path. This needs to be removed before comparison
    without_model_path = copy.deepcopy(existing_configs)
    for dict in without_model_path:
        del dict["trained_model_path"]
    # Compare the dictionaries to find a match
    matches = [this_config == config for config in without_model_path]
    # If there's are matches then return all filenames that match it
    match_dict = None
    match_filenames = []
    for ind, match in enumerate(matches):
        if match:
            new_match_dict = existing_configs[ind]
            if not new_match_dict["trained_model_path"]:
                logger.log(logging.WARN, f"Config file doesn't contain the trained path: {config_filenames[ind]}")
            del new_match_dict["trained_model_path"]
            # Just double checking that the previous matching worked
            assert match_dict is None or match_dict == new_match_dict
            match_dict = new_match_dict
            match_filenames.append(config_filenames[ind])
    return match_dict, match_filenames


def gen_model_filename(config, iter):
    date_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"{date_str}_{config['task_name']}_{str(iter)}"
    path = base_dir.joinpath(config['output_dir']) / filename
    return path


def execute_models(params, use_cache=True):
    # Read in existing models
    trained_folder = base_dir / "models" / "trained"
    existing_config_fns = utils.get_filenames(trained_folder, ".yaml", keep_ext=True, full_path=True)
    existing_configs = []
    for filename in existing_config_fns:
        with open(filename, 'r') as file:
            existing_configs.append(yaml.load(file))
    # Get list of individual tasks
    search_list = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]
    shuffle(search_list)
    # Run training
    for ind, this_config in enumerate(search_list):
        # Check for existing model files
        (existing_config, config_fn) = get_existing_config(this_config, existing_configs, existing_config_fns)
        if use_cache and existing_config:
            # The model file already exists
            logger.log(logging.INFO, f"This model has already been trained and is stored in {config_fn}/.py - training skipped.")
            continue
        # Set the seed
        utils.set_seed(this_config["seed"])
        # The model doesn't exist yet - train it
        logger.log(logging.INFO, "=======================================================================================================")
        logger.log(logging.INFO, f"Training new config: {this_config}")
        # Set up MLFlow tracking of this config
        mlflow.set_experiment(experiment_name=this_config["task_name"])
        mlflow.start_run()
        for param, val in this_config.items():
            mlflow.log_param(param, val)
        # Fit model
        model = fit(this_config)
        mlflow.end_run()
        # Save the model file
        filename = gen_model_filename(this_config, ind)
        # print(filename)
        filename.with_suffix(".pt")
        torch.save(model, filename.with_suffix(".pt"))
        # Save the config file
        this_config["trained_model_path"] = str(filename.with_suffix(".pt"))
        file = open(filename.with_suffix(".yaml"), 'w')
        yaml.dump(this_config, file)
        mlflow.end_run()


if __name__ == "__main__":
    # Set up logging and MLFlow tracking
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    config_filename = sys.argv[1]
    config_file = open(config_filename, 'r')
    params = yaml.load(config_file)
    use_cache = False
    for pretrained in [True, False]:
        for i in range(6):
            template_dir = base_dir / "models" / "templates"
            make_vanilla_model(template_dir, pretrained=pretrained, trainable_backbone_layers=i)
    execute_models(params, use_cache)


