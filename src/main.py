import logging

import os
import torch
import src.util.util as utils
import sys
import yaml
import itertools
from random import shuffle

from src.models.make_base_models import make_vanilla_model
from src.models.train_model import fit
from datetime import datetime
import copy
import pathlib

#TODO: Not sure if it is good practice to leave this here to be executed at import.
module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.absolute()


def get_existing_config(this_config, existing_configs, config_filenames):
    assert len(existing_configs) == len(config_filenames)
    # Each config file has one extra entry: Model_path. This needs to be removed before comparison
    without_model_path = copy.deepcopy(existing_configs)
    for dict in without_model_path:
        del dict["trained_model_path"]
    # Compare the dictionaries to find a match
    matches = [this_config == config for config in without_model_path]
    match_dict = None
    match_filename = None
    # If there's a match then return the config dict and its filename
    if any(matches):
        match_ind = [i for i,x in enumerate(matches) if x]
        assert len(match_ind) == 1, "There should be only one matching config"
        match_dict = existing_configs[match_ind[0]]
        match_filename = config_filenames[match_ind[0]]
    return match_dict, match_filename


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
            logging.log(logging.INFO, f"This model has already been trained and is stored in {config_fn}/.py - training skipped.")
            continue
        # Set the seed
        utils.set_seed(this_config["seed"])
        # The model doesn't exist yet - train it
        print("=======================================================================================================")
        print(f"Training new config: {this_config}")
        model = fit(this_config)
        # Save the model file
        filename = gen_model_filename(this_config, ind)
        # print(filename)
        filename.with_suffix(".pt")
        torch.save(model, filename.with_suffix(".pt"))
        # Save the config file
        this_config["trained_model_path"] = str(filename.with_suffix(".pt"))
        file = open(filename.with_suffix(".yaml"), 'w')
        yaml.dump(this_config, file)

    # Run validation
    # Here we read in model files and perform evaluation on them


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    config_filename = sys.argv[1]
    config_file = open(config_filename, 'r')
    params = yaml.load(config_file)
    use_cache = False
    for i in range(6):
        template_dir = base_dir / "models" / "templates"
        make_vanilla_model(template_dir, trainable_backbone_layers=i)
    execute_models(params, use_cache)


