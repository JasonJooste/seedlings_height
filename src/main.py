import copy
import itertools
import logging
import pathlib
import sys
from datetime import datetime
from random import shuffle

import mlflow
import yaml

import src.util.util as utils
from src.models.make_base_models import *
from src.models.train_model import fit_and_test

module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.absolute()
logger = logging.getLogger(__name__)


def get_existing_config(this_config, existing_configs, config_filenames):
    """Return a list of files whose configs match the given config and the config. If no files match return None"""
    assert len(existing_configs) == len(config_filenames)
    # Each config file has a number of extra entries that need to be removed first
    without_model_path = copy.deepcopy(existing_configs)
    for conf_dict in without_model_path:
        conf_dict.pop("trained_model_path", None)
        conf_dict.pop("best_model_epoch", None)
        conf_dict.pop("test_MAP", None)
        conf_dict.pop("run_id", None)
        conf_dict.pop("experiment_id", None)
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
    """Generate a filename for the model that includes the configuration and time"""
    date_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"{date_str}_{config['task_name']}_{str(iter)}"
    path = base_dir.joinpath(config['output_dir']) / filename
    return path


def execute_models(params, use_cache=True):
    """Train, test and save all hyperparameter configurations contained in a configuration dict
    """
    # Read in existing models
    trained_folder = base_dir / "models" / "trained"
    existing_config_fns = utils.get_filenames(trained_folder, ".yaml", keep_ext=True, full_path=True)
    existing_configs = []
    for filename in existing_config_fns:
        with open(filename, 'r') as file:
            existing_configs.append(yaml.safe_load(file))
    # Get list of individual tasks by taking the cartesian product of all iterable entries
    search_list = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]
    shuffle(search_list)
    # Run training
    for ind, this_config in enumerate(search_list):
        filename = gen_model_filename(this_config, ind)
        # Set up logging for this run
        handler = logging.FileHandler(filename.with_suffix(".log"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        # Check for existing model files
        existing_config, config_fn = get_existing_config(this_config, existing_configs, existing_config_fns)
        if use_cache and existing_config:
            # The model file already exists
            logger.log(logging.INFO,
                       f"This model has already been trained and is stored in {config_fn}/.py - training skipped.")
            continue
        # Set the seed
        utils.set_seed(this_config["seed"])
        # The model doesn't exist yet - train it
        logger.log(logging.INFO,
                   "="*80)
        logger.log(logging.INFO, f"Model {ind} from {len(search_list)} ({ind / len(search_list) * 100 :.0f}%)")
        logger.log(logging.INFO, f"Training new config: {this_config}")
        # Set up MLFlow tracking of this config
        mlflow.set_experiment(experiment_name=this_config["task_name"])
        this_run = mlflow.start_run()
        for param, val in this_config.items():
            mlflow.log_param(param, val)
        # Fit model
        model, best_model_epoch, test_MAP = fit_and_test(this_config)
        # Save the model file
        if not this_config["develop"]:
            torch.save(model, filename.with_suffix(".pt"))
        # Save the config file
        this_config["trained_model_path"] = str(filename.with_suffix(".pt"))
        this_config["best_model_epoch"] = best_model_epoch
        this_config["test_MAP"] = test_MAP
        this_config["run_id"] = this_run.info.run_id
        this_config["experiment_id"] = this_run.info.experiment_id
        file = open(filename.with_suffix(".yaml"), 'w')
        yaml.dump(this_config, file)
        mlflow.end_run()
        # Stop the log to file
        root_logger.handlers.pop()


def run(config_filename):
    """ Run the seedlings analysis for a given config file"""
    config_file = open(config_filename, 'r')
    params = yaml.load(config_file)
    if not "develop" in params:
        params["develop"] = [True]
        # Too much logging from packages still.
        # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Set up logging and MLFlow tracking
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    use_cache = True
    rebuild = True
    if rebuild:
        template_dir = base_dir / "models" / "templates"
        build_models(template_dir)
    execute_models(params, use_cache)


def build_models(template_dir):
    """Build the modified Faster-RCNN models with new parameters for different input types.

    These models will later be loaded and trained with a given set of hyperparameters"""
    # # Build the models
    # ### Vanilla models
    make_vanilla_model(template_dir, pretrained=True, trainable_backbone_layers=5)
    make_vanilla_model(template_dir, pretrained=False, trainable_backbone_layers=5, pretrained_backbone=True)
    make_vanilla_model(template_dir, pretrained=False, trainable_backbone_layers=5)
    # # Vanilla model with single feature map from backbone (not FPN)
    make_normal_backbone_model(template_dir, pretrained=True, trainable_backbone_layers=5)
    # ### Final layer model
    make_final_layer_model(template_dir, pretrained=True, trainable_backbone_layers=5)
    make_final_layer_model(template_dir, pretrained=False, trainable_backbone_layers=5, pretrained_backbone=True)
    make_final_layer_model(template_dir, pretrained=False, trainable_backbone_layers=5)
    # ### First layer model
    make_first_layer_model(template_dir, pretrained=True, trainable_backbone_layers=5)
    make_first_layer_model(template_dir, pretrained=False, trainable_backbone_layers=5, pretrained_backbone=True)
    make_first_layer_model(template_dir, pretrained=False, trainable_backbone_layers=5)
    # ### Pre roi models
    # # [1,2,3,4,5] - 256
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=256)
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=192)
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=128)
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=64)
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=32)
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=16)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=True, pooling_layer=True, out_channels=256)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=False, pooling_layer=True, out_channels=256)
    # [1,2,3,4,5] - 64
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=True, out_channels=64)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=True, pooling_layer=True, out_channels=64)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=False, pooling_layer=True, out_channels=64)
    # [1,2,3,4] - 256
    make_pre_rpn_pretrained_model(template_dir, [1, 2, 3, 4], pooling_layer=False, out_channels=256)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=True, pooling_layer=False, out_channels=256)
    make_pre_roi_model(template_dir, [1, 2, 3, 4], pretrained=False, pooling_layer=False, out_channels=256)
    # [4,5] - 256
    make_pre_rpn_pretrained_model(template_dir, [4], pooling_layer=True, out_channels=256)
    make_pre_roi_model(template_dir, [4], pretrained=True, pooling_layer=True, out_channels=256)
    make_pre_roi_model(template_dir, [4], pretrained=False, pooling_layer=True, out_channels=256)
    # [4] - 256
    make_pre_rpn_pretrained_model(template_dir, [4], pooling_layer=False, out_channels=256)
    make_pre_roi_model(template_dir, [4], pretrained=True, pooling_layer=False, out_channels=256)
    make_pre_roi_model(template_dir, [4], pretrained=False, pooling_layer=False, out_channels=256)
    # [4] - 64
    make_pre_rpn_pretrained_model(template_dir, [4], pooling_layer=False, out_channels=64)
    make_pre_roi_model(template_dir, [4], pretrained=True, pooling_layer=False, out_channels=64)
    make_pre_roi_model(template_dir, [4], pretrained=False, pooling_layer=False, out_channels=64)


if __name__ == "__main__":
    config_filenames = sys.argv[1:]
    for config_filname in config_filenames:
        print("*" * 80)
        print(f"Executing config file {config_filname}")
        run(config_filname)
