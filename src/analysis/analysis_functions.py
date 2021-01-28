import math
import pathlib
import logging
from itertools import cycle

import mlflow
import pandas as pd
import yaml

module_path = pathlib.Path(__file__).parent
base_dir = module_path.parent.parent.absolute()
logger = logging.getLogger(__name__)
# Not a wonderful solution. This makes sure that the server is always set
mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")

def get_dataframe(directory, taskname=None):
    """
    Read in all of the config files of trained models and organise them as a pandas dataframe
    :param taskname: the name of the task (experiment) that we're limiting ourselves to
    :param directory: The directory of the trained models
    :return:
    """
    # Convert to pathlib path
    directory = pathlib.Path(directory)
    if not directory.exists():
        raise Exception(f"Directory doesn't exist: {str(directory.absolute())}")
    # Get all of the yaml files
    config_files = directory.glob("*.yaml")
    configs = []
    for config_file in config_files:
        with config_file.open(mode='r') as f:
            config_dict = yaml.safe_load(f)
            # Check that the name matches the desired task name
            if not taskname or config_dict["task_name"] == taskname:
                configs.append(config_dict)
    if not configs:
        return None
    # Convert list of individual config dicts into a pandas df
    df = pd.DataFrame(configs)
    return df


def get_runs_data(df, verbose=True):
    """
    Take a dataframe of our run logs, access any mlflow data that was stored for these runs and return a dataframe
    augmented with the metrics
    :param df:
    :return: df
    """
    run_ids = df["run_id"]
    rows = []
    if verbose:
        #TODO: Using print statemetns here because logging doesn't work nicely with notebooks. This could be fixed
        print("Reading run info from server")
    for i, ind in enumerate(run_ids.index):
        if verbose and not i % 10:
            print(f"Run {i} of {len(run_ids)} ({i/len(run_ids) * 100:.2f}%)")
        run_id = run_ids[ind]
        if type(run_id) is str:
            metric_data = get_run_metric_data(run_id)
            # Associate the metric data with the correct row
            metric_data = metric_data.set_index([[ind]])
            rows.append(metric_data)
        else:
            assert math.isnan(run_id), "Should either be a string or NaN"
    if verbose:
        print("Finished reading run info from server")
    metrics = pd.concat(rows)
    # We don't want to modify the existing dataframe
    new_df = df.copy(deep=True)
    # Change the existing column index to be heirarchical
    new_df.columns = pd.MultiIndex.from_product([["params"], new_df.columns])
    new_df = pd.concat([new_df, metrics], axis=1)
    return new_df


def get_run_metric_data(run_id: str) -> pd.DataFrame:
    """
    Get all data from the metrics of a run in a single vector.

    Each metric gets assigned a timepoint
    :param run_id: The run id
    :return: A single row pandas dataframe with hierarchical indexing over metrics
    """
    client = mlflow.tracking.MlflowClient()
    run = mlflow.get_run(run_id)
    # Get the list of metrics
    metrics = run.data.metrics.keys()
    values = []
    indexes = []
    for metric in metrics:
        metric_hist = client.get_metric_history(run_id, metric)
        metric_names = [f"{metric}_{ind}" for ind in range(len(metric_hist))]
        index_tuples = zip(cycle([metric]), metric_names)
        values.extend([m.value for m in metric_hist])
        indexes.extend(index_tuples)
    index = pd.MultiIndex.from_tuples(indexes, names=["metric", "metric_dp"])
    # Needs a list around the values to make this a row instead of a column
    run_df = pd.DataFrame([values], columns=index)
    return run_df