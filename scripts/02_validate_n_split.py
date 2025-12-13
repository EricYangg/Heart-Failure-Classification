# validate_n_split.py
# author: Mara Sanchez
# date: 2024-12-02

import click
import warnings
import os
import numpy as np
import pandas as pd
import json
import logging
from sklearn import set_config
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")
warnings.filterwarnings("ignore", category=UserWarning, module="deepchecks")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")
from src.validate_data_schema import validate_data_schema


@click.command()
@click.option("--logs-to", type=str, help="Path to write validation logs")
@click.option("--raw-data", type=str, help="Path to raw data")
@click.option(
    "--data-to",
    type=str,
    help="Path to directory where validated data will be written to",
)
@click.option("--seed", type=int, help="Random seed", default=123)
def main(logs_to, raw_data, data_to, seed):
    """This script validates that our data is within the ranges established for each feature,
    and then splits the filtered out data into train and test sets. Additionally, it saves
    the train and test data in the given path for future EDA and preprocessing."""

    np.random.seed(seed)
    set_config(transform_output="pandas")

    # DATA VALIDATION SECTION

    df = pd.read_csv(raw_data)

    # Configure logging
    logging.basicConfig(
        filename=logs_to + "/validation_errors.log",
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    # Data validation
    validated_data = validate_data_schema(df)

    # Create the split
    train_df, test_df = train_test_split(validated_data, test_size=0.3)

    # Save the files in the path given
    train_df.to_csv(os.path.join(data_to, "heart_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "heart_test.csv"), index=False)


if __name__ == "__main__":
    main()
