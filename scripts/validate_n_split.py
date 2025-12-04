# validate_n_split.py
# author: Mara Sanchez
# date: 2024-12-02

import click
import warnings
import os
import numpy as np
import pandas as pd
#from sklearn.exceptions import UndefinedMetricWarning
import json
import logging
from sklearn import set_config
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")
warnings.filterwarnings("ignore", category=UserWarning, module="deepchecks")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")
import pandera as pa

@click.command()
@click.option('--logs-to', type=str, help="Path to write validation logs")
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where validated data will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(logs_to, raw_data, data_to, seed):
    '''This script validates that our data is within the ranges established for each feature,
    and then splits the filtered out data into train and test sets. Additionally, it saves
    the train and test data in the given path for future EDA and preprocessing.'''
    
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
    
    schema = pa.DataFrameSchema(
        {
            "Age": pa.Column(int, pa.Check.between(20, 90)),
            "Sex": pa.Column(str, pa.Check.isin(["F", "M"])),
            "ChestPainType": pa.Column(str, pa.Check.isin(["TA", "ATA", "NAP", "ASY"])),
            "RestingBP": pa.Column(int, pa.Check.between(80, 230)),
            "Cholesterol": pa.Column(int, pa.Check.between(50, 400)),
            "FastingBS": pa.Column(int, pa.Check.isin([0, 1])),
            "RestingECG": pa.Column(str, pa.Check.isin(["Normal", "ST", "LVH"])),
            "MaxHR": pa.Column(int, pa.Check.between(60, 202)),
            "ExerciseAngina": pa.Column(str, pa.Check.isin(["Y", "N"])),
            "Oldpeak": pa.Column(float, pa.Check.between(-4.0, 6.2)),
            "ST_Slope": pa.Column(str, pa.Check.isin(["Up", "Flat", "Down"])),
            "HeartDisease": pa.Column(int, pa.Check.isin([0, 1])),
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
        ],
        drop_invalid_rows=False,
    )
    
    # Make a copy so original is untouched
    data = df.copy()
    error_cases = pd.DataFrame()
    
    try:
        # Validate with lazy errors (UBC: lazy=True)
        validated_data = schema.validate(data, lazy=True)

    except pa.errors.SchemaErrors as e:
        # Find invalid rows
        error_cases = e.failure_cases

        # Log JSON message
        error_message = json.dumps(e.message, indent=2)
        logging.error("\n" + error_message)

    # Filter out invalid rows
    if not error_cases.empty:
        invalid_indices = error_cases["index"].dropna().unique()
        validated_data = (
            data.drop(index=invalid_indices)
                .reset_index(drop=True)
                .drop_duplicates()
                .dropna(how="all")
        )
        print(f"Dropped {len(invalid_indices)} invalid rows.")
    else:
        validated_data = data
        print("No validation errors found.")
        
        
    # Create the split    
    train_df, test_df = train_test_split(validated_data, test_size=0.3)
  
    # Save the files in the path given
    train_df.to_csv(os.path.join(data_to, "heart_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "heart_test.csv"), index=False)

if __name__ == '__main__':
    main()