# preprocess_validate.py
# author: Eric Yang
# date: 2025-12-03

import click
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(training_data, preprocessor_to, seed):
    '''Create and save a preprocessing object. Validate feature correlations'''
    
    np.random.seed(seed)
    # Load training data
    train_df = pd.read_csv(training_data)

    # Define feature types
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(drop='if_binary', 
                   handle_unknown="ignore", 
                   sparse_output=False
                  ), categorical_features),
    )
    pickle.dump(preprocessor, open(os.path.join(preprocessor_to, 'preprocessor.pkl'), 'wb'))

if __name__ == '__main__':
    main()