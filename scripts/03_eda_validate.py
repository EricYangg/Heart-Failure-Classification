# eda.py 
# author: Eric Yang
# date: 2025-12-03

import click
import os
import altair_ally as aly
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore", module="altair")
from src.validate_feature_correlation import validate_feature_correlation

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--data-to', type=str, help="Path to directory where split data will be written to")
def main(training_data, test_data, plot_to, data_to):
    '''This script performs initial exploratory data analysis on the processed training data. 
        Creates box plots, distributions for numeric and categorical features, and correlation between numeric features.
        Then saves each plot. Also performs feature correlation checks using Deepchecks.
        The processed training data is split into train and validation sets and saves them to the specified directory.'''
    
    # Load processed training data
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(test_data)

    # Create a copy of train_df for EDA purposes
    eda_df = train_df.copy()
    eda_df['HeartDisease'] = eda_df['HeartDisease'].astype(bool)

    # Create box plots for all numeric features
    numeric_cols = eda_df.select_dtypes(include='number').columns

    fig, axes = plt.subplots(
        nrows=len(numeric_cols)//2 + len(numeric_cols)%2, 
        ncols=2,
        figsize=(10, 6)
    )

    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=eda_df[col], ax=ax)
        ax.set_title(col, fontsize=10)

    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)
    fig.suptitle("Boxplots of All Numeric Features", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_to, 'boxplots_numeric_features.png'), dpi=120)
    plt.close(fig)

    # Create distribution plots for numeric features colored by HeartDisease
    aly.alt.data_transformers.enable('vegafusion')

    numeric_chart = (
        aly.dist(eda_df, color='HeartDisease')
            .properties(
                title="Distributions of Numeric Features by Heart Disease Status"
            )
            .configure_title(
                anchor="middle", 
                fontSize=16
            )
    )

    numeric_chart.save(os.path.join(plot_to, "numeric_dist_combined.png"))

    # Create distribution plots for categorical features colored by HeartDisease
    cat_chart = (
        aly.dist(
            eda_df.assign(
                HeartDisease=lambda eda_df: eda_df['HeartDisease'].astype(object)
            ), 
            dtype='object', 
            color='HeartDisease'
        )
            .properties(
                title="Distributions of Categorical Features by Heart Disease Status"
            )
            .configure_title(
                anchor="middle", 
                fontSize=16
            )
    )
    
    cat_chart.save(os.path.join(plot_to, "categorical_dist_combined.png"))

    # Create correlation heatmap for numeric features
    corr = eda_df.corr(numeric_only=True)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f", 
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={'shrink': 0.8}
    )

    plt.title("Correlation Heatmap of All Numeric Features", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_to, "correlation_heatmap_numeric_features.png"), dpi=120)
    plt.close()

    # Deepchecks: Feature correlation checks
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    feat_lab_threshold = 0.9
    feat_feat_threshold = 0.95
    feat_feat_n_pairs = 0
    # Validate feature correlations
    check_feat_lab_corr_result, check_feat_feat_corr_result = validate_feature_correlation(
        train_df=train_df,
        label="HeartDisease",
        categorical_features=categorical_features,
        feat_lab_threshold=feat_lab_threshold,
        feat_feat_threshold=feat_feat_threshold,
        feat_feat_n_pairs=feat_feat_n_pairs
    )

    print("\n--- Deepchecks Validation Summary ---")
    print("SUCCESS: Feature correlation checks passed.")

    # Create the split
    X_train = train_df.drop(columns=["HeartDisease"])
    X_test = test_df.drop(columns=["HeartDisease"])
    y_train = train_df["HeartDisease"]
    y_test = test_df["HeartDisease"]

    # Save the files in the path given
    X_train.to_csv(os.path.join(data_to, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_to, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_to, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_to, "y_test.csv"), index=False)

if __name__ == '__main__':
    main()