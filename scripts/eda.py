# eda.py 
# author: Eric Yang
# date: 2025-12-03

import click
import os
import altair_ally as aly
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="altair")

@click.command()
@click.option('--data-from', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")

def main(data_from, plot_to):
    '''This script performs initial exploratory data analysis on the processed training data. 
        Creates box plots, distributions for numeric and categorical features, and correlation between numeric features.
        Then saves each plot.'''
    
    # Load processed training data
    train_df = pd.read_csv(data_from)

    train_df['HeartDisease'] = train_df['HeartDisease'].astype('bool')

    # Create box plots for all numeric features
    numeric_cols = train_df.select_dtypes(include='number').columns

    fig, axes = plt.subplots(
        nrows=len(numeric_cols)//2 + len(numeric_cols)%2, 
        ncols=2,
        figsize=(10, 6)
    )

    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=train_df[col], ax=ax)
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
        aly.dist(train_df, color='HeartDisease')
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
            train_df.assign(
                HeartDisease=lambda train_df: train_df['HeartDisease'].astype(object)
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
    corr = train_df.corr(numeric_only=True)

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

if __name__ == '__main__':
    main()