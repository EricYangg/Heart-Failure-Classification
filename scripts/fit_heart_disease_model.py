# fit_heart_disease_model.py
# author: Omar Ramos
# date: 2025-12-02

import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import warnings
import click
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Start of code block copied from another author ---
# Title: Function to consolidate cross validation scores into a pandas series.
# Author: Varada Kolhatkar & Michael Gelbart
# Source: https://pages.github.ubc.ca/mds-2025-26/DSCI_571_sup-learn-1_students/README.html 
# Taken from: DSCI-571: Laboratory 2
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], 
                                                std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)
# --- End of code block copied from another author ---

def build_pipe(preprocessor, model):
    """
    Create and return a pipeline composed of a preprocessor and a model estimator.

    Parameters
    ----------
    preprocessor : A transformer object
        A scikit-learn column transformer object that handles data preprocessing
        (e.g., scaling, encoding).

    model : Any type of estimator object
        A scikit-learn estimator such as a classifier or regressor.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline combining the preprocessor and the model.
    """
    return make_pipeline(preprocessor, model)

@click.command()
@click.option('--x-train-data', type=str, help="Relative path to read X training data")
@click.option('--y-train-data', type=str, help="Relative path to read y training data")
@click.option('--x-test-data', type=str, help="Relative path to read X test data")
@click.option('--y-test-data', type=str, help="Relative path to read y test data")
@click.option('--preprocessor', type=str, help="Relative path to read preprocessor object")
@click.option('--pipeline-to', type=str, help="Relative path to write model to")
@click.option('--results-to', type=str, help="Relative path to write results to")
@click.option('--figures-to', type=str, help="Relative path to save figures to")
@click.option('--seed', type=int, help="Random seed", default=123)
@click.option('--cv-folds', type=int, help="Number of cross validation folds", default=5)
def main(x_train_data, y_train_data, x_test_data, y_test_data, preprocessor, pipeline_to, results_to, figures_to, seed, cv_folds):
    # General variables
    classification_metrics = ["accuracy", "precision", "recall", "f1"]
    preprocessor = pickle.load(open(preprocessor, "rb"))
    X_train = pd.read_csv(x_train_data)
    y_train = pd.read_csv(y_train_data)
    X_test = pd.read_csv(x_test_data)
    y_test = pd.read_csv(y_test_data)

    # Define models to evaluate
    models = {
        "dummy": DummyClassifier(random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "kNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=seed),
        # "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(random_state=seed, 
                                                  max_iter=1000)
    }

    # Execute cross-validation for each model and store results
    results_dict = {}

    for estimator in models:
        # print("Running CV for: " + estimator)
        results_dict[estimator] = mean_std_cross_val_scores(
            build_pipe(preprocessor, models[estimator]),
            X_train,
            y_train,
            cv=5,
            return_train_score=True,
            scoring=classification_metrics
        )
        cv_results_df = pd.DataFrame(results_dict).reset_index() # T

    cv_results_df.to_csv(os.path.join(results_to, 
                                      "cv_results_df.csv"), index=False)

    heart_lr = make_pipeline(preprocessor, 
                             LogisticRegression(random_state=seed, 
                                                max_iter=1000))

    heart_lr_fit = heart_lr.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, 
                           "hearth_lr_fit_pipeline.pickle"), 'wb'
                           ) as f:
        pickle.dump(heart_lr_fit, f)

    fit_conf_mat_logreg = ConfusionMatrixDisplay.from_estimator(
        heart_lr_fit,
        X_train,
        y_train,
        values_format="d",
        cmap='Blues'
    )

    # Save Fit Confusion Matrix as image
    fit_conf_mat_logreg.figure_.savefig(
        os.path.join(figures_to, "fit_confusion_matrix_logreg.png"),
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fit_conf_mat_logreg.figure_)  # clean up the figure

    # Save Fit Confusion Matrix as CSV
    fit_cm_logreg_array = fit_conf_mat_logreg.confusion_matrix  # numpy array
    fit_cm_logreg_df = pd.DataFrame(
        fit_cm_logreg_array,
        columns=[f"Pred_{c}" for c in fit_conf_mat_logreg.display_labels],
        index=[f"True_{c}" for c in fit_conf_mat_logreg.display_labels]
    )

    fit_cm_logreg_df.to_csv(os.path.join(results_to, 
                                         "fit_confusion_matrix_logreg.csv"))

    # Evaluate on Test Set
    y_pred = heart_lr.predict(X_test)

    # Generate Classification Report for Test set

    # classification_rep = classification_report(y_test, y_pred)
    # print(classification_rep)
    eval_report_dict = classification_report(y_test, 
                                             y_pred, 
                                             output_dict=True)
    
    eval_report_df = pd.DataFrame(eval_report_dict).T

    eval_report_df.to_csv(os.path.join(results_to, 
                                       "eval_classification_report_logreg.csv"))

    # Confusion Matrix for Logistic Regression model on Test Set
    eval_conf_mat_logreg = ConfusionMatrixDisplay.from_estimator(
        heart_lr.fit(X_test, y_test),
        X_test,
        y_test,
        values_format="d",
        cmap='Blues'
    )

    # Save eval Confusion Matrix as image
    eval_conf_mat_logreg.figure_.savefig(
        os.path.join(figures_to, "eval_confusion_matrix_logreg.png"),
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(eval_conf_mat_logreg.figure_)  # clean up the figure

    # Save eval Confusion Matrix as CSV
    eval_cm_logreg_array = eval_conf_mat_logreg.confusion_matrix  # numpy array
    eval_cm_logreg_df = pd.DataFrame(
        eval_cm_logreg_array,
        columns=[f"Pred_{c}" for c in fit_conf_mat_logreg.display_labels],
        index=[f"True_{c}" for c in fit_conf_mat_logreg.display_labels]
    )

    eval_cm_logreg_df.to_csv(os.path.join(results_to, 
                                          "eval_confusion_matrix_logreg.csv"))

if __name__ == '__main__':
    main()
