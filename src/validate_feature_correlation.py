import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")
warnings.filterwarnings("ignore", category=UserWarning, module="deepchecks")
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation

def validate_feature_correlation(train_df: pd.DataFrame, label: str, categorical_features: list, feat_lab_threshold: float, feat_feat_threshold: float, feat_feat_n_pairs: int):
    """
    Validate feature correlations in the training dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        The training dataset.
    label : str
        The name of the label column.
    categorical_features : list
        List of categorical feature names.
    feat_lab_threshold : float
        Threshold for feature-label correlation.
    feat_feat_threshold : float
        Threshold for feature-feature correlation.
    feat_feat_n_pairs : int
        Maximum number of feature-feature pairs allowed above the threshold.
    
    Returns
    -------
    tuple
        A tuple containing the results of feature-label and feature-feature correlation check objects.

    Raises
    ------
    ValueError
        If any feature-label or feature-feature correlation exceeds the defined thresholds.
    """

    heart_train_ds = Dataset(
        train_df,
        label=label,
        cat_features=categorical_features
    )

    # Check: Feature-label correlation
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(feat_lab_threshold)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=heart_train_ds)

    # Check: Feature-feature correlation
    check_feat_feat_corr = (
        FeatureFeatureCorrelation()
        .add_condition_max_number_of_pairs_above_threshold(n_pairs=feat_feat_n_pairs, threshold=feat_feat_threshold)
    )
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=heart_train_ds)

    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-label correlation exceeds the maximum acceptable threshold.")

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")
    
    return check_feat_lab_corr_result, check_feat_feat_corr_result