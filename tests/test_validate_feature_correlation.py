import pandas as pd
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_feature_correlation import validate_feature_correlation

# --- Helper Fixtures and Data ---
@pytest.fixture(scope="module")
def categorical_features():
    """Returns the list of categorical features in the dataset."""
    return ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# 1. Success Case Fixture (Low Correlation)
@pytest.fixture
def dummy_data_low_corr(categorical_features):
    """Returns a DataFrame with very low feature-label and feature-feature correlation (should pass)."""
    n_samples = 100
    np.random.seed(123)
    
    data = {
        'Age': np.random.randint(20, 80, n_samples),
        'Sex': np.random.choice(['M', 'F'], n_samples),
        'ChestPainType': np.random.choice(['ASY', 'ATA', 'NAP', 'TA'], n_samples),
        'RestingBP': np.random.randint(100, 180, n_samples),
        'Cholesterol': np.random.randint(150, 400, n_samples),
        'FastingBS': np.random.choice([0, 1], n_samples),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
        'MaxHR': np.random.randint(90, 200, n_samples),
        'ExerciseAngina': np.random.choice(['N', 'Y'], n_samples),
        'Oldpeak': np.round(np.random.uniform(0, 3.5, n_samples), 1),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
        'HeartDisease': np.random.choice([0, 1], n_samples)
    }
    df = pd.DataFrame(data)
    for col in categorical_features:
        df[col] = df[col].astype('object')
    return df

# 2. Failure Case Fixture (High Feature-Label Correlation)
@pytest.fixture
def minimal_failing_feat_lab_corr_df(categorical_features):
    """Hardcoded DataFrame where 'Age' is perfectly correlated with 'HeartDisease'."""
    # Increased to 10 rows to ensure statistical calculations work
    df = pd.DataFrame({
        'Age': [20, 30, 40, 50, 60, 20, 30, 40, 50, 60],
        'HeartDisease': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1] # Correlation with Age is clear
    })
    
    # Fill required columns with constant valid data
    for col in ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
        df[col] = 120
    df['FastingBS'] = 0
    
    for col in categorical_features:
        df[col] = 'Normal' # or appropriate dummy string
        df[col] = df[col].astype('object')
        
    return df

# 3. Failure Case Fixture (High Feature-Feature Correlation)
@pytest.fixture
def minimal_failing_feat_feat_corr_df(categorical_features):
    """Hardcoded DataFrame where 'RestingBP' and 'MaxHR' are perfectly correlated."""
    # Increased to 10 rows
    df = pd.DataFrame({
        'RestingBP': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        'MaxHR':     [100, 110, 120, 130, 140, 150, 160, 170, 180, 190], # Perfect Correlation (1.0)
        'HeartDisease': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Fill required columns
    df['Age'] = 50
    df['Cholesterol'] = 200
    df['FastingBS'] = 0
    df['Oldpeak'] = 0.0
    
    for col in categorical_features:
        df[col] = 'Normal'
        df[col] = df[col].astype('object')

    return df


# CORE Function Logic tests

def test_01_core_logic_passes(dummy_data_low_corr, categorical_features):
    """Test 01: Confirms the function passes when data is compliant (low correlation)."""
    result_lab, result_feat = validate_feature_correlation(
        train_df=dummy_data_low_corr,
        label='HeartDisease',
        categorical_features=categorical_features,
        feat_lab_threshold=0.8,
        feat_feat_threshold=0.95,
        feat_feat_n_pairs=100
    )
    assert result_lab.passed_conditions() == True
    assert result_feat.passed_conditions() == True

def test_02_core_logic_fails_feat_lab_corr(minimal_failing_feat_lab_corr_df, categorical_features):
    """Test 02: Confirms ValueError is raised for high Feature-Label correlation."""
    with pytest.raises(ValueError, match="Feature-label correlation exceeds the maximum acceptable threshold."):
        validate_feature_correlation(
            train_df=minimal_failing_feat_lab_corr_df,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.01, # Guaranteed failure threshold
            feat_feat_threshold=0.95,
            feat_feat_n_pairs=100
        )

def test_03_core_logic_fails_feat_feat_corr(minimal_failing_feat_feat_corr_df, categorical_features):
    """Test 03: Confirms ValueError is raised for high Feature-Feature correlation."""
    with pytest.raises(ValueError, match="Feature-feature correlation exceeds the maximum acceptable threshold."):
        validate_feature_correlation(
            train_df=minimal_failing_feat_feat_corr_df,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.95,
            feat_feat_threshold=0.01, # Guaranteed failure threshold
            feat_feat_n_pairs=0
        )


# Input Validation Tests

# --- Input: train_df (DataFrame) ---

def test_04_validate_df_type(dummy_data_low_corr):
    """Test 04: Confirms TypeError is raised if train_df is not a pandas DataFrame."""
    with pytest.raises(TypeError, match="Input 'train_df' must be a pandas DataFrame."):
        validate_feature_correlation(
            train_df={'col1': [1, 2]},  # Wrong type
            label='HeartDisease',
            categorical_features=[],
            feat_lab_threshold=0.5, feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

def test_05_validate_df_empty(categorical_features):
    """Test 05: Confirms ValueError is raised if the input DataFrame is empty."""
    empty_df = pd.DataFrame(columns=['HeartDisease', 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'] + categorical_features)
    
    with pytest.raises(ValueError, match="Input 'train_df' cannot be empty."):
        validate_feature_correlation(
            train_df=empty_df,  # Empty DataFrame
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.5, feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

# --- Input: label (str) ---

def test_06_validate_label_missing_column(dummy_data_low_corr, categorical_features):
    """Test 06: Confirms ValueError is raised if the label column is missing."""
    with pytest.raises(ValueError, match="Label column 'MissingColumn' not found in the DataFrame columns."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='MissingColumn',  # Missing column
            categorical_features=categorical_features,
            feat_lab_threshold=0.5, feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

# --- Input: categorical_features (list) ---

def test_07_validate_cat_features_type(dummy_data_low_corr):
    """Test 07: Confirms TypeError is raised if categorical_features is not a list."""
    with pytest.raises(TypeError, match="Input 'categorical_features' must be a list."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features='not_a_list', # Wrong type (string)
            feat_lab_threshold=0.5, feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

# --- Input: feat_lab_threshold (float) ---

def test_08_validate_lab_thresh_type(dummy_data_low_corr, categorical_features):
    """Test 08: Confirms TypeError is raised if feat_lab_threshold is not a number."""
    with pytest.raises(TypeError, match="Input 'feat_lab_threshold' must be a number"):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold='high',  # Wrong type (string)
            feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

def test_09_validate_lab_thresh_range_high(dummy_data_low_corr, categorical_features):
    """Test 09: Confirms ValueError is raised when feat_lab_threshold is outside [0, 1] range (High)."""
    with pytest.raises(ValueError, match="Input 'feat_lab_threshold' must be between 0 and 1, inclusive."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=2.5,  # > 1
            feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

def test_10_validate_lab_thresh_range_low(dummy_data_low_corr, categorical_features):
    """Test 10: Confirms ValueError is raised when feat_lab_threshold is outside [0, 1] range (Low)."""
    with pytest.raises(ValueError, match="Input 'feat_lab_threshold' must be between 0 and 1, inclusive."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=-0.1,  # < 0
            feat_feat_threshold=0.5, feat_feat_n_pairs=1
        )

# --- Input: feat_feat_threshold (float) ---

def test_11_validate_feat_thresh_type(dummy_data_low_corr, categorical_features):
    """Test 11: Confirms TypeError is raised if feat_feat_threshold is not a number."""
    with pytest.raises(TypeError, match="Input 'feat_feat_threshold' must be a number, int or float."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.5,
            feat_feat_threshold='low',  # Wrong type (string)
            feat_feat_n_pairs=1
        )

def test_12_validate_feat_thresh_range_high(dummy_data_low_corr, categorical_features):
    """Test 12: Confirms ValueError is raised when feat_feat_threshold is outside [0, 1] range (High)."""
    with pytest.raises(ValueError, match="Input 'feat_feat_threshold' must be between 0 and 1, inclusive."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.8,
            feat_feat_threshold=2.5,  # > 1
            feat_feat_n_pairs=1
        )

def test_13_validate_feat_thresh_range_low(dummy_data_low_corr, categorical_features):
    """Test 13: Confirms ValueError is raised when feat_feat_threshold is outside [0, 1] range (Low)."""
    with pytest.raises(ValueError, match="Input 'feat_feat_threshold' must be between 0 and 1, inclusive."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.8,
            feat_feat_threshold=-0.1,  # < 0
            feat_feat_n_pairs=1
        )

# --- Input: feat_feat_n_pairs (int) ---

def test_14_validate_n_pairs_type(dummy_data_low_corr, categorical_features):
    """Test 14: Confirms TypeError is raised if feat_feat_n_pairs is not an integer."""
    with pytest.raises(TypeError, match="Input 'feat_feat_n_pairs' must be an integer"):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.5,
            feat_feat_threshold=0.5,
            feat_feat_n_pairs=3.14  # Wrong type (float)
        )

def test_15_validate_n_pairs_range(dummy_data_low_corr, categorical_features):
    """Test 15: Confirms ValueError is raised if feat_feat_n_pairs is negative."""
    with pytest.raises(ValueError, match="Input 'feat_feat_n_pairs' cannot be negative."):
        validate_feature_correlation(
            train_df=dummy_data_low_corr,
            label='HeartDisease',
            categorical_features=categorical_features,
            feat_lab_threshold=0.8,
            feat_feat_threshold=0.95,
            feat_feat_n_pairs=-5 # Invalid value (< 0)
        )