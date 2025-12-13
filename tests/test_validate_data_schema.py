import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.validate_data_schema import validate_data_schema

# --- Fixtures ---


@pytest.fixture
def valid_heart_df():
    return pd.DataFrame(
        {
            "Age": [45, 60],
            "Sex": ["M", "F"],
            "ChestPainType": ["ATA", "ASY"],
            "RestingBP": [120, 140],
            "Cholesterol": [200, 240],
            "FastingBS": [0, 1],
            "RestingECG": ["Normal", "ST"],
            "MaxHR": [150, 130],
            "ExerciseAngina": ["N", "Y"],
            "Oldpeak": [1.0, 2.3],
            "ST_Slope": ["Up", "Flat"],
            "HeartDisease": [0, 1],
        }
    )


@pytest.fixture
def invalid_age_df(valid_heart_df):
    df = valid_heart_df.copy()
    df.loc[0, "Age"] = 150  # Invalid
    return df


# --- Core functionality tests ---


def test_valid_data_passes(valid_heart_df):
    cleaned = validate_data_schema(valid_heart_df)
    assert cleaned.equals(valid_heart_df)


def test_invalid_age_row_dropped(invalid_age_df):
    cleaned = validate_data_schema(invalid_age_df)
    assert len(cleaned) == 1


# --- Input validation tests ---


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        validate_data_schema("not a dataframe")


def test_empty_dataframe():
    with pytest.raises(ValueError):
        validate_data_schema(pd.DataFrame())
