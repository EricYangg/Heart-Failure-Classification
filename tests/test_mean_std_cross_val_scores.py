import pandas as pd
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.mean_std_cross_val_scores as mscv


# Define a fake model object for validation
class FakeModel:
    def fit(self, X, y):
        return self

# 1. Success return of series (index and formatting)
def test_returns_series_with_correct_index_and_format(monkeypatch):
    # Fake cross_validate
    def fake_cross_validate(model, X, y, **kwargs):
        return {
            "fit_time": np.array([0.1, 0.2, 0.3]),
            "score_time": np.array([1.0, 1.0, 1.0]),
            "test_score": np.array([1.0, 2.0, 3.0]),
        }

    monkeypatch.setattr(mscv, "cross_validate", fake_cross_validate)

    # Dummy inputs (they won't be used by fake_cross_validate)
    X = np.zeros((3, 2))
    y = np.zeros(3)

    result = mscv.mean_std_cross_val_scores(FakeModel(), X, y, cv=3)

    # Type check
    assert isinstance(result, pd.Series)

    # Index check – same keys as fake_cross_validate
    assert set(result.index) == {"fit_time", "score_time", "test_score"}

    # Value check – formatting of one field
    # For [1, 2, 3] with ddof=1: mean=2.0, std=1.0
    assert result["test_score"] == "2.000 (+/- 1.000)"

# 2. Success forwarding of **kwargs
def test_kwargs_are_passed_to_cross_validate(monkeypatch):
    received_kwargs = {}

    def fake_cross_validate(model, X, y, **kwargs):
        received_kwargs.update(kwargs)
        return {"test_score": np.array([0.5, 0.6])}

    monkeypatch.setattr(mscv, "cross_validate", fake_cross_validate)

    X = np.zeros((2, 2))
    y = np.zeros(2)

    mscv.mean_std_cross_val_scores(
        FakeModel(),
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )

    assert received_kwargs["cv"] == 5
    assert received_kwargs["scoring"] == "accuracy"
    assert received_kwargs["n_jobs"] == -1
    assert received_kwargs["return_train_score"] is True


# Input Validation Tests
    
def test_invalid_model_raises_typeerror():
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(TypeError, match="model must implement a 'fit'"):
        mscv.mean_std_cross_val_scores(object(), X, y)

def test_invalid_X_type():
    with pytest.raises(TypeError, match="X_train must be a numpy array"):
        mscv.mean_std_cross_val_scores(FakeModel(), "not a df", [1,2,3])

def test_empty_X():
    with pytest.raises(ValueError, match="X_train must contain observations"):
        mscv.mean_std_cross_val_scores(FakeModel(), np.array([]), np.array([]))

def test_X_invalid_shape():
    X = np.array([1,2,3])  # 1D
    y = np.array([1,0,1])

    with pytest.raises(ValueError, match="at least one feature column"):
        mscv.mean_std_cross_val_scores(FakeModel(), X, y)

def test_invalid_y_type():
    X = np.zeros((5, 2))

    with pytest.raises(TypeError, match="y_train must be"):
        mscv.mean_std_cross_val_scores(FakeModel(), X, {"invalid": "dict"})

def test_y_not_1d():
    X = np.zeros((5, 2))
    y = np.zeros((5, 1))

    with pytest.raises(ValueError, match="1-dimensional"):
        mscv.mean_std_cross_val_scores(FakeModel(), X, y)

def test_X_y_length_mismatch():
    X = np.zeros((10, 2))
    y = np.zeros(9)

    with pytest.raises(ValueError, match="same number of rows"):
        mscv.mean_std_cross_val_scores(FakeModel(), X, y)
