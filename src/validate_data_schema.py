import pandas as pd
import pandera as pa


def validate_data_schema(df: pd.DataFrame):
    """
    Validate the heart disease dataset using predefined schema rules with pandera
    and return a cleaned DataFrame with invalid rows removed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw heart disease dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing only valid rows/columns.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is empty.
    """

    # --- Input validation (raw df) ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

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
            pa.Check(lambda df: ~df.duplicated().any()),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any()),
        ],
        drop_invalid_rows=False,
    )

    try:
        validated = schema.validate(df, lazy=True)
        return validated.reset_index(drop=True)

    except pa.errors.SchemaErrors as e:
        invalid_idx = e.failure_cases["index"].dropna().unique()
        cleaned = (
            df.drop(index=invalid_idx)
            .drop_duplicates()
            .dropna(how="all")
            .reset_index(drop=True)
        )
        return cleaned
