import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = True,
):
    """
    Prepare the bank churn dataset for machine learning.

    The dataset is split into training and validation sets.
    Numeric features can be scaled, and categorical features
    are one-hot encoded.

    Args:
        raw_df: Raw bank churn dataset.
        scale_numeric: Whether to scale numeric features.

    Returns:
        X_train, y_train, X_val, y_val,
        input_cols, scaler, encoder.
    """
    train_df, val_df = split_dataset(raw_df)

    train_X, train_y, input_cols = split_features_and_target(train_df)
    val_X, val_y, _ = split_features_and_target(val_df)

    train_X, val_X, scaler, encoder = preprocess_features(
        train_X,
        val_X,
        scale_numeric,
    )

    return (
        train_X,
        train_y,
        val_X,
        val_y,
        input_cols,
        scaler,
        encoder,
    )


def split_dataset(df: pd.DataFrame):
    """
    Split dataset into training and validation sets.

    Args:
        df: Raw dataset.

    Returns:
        Training and validation datasets.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Exited"],
    )

    return train_df, val_df


def split_features_and_target(df: pd.DataFrame):
    """
    Split dataset into features and target.

    The 'Surname' column is excluded.

    Args:
        df: Dataset.

    Returns:
        Features, target and input column names.
    """
    target_col = "Exited"

    input_cols = [
        col for col in df.columns
        if col not in ["Exited", "Surname"]
    ]

    X = df[input_cols].copy()
    y = df[target_col].copy()

    return X, y, input_cols


def preprocess_features(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    scale_numeric: bool,
):
    """
    Apply preprocessing to feature datasets.

    Args:
        train_X: Training features.
        val_X: Validation features.
        scale_numeric: Whether to scale numeric features.

    Returns:
        Processed train and validation features,
        fitted scaler and encoder.
    """
    numeric_cols = train_X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_X.select_dtypes(include="object").columns.tolist()

    scaler = None

    if scale_numeric:
        train_X, val_X, scaler = scale_numeric_features(
            train_X,
            val_X,
            numeric_cols,
        )

    train_X, val_X, encoder = encode_categorical_features(
        train_X,
        val_X,
        categorical_cols,
    )

    return train_X, val_X, scaler, encoder


def scale_numeric_features(
    train_X,
    val_X,
    numeric_cols,
):
    """
    Scale numeric features using MinMaxScaler.

    Args:
        train_X: Training features.
        val_X: Validation features.
        numeric_cols: Numeric columns.

    Returns:
        Processed datasets and fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_X[numeric_cols])

    train_X[numeric_cols] = scaler.transform(train_X[numeric_cols])
    val_X[numeric_cols] = scaler.transform(val_X[numeric_cols])

    return train_X, val_X, scaler


def encode_categorical_features(
    train_X,
    val_X,
    categorical_cols,
):
    """
    One-hot encode categorical features.

    Args:
        train_X: Training features.
        val_X: Validation features.
        categorical_cols: Categorical columns.

    Returns:
        Processed datasets and fitted encoder.
    """
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoder.fit(train_X[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = encoder.transform(train_X[categorical_cols])
    val_encoded = encoder.transform(val_X[categorical_cols])

    train_X = train_X.drop(columns=categorical_cols)
    val_X = val_X.drop(columns=categorical_cols)

    train_X[encoded_cols] = train_encoded
    val_X[encoded_cols] = val_encoded 

    return train_X, val_X, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols,
    scaler,
    encoder,
):
    """
    Apply fitted preprocessing to new data.

    Args:
        new_df: New dataset.
        input_cols: Feature columns used during training.
        scaler: Fitted scaler.
        encoder: Fitted encoder.

    Returns:
        Processed feature dataframe.
    """
    X = new_df[input_cols].copy()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded = encoder.transform(X[categorical_cols])

    X = X.drop(columns=categorical_cols)
    X[encoded_cols] = encoded

    return X