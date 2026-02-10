"""
File I/O utilities: CSV loading with encoding detection and column validation.
"""

from typing import List, Optional, Tuple

import pandas as pd


# Common encodings to try when reading CSV files
DEFAULT_CSV_ENCODINGS: List[str] = [
    "utf-8",
    "utf-8-sig",
    "gbk",
    "gb2312",
    "latin-1",
]


def load_csv_with_encoding(
    file_path: str,
    encodings: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file, trying multiple encodings until one succeeds.

    Args:
        file_path: Path to the CSV file.
        encodings: List of encoding names to try. Defaults to DEFAULT_CSV_ENCODINGS.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file could not be decoded with any of the given encodings.
    """
    encodings = encodings or DEFAULT_CSV_ENCODINGS
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
        except Exception as e:
            last_error = e
            if "codec can't decode" not in str(e).lower():
                raise

    raise ValueError(
        f"Could not read CSV with any of the tried encodings: {encodings}. "
        f"Last error: {last_error}"
    )


def validate_csv_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_on_empty: bool = False,
) -> Tuple[bool, str]:
    """
    Check that a DataFrame contains all required columns.

    Args:
        df: The DataFrame to validate.
        required_columns: List of column names that must be present.
        raise_on_empty: If True, raise ValueError on failure; otherwise return (False, message).

    Returns:
        (True, "") if valid; otherwise (False, error_message).

    Raises:
        ValueError: If raise_on_empty is True and validation fails.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        msg = (
            f"CSV is missing required columns: {', '.join(missing)}. "
            f"Existing columns: {', '.join(df.columns.tolist())}. "
            f"Required: {', '.join(required_columns)}."
        )
        if raise_on_empty:
            raise ValueError(msg)
        return False, msg

    if df.empty:
        msg = "CSV has no rows. Please provide at least one row of data."
        if raise_on_empty:
            raise ValueError(msg)
        return False, msg

    return True, ""


def warn_empty_column(df: pd.DataFrame, column_name: str) -> None:
    """
    Log a warning if a column has empty or NaN values in any row.

    Args:
        df: The DataFrame to check.
        column_name: Name of the column.
    """
    if column_name not in df.columns:
        return
    empty = df[df[column_name].isna() | (df[column_name].astype(str).str.strip() == "")]
    if len(empty) > 0:
        print(f"  Warning: Column '{column_name}' has {len(empty)} empty row(s); they will be skipped.")

