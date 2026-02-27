import pandas as pd
import numpy as np


def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    # Create bins using expected (training) data
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    # Prevent duplicate bin edges
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_ratio = expected_counts / len(expected)
    actual_ratio = actual_counts / len(actual)

    psi_value = np.sum(
        (expected_ratio - actual_ratio) *
        np.log((expected_ratio + 1e-6) / (actual_ratio + 1e-6))
    )

    return psi_value

def detect_drift(train_df, new_df, threshold=0.2):
    drift_report = {}

    numeric_columns = train_df.select_dtypes(include=["int64", "float64"]).columns

    for column in numeric_columns:
        psi = calculate_psi(train_df[column], new_df[column])
        drift_report[column] = psi

    drift_detected = any(value > threshold for value in drift_report.values())

    return drift_detected, drift_report