from pathlib import Path
from src.data.config import RESULTS_FOLDER
import re
import os
import shutil
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default="evaluation_results.csv")
argparser.add_argument("--csv_compare", type=str, default="evaluation_results_no_hr.csv")

args = argparser.parse_args().__dict__

hr_csv = Path(RESULTS_FOLDER / args["csv"])
no_hr_csv = Path(RESULTS_FOLDER / args["csv_compare"])

def compare_csv_results(hr_csv: Path, no_hr_csv: Path, metric: str = "rmse"):
    import pandas as pd

    df_hr = pd.read_csv(hr_csv)
    df_no_hr = pd.read_csv(no_hr_csv)

    assert len(df_hr) == len(df_no_hr), "CSV files must have the same number of rows for comparison."
    assert hr_csv.columns.equals(no_hr_csv.columns), "CSV files must have the same columns for comparison."
    assert all(df_hr['filename'] == df_no_hr['filename']), "CSV files must have the same filenames in the same order."

    comparison_data = {
        'filename': df_hr['filename'],
        f'{metric}_with_hr': df_hr[metric],
        f'{metric}_without_hr': df_no_hr[metric],
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df[f'{metric}_diff'] = comparison_df[f'{metric}_with_hr'] - comparison_df[f'{metric}_without_hr']

    # get the number of cases where excluding high-res improved performance
    improved_cases = (comparison_df[f'{metric}_diff'] > 0).sum()
    print(f"Number of cases where excluding high-res improved performance: {improved_cases} out of {len(comparison_df)}")

    # get the number of cases where excluding high-res worsened performance
    worsened_cases = (comparison_df[f'{metric}_diff'] < 0).sum()
    print(f"Number of cases where excluding high-res worsened performance: {worsened_cases} out of {len(comparison_df)}")

    # mean rmse with and without high-res
    mean_rmse_with_hr = comparison_df[f'{metric}_with_hr'].mean()
    mean_rmse_without_hr = comparison_df[f'{metric}_without_hr'].mean()
    print(f"Mean {metric} with high-res: {mean_rmse_with_hr}")
    print(f"Mean {metric} without high-res: {mean_rmse_without_hr}")

    return comparison_df

if __name__ == "__main__":
    _ = compare_csv_results(hr_csv, no_hr_csv, metric="rmse")
    _ = compare_csv_results(hr_csv, no_hr_csv, metric="r2")
    