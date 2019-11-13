#
# Prerequisites:
# pip3 install pandas
#
# Usage:
# python3 main-visualize.py </path/to/random_result.csv> </path/to/bo_result.csv>
#

from typing import Tuple
import pandas as pd


def process_args() -> Tuple[str, str]:
    pass


def read_csv_as_data_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None)


def calculate_stats(data_frame: pd.DataFrame) -> pd.DataFrame:
    num_iters = data_frame.shape[0]

    mean_array = []
    lower_array = []
    upper_array = []

    for iter in range(num_iters):
        mean = data_frame.loc[iter].mean()
        stdev = data_frame.loc[iter].std()

        mean_array.append(mean)
        lower_array.append(mean - stdev)
        upper_array.append(mean + stdev)

    return pd.DataFrame({
        "mean": mean_array,
        "lower": lower_array,
        "upper": upper_array,
    })


if __name__ == "__main__":
    rand_csv_path, bo_csv_path = process_args()

    rand_data_frame = read_csv_as_data_frame(rand_csv_path)
    bo_data_frame = read_csv_as_data_frame(bo_csv_path)

    rand_stats = calculate_stats(rand_data_frame)
    bo_stats = calculate_stats(bo_data_frame)
