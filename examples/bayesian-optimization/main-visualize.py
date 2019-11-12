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
    data_frame = pd.read_csv(path, header=None)
    print(data_frame)


if __name__ == "__main__":
    random_csv_path, bo_csv_path = process_args()

    random_data_frame = read_csv_as_data_frame(random_csv_path)
    bo_data_frame = read_csv_as_data_frame(bo_csv_path)
