#
# Prerequisites:
# pip3 install matplotlib pandas seaborn
#
# Usage:
# python3 main-visualize.py </path/to/rand_result.csv> </path/to/bo_result.csv> </path/to/output.pdf>
#

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


def process_args() -> Tuple[str, str, str]:
    assert len(sys.argv) >= 4

    return sys.argv[1], sys.argv[2], sys.argv[3]


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


def visualize_stats(path: str, data: Dict[str, pd.DataFrame]) -> None:
    FIG_SIZE = (4, 4)
    CONFIDENT_REGION_ALPHA = 0.2
    X_TICKS_SKIP = 2
    DPI = 300

    num_iters = next(iter(data.values())).shape[0]

    sns.set()
    sns.set_context()

    plt.rcParams['font.sans-serif'] = ["Linux Biolinum"]

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)

    for name, data_frame in data.items():
        ax.fill_between(range(1, num_iters + 1),
                        data_frame["lower"],
                        data_frame["upper"],
                        alpha=CONFIDENT_REGION_ALPHA,
                        label=name)

    for name, data_frame in data.items():
        ax.plot(range(1, num_iters + 1), data_frame["mean"], label=name)

    ax.set_xlim([1, num_iters])
    ax.set_xticks(range(1, num_iters + 1, X_TICKS_SKIP))
    ax.set_xlabel("#iterations")
    ax.set_ylabel("Function value")

    ax.legend(data.keys())

    fig.tight_layout()

    plt.savefig(path)


if __name__ == "__main__":
    rand_csv_path, bo_csv_path, out_path = process_args()

    rand_data_frame = read_csv_as_data_frame(rand_csv_path)
    bo_data_frame = read_csv_as_data_frame(bo_csv_path)

    rand_stats = calculate_stats(rand_data_frame)
    bo_stats = calculate_stats(bo_data_frame)

    visualize_stats(out_path, {
        "Random Sampling": rand_stats,
        "Bayesian Optimization": bo_stats,
    })
