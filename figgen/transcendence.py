# %%
import os
import random

from figgen.glicko2 import GlickoCalc

os.environ["DISPLAY"] = ""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wandb.apis import PublicApi
from wandb.apis.public.artifacts import ArtifactType
from wandb.sdk import Artifact

from figgen import DataAnalyzer


class TranscendenceDataAnalyzer(DataAnalyzer):
    def fetch_and_process_skill_level_data_for_temperature(self, df: pd.DataFrame):
        row = df.iloc[0]

        glicko = GlickoCalc()
        temperature = row["temperature"]

        stockfish_index = "one" if row["player_one"].startswith("Stockfish") else "two"
        stockfish_level = int(row[f"player_{stockfish_index}"].split(" ")[1])

        for idx, row in df.iterrows():
            if row["player_one"].startswith("Stockfish"):
                nanogpt_index = "two"
                stockfish_index = "one"
            else:
                nanogpt_index = "one"
                stockfish_index = "two"

            # assert stockfish_level == int(row[f"player_{stockfish_index}"].split(" ")[1]), "Stockfish level should be the same for all rows in the dataframe"
            assert (
                temperature == row["temperature"]
            ), "Temperature should be the same for all rows in the dataframe"

            if row[f"player_{nanogpt_index}_score"] == "1":
                glicko.glicko2_update(0, stockfish_level)
            elif row[f"player_{nanogpt_index}_score"] == "0":
                glicko.glicko2_update(1, stockfish_level)
            else:
                glicko.glicko2_update(2, stockfish_level)

        return glicko.current_elo, glicko.current_deviation, temperature, 0  # TODO

    def get_table(self, run_id):
        runs = self.get_runs([run_id])
        artifacts = runs[0].logged_artifacts()

        dfs = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if "2000_2000" not in table_name:  #!!! HACK, need to fix
                continue
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                dfs.append(df)

        return dfs


# %%
def custom_error_bar_fn(x):
    two_mean_minus_std = x.max()
    std = x.min()
    mean = (two_mean_minus_std + std) / 2

    return (mean - 2 * std, mean + 2 * std)


if __name__ == "__main__":
    analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project="770-Testing-Eval"
    )
    # https://wandb.ai/project-eval/770-Testing-Eval/runs/2vev6jt5/overview?nw=nwuserezipe 770-High-Elo-2000-Eval at April 28, 4:32pm

    # table = analyzer.get_table("r5gi54js")
    def temperature_sampling_experiment(groupby, y_label, data):
        analyzer.visualize_lineplot_groupby(
            f"{y_label}s of NanoGPT across Temperature",
            "Temperature",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
            custom_error_bar_fn=custom_error_bar_fn,
        )

    dfs = analyzer.get_table("2vev6jt5")
    sample_data = []
    for df in dfs:
        (
            elo,
            dev,
            temperature,
            stockfish_level,
        ) = analyzer.fetch_and_process_skill_level_data_for_temperature(df)
        sample_data += [
            {
                "Temperature": temperature,
                "Stockfish Skill Level": stockfish_level,
                "Rating": 2 * elo - dev,
            },
            {
                "Temperature": temperature,
                "Stockfish Skill Level": stockfish_level,
                "Rating": dev,
            },  # super hacky
        ]

    groupby = "Stockfish Skill Level"  # Key for the temperature group
    y_label = "Rating"

    ######################
    # Example Usage: Win Percentage by Stockfish Level
    #######################
    # groupby = "Stockfish Skill Level"  # Key for the temperature group
    # y_label = "Win Percentage"

    # sample_data = [
    #     {"Temperature": 0.5, groupby: "0", y_label: 1000 / 4000.0},
    #     {"Temperature": 0.5, groupby: "1", y_label: 1200 / 4000.0},
    #     {"Temperature": 0.5, groupby: "2", y_label: 1400 / 4000.0},
    #     {"Temperature": 0.2, groupby: "0", y_label: 1100 / 4000.0},
    #     {"Temperature": 0.2, groupby: "1", y_label: 1300 / 4000.0},
    #     {"Temperature": 0.2, groupby: "2", y_label: 1500 / 4000.0},
    #     {"Temperature": 0.1, groupby: "0", y_label: 1200 / 4000.0},
    #     {"Temperature": 0.1, groupby: "1", y_label: 1400 / 4000.0},
    #     {"Temperature": 0.1, groupby: "2", y_label: 1600 / 4000.0},
    # ]
    # sample_data = sum(
    #     [
    #         [
    #             {
    #                 "Temperature": d["Temperature"],
    #                 groupby: d[groupby],
    #                 y_label: min(d[y_label] + 0.01 * np.random.randn(), 1.0),
    #             }
    #             for d in sample_data
    #         ]
    #         for _ in range(3)
    #     ],
    #     [],
    # )

    temperature_sampling_experiment(groupby, y_label, sample_data)

    ######################
    # Example Usage: Chess Rating by Model Size
    #######################
    groupby = "Model Size"  # Key for the temperature group
    y_label = "Chess Rating"

    sample_data = [
        {"Temperature": 0.5, groupby: "Small", y_label: 1000},
        {"Temperature": 0.5, groupby: "Medium", y_label: 1200},
        {"Temperature": 0.5, groupby: "Large", y_label: 1400},
        {"Temperature": 0.2, groupby: "Small", y_label: 1100},
        {"Temperature": 0.2, groupby: "Medium", y_label: 1300},
        {"Temperature": 0.2, groupby: "Large", y_label: 1500},
        {"Temperature": 0.1, groupby: "Small", y_label: 1200},
        {"Temperature": 0.1, groupby: "Medium", y_label: 1400},
        {"Temperature": 0.1, groupby: "Large", y_label: 1600},
    ]
    sample_data = sum(
        [
            [
                {
                    "Temperature": d["Temperature"],
                    groupby: d[groupby],
                    y_label: d[y_label] + 100 * np.random.randn(),
                }
                for d in sample_data
            ]
            for _ in range(3)
        ],
        [],
    )
    temperature_sampling_experiment(groupby, y_label, sample_data)
