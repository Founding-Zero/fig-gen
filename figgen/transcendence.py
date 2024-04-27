# %%
import os
import random

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
    # TODO implement this
    def fetch_and_process_skill_level_data(self, data_header):
        desired_group = defaultdict(list)
        for run in self.runs:
            if (
                data_header in self.histories[run.id].columns
                and len(self.histories[run.id][data_header]) > self.min_length
            ):
                desired_group[run.config["skill_level_vals"]].append(
                    self.histories[run.id][data_header]
                    .iloc[1 : self.min_length]
                    .tolist()
                )

        desired_data = {}
        for skill_level, runs in desired_group.items():
            transposed_runs = list(zip(*runs))
            desired_data[skill_level] = transposed_runs

        records = []
        for skill_level, episode in desired_data.items():
            for episode_index, values in enumerate(episode):
                for value in values:
                    records.append(
                        {
                            "Episode": episode_index,
                            "skill_level": skill_level,
                            data_header: value,
                        }
                    )
        return pd.DataFrame(records)

    def get_table(self, run_id):
        runs = self.get_runs([run_id])
        artifacts = runs[0].logged_artifacts()

        for artifact in artifacts:
            table_name = next(iter(artifacts[0].manifest.entries))
            table = artifact.get(table_name)
            df = pd.DataFrame(data=table.data, columns=table.columns)
            print(df)
            return df


# %%
import pandas as pd

if __name__ == "__main__":
    analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project="Elo-Testing"
    )

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
        )

    ######################
    # Example Usage: Win Percentage by Stockfish Level
    #######################
    groupby = "Stockfish Skill Level"  # Key for the temperature group
    y_label = "Win Percentage"

    sample_data = [
        {"Temperature": 0.5, groupby: "0", y_label: 1000 / 4000.0},
        {"Temperature": 0.5, groupby: "1", y_label: 1200 / 4000.0},
        {"Temperature": 0.5, groupby: "2", y_label: 1400 / 4000.0},
        {"Temperature": 0.2, groupby: "0", y_label: 1100 / 4000.0},
        {"Temperature": 0.2, groupby: "1", y_label: 1300 / 4000.0},
        {"Temperature": 0.2, groupby: "2", y_label: 1500 / 4000.0},
        {"Temperature": 0.1, groupby: "0", y_label: 1200 / 4000.0},
        {"Temperature": 0.1, groupby: "1", y_label: 1400 / 4000.0},
        {"Temperature": 0.1, groupby: "2", y_label: 1600 / 4000.0},
    ]
    sample_data = sum(
        [
            [
                {
                    "Temperature": d["Temperature"],
                    groupby: d[groupby],
                    y_label: min(d[y_label] + 0.01 * np.random.randn(), 1.0),
                }
                for d in sample_data
            ]
            for _ in range(3)
        ],
        [],
    )

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
