import json
import os
import pickle
import random

from figgen.glicko2 import GlickoCalc

os.environ["DISPLAY"] = ""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transcendence import TranscendenceDataAnalyzer
from wandb.apis import PublicApi
from wandb.apis.public.artifacts import ArtifactType
from wandb.sdk import Artifact

from figgen import DataAnalyzer


def fetch_and_process_illegal_move_percentage(df):
    games_played = 0
    # games_played = 20
    games_lost_due_to_illegal_moves = 0
    for idx, row in df.iterrows():
        games_played += 1
        if row["player_one"].startswith("Stockfish"):
            nanogpt_index = "two"
            stockfish_index = "one"
        else:
            nanogpt_index = "one"
            stockfish_index = "two"

        if (
            row[f"player_{nanogpt_index}_score"] == "0"
            and row[f"player_{nanogpt_index}_failed_to_find_legal_move"]
        ):
            games_lost_due_to_illegal_moves += 1

    return games_played, games_lost_due_to_illegal_moves


def illegal_move_experiment(groupby, y_label, x_label, data):
    title = "Loss Percentage Due to Illegal Moves By Trained High Elo"
    analyzer.visualize_one_lineplot_groupby(
        title,
        x_label,
        y_label,
        groupby,
        pd.DataFrame(data),
        x_label="Training Iteration Number",
        y_label=f"Chess {y_label}",
        x_ticks_by_data=True,
    )


if __name__ == "__main__":
    print("Illegal Move Percentage Distribution")
    analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project="50M-Training"
    )
    # analyzer = TranscendenceDataAnalyzer(
    #     wandb_entity="project-eval", wandb_project="transcendence-Eval-Full"
    # )

    runs_in_project_per_high_elo = {
        # "1000": ["t2uz57fr", "akbd8k2v"],
        # "1100": ["jfwepvp6", "2nlzvq3l"],
        # "1200": ["asqcagmw", "6cuuhiff"],
        # "1300": ["u56krymr", "ar2ta0cs"],
        # "1400": ["lhyji88c", "zgyf6ngk"],
        # "1500": ["z1xdz9c9", "zyqaxg1h"],
        "1000": ["c6e7g14h"],
        "1100": ["fygd29kg"],
        "1300": ["cetrd1vi"],
        "1500": ["3j84hvqe"],
        # "1600": ["uymkl9hv"],
        # "1700": ["xsons03b"],
        # "1800": ["uf9f13c4"],
        # "1900": ["ayjt31xc"],
        # "2000": ["at0nlgx4"],
        # "2100": ["tywi2vuu"],
        # "2200": ["titqh8nb"],
        # "2300": ["w86a0m14"],
    }

    if os.path.exists("cached_dfs_by_high_elo_new_1000_1500.pkl"):
        # if os.path.exists('cached_dfs_by_high_elo_160000.pkl'):
        # if os.path.exists('cached_dfs_by_high_elo_1600_2300.pkl'):
        # if os.path.exists('cached_dfs_by_high_elo_1000_1500.pkl'):
        dfs_by_high_elo = pickle.load(
            open("cached_dfs_by_high_elo_new_1000_1500.pkl", "rb")
        )
        # dfs_by_high_elo = pickle.load(open('cached_dfs_by_high_elo_160000.pkl', 'rb'))
        # dfs_by_high_elo = pickle.load(open('cached_dfs_by_high_elo_1600_2300.pkl', 'rb'))
        # dfs_by_high_elo = pickle.load(open('cached_dfs_by_high_elo_1000_1500.pkl', 'rb'))
    else:
        dfs_by_high_elo = {}
        # dfs_by_high_elo_dict_list = defaultdict(list)
        for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
            for run_id in runs_in_project_per_high_elo[high_elo]:
                dfs_by_high_elo[high_elo] = analyzer.get_table_by_iter_num(run_id)
                # dfs_by_high_elo_dict_list[high_elo].append(analyzer.get_table_by_iter_num(run_id))

            # if len(dfs_by_high_elo_dict_list[high_elo]) > 1:
            #     df_list = [dfs_by_high_elo_dict_list[high_elo][0]]
            #     for i in range(1 , len(dfs_by_high_elo_dict_list[high_elo])):
            #         for key, value in dfs_by_high_elo_dict_list[high_elo][i].items():
            #             dfs_by_high_elo_dict_list[high_elo][0][key] = pd.concat([dfs_by_high_elo_dict_list[high_elo][0][key], value])

            # dfs_by_high_elo[high_elo] = dfs_by_high_elo_dict_list[high_elo][0]
        with open("cached_dfs_by_high_elo_new_1000_1500.pkl", "wb") as fin:
            # with open('cached_dfs_by_high_elo_160000.pkl', 'wb') as fin:
            # with open('cached_dfs_by_high_elo_1600_2300.pkl', 'wb') as fin:
            # with open('cached_dfs_by_high_elo_1000_1500.pkl', 'wb') as fin:
            pickle.dump(dfs_by_high_elo, fin)

    sample_data = []
    for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
        dfs = dfs_by_high_elo[high_elo]
        for iter_num, df in dfs.items():
            (
                games_played,
                games_lost_due_to_illegal_moves,
            ) = fetch_and_process_illegal_move_percentage(df)
            games_loss_percentage_due_to_illegal_moves = (
                games_lost_due_to_illegal_moves / games_played * 100
            )
            sample_data += [
                {
                    "Iteration Num": iter_num,
                    "High Elo": high_elo,
                    "Loss Percentage due to Illegal Moves": games_loss_percentage_due_to_illegal_moves,
                },
            ]

    groupby = "High Elo"  # Key for the temperature group
    y_label = "Loss Percentage due to Illegal Moves"
    x_label = "Iteration Num"
    illegal_move_experiment(groupby, y_label, x_label, sample_data)
