# %%
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
import torch
from wandb.apis import PublicApi
from wandb.apis.public.artifacts import ArtifactType
from wandb.sdk import Artifact

from figgen import DataAnalyzer


class TranscendenceDataAnalyzer(DataAnalyzer):
    def get_table_by_iter_num(self, run_id):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()
        dfs_by_iter_num = {}
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet":  # should be the end of the list of tables
                break
            if table_name.startswith(
                f"eval_table_iter=210000"
            ):  # should be the end of the list of tables
                break
            # if table_name.startswith(f"eval_table_iter={iter_num}"):
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                row = df.iloc[0]
                if row["temperature"] == 0.001:
                    part = table_name.split("=")[1]
                    iter_num = part.split("_")[0]
                    dfs_by_iter_num[int(iter_num)] = df

        return dfs_by_iter_num

    def fetch_and_process(self, df: pd.DataFrame):
        row = df.iloc[0]

        glicko = GlickoCalc()
        temperature = row["temperature"]
        for idx, row in df.iterrows():
            if row["player_one"].startswith("Stockfish"):
                nanogpt_index = "two"
                stockfish_index = "one"
            else:
                nanogpt_index = "one"
                stockfish_index = "two"
            stockfish_level = int(row[f"player_{stockfish_index}"].split(" ")[1])
            assert (
                temperature == row["temperature"]
            ), "Temperature should be the same for all rows in the dataframe"

            if row[f"player_{nanogpt_index}_score"] == "1":
                glicko.glicko2_update(0, stockfish_level)
            elif row[f"player_{nanogpt_index}_score"] == "0":
                glicko.glicko2_update(1, stockfish_level)
            else:
                glicko.glicko2_update(2, stockfish_level)

        return glicko.current_elo, glicko.current_deviation  # TODO

    def get_table(self, run_id):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()
        dfs = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet":  # should be the end of the list of tables
                break
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                dfs.append(df)

        return dfs

    def get_table_for_iter_num(self, run_id, iter_num):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()
        dfs = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet":  # should be the end of the list of tables
                break
            if table_name.startswith(f"eval_table_iter={iter_num}"):
                table = artifact.get(table_name)
                if table is not None:
                    df = pd.DataFrame(data=table.data, columns=table.columns)
                    dfs.append(df)

        return dfs

    def get_all_charts_sort_by_ckpt(self, run_id):
        runs = self.get_runs([run_id])

        def create_nested_dict(run):
            nested_dict = {}
            for key, value in run.summary.items():
                parts = key.split("/")  # Split the path into components
                if key.startswith("tactics_eval"):
                    parts[2], parts[3] = parts[3], parts[2]
                parts = [
                    part.strip() for part in parts if part
                ]  # Remove any empty strings

                # Start with the top-level dictionary and iteratively go deeper
                current_level = nested_dict
                for part in parts[:-1]:  # Go up to the second last part to build keys
                    if part not in current_level:
                        current_level[
                            part
                        ] = {}  # Create a new dictionary if the key does not exist
                    current_level = current_level[
                        part
                    ]  # Move to the next level of the dictionary

                # Set the value at the deepest level
                last_key = parts[-1]
                current_level[last_key] = value

            return nested_dict

        # Convert and write JSON object to file
        with open("sample.json", "w") as outfile:
            json.dump(create_nested_dict(runs[0]), outfile)
        return create_nested_dict(runs[0])

    def get_df_from_panels_per_ckpt(self, data_by_big_row):
        dfs = {}
        avg_big_row = {}
        avg_big_row_indexes = [
            "accuracy",
            "correctly solved percentage",
            "solved length",
            "length",
        ]
        for model_key, metric_dict in data_by_big_row[
            "avg"
        ].items():  # panels is a list of panel items
            column_name = " ".join(model_key.split("_"))
            avg_big_row[column_name] = []
            avg_big_row[column_name].append(metric_dict["accuracy"])
            avg_big_row[column_name].append(metric_dict["correctly_solved_percentage"])
            avg_big_row[column_name].append(metric_dict["solved_length"])
            avg_big_row[column_name].append(metric_dict["length"])

        # turn big_row into pandas data frame:
        avg_big_row_df = pd.DataFrame(avg_big_row, index=avg_big_row_indexes)
        dfs["avg"] = avg_big_row_df

        for i in range(1, 11):
            big_row = {}
            big_row_indexes = [
                "accuracy",
                "correctly solved percentage",
                "avg solved len",
                "total evals",
            ]
            if str(i) in data_by_big_row.keys():
                for model_key, metric_dict in data_by_big_row[
                    str(i)
                ].items():  # panels is a list of panel items
                    column_name = " ".join(model_key.split("_"))
                    if metric_dict["total_evals"] < 100:
                        metric_dict["accuracy"] = None
                        metric_dict["correctly_solved_percentage"] = None
                        metric_dict["avg_solved_len"] = None
                        metric_dict["total_evals"] = None
                        continue

                    big_row[column_name] = []
                    big_row[column_name].append(metric_dict["accuracy"])
                    big_row[column_name].append(
                        metric_dict["correctly_solved_percentage"]
                    )
                    big_row[column_name].append(metric_dict["avg_solved_len"])
                    big_row[column_name].append(metric_dict["total_evals"])

            # turn big_row into pandas data frame:
            big_row_df = pd.DataFrame(big_row, index=big_row_indexes)
            dfs[str(i)] = big_row_df

        combined_df = pd.concat(
            dfs, keys=dfs.keys(), names=["Length Number", "Eval Metric"]
        )
        return combined_df


# %%
def custom_error_bar_fn(x):
    two_mean_minus_std = x.max()
    std = x.min()
    mean = (two_mean_minus_std + std) / 2

    return (mean - 2 * std, mean + 2 * std)


if __name__ == "__main__":
    analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project="transcendence-Eval-Full"
    )
    # tactics_analyzer = TranscendenceDataAnalyzer(
    #     wandb_entity="project-eval", wandb_project="tactics_eval"
    # )
    # analyzer = TranscendenceDataAnalyzer(
    #     wandb_entity="project-eval", wandb_project="50M-Training"
    # )
    # https://wandb.ai/project-eval/770-Testing-Eval/runs/2vev6jt5/overview?nw=nwuserezipe 770-High-Elo-2000-Eval at April 28, 4:32pm
    # table = analyzer.get_table("r5gi54js")

    def plot_trajectory_lineplot(groupby, x_label, y_label, data):
        analyzer.visualize_one_lineplot_groupby(
            f"ChessFormer Training Trajectories",
            x_label,
            y_label,
            groupby,
            data,
            x_label=x_label,
            y_label=y_label,
            x_ticks_by_data=True,
            # custom_error_bar_fn=custom_error_bar_fn,
        )

    def temperature_sampling_experiment(groupby, y_label, data):
        analyzer.visualize_lineplot_groupby(
            f"{y_label}s of ChessFormer across Temperature",
            "Temperature",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
            # custom_error_bar_fn=custom_error_bar_fn,
        )

    def game_length_sampling_experiment(groupby, y_label, data, trained_game_lengths):
        analyzer.visualize_lineplot_groupby_with_dotted_line(
            f"{y_label}s of NanoGPT across Game Lengths",
            "Starting_Moves",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
            dotted_line_x=trained_game_lengths,
            custom_error_bar_fn=None,
        )

    def win_condition_experiment(groupby, y_label, data, category, plot_num):
        title = (
            f"{y_label}s of NanoGPT Win Conditioning 1000-1500"
            if plot_num == 1
            else f"{y_label}s of NanoGPT Win Conditioning 1600-2100"
        )
        analyzer.visualize_barplot_groupby(
            title,
            category,
            y_label,
            groupby,
            pd.DataFrame(data),
            x_label="Trained High Elos",
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
        )

    def probability_distribution_of_moves_experiment(groupby, y_label, data):
        title = "Move Probability Distribution by Temperature"
        analyzer.visualize_three_histogram_plot_groupby(
            title,
            y_label,
            groupby,
            pd.DataFrame(data),
            x_label="",
            y_label=f"",
            x_ticks_by_data=True,
        )

    def func(groupby, y_label, data):
        title = "Simple Probability Plot"
        analyzer.visualize_histplot_groupby(
            title,
            y_label,
            groupby,
            pd.DataFrame(data),
            x_label="",
            y_label=f"",
            x_ticks_by_data=True,
        )

    def plot_game_trajectory():
        game_trajectories = torch.load("game_trajectories.pt")
        groupby = "Game Number"
        x_label = "Move Number"
        y_label = "States"
        plot_trajectory_lineplot(groupby, x_label, y_label, game_trajectories)

    # plot_game_trajectory()

    def plot_glicko_across_4_elos():
        runs_in_project_per_high_elo = {
            # "1000": ["c6e7g14h"],
            # "1100": ["fygd29kg"],
            # "1300": ["cetrd1vi"],
            # "1500": ["3j84hvqe"],
            "1000": ["0dlifhse"],
            # "1100": ["v6dn5uw3"],
            "1300": ["pkr0nh7n"],
            "1500": ["0osbdtn4"],
        }
        sample_data = []

        if os.path.exists("cached_dfs_by_4_high_elos_full.pkl"):
            dfs_by_high_elo = pickle.load(
                open("cached_dfs_by_4_high_elos_full.pkl", "rb")
            )
        else:
            dfs_by_high_elo = {}
            for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
                dfs = []
                for run_id in runs_in_project_per_high_elo[high_elo]:
                    dfs += analyzer.get_table_for_iter_num(run_id, 100000)

                dfs_by_high_elo[high_elo] = dfs
            with open("cached_dfs_by_4_high_elos_full.pkl", "wb") as fin:
                pickle.dump(dfs_by_high_elo, fin)

        for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
            dfs = dfs_by_high_elo[high_elo]
            df_by_temp = {}
            for df in dfs:
                temp = df.loc[0, "temperature"]
                if (
                    str(temp) in df_by_temp
                ):  # temperture is the same for all rows in a table (df)
                    df_by_temp[str(df.loc[0, "temperature"])] = pd.concat(
                        [df_by_temp[str(df.loc[0, "temperature"])], df]
                    )
                else:
                    df_by_temp[str(df.loc[0, "temperature"])] = df

            df_by_temp_sorted = dict(sorted(df_by_temp.items()))
            for temperature, df in df_by_temp_sorted.items():
                (glicko_elo, dev) = analyzer.fetch_and_process(df)
                sample_data += [
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo,
                    },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo - dev,
                    },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo + dev,
                    },  # super hacky
                ]

        groupby = "High_Elo"
        y_label = "Rating"
        temperature_sampling_experiment(groupby, y_label, sample_data)

    # plot_glicko_across_4_elos()

    def plot_glicko_across_6_elos():
        runs_in_project_per_high_elo = {
            "1000": ["t2uz57fr", "akbd8k2v"],
            # "1100": ["jfwepvp6", "2nlzvq3l"],
            # "1200": ["asqcagmw", "6cuuhiff"],
            "1300": ["u56krymr", "ar2ta0cs"],
            # "1400": ["lhyji88c", "zgyf6ngk"],
            "1500": ["z1xdz9c9", "zyqaxg1h"],
        }
        sample_data = []

        if os.path.exists("cached_dfs_by_high_elo.pkl"):
            dfs_by_high_elo = pickle.load(open("cached_dfs_by_high_elo.pkl", "rb"))
        else:
            dfs_by_high_elo = {}
            for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
                dfs = []  # all df per high elo
                # TODO: Use pickle, if path does not exist, compute, else pickle.load into dfs
                for run_id in runs_in_project_per_high_elo[high_elo]:
                    dfs += analyzer.get_table(run_id)

                dfs_by_high_elo[high_elo] = dfs
            with open("cached_dfs_by_high_elo.pkl", "wb") as fin:
                pickle.dump(dfs_by_high_elo, fin)

        for high_elo in runs_in_project_per_high_elo:  # iterate over the keys
            # TODO: Aggregate the skill levels for this high_elo
            # TODO: Combine tables with the same temperatures. Then calc the glicko_elo based on all games of that temperature
            #   currently, there are 16 tables per high_elo (2 for each temp) --> need to become 8 (1 for each temp)
            dfs = dfs_by_high_elo[high_elo]
            df_by_temp = {}
            for df in dfs:
                temp = df.loc[0, "temperature"]
                # print(type(temp))
                # print(temp)
                if (
                    str(temp) in df_by_temp
                ):  # temperture is the same for all rows in a table (df)
                    x = df_by_temp[str(df.loc[0, "temperature"])]
                    df_by_temp[str(df.loc[0, "temperature"])] = pd.concat(
                        [df_by_temp[str(df.loc[0, "temperature"])], df]
                    )
                    # print(type(x))
                    # print("Hello: ", len(df_by_temp[str(df.loc[0, 'temperature'])]))
                else:
                    df_by_temp[str(df.loc[0, "temperature"])] = df
                    # print(len(df_by_temp[str(df.loc[0, 'temperature'])]))
                # print(len(df_by_temp.keys()))

            # TODO: sort the df_by_temp because there are duplicate 0.5, 0.75, 1.0, 1.5 in the beginning
            df_by_temp_sorted = dict(sorted(df_by_temp.items()))
            for temperature, df in df_by_temp_sorted.items():
                # print(type(df))
                (glicko_elo, dev) = analyzer.fetch_and_process(df)
                sample_data += [
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo,
                    },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo - dev,
                    },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": glicko_elo + dev,
                    },  # super hacky
                ]

        groupby = "High_Elo"  # Key for the temperature group
        y_label = "Rating"
        temperature_sampling_experiment(groupby, y_label, sample_data)

    plot_glicko_across_6_elos()

    def plot_glicko_by_num_starting_moves():
        # runs_in_project_per_trained_game_lengths = { # 0 --> 35
        #     30: ["w05vky0w"],
        #     35: ["0dw332co"],
        #     40: ["cnah42me"],
        #     45: ["ycjjlm6e"],
        #     50: ["xqz2weko"],
        #     55: ["1zbajuca"],
        #     60: ["9obr37lb"],
        #     65: ["80o97opc"],
        # }
        # runs_in_project_per_trained_game_lengths = { # 0 --> 85
        #     30: ["gcwzltax"],
        #     35: ["8i0jrd8y"],
        #     40: ["y5r6vsct"],
        #     45: ["lebz9wrg"],
        #     50: ["drl2mdne"],
        #     55: ["wpdmm2h3"],
        #     60: ["vuu6r467"],
        #     65: ["86h60ypq"],
        # }

        # runs_in_project_per_trained_game_lengths = { # 0 --> 85 && game_lengths 40 - 95 && Positional Embeddings && W/Random Moves
        #     30: ["rcy1kfrb", "kszox6pb"],
        #     35: ["kcy3w25y", "mlwz4rt2"],
        #     40: ["gve4pr73", "2ilyv5g3"],
        #     45: ["pdjdaqf7", "wxhy999z"],
        #     50: ["0teq0qtk", "w75wrjx2"],
        #     55: ["temt8epb", "ld8louk1"],
        #     60: ["nm7d93u4", "ssva9orf"],
        #     65: ["gbx2lyb5", "clk6hzmw"],
        # }

        runs_in_project_per_trained_game_lengths = (
            {  # 0 --> 85 && Positional Embeddings && W/O Random Moves
                30: ["zbf7anqn"],
                35: ["1zk2yzln"],
                40: ["eailn4yv"],
                45: ["v6rw4ulb"],
                50: ["jckzgdqu"],
                55: ["sxr7bmdv"],
                60: ["bntx63sl"],
                65: ["6o4ihay1"],
            }
        )

        sample_data = []
        for (
            trained_game_length,
            run_ids,
        ) in runs_in_project_per_trained_game_lengths.items():
            dfs = []
            for run_id in run_ids:
                dfs += analyzer.get_table(run_id)

            for df in dfs:
                (glicko_elo, dev) = analyzer.fetch_and_process(df)
                game_len = df.loc[0, "game_total_moves"]
                print("Game Length: ", game_len)
                num_start_moves = df.loc[0, "num_start_moves"]
                print("Number of Start Moves: ", num_start_moves)

                groupby = "trained_game_length"  # Key for the temperature group
                y_label = "Rating"
                sample_data += [
                    {
                        "Starting_Moves": num_start_moves,
                        "trained_game_length": trained_game_length,
                        # "Rating": 2 * glicko_elo - dev,
                        "Rating": glicko_elo,
                    },
                    {
                        "Starting_Moves": num_start_moves,
                        "trained_game_length": trained_game_length,
                        "Rating": glicko_elo - dev,
                        # "Rating": dev,
                    },  # super hacky
                    {
                        "Starting_Moves": num_start_moves,
                        "Low_Elo": trained_game_length,
                        "Rating": glicko_elo + dev,
                    },  # super hacky
                ]

        game_length_sampling_experiment(
            groupby,
            y_label,
            sample_data,
            runs_in_project_per_trained_game_lengths.keys(),
        )

    # plot_glicko_by_num_starting_moves()

    win_conditions_runs_per_high_elo_1 = {
        1000: ["13ynfpra"],
        1100: ["67v6rzfc"],
        1200: ["cvgpd24s"],
        1300: ["lbizo04p"],
        1400: ["60je9jab"],
        1500: ["vefofn98"],
    }
    win_conditions_runs_per_high_elo_2 = {
        1600: ["1esbn5yp"],
        1700: ["ypemclgv"],
        1800: ["bdd2k2ia"],
        1900: ["oxgy04r0"],
        2000: ["lwlloyu0"],
        2100: ["5hf9k0ny"],
    }
    no_win_conditions_runs_per_high_elo_1 = {
        1000: ["t2uz57fr", "akbd8k2v"],
        1100: ["jfwepvp6", "2nlzvq3l"],
        1200: ["asqcagmw", "6cuuhiff"],
        1300: ["u56krymr", "ar2ta0cs"],
        1400: ["lhyji88c", "zgyf6ngk"],
        1500: ["z1xdz9c9", "zyqaxg1h"],
    }
    no_win_conditions_runs_per_high_elo_2 = {
        # 1600: ["fc880mfe"],
        # 1700: ["rwcsa409"],
        # 1800: ["cydu8n2q"],
        # 1900: ["emtr89mq"],
        # 2000: ["8za606nb"],
        # 2100: ["j52idlnk"],
        1600: ["l6d4aysn"],
        1700: ["fi06dre7"],
        1800: ["2paeibpl"],
        1900: ["6imrtoj2"],
        2000: ["dbvnpraj"],
        2100: ["d4jwtk3h"],
    }

    def plot_win_conditioning(
        win_conditions_runs_per_high_elo, no_win_conditions_runs_per_high_elo, plot_num
    ):
        sample_data = []
        for elo, run_ids in win_conditions_runs_per_high_elo.items():
            dfs = []
            for run_id in run_ids:
                dfs += analyzer.get_table(run_id)

            for df in dfs:
                # df = df.drop(df[df['temperature'] != 0.001].index)
                (glicko_elo, dev) = analyzer.fetch_and_process(df)

                groupby = "High_Elo"
                category = "Condition"
                y_label = "Rating"
                sample_data += [
                    {
                        "High_Elo": elo,
                        "Rating": glicko_elo,
                        "Condition": "Win_Conditioning",
                    },
                    {
                        "High_Elo": elo,
                        "Rating": glicko_elo - dev,
                        "Condition": "Win_Conditioning",
                    },
                    {
                        "High_Elo": elo,
                        "Rating": glicko_elo + dev,
                        "Condition": "Win_Conditioning",
                    },
                ]

        for elo, run_ids in no_win_conditions_runs_per_high_elo.items():
            dfs = []
            for run_id in run_ids:
                dfs += analyzer.get_table(run_id)

            desired_dfs = []
            for df in dfs:
                row = df.iloc[0]
                if row["temperature"] == 0.001:
                    desired_dfs.append(df)

            df = pd.concat(desired_dfs)
            # print(f"Elo: {elo}\n", df)
            if df.shape[0] == 0:
                continue

            (glicko_elo, dev) = analyzer.fetch_and_process(df)

            groupby = "High_Elo"
            category = "Condition"
            y_label = "Rating"
            sample_data += [
                {
                    "High_Elo": elo,
                    "Rating": glicko_elo,
                    "Condition": "No_Win_Conditioning",
                },
                {
                    "High_Elo": elo,
                    "Rating": glicko_elo - dev,
                    "Condition": "No_Win_Conditioning",
                },
                {
                    "High_Elo": elo,
                    "Rating": glicko_elo + dev,
                    "Condition": "No_Win_Conditioning",
                },
            ]

        # print("Sample Data:", sample_data)

        win_condition_experiment(groupby, y_label, sample_data, category, plot_num)

    # plot_win_conditioning(win_conditions_runs_per_high_elo_1, no_win_conditions_runs_per_high_elo_1, 1)
    # plot_win_conditioning(win_conditions_runs_per_high_elo_2, no_win_conditions_runs_per_high_elo_2, 2)

    def record_tactics_eval_in_table():
        tactics_eval_runs = {
            "fork_sacrifice": ["k93zmql1"],
            "attraction_fork": ["1y801btw"],
            "attraction_sacrifice": ["wx1z3l5w"],
            "kingsideAttack_sacrifice": ["jvzjjlhm"],
        }
        sample_data = []
        for data_key, run_ids in tactics_eval_runs.items():
            for run_id in run_ids:
                panels_by_ckpt = tactics_analyzer.get_all_charts_sort_by_ckpt(run_id)

            for ckpt_iter_num in [20000, 60000, 100000, 120000]:
                full_df = tactics_analyzer.get_df_from_panels_per_ckpt(
                    panels_by_ckpt["tactics_eval"][str(ckpt_iter_num)]
                )
                print(full_df)
                latex_table = full_df.to_latex(
                    index=True,  # To not include the DataFrame index as a column in the table
                    caption=f"Comparison of Tactics Model Performance Metrics (ckpt {ckpt_iter_num}.pt)",  # The caption to appear above the table in the LaTeX document
                    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
                    # position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
                    column_format="|c|c|c|c|",  # The format of the columns: center-aligned with vertical lines between them
                    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
                    float_format="{:0.2f}".format,  # Formats floats to two decimal places
                )
                # latex_customized = combined_df.to_latex(
                #     index=True,  # Include the index in the output
                #     caption="Concatenated DataFrame",
                #     label="tab:concatenated_dataframe",
                #     column_format="|c|c|c|",  # For three columns with vertical lines
                #     escape=False  # Keep special characters as is
                # )
                with open(f"{data_key}_ckpt_{ckpt_iter_num}.tex", "w") as f:
                    f.write(latex_table)
                print(f"ckpt_{ckpt_iter_num}.pt:")
                print(latex_table)

    # record_tactics_eval_in_table()

    def plot_move_probabilities_by_temperature():
        groupby = "Pecentage"
        y_label = "Probability"

        sample_data = [
            # {groupby: "60%", y_label: 60},
            # {groupby: "40%", y_label: 40},
            # {groupby: "0%", y_label: 0},
            {groupby: "5%", y_label: 5},
            {groupby: "90%", y_label: 90},
            {groupby: "5%", y_label: 5},
            # {groupby: "30%", y_label: 30},
            # {groupby: "40%", y_label: 40},
            # {groupby: "30", y_label: 30},
        ]
        func(groupby, y_label, sample_data)

        # 1.0
        # h5f4 0.3400000000000001
        # f8e8 0.13999999999999996
        # c6g6 0.10000000000000002
        # d4g4 0.09000000000000002
        # 0.75
        # h5f4 0.36
        # f8e8 0.22000000000000003
        # c6g6 0.13
        # d4g4 0.030000000000000006
        # 0.001
        # h5f4 0.0
        # f8e8 1.0
        # c6g6 0.0
        # d4g4 0.0
        # sample_data = [
        #     {"Temperature": "1.0", groupby: "f8e8", y_label: 0.3400000000000001},
        #     {"Temperature": "1.0", groupby: "h5f4", y_label: 0.13999999999999996},
        #     {"Temperature": "1.0", groupby: "c6g6", y_label: 0.10000000000000002},
        #     {"Temperature": "1.0", groupby: "d4g4", y_label: 0.09000000000000002},
        #     {"Temperature": "0.75", groupby: "f8e8", y_label: 0.36},
        #     {"Temperature": "0.75", groupby: "h5f4", y_label: 0.22000000000000003},
        #     {"Temperature": "0.75", groupby: "c6g6", y_label: 0.13},
        #     {"Temperature": "0.75", groupby: "d4g4", y_label: 0.030000000000000006},
        #     {"Temperature": "0.001", groupby: "f8e8", y_label: 1.0},
        #     {"Temperature": "0.001", groupby: "h5f4", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "c6g6", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "d4g4", y_label: 0.0},
        # ]
        # probability_distribution_of_moves_experiment(groupby, y_label, sample_data)

        # {'pgn': '[White "Magnus Carlsen"]\n[Black "Stockfish"]\n\n1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. d3 Bc5 5. h3 d6 6. a3 Bf5 7. g3 O-O 8. b4 Bd4 9. Bb2 Bxc3 10. Bxc3 e4 11. b5 Ne7 12. Nh4 exd3 13. Bxf6 gxf6 14. Nxf5 Nxf5 15. e3 d5 16. Bxd3 dxc4 17. Bxf5 Qxd1 18. Rxd1 Rad8 19. Bc2 Rxd1 20. Bxd1 Rd8 21. Bc2 c3 22. O-O Rd2 23. Be4 c2 24. Kg2 Rd1 25. Bxc2 Rxf1 26. Kxf1 c6 27. Bd1 cxb5 28. f4 a5 29. g4 b4 30. a4 b3 31. Bxb3 b6 32. Kg1 b5 33. h4 bxa4 34. Ba2 a3 35. g5 fxg5 36. fxg5 Kg7 37. Kh2 f6 38. Kg3 fxg5 39. e4 gxh4 40. Kh3 Kg6 41. Bg8 Kg5 42. Bf7 h6 43. Bd5 Kf4 44. Kh2 Ke3 45. e5 Kd4 46. Ba2 Kxe5 47. Kh1 Kd4 48. Kg2 Kc3 49. Bd5 Kb2 50. Ba8 a2 51. Be4 a1=Q 52. Kh2 Qa2 53. Bh1 Ka3 54. Kh3 Qb3 55. Kxh4 Qb4 56. Kh3 a4 57. Be4 Kb3 58. Bb1 a3 59. Ba2+ Kxa2 60. Kg3 Kb3 61. Kh2 a2 62. Kg3 a1=Q 63. Kf3 Qaa3 64. Kg2 Ka4 65. Kh1 Qbb2 66. Kg1 Qaa1#', 'dist': {'0.001': Counter({'b3a4': 100}), '0.75': Counter({'b3a4': 24, None: 22, 'b3c4': 15, 'b3a2': 13, 'b3c2': 11, 'b3b2': 8, 'a3a2': 3, 'h6h5': 2, 'a3b2': 1, 'a3a1': 1}), '1': Counter({None: 35, 'b3a2': 15, 'b3c4': 10, 'b3c2': 10, 'b3a4': 9, 'b3b2': 5, 'h6h5': 4, 'b3c3': 3, 'a3b2': 2, 'a3a2': 2, 'b4d2': 2, 'b4a4': 1, 'b4c3': 1, 'b4d4': 1})}, 'half_move_clock': 127, 'advantages': {'a3a1': (0.999, 99.97), 'a3c1': (0.999, 99.97), 'b4g4': (0.999, 99.97), 'b4f4': (0.999, 99.97), 'a3a6': (0.999, 99.97), 'b3a2': (0.999, 99.97), 'b3b2': (0.999, 99.97), 'b3a4': (0.999, 99.97), 'b4c3': (0.999, 99.97), 'b4e4': (0.999, 99.97), 'b3c4': (0.999, 99.97), 'b3c2': (0.999, 99.97), 'a3a8': (0.998, 99.96), 'h6h5': (0.998, 99.96), 'b4h4': (0.998, 99.96), 'b4a4': (0.998, 99.96), 'b4d4': (0.998, 99.96), 'a3a4': (0.998, 99.96), 'a3a5': (0.998, 99.96), 'b3c3': (0.998, 99.96), 'a3a7': (0.998, 99.96), 'b4d6': (0.998, 99.96), 'b4c4': (0.998, 99.96), 'b4f8': (0.998, 99.96), 'b4d2': (0.998, 99.96), 'a3b2': (0.998, 99.96), 'b4b7': (0.998, 99.96), 'b4a5': (0.998, 99.96), 'b4c5': (0.998, 99.96), 'a3a2': (0.998, 99.96), 'b4e7': (0.998, 99.96), 'b4b5': (0.998, 99.96), 'b4e1': (0.998, 99.96), 'b4b8': (0.998, 99.96), 'b4b6': (0.998, 99.96)}}
        # b3a2 0.0
        # b3c4 1.0
        # b3c2 0.0
        # b3a2 0.13
        # b3c4 0.15
        # b3c2 0.11000000000000001
        # b3a2 0.15
        # b3c4 0.10000000000000002
        # b3c2 0.10000000000000002
        # sample_data = [
        #     {"Temperature": "1.0", groupby: "b3c4", y_label: 0.10000000000000002},
        #     {"Temperature": "1.0", groupby: "b3a2", y_label: 0.15},
        #     {"Temperature": "1.0", groupby: "b3c2", y_label: 0.10000000000000002},
        #     {"Temperature": "0.75", groupby: "b3c4", y_label: 0.15},
        #     {"Temperature": "0.75", groupby: "b3a2", y_label: 0.13},
        #     {"Temperature": "0.75", groupby: "b3c2", y_label: 0.11000000000000001},
        #     {"Temperature": "0.001", groupby: "b3c4", y_label: 1.0},
        #     {"Temperature": "0.001", groupby: "b3a2", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "b3c2", y_label: 0.0},
        # ]
        # probability_distribution_of_moves_experiment(groupby, y_label, sample_data)

        # {'pgn': '[White "Magnus Carlsen"]\n[Black "Stockfish"]\n\n1. c4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. d3 Bc5 5. e3 d6 6. Be2 Bf5 7. a3 O-O 8. d4 exd4 9. exd4 Bxd4 10. Nxd4 Nxd4 11. Qxd4 c5 12. Qf4 Qd7 13. h4 Rae8 14. Qd2 Ne4 15. Nxe4 Bxe4 16. f3 Bf5 17. Qc3 Re7 18. Kf1 Rfe8 19. Bd1 Re1 20. Qxe1 Rxe1 21. Kxe1 Qe7 22. Kf1 Bd3 23. Kg1 Qe1 24. Kh2 Qxh1 25. Kxh1 Bxc4 26. Bc2 Bd5 27. Bd2 Bxf3 28. Bxh7+ Kxh7 29. b4 Bxg2 30. Kxg2 cxb4 31. Bxb4 d5 32. Kg3 d4 33. Kf4 d3 34. a4 d2 35. Bxd2 b5 36. h5 bxa4 37. Kg3 a3 38. Rxa3 a5 39. Kh4 a4 40. Bf4 f5 41. Bd2 f4 42. Bxf4 g5 43. Kxg5 Kg7 44. Rxa4 Kf7 45. Kf5 Ke7 46. Ra7+ Kd8 47. Rc7 Ke8 48. Ke6 Kf8 49. Kf5 Kg8 50. h6 Kh8 51. Kg6 Kg8 52. Rc8#', 'dist': {'0.001': Counter({'e1h4': 100}), '0.75': Counter({'e1h4': 33, 'e1e5': 25, 'e1d1': 18, 'e1h1': 15, 'e1f2': 4, 'g7g5': 2, None: 1, 'd3c4': 1, 'f7f5': 1}), '1': Counter({'e1e5': 31, 'e1h4': 21, 'e1d1': 16, 'e1h1': 8, 'e1f2': 7, 'd3c4': 5, None: 3, 'd3e2': 2, 'g7g5': 2, 'e1g3': 1, 'h7h6': 1, 'd6d5': 1, 'e1e6': 1, 'g7g6': 1})}, 'half_move_clock': 47, 'advantages': {'e1h4': (0.694, 2.22), 'e1f2': (0.624, 1.38), 'd3f1': (0.5, 0.0), 'e1e5': (0.334, -1.87), 'e1e6': (0.313, -2.14), 'e1e7': (0.284, -2.51), 'e1a5': (0.275, -2.63), 'e1e8': (0.26, -2.84), 'e1h1': (0.117, -5.48), 'e1f1': (0.104, -5.84), 'e1d1': (0.083, -6.53), 'g7g5': (0.071, -6.98), 'e1g3': (0.07, -7.03), 'e1d2': (0.069, -7.06), 'e1c3': (0.069, -7.08), 'e1e2': (0.067, -7.16), 'b7b5': (0.066, -7.18), 'e1e4': (0.066, -7.18), 'e1b4': (0.066, -7.2), 'h7h6': (0.066, -7.2), 'd3f5': (0.065, -7.22), 'a7a6': (0.065, -7.24), 'e1g1': (0.065, -7.25), 'e1e3': (0.064, -7.27), 'f7f6': (0.064, -7.28), 'd3e4': (0.064, -7.29), 'h7h5': (0.064, -7.29), 'd3c4': (0.063, -7.33), 'f7f5': (0.063, -7.34), 'b7b6': (0.063, -7.35), 'd6d5': (0.062, -7.4), 'a7a5': (0.062, -7.4), 'g8f8': (0.062, -7.37), 'd3e2': (0.062, -7.39), 'd3b1': (0.061, -7.41), 'd3g6': (0.061, -7.42), 'g7g6': (0.061, -7.42), 'g8h8': (0.058, -7.57), 'd3c2': (0.053, -7.82)}}
        # e1e5 0.0
        # e1h4 1.0
        # e1d1 0.0
        # e1h1 0.0
        # 0.334 #4b3a8f
        # 0.694 #80007f
        # 0.083 #2a5f99
        # 0.117 #2b5e99
        # e1e5 0.24999999999999997
        # e1h4 0.33
        # e1d1 0.17999999999999997
        # e1h1 0.15
        # 0.334 #4b3a8f
        # 0.694 #80007f
        # 0.083 #2a5f99
        # 0.117 #2b5e99
        # e1e5 0.31
        # e1h4 0.21
        # e1d1 0.15999999999999998
        # e1h1 0.07999999999999999
        # 0.334 #4b3a8f
        # 0.694 #80007f
        # 0.083 #2a5f99
        # 0.117 #2b5e99
        # sample_data = [
        #     {"Temperature": "1.0", groupby: "e1h4", y_label: 0.21},
        #     {"Temperature": "1.0", groupby: "e1e5", y_label: 0.31},
        #     {"Temperature": "1.0", groupby: "e1d1", y_label: 0.15999999999999998},
        #     {"Temperature": "1.0", groupby: "e1h1", y_label: 0.07999999999999999},
        #     {"Temperature": "0.75", groupby: "e1h4", y_label: 0.33},
        #     {"Temperature": "0.75", groupby: "e1e5", y_label: 0.24999999999999997},
        #     {"Temperature": "0.75", groupby: "e1d1", y_label: 0.17999999999999997},
        #     {"Temperature": "0.75", groupby: "e1h1", y_label: 0.15},
        #     {"Temperature": "0.001", groupby: "e1h4", y_label: 1.0},
        #     {"Temperature": "0.001", groupby: "e1e5", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "e1d1", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "e1h1", y_label: 0.0},
        # ]
        # probability_distribution_of_moves_experiment(groupby, y_label, sample_data)

        # {'pgn': '[White "Magnus Carlsen"]\n[Black "Stockfish"]\n\n1. c4 e5 2. g3 Nf6 3. Nc3 Bc5 4. d3 Ng4 5. e3 Qf6 6. Qxg4 d6 7. Qxc8+ Qd8 8. Qxd8+ Kxd8 9. a3 Nc6 10. Bg2 Na5 11. Rb1 Nb3 12. Nf3 f6 13. Nd2 Nxd2 14. Kxd2 Re8 15. a4 a5 16. Bh3 Bb4 17. Ra1 c5 18. Ke2 g5 19. Bg2 h5 20. h3 h4 21. Nd5 hxg3 22. Kd1 gxf2 23. Bf1 Rh8 24. e4 Rh4 25. Be3 g4 26. Bg2 gxh3 27. Bf1 h2 28. b3 Rg4 29. Ke2 Rg1 30. Nxb4 Rxh1 31. Na2 Rg1 32. Nc3 h1=Q 33. Rc1 Rxf1 34. Bg5 Rxc1 35. Ke3 f1=Q 36. Ne2 Qxe2# 37. Kxe2', 'dist': {'0.001': Counter({'h1g1': 100}), '0.75': Counter({'h1g1': 48, 'h1f1': 21, None: 13, 'd8c7': 6, 'd8d7': 6, 'f6f5': 2, 'b7b6': 2, 'a8c8': 1, 'd8e7': 1}), '1': Counter({'h1g1': 33, None: 25, 'h1f1': 20, 'f6f5': 7, 'd8e7': 6, 'd8d7': 5, 'd8c7': 2, 'b7b6': 1, 'd8e8': 1})}, 'half_move_clock': 61, 'advantages': {'h1g1': (0.809, 3.92), 'd8c7': (0.781, 3.45), 'd8e7': (0.753, 3.03), 'a8c8': (0.75, 2.99), 'd8e8': (0.743, 2.89), 'a8a6': (0.719, 2.55), 'a8b8': (0.707, 2.39), 'b7b6': (0.7, 2.3), 'a8a7': (0.657, 1.76), 'f6f5': (0.628, 1.42), 'b7b5': (0.621, 1.34), 'd6d5': (0.603, 1.13), 'h1f1': (0.163, -4.44), 'd8d7': (0.088, -6.36), 'd8c8': (0.086, -6.43)}}
        # h1g1 1.0
        # h1f1 0.0
        # f6f5 0.0
        # 0.809 #8f0070
        # 0.163 #2e5a98
        # 0.628 #7e0280
        # h1g1 0.4800000000000001
        # h1f1 0.21
        # f6f5 0.02
        # 0.809 #8f0070
        # 0.163 #2e5a98
        # 0.628 #7e0280
        # h1g1 0.33
        # h1f1 0.19999999999999996
        # f6f5 0.06999999999999999
        # 0.809 #8f0070
        # 0.163 #2e5a98
        # 0.628 #7e0280
        # sample_data = [
        #     {"Temperature": "1.0", groupby: "h1g1", y_label: 0.33},
        #     {"Temperature": "1.0", groupby: "h1f1", y_label: 0.19999999999999996},
        #     {"Temperature": "1.0", groupby: "f6f5", y_label: 0.06999999999999999},

        #     {"Temperature": "0.75", groupby: "h1g1", y_label: 0.4800000000000001},
        #     {"Temperature": "0.75", groupby: "h1f1", y_label: 0.21},
        #     {"Temperature": "0.75", groupby: "f6f5", y_label: 0.02},

        #     {"Temperature": "0.001", groupby: "h1g1", y_label: 1.0},
        #     {"Temperature": "0.001", groupby: "h1f1", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "f6f5", y_label: 0.0},

        # ]
        # probability_distribution_of_moves_experiment(groupby, y_label, sample_data)

        # {'pgn': '[White "Magnus Carlsen"]\n[Black "Stockfish"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. Qe2 Bc5 5. Bxc6 dxc6 6. a4 O-O 7. h3 Be6 8. O-O Qd6 9. a5 a6 10. Rd1 Rad8 11. c3 b5 12. axb6 cxb6 13. Re1 a5 14. Na3 b5 15. Nc2 a4 16. d4 exd4 17. e5 Qd5 18. exf6 gxf6 19. Nfxd4 Bxd4 20. Rd1 Qc5 21. Rxd4 Rxd4 22. Be3 Re4 23. Qf1 Qe5 24. Bd4 Qf4 25. Qd1 Rfe8 26. f3 Re2 27. Qxe2 Bxh3 28. Qxe8+ Kg7 29. Ne3 Qg3 30. Qe7 Qxg2 31. Nxg2 Bxg2 32. Qxf6+ Kf8 33. Qd6+ Kg8 34. Qd8#', 'dist': {'0.001': Counter({'d4e4': 100}), '0.75': Counter({'d4e4': 29, 'd4d5': 17, 'd4d8': 11, 'f8d8': 10, 'e6c4': 7, 'd4c4': 5, 'd4d6': 5, 'c5d5': 3, 'c5e5': 3, 'c5f5': 1, 'b5b4': 1, 'd4h4': 1, 'c5c4': 1, 'd4d1': 1, 'a4a3': 1, None: 1, 'f8e8': 1, 'e6b3': 1, 'd4d7': 1}), '1': Counter({'d4e4': 25, 'f8d8': 11, None: 9, 'e6c4': 7, 'd4d8': 7, 'd4d5': 5, 'd4c4': 5, 'c5d5': 4, 'c5e5': 4, 'd4g4': 3, 'd4d1': 3, 'c5c4': 3, 'e6b3': 3, 'e6h3': 2, 'd4b4': 1, 'd4d3': 1, 'f6f5': 1, 'd4h4': 1, 'c5f5': 1, 'b5b4': 1, 'c5h5': 1, 'd4d6': 1, 'f8e8': 1})}, 'half_move_clock': 43, 'advantages': {'d4e4': (0.643, 1.6), 'a4a3': (0.187, -4.0), 'd4d2': (0.169, -4.32), 'f8e8': (0.169, -4.32), 'f8d8': (0.167, -4.36), 'c5e5': (0.167, -4.37), 'c5d5': (0.166, -4.39), 'g8g7': (0.166, -4.39), 'c5c4': (0.163, -4.44), 'e6d7': (0.161, -4.48), 'd4d1': (0.16, -4.5), 'f8b8': (0.159, -4.52), 'e6g4': (0.157, -4.57), 'e6c8': (0.155, -4.6), 'g8h8': (0.153, -4.64), 'c5f5': (0.153, -4.64), 'e6c4': (0.151, -4.68), 'c5d6': (0.151, -4.69), 'f8c8': (0.15, -4.71), 'h7h5': (0.149, -4.73), 'f6f5': (0.148, -4.76), 'd4c4': (0.148, -4.76), 'e6h3': (0.147, -4.78), 'd4d5': (0.145, -4.81), 'e6f5': (0.145, -4.81), 'b5b4': (0.145, -4.82), 'c5e7': (0.144, -4.84), 'f8a8': (0.141, -4.9), 'h7h6': (0.139, -4.95), 'e6a2': (0.128, -5.22), 'c5b6': (0.127, -5.23), 'c5a7': (0.12, -5.41), 'd4b4': (0.119, -5.44), 'e6d5': (0.117, -5.5), 'd4h4': (0.116, -5.51), 'd4g4': (0.115, -5.55), 'e6b3': (0.113, -5.59), 'd4f4': (0.113, -5.6), 'c5c3': (0.11, -5.68), 'd4d7': (0.109, -5.7), 'd4d3': (0.104, -5.85), 'c5g5': (0.103, -5.89), 'd4d6': (0.099, -6.01), 'd4d8': (0.098, -6.03), 'c5h5': (0.098, -6.03), 'c5a3': (0.098, -6.03), 'c5b4': (0.094, -6.16)}}
        # d4e4 1.0
        # f8d8 0.0
        # e6c4 0.0
        # 0.643 #7f0180
        # 0.167 #2f5a98
        # 0.151 #2d5b98
        # d4e4 0.29000000000000004
        # f8d8 0.10000000000000002
        # e6c4 0.06999999999999999
        # 0.643 #7f0180
        # 0.167 #2f5a98
        # 0.151 #2d5b98
        # d4e4 0.24999999999999997
        # f8d8 0.11000000000000001
        # e6c4 0.06999999999999999
        # 0.643 #7f0180
        # 0.167 #2f5a98
        # 0.151 #2d5b98
        # sample_data = [
        #     {"Temperature": "1.0", groupby: "d4e4", y_label: 0.24999999999999997},
        #     {"Temperature": "1.0", groupby: "f8d8", y_label: 0.11000000000000001},
        #     {"Temperature": "1.0", groupby: "e6c4", y_label: 0.06999999999999999},

        #     {"Temperature": "0.75", groupby: "d4e4", y_label: 0.29000000000000004},
        #     {"Temperature": "0.75", groupby: "f8d8", y_label: 0.10000000000000002},
        #     {"Temperature": "0.75", groupby: "e6c4", y_label: 0.06999999999999999},

        #     {"Temperature": "0.001", groupby: "d4e4", y_label: 1.0},
        #     {"Temperature": "0.001", groupby: "f8d8", y_label: 0.0},
        #     {"Temperature": "0.001", groupby: "e6c4", y_label: 0.0},

        # ]
        # probability_distribution_of_moves_experiment(groupby, y_label, sample_data)

    plot_move_probabilities_by_temperature()

    # sample_data = [
    #     {"Game_Length": 0, groupby: "5", y_label: 1000},
    #     {"Game_Length": 0, groupby: "15", y_label: 1200},
    #     {"Game_Length": 0, groupby: "25", y_label: 1400},
    #     {"Game_Length": 40, groupby: "5", y_label: 1100},
    #     {"Game_Length": 40, groupby: "15", y_label: 1300},
    #     {"Game_Length": 40, groupby: "25", y_label: 1500},
    #     {"Game_Length": 50, groupby: "5", y_label: 1200},
    #     {"Game_Length": 50, groupby: "15", y_label: 1400},
    #     {"Game_Length": 50, groupby: "25", y_label: 1400},
    #     {"Game_Length": 60, groupby: "5", y_label: 1300},
    #     {"Game_Length": 60, groupby: "15", y_label: 1500},
    #     {"Game_Length": 60, groupby: "25", y_label: 1700},
    # ]
    # sample_data = sum(
    #     [
    #         [
    #             {
    #                 "Game_Length": d["Game_Length"],
    #                 groupby: d[groupby],
    #                 y_label: d[y_label] + 100 * np.random.randn(),
    #             }
    #             for d in sample_data
    #         ]
    #         for _ in range(3)
    #     ],
    #     [],
    # )

    # plot_glicko_by_game_length()

    # dfs = analyzer.get_table("2vev6jt5")
    # sample_data = []
    # for df in dfs:
    #     (
    #         elo,
    #         dev,
    #         temperature,
    #         stockfish_level,
    #     ) = analyzer.fetch_and_process(df)
    #     sample_data += [
    #         {
    #             "Temperature": temperature,
    #             "Stockfish Skill Level": stockfish_level,
    #             "Rating": 2 * elo - dev,
    #         },
    #         {
    #             "Temperature": temperature,
    #             "Stockfish Skill Level": stockfish_level,
    #             "Rating": dev,
    #         },  # super hacky
    #     ]

    # groupby = "Stockfish Skill Level"  # Key for the temperature group
    # y_label = "Rating"
    # temperature_sampling_experiment(groupby, y_label, sample_data)

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

    # temperature_sampling_experiment(groupby, y_label, sample_data)

    ######################
    # Example Usage: Chess Rating by Model Size
    #######################
    # groupby = "Model Size"  # Key for the temperature group
    # y_label = "Chess Rating"

    # sample_data = [
    #     {"Temperature": 0.5, groupby: "Small", y_label: 1000},
    #     {"Temperature": 0.5, groupby: "Medium", y_label: 1200},
    #     {"Temperature": 0.5, groupby: "Large", y_label: 1400},
    #     {"Temperature": 0.2, groupby: "Small", y_label: 1100},
    #     {"Temperature": 0.2, groupby: "Medium", y_label: 1300},
    #     {"Temperature": 0.2, groupby: "Large", y_label: 1500},
    #     {"Temperature": 0.1, groupby: "Small", y_label: 1200},
    #     {"Temperature": 0.1, groupby: "Medium", y_label: 1400},
    #     {"Temperature": 0.1, groupby: "Large", y_label: 1600},
    # ]
    # sample_data = sum(
    #     [
    #         [
    #             {
    #                 "Temperature": d["Temperature"],
    #                 groupby: d[groupby],
    #                 y_label: d[y_label] + 100 * np.random.randn(),
    #             }
    #             for d in sample_data
    #         ]
    #         for _ in range(3)
    #     ],
    #     [],
    # )
    # temperature_sampling_experiment(groupby, y_label, sample_data)

# %%
