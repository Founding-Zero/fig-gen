# %%
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
from wandb.apis import PublicApi
from wandb.apis.public.artifacts import ArtifactType
from wandb.sdk import Artifact

from figgen import DataAnalyzer


class TranscendenceDataAnalyzer(DataAnalyzer):
    def fetch_and_process_skill_level_data_for_temperature(self, df: pd.DataFrame):
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

        return glicko.current_elo, glicko.current_deviation # TODO

    def get_table(self, run_id):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()

        dfs = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            # if "2000_2000" not in table_name:  #!!! HACK, need to fix
            #     continue
            if table_name == "0000.parquet": # should be the end of the list of tables
                break
            table = artifact.get(table_name)
            if table is not None:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                dfs.append(df)

        return dfs
    
    def get_table_game_length(self, run_id):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()

        dfs = {}
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            # if "2000_2000" not in table_name:  #!!! HACK, need to fix
            #     continue
            if table_name == "0000.parquet": # should be the end of the list of tables
                break
            # print(table_name)
            # game_length = table_name[-2]
            # print("GAME LENGTH: ", game_length)
            # game_length = table_name[length_index + 17 : length_index + 17 + 2]
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
        wandb_entity="project-eval", wandb_project="transcendence-Eval-Full"
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
        
    def game_length_sampling_experiment(groupby, y_label, data):
        analyzer.visualize_lineplot_groupby_with_dotted_line(
            f"{y_label}s of NanoGPT across Game Lengths",
            "Starting_Moves",
            y_label,
            groupby,
            pd.DataFrame(data),
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
            custom_error_bar_fn=None,
            dotted_line_x=10,
        )
        
    def plot_glicko_across_6_elos():
        runs_in_project = ["t2uz57fr", "akbd8k2v", "jfwepvp6", "2nlzvq3l", "asqcagmw", "6cuuhiff", "u56krymr", "ar2ta0cs", "lhyji88c", "zgyf6ngk", "z1xdz9c9", "zyqaxg1h"]
        runs_in_project_per_high_elo = {
            "1000": ["t2uz57fr", "akbd8k2v"],
            "1100": ["jfwepvp6", "2nlzvq3l"],
            "1200": ["asqcagmw", "6cuuhiff"],
            "1300": ["u56krymr", "ar2ta0cs"],
            "1400": ["lhyji88c", "zgyf6ngk"],
            "1500": ["z1xdz9c9", "zyqaxg1h"],
        }
        sample_data = []
        # df_by_temp = {
        #     0.001: pd.DataFrame(),
        #     0.01: pd.DataFrame(),
        #     0.1: pd.DataFrame(),
        #     0.3: pd.DataFrame(),
        #     0.5: pd.DataFrame(),
        #     0.75: pd.DataFrame(),
        #     1.0: pd.DataFrame(),
        #     1.5: pd.DataFrame(),
        # }
        if os.path.exists('cached_dfs_by_high_elo.pkl'):
            dfs_by_high_elo = pickle.load(open('cached_dfs_by_high_elo.pkl', 'rb'))
        else:
            dfs_by_high_elo = {}
            for high_elo in runs_in_project_per_high_elo: # iterate over the keys
                dfs = [] # all df per high elo
                #TODO: Use pickle, if path does not exist, compute, else pickle.load into dfs
                for run_id in runs_in_project_per_high_elo[high_elo]:
                    dfs += analyzer.get_table(run_id)
                    
                dfs_by_high_elo[high_elo] = dfs
            with open('cached_dfs_by_high_elo.pkl', 'wb') as fin:
                pickle.dump(dfs_by_high_elo, fin)
                

        for high_elo in runs_in_project_per_high_elo: # iterate over the keys
                # TODO: Aggregate the skill levels for this high_elo
                # TODO: Combine tables with the same temperatures. Then calc the glicko_elo based on all games of that temperature
                #   currently, there are 16 tables per high_elo (2 for each temp) --> need to become 8 (1 for each temp)
            dfs = dfs_by_high_elo[high_elo] 
            df_by_temp = {}
            for df in dfs:
                temp = df.loc[0, 'temperature']
                # print(type(temp))
                # print(temp)
                if str(temp) in df_by_temp: # temperture is the same for all rows in a table (df)
                    x = df_by_temp[str(df.loc[0, 'temperature'])]
                    df_by_temp[str(df.loc[0, 'temperature'])] = pd.concat([df_by_temp[str(df.loc[0, 'temperature'])], df]) 
                    # print(type(x))
                    # print("Hello: ", len(df_by_temp[str(df.loc[0, 'temperature'])]))
                else:
                    df_by_temp[str(df.loc[0, 'temperature'])] = df
                    # print(len(df_by_temp[str(df.loc[0, 'temperature'])]))
                # print(len(df_by_temp.keys()))
            
            # TODO: sort the df_by_temp because there are duplicate 0.5, 0.75, 1.0, 1.5 in the beginning
            df_by_temp_sorted = dict(sorted(df_by_temp.items()))
            for temperature, df in df_by_temp_sorted.items():
                # print(type(df))
                (
                    glicko_elo,
                    dev
                ) = analyzer.fetch_and_process_skill_level_data_for_temperature(df)
                sample_data += [
                    # {
                    #     "Temperature": temperature,
                    #     "High_Elo": high_elo,
                    #     "Rating": glicko_elo,
                    # },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": 2 * glicko_elo - dev,
                    },
                    {
                        "Temperature": temperature,
                        "High_Elo": high_elo,
                        "Rating": dev,
                    },  # super hacky
                ]
                    

        groupby = "High_Elo"  # Key for the temperature group
        y_label = "Rating"
        temperature_sampling_experiment(groupby, y_label, sample_data)
    # plot_glicko_across_6_elos()
    
    # def plot_glicko_by_game_length():
    runs_in_project_per_low_elo = {
        # "1300": ["baapshvl"],
        "1700": ["mppogx97"],
        "1800": ["800eydo6"],
        "1900": ["40rpczem"],
        "2000": ["jxatkohv"],
    }
    
    # for run_id in runs_in_project: # iterate over the keys
    #     dfs = analyzer.get_table(run_id)
    # dfs = analyzer.get_table("ps88jtca")
    # dfs = analyzer.get_table("ppg3yrog") #1700 Elo
    # dfs = analyzer.get_table("dfxh8i7a") #1800 Elo
    sample_data = []
    for low_elo, run_ids in runs_in_project_per_low_elo.items():
        dfs = []
        for run_id in run_ids:
            dfs += analyzer.get_table(run_id) #1900 Elo
            
        for df in dfs:
            (
                glicko_elo,
                dev
            ) = analyzer.fetch_and_process_skill_level_data_for_temperature(df)
            game_len = df.loc[0, 'game_total_moves']
            print("Game Length: ", game_len)
            num_start_moves = df.loc[0, 'num_start_moves']
            print("Number of Start Moves: ", num_start_moves)
            
            groupby = "Low_Elo"  # Key for the temperature group
            y_label = "Rating"
            sample_data += [
                {
                    "Starting_Moves": num_start_moves,
                    "Low_Elo": low_elo,
                    # "Rating": 2 * glicko_elo - dev,
                    "Rating": glicko_elo,
                },
                {
                    "Starting_Moves": num_start_moves,
                    "Low_Elo": low_elo,
                    "Rating": glicko_elo - dev,
                    # "Rating": dev,
                },  # super hacky
                {
                    "Starting_Moves": num_start_moves,
                    "Low_Elo": low_elo,
                    "Rating": glicko_elo + dev,
                },  # super hacky
            ]
    # sample_data = sum(
    #     [
    #         [
    #             {
    #                 "Starting_Moves": d["Starting_Moves"],
    #                 groupby: d[groupby],
    #                 y_label: d[y_label] + 100 * np.random.randn(),
    #             }
    #             for d in sample_data
    #         ]
    #         for _ in range(3)
    #     ],
    #     [],
    # )  
    game_length_sampling_experiment(groupby, y_label, sample_data)
        
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
    #     ) = analyzer.fetch_and_process_skill_level_data_for_temperature(df)
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
