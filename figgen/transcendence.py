# %%
import os
import pickle
import random
import json

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

        return glicko.current_elo, glicko.current_deviation # TODO

    def get_table(self, run_id):
        runs = self.get_runs([run_id])
        # run_title = runs[0].config.get("title") or runs[0].summary.get("title")
        artifacts = runs[0].logged_artifacts()
        dfs = []
        for artifact in artifacts:
            table_name = next(iter(artifact.manifest.entries))
            if table_name == "0000.parquet": # should be the end of the list of tables
                break
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
                parts = key.split('/')  # Split the path into components
                if key.startswith("tactics_eval"):
                    parts[2], parts[3] = parts[3], parts[2]
                parts = [part.strip() for part in parts if part]  # Remove any empty strings

                # Start with the top-level dictionary and iteratively go deeper
                current_level = nested_dict
                for part in parts[:-1]:  # Go up to the second last part to build keys
                    if part not in current_level:
                        current_level[part] = {}  # Create a new dictionary if the key does not exist
                    current_level = current_level[part]  # Move to the next level of the dictionary

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
        avg_big_row_indexes = ["accuracy", "correctly solved percentage", "solved length", "length"]
        for model_key, metric_dict in data_by_big_row["avg"].items(): # panels is a list of panel items
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
            big_row_indexes = ["accuracy", "correctly solved percentage", "avg solved len", "total evals"]
            if str(i) in data_by_big_row.keys():
                for model_key, metric_dict in data_by_big_row[str(i)].items(): # panels is a list of panel items
                    column_name = " ".join(model_key.split("_"))
                    if metric_dict["total_evals"] < 100:
                        metric_dict["accuracy"] = None
                        metric_dict["correctly_solved_percentage"] = None
                        metric_dict["avg_solved_len"] = None
                        metric_dict["total_evals"] = None
                        continue
                        
                    big_row[column_name] = []
                    big_row[column_name].append(metric_dict["accuracy"])
                    big_row[column_name].append(metric_dict["correctly_solved_percentage"])
                    big_row[column_name].append(metric_dict["avg_solved_len"])
                    big_row[column_name].append(metric_dict["total_evals"])
                
            # turn big_row into pandas data frame:
            big_row_df = pd.DataFrame(big_row, index=big_row_indexes)
            dfs[str(i)] = big_row_df    
        
        combined_df = pd.concat(dfs, keys=dfs.keys(), names=['Length Number', 'Eval Metric'])
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
    tactics_analyzer = TranscendenceDataAnalyzer(
        wandb_entity="project-eval", wandb_project="tactics_eval"
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
        title = f"{y_label}s of NanoGPT Win Conditioning 1000-1500" if plot_num == 1 else f"{y_label}s of NanoGPT Win Conditioning 1600-2100"
        analyzer.visualize_barplot_groupby(
            title,
            category,
            y_label,
            groupby,
            pd.DataFrame(data),
            x_label = "Trained High Elos",
            y_label=f"Chess {y_label}",
            x_ticks_by_data=True,
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
                ) = analyzer.fetch_and_process(df)
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
        
        runs_in_project_per_trained_game_lengths = { # 0 --> 85 && Positional Embeddings && W/O Random Moves
            30: ["zbf7anqn"],
            35: ["1zk2yzln"],
            40: ["eailn4yv"],
            45: ["v6rw4ulb"],
            50: ["jckzgdqu"],
            55: ["sxr7bmdv"],
            60: ["bntx63sl"],
            65: ["6o4ihay1"],
        }
        
        sample_data = []
        for trained_game_length, run_ids in runs_in_project_per_trained_game_lengths.items():
            dfs = []
            for run_id in run_ids:
                dfs += analyzer.get_table(run_id)
                
            for df in dfs:
                (
                    glicko_elo,
                    dev
                ) = analyzer.fetch_and_process(df)
                game_len = df.loc[0, 'game_total_moves']
                print("Game Length: ", game_len)
                num_start_moves = df.loc[0, 'num_start_moves']
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

        game_length_sampling_experiment(groupby, y_label, sample_data, runs_in_project_per_trained_game_lengths.keys())
        
    plot_glicko_by_num_starting_moves()
    
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

    def plot_win_conditioning(win_conditions_runs_per_high_elo, no_win_conditions_runs_per_high_elo, plot_num):
        sample_data = []
        for elo, run_ids in win_conditions_runs_per_high_elo.items():
            dfs = []
            for run_id in run_ids:
                dfs += analyzer.get_table(run_id)
                
            for df in dfs:
                # df = df.drop(df[df['temperature'] != 0.001].index)
                (
                    glicko_elo,
                    dev
                ) = analyzer.fetch_and_process(df)
                
                
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
                if row['temperature'] == 0.001:
                    desired_dfs.append(df)
                    
            df = pd.concat(desired_dfs)
            # print(f"Elo: {elo}\n", df) 
            if df.shape[0] == 0:
                continue
            
            (
                glicko_elo,
                dev
            ) = analyzer.fetch_and_process(df)
            
            
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
    
    plot_win_conditioning(win_conditions_runs_per_high_elo_1, no_win_conditions_runs_per_high_elo_1, 1)
    plot_win_conditioning(win_conditions_runs_per_high_elo_2, no_win_conditions_runs_per_high_elo_2, 2)
    
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
                full_df = tactics_analyzer.get_df_from_panels_per_ckpt(panels_by_ckpt["tactics_eval"][str(ckpt_iter_num)])
                print(full_df)
                latex_table = full_df.to_latex(
                    index=True,  # To not include the DataFrame index as a column in the table
                    caption=f"Comparison of Tactics Model Performance Metrics (ckpt {ckpt_iter_num}.pt)",  # The caption to appear above the table in the LaTeX document
                    label="tab:model_comparison",  # A label used for referencing the table within the LaTeX document
                    # position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
                    column_format="|c|c|c|c|",  # The format of the columns: center-aligned with vertical lines between them
                    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
                    float_format="{:0.2f}".format  # Formats floats to two decimal places
                )
                # latex_customized = combined_df.to_latex(
                #     index=True,  # Include the index in the output
                #     caption="Concatenated DataFrame",
                #     label="tab:concatenated_dataframe",
                #     column_format="|c|c|c|",  # For three columns with vertical lines
                #     escape=False  # Keep special characters as is
                # )
                with open(f"{data_key}_ckpt_{ckpt_iter_num}.tex", 'w') as f:
                    f.write(latex_table)
                print(f"ckpt_{ckpt_iter_num}.pt:")
                print(latex_table)

    record_tactics_eval_in_table()
    
     
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