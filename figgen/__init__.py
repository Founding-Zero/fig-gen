import os
import tempfile
from collections import defaultdict

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv


class DataAnalyzer:
    def __init__(
        self,
        wandb_entity,
        wandb_project,
        min_length=100,
        color_scheme="viridis_r",
        # color_scheme="magma",
        export_to_wandb=False,
    ):
        load_dotenv()
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.api_key = os.environ["WANDB_API_KEY"]
        self.api = wandb.Api()
        self.histories = {}
        self.min_length = min_length
        self.color_scheme = color_scheme
        self.export_to_wandb = export_to_wandb
        if self.export_to_wandb and self.api_key:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def get_runs(self, run_ids=None):
        if run_ids is not None:
            self.runs = [
                self.api.run(f"{self.wandb_entity}/{self.wandb_project}/{run_id}")
                for run_id in run_ids
            ]
        else:
            self.runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")
        return self.runs

    def send_to_wandb(self, fig, title):
        if self.export_to_wandb:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_path = tmpfile.name
                fig.savefig(fig_path)
                wandb.log({title: wandb.Image(fig_path)})
                os.unlink(fig_path)

    def get_histories(self):
        for run in self.runs:
            self.histories[run.id] = run.history()

    def visualize_one_lineplot_groupby(
        self,
        title: str,
        x_key: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
        custom_error_bar_fn=None,
    ):
        """Visualize a dataframe with this form. Here, Temperature is the x-axis, Rating is the y-axis, and Stockfish Skill Level is the group key.:

        stockfish_groupby = "Stockfish Skill Level" # Key for the temperature group
        sample_data = [
            {"Temperature": 0.3, stockfish_groupby: "0", "Rating": 1000},
            {"Temperature": 0.3, stockfish_groupby: "1", "Rating": 1200},
            {"Temperature": 0.3, stockfish_groupby: "2", "Rating": 1400},
            {"Temperature": 0.2, stockfish_groupby: "0", "Rating": 1100},
            {"Temperature": 0.2, stockfish_groupby: "1", "Rating": 1300},
            {"Temperature": 0.2, stockfish_groupby: "2", "Rating": 1500},
            {"Temperature": 0.1, stockfish_groupby: "0", "Rating": 1200},
            {"Temperature": 0.1, stockfish_groupby: "1", "Rating": 1400},
            {"Temperature": 0.1, stockfish_groupby: "2", "Rating": 1600},
        ]

        # add noise to data for error bars
        data = pd.DataFrame.from_dict(
            sum(
                [
                    [
                        {
                            "Temperature": d["Temperature"],
                            stockfish_groupby: d[stockfish_groupby],
                            "Rating": d["Rating"] + 100 * np.random.randn(),
                        }
                        for d in sample_data
                    ]
                    for _ in range(3)
                ],
                [],
            )
        )

        visualize_lineplot_groupby("Chess Ratings of NanoGPT across Temperature", "Temperature", "Rating", stockfish_groupby, data, x_ticks_by_data=True)

        """
        groups = data[group_key].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            matplotlib.rcParams.update({"font.size": 40})
            matplotlib.rcParams.update(
                {
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsfonts}",
                    "font.family": "Computer Modern Serif",
                }
            )
            fig, ax = plt.subplots(figsize=(30, 20))
            # fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar=("ci", 95)
                if custom_error_bar_fn is None
                else custom_error_bar_fn,
            )
            # plt.title(title, fontsize=25, pad=30)
            plt.title(title, fontsize="large")
            # plt.xlabel(x_label or x_key.capitalize(), fontsize=20, labelpad=30)
            plt.xlabel(x_label or x_key.capitalize(), fontsize="large")
            # plt.ylabel(y_label or y_key.capitalize(), fontsize=20, labelpad=30)
            plt.ylabel(y_label or y_key.capitalize(), fontsize="large")
            # plt.legend().set_visible(False)

            # plt.xlim(left=0, right=self.min_length)
            if x_ticks_by_data:
                plt.xticks(data[x_key].unique())
                # plt.xticks(fontsize=15)
                plt.xticks(rotation=45)

            # ax.axhline(y=int(groups[0]), color=palette[0], linestyle='--', linewidth=1)

            # ax.text(2, int(groups[0]), f"Max Rating Seen During Training: {groups[0]}", color='black', va='bottom', ha='right')

            # Get handles and labels
            # handles, labels = ax.get_legend_handles_labels()

            # Recreate the legend with new font size
            # ax.legend(handles=handles, labels=labels, title='Trainined High Elo', fontsize='16', title_fontsize='16')

            # plt.yticks(data[y_key].unique())
            # plt.ylim(bottom=0)
            plt.grid(True)

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")

    def visualize_lineplot_groupby(
        self,
        title: str,
        x_key: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
        custom_error_bar_fn=None,
    ):
        """Visualize a dataframe with this form. Here, Temperature is the x-axis, Rating is the y-axis, and Stockfish Skill Level is the group key.:

        stockfish_groupby = "Stockfish Skill Level" # Key for the temperature group
        sample_data = [
            {"Temperature": 0.3, stockfish_groupby: "0", "Rating": 1000},
            {"Temperature": 0.3, stockfish_groupby: "1", "Rating": 1200},
            {"Temperature": 0.3, stockfish_groupby: "2", "Rating": 1400},
            {"Temperature": 0.2, stockfish_groupby: "0", "Rating": 1100},
            {"Temperature": 0.2, stockfish_groupby: "1", "Rating": 1300},
            {"Temperature": 0.2, stockfish_groupby: "2", "Rating": 1500},
            {"Temperature": 0.1, stockfish_groupby: "0", "Rating": 1200},
            {"Temperature": 0.1, stockfish_groupby: "1", "Rating": 1400},
            {"Temperature": 0.1, stockfish_groupby: "2", "Rating": 1600},
        ]

        # add noise to data for error bars
        data = pd.DataFrame.from_dict(
            sum(
                [
                    [
                        {
                            "Temperature": d["Temperature"],
                            stockfish_groupby: d[stockfish_groupby],
                            "Rating": d["Rating"] + 100 * np.random.randn(),
                        }
                        for d in sample_data
                    ]
                    for _ in range(3)
                ],
                [],
            )
        )

        visualize_lineplot_groupby("Chess Ratings of NanoGPT across Temperature", "Temperature", "Rating", stockfish_groupby, data, x_ticks_by_data=True)

        """
        groups = data[group_key].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            matplotlib.rcParams.update({"font.size": 50})
            matplotlib.rcParams.update(
                {
                    "font.size": 50,
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsfonts}",
                    "font.family": "Computer Modern Serif",
                }
            )
            # Set the font properties
            # font_path = 'path/to/Helvetica.ttf'  # Update with the correct path to your Helvetica font
            # prop = font_manager.FontProperties(fname=font_path, size=12)
            fig, axes = plt.subplots(1, 3, figsize=(50, 15))
            # fig.suptitle(title, fontsize=40, x=0.49)
            fig.subplots_adjust(left=0.05, right=0.99)
            fig.legend().set_visible(False)

            sns.lineplot(
                data=data[data[group_key] == "1000"],
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar=("ci", 95)
                if custom_error_bar_fn is None
                else custom_error_bar_fn,
                ax=axes[0],
            )

            # y_values = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600] # New model old stockfish
            # axes[0].set_yticklabels(list(map(str, [800] + y_values))) # New model old stockfish
            y_values = [
                600,
                800,
                1000,
                1200,
                1400,
                1600,
                1800,
            ]  # Old model new stockfish
            # y_values = [400, 600, 800, 1000, 1200, 1400, 1600] # Old model new stockfish
            # axes[0].set_yticklabels(list(map(str, y_values))) # Old model new stockfish
            # axes[0].set_yticklabels(list(map(str, [600] + y_values))) # Old model new stockfish
            # axes[0].xaxis.label.set_size(45)
            # axes[0].yaxis.label.set_size(45)
            axes[0].set_ylim([600, 1800])
            axes[0].tick_params(axis="x")
            axes[0].tick_params(axis="y")
            axes[0].axhline(
                y=int(groups[0]), color=palette[0], linestyle="--", linewidth=4
            )
            axes[0].text(
                7,
                int(groups[0]),
                f"Max Rating Seen During Training: {groups[0]}",
                color="black",
                va="bottom",
                ha="right",
                fontsize=60,
                fontfamily="Latin Modern Roman",
            )
            axes[0].legend().set_visible(False)
            axes[0].grid(True)
            axes[0].set_xlabel(r"Temperature ($\tau$)", labelpad=10)
            axes[0].set_ylabel("Rating", labelpad=10)
            # axes[0].margins(y=0.26)

            sns.lineplot(
                data=data[data[group_key] == "1300"],
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar=("ci", 95)
                if custom_error_bar_fn is None
                else custom_error_bar_fn,
                ax=axes[1],
            )
            # axes[1].set_yticklabels(list(map(str, [800] + y_values))) # New model old stockfish
            # axes[1].set_yticklabels(list(map(str, [600] + y_values + [1600]))) # Old model new stockfish
            # axes[1].set_yticklabels(list(map(str, y_values))) # Old model new stockfish
            # axes[1].xaxis.label.set_size(45)
            # axes[1].yaxis.label.set_size(45)
            axes[1].set_ylim([600, 1800])
            axes[1].tick_params(axis="x")
            axes[1].tick_params(axis="y")
            axes[1].axhline(
                y=int(groups[1]), color=palette[1], linestyle="--", linewidth=4
            )
            axes[1].text(
                7,
                int(groups[1]),
                f"Max Rating Seen During Training: {groups[1]}",
                color="black",
                va="bottom",
                ha="right",
                fontsize=60,
                fontfamily="Latin Modern Roman",
            )
            axes[1].legend().set_visible(False)
            axes[1].grid(True)
            axes[1].set_xlabel(r"Temperature ($\tau$)", labelpad=10)
            axes[1].set_ylabel("Rating", labelpad=10)
            # axes[1].margins(y=0.1)

            sns.lineplot(
                data=data[data[group_key] == "1500"],
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar=("ci", 95)
                if custom_error_bar_fn is None
                else custom_error_bar_fn,
                ax=axes[2],
            )
            # axes[2].set_yticklabels(list(map(str, [800] + y_values))) # New model old stockfish
            # axes[2].set_yticklabels(list(map(str, y_values + [2000]))) # Old model new stockfish
            # axes[0].set_yticklabels(list(map(str, y_values + [1800]))) # Old model new stockfish
            # axes[2].xaxis.label.set_size(45)
            # axes[2].yaxis.label.set_size(45)
            axes[2].set_ylim([600, 1800])
            axes[2].tick_params(axis="x")
            axes[2].tick_params(axis="y")
            axes[2].axhline(
                y=int(groups[2]), color=palette[2], linestyle="--", linewidth=4
            )
            axes[2].text(
                0,
                int(groups[2]),
                f"Max Rating Seen During Training: {groups[2]}",
                color="black",
                va="bottom",
                ha="left",
                fontsize=60,
                fontfamily="Latin Modern Roman",
            )
            axes[2].legend().set_visible(False)
            axes[2].grid(True)
            axes[2].set_xlabel(r"Temperature ($\tau$)", labelpad=10)
            axes[2].set_ylabel("Rating", labelpad=10)
            # axes[2].margins(y=0.15)

        if self.export_to_wandb:
            self.send_to_wandb(fig, title)
        else:
            plt.savefig(f"{title}.png")

    def visualize_lineplot_groupby_with_dotted_line(
        self,
        title: str,
        x_key: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
        dotted_line_x: list = [],
        custom_error_bar_fn=None,
    ):
        """Visualize a dataframe with this form. Here, Temperature is the x-axis, Rating is the y-axis, and Stockfish Skill Level is the group key.:

        stockfish_groupby = "Stockfish Skill Level" # Key for the temperature group
        sample_data = [
            {"Temperature": 0.3, stockfish_groupby: "0", "Rating": 1000},
            {"Temperature": 0.3, stockfish_groupby: "1", "Rating": 1200},
            {"Temperature": 0.3, stockfish_groupby: "2", "Rating": 1400},
            {"Temperature": 0.2, stockfish_groupby: "0", "Rating": 1100},
            {"Temperature": 0.2, stockfish_groupby: "1", "Rating": 1300},
            {"Temperature": 0.2, stockfish_groupby: "2", "Rating": 1500},
            {"Temperature": 0.1, stockfish_groupby: "0", "Rating": 1200},
            {"Temperature": 0.1, stockfish_groupby: "1", "Rating": 1400},
            {"Temperature": 0.1, stockfish_groupby: "2", "Rating": 1600},
        ]

        # add noise to data for error bars
        data = pd.DataFrame.from_dict(
            sum(
                [
                    [
                        {
                            "Temperature": d["Temperature"],
                            stockfish_groupby: d[stockfish_groupby],
                            "Rating": d["Rating"] + 100 * np.random.randn(),
                        }
                        for d in sample_data
                    ]
                    for _ in range(3)
                ],
                [],
            )
        )

        visualize_lineplot_groupby("Chess Ratings of NanoGPT across Temperature", "Temperature", "Rating", stockfish_groupby, data, x_ticks_by_data=True)

        """
        groups = data[group_key].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar=("ci", 95)
                if custom_error_bar_fn is None
                else custom_error_bar_fn,
            )

            plt.title(title, fontsize="large")
            plt.xlabel(x_label or x_key.capitalize(), fontsize="large")
            plt.ylabel(y_label or y_key.capitalize(), fontsize="large")

            # plt.xlim(left=0, right=self.min_length)
            if x_ticks_by_data:
                plt.xticks(data[x_key].unique())

            plt.ylim(bottom=0)
            plt.grid(True)

            # if dotted_line_x is not None:
            i = 0
            for trained_game_length in dotted_line_x:
                ax.axvline(
                    x=trained_game_length, color=palette[i], linestyle="--", linewidth=1
                )
                i += 1

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")

    def visualize_simple_barplot_groupby(
        self,
        title: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        # x_ticks_by_data: bool = False,
    ):
        groups = data[group_key].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(
                data=data,
                x=group_key,
                y=y_key,
                hue=None,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
            )

            plt.title(title, fontsize="large")
            # plt.xlabel('High Elo')
            # plt.ylabel('Rating')
            # plt.legend(title='Conditioning')
            plt.xticks(rotation=45)
            plt.xlabel(x_label, fontsize="large")
            plt.ylabel(y_label, fontsize="large")

            # plt.xlim(left=0, right=self.min_length)
            # if x_ticks_by_data:
            #     plt.xticks(data[group_key].unique())

            # plt.ylim(bottom=0)
            plt.grid(True)

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")

    def visualize_barplot_groupby(
        self,
        title: str,
        category: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
    ):
        groups = data[category].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(16, 9))
            sns.barplot(
                data=data,
                x=group_key,
                y=y_key,
                hue=category,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
                # errorbar=("ci", 95)
            )

            plt.title(title, fontsize="large")
            # plt.xlabel('High Elo')
            # plt.ylabel('Rating')
            plt.legend(title="Conditioning")
            plt.xticks(rotation=45)
            plt.xlabel(x_label, fontsize="large")
            plt.ylabel(y_label, fontsize="large")

            # plt.xlim(left=0, right=self.min_length)
            # if x_ticks_by_data:
            #     plt.xticks(data[group_key].unique())

            # plt.ylim(bottom=0)
            plt.grid(True)

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")

    def visualize_histplot_groupby(
        self,
        title: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
    ):
        # groups = data[group_key].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        # color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        color_dict = {
            "1": "#85007A",
            "0": "#2F5997",
            "2": "#2F5997",
            # "5.0%": "#2F5997",
        }
        # color_dict = {
        #     "40%": "#85007A",
        #     "60%": "#2F5997",
        #     "0%": "#473F90",
        # }
        with sns.axes_style("darkgrid"):
            matplotlib.rcParams.update({"font.size": 50})
            matplotlib.rcParams.update(
                {
                    "font.size": 50,
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsfonts}",
                    "font.family": "Computer Modern Serif",
                }
            )
            fig, ax = plt.subplots(1, 1, figsize=(20, 8))
            sns.barplot(
                data=data,
                x=data.index,
                y=y_key,
                hue=None,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
                # errorbar=("ci", 95)
            )

            ax.set_ylim([0, 100])
            ax.set_xticklabels([5, 90, 5])
            ax.tick_params(axis="x")
            ax.tick_params(axis="y")
            ax.legend().set_visible(False)
            ax.grid(True)
            ax.set_ylabel(r"$\mathbb{P}(Y \mid X)$")
            ax.set_xlabel("")

            # plt.ylim(bottom=0)
            plt.grid(True)

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")

    def visualize_three_histogram_plot_groupby(
        self,
        title: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
    ):
        groups = data[group_key].unique()
        # palette = sns.dark_palette("seagreen", n_colors=len(groups))
        # palette = sns.diverging_palette(171, 80, s=74, l=50, sep=10, n=len(groups), center='dark')
        # palette = sns.diverging_palette(220, 10, s=74, l=50, sep=10, n=len(groups), center='dark')
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        # palette = sns.cubehelix_palette(n_colors=len(groups), start=0.5, rot=-0.95)
        # palette = sns.color_palette(self.color_scheme, n_colors=(4*len(groups)))
        # filtered_palette = palette[0::4]
        # palette = filtered_palette
        # palette = sns.cubehelix_palette(n_colors=len(groups))
        # color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        color_dict = {
            "h5f4": "#305897",
            "f8e8": "#85007A",
            "c6g6": "#2F5997",
            "d4g4": "#473F90",
        }
        # color_dict_1 = {
        #     "b3a2": "#c60039",
        #     "b3c4": "#c60039",
        #     "b3c2": "#c60039",
        # }
        # color_dict = {
        #     "e1e5": "#4b3a8f",
        #     "e1h4": "#80007f",
        #     "e1d1": "#2a5f99",
        #     "e1h1": "#2b5e99",
        # }
        # color_dict_5 = {
        #     "h1g1": "#8f0070",
        #     "h1f1": "#2e5a98",
        #     "f6f5": "#7e0280",
        # }
        # color_dict_7 = {
        #     "d4e4": "#7f0180",
        #     "f8d8": "#2f5a98",
        #     "e6c4": "#2d5b98",
        # }
        with sns.axes_style("darkgrid"):
            matplotlib.rcParams.update({"font.size": 50})
            matplotlib.rcParams.update(
                {
                    "font.size": 50,
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsfonts}",
                    "font.family": "Computer Modern Serif",
                }
            )
            fig, axes = plt.subplots(1, 3, figsize=(50, 6))
            fig.subplots_adjust(left=0.05, right=0.99)
            fig.legend().set_visible(False)
            sns.barplot(
                data=data[data["Temperature"] == "0.75"],
                x=group_key,
                y=y_key,
                hue=group_key,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
                # errorbar=("ci", 95)
                ax=axes[0],
            )
            axes[0].set_ylim([0, 1])
            axes[0].tick_params(axis="x")
            axes[0].tick_params(axis="y")
            axes[0].legend().set_visible(False)
            axes[0].grid(True)
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
            # axes[0].xticks(x, weight = 'bold')
            # axes[0].margins(y=0.26)

            sns.barplot(
                data=data[data["Temperature"] == "1.0"],
                x=group_key,
                y=y_key,
                hue=group_key,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
                # errorbar=("ci", 95)
                ax=axes[1],
            )
            axes[1].set_ylim([0, 1])
            axes[1].tick_params(axis="x")
            axes[1].tick_params(axis="y")
            axes[1].legend().set_visible(False)
            axes[1].grid(True)
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
            # axes[1].margins(y=0.26)

            sns.barplot(
                data=data[data["Temperature"] == "0.001"],
                x=group_key,
                y=y_key,
                hue=group_key,
                palette=color_dict,
                errorbar=("pi", 50),
                capsize=0.4,
                err_kws={"color": "0", "linewidth": 2.5},
                # errorbar=("ci", 95)
                ax=axes[2],
            )
            axes[2].set_ylim([0, 1])
            axes[2].tick_params(axis="x")
            axes[2].tick_params(axis="y")
            axes[2].legend().set_visible(False)
            axes[2].grid(True)
            axes[2].set_xlabel("")
            axes[2].set_ylabel("")
            # axes[1].margins(y=0.26)

            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")
