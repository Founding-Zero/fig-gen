from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from figgen import DataAnalyzer


class BenchMARLDataAnalyzer(DataAnalyzer):
    def fetch_and_process_sigma_data(self, data_header):
        self.get_runs()
        self.get_histories()

        desired_group = defaultdict(list)
        for run in self.runs:
            if (
                data_header in self.histories[run.id].columns
                and len(self.histories[run.id][data_header]) > self.min_length
            ):
                desired_group[run.config["task_config"]["sigma_vals"]].append(
                    self.histories[run.id][data_header]
                    .iloc[1 : self.min_length]
                    .tolist()
                )

        desired_data = {}
        for sigma, runs in desired_group.items():
            transposed_runs = list(zip(*runs))
            desired_data[sigma] = transposed_runs

        records = []
        for sigma, episode in desired_data.items():
            for episode_index, values in enumerate(episode):
                for value in values:
                    records.append(
                        {"Episode": episode_index, "Sigma": sigma, data_header: value}
                    )
        return pd.DataFrame(records)

    def visualize_all_sigma_data(self, data, title):
        sigma_groups = data["Sigma"].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups))
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x="Episode",
                y=f"{title}",
                hue="Sigma",
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar="se",
            )
            plt.title(f"{title} across all Sigma groups")
            plt.xlabel("Episodes")
            plt.ylabel(f"{title}")
            plt.xlim(left=0, right=self.min_length)
            plt.ylim(bottom=0)
            plt.grid(True)
            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.show()

    def visualize_individual_sigma_data(self, data, title):
        sigma_groups = data["Sigma"].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups))
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        self.visualize_all_sigma_data(data, title)

        with sns.axes_style("darkgrid"):
            # Iterate through each sigma group to create individual plots
            for highlighted_sigma in sigma_groups:
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot each sigma group
                for sigma in sigma_groups:
                    subset = data[data["Sigma"] == sigma]
                    if sigma == highlighted_sigma:
                        # Highlight the selected sigma group
                        sns.lineplot(
                            data=subset,
                            x="Episode",
                            y=f"{title}",
                            color=color_dict[sigma],
                            label=f"{sigma}",
                            linewidth=2.5,
                            errorbar="se",
                        )
                    else:
                        # Dim other sigma groups
                        sns.lineplot(
                            data=subset,
                            x="Episode",
                            y=f"{title}",
                            color=color_dict[sigma],
                            label=f"{sigma}",
                            linewidth=1,
                            errorbar=None,
                            alpha=0.4,
                        )

                plt.title(f"{title} (Sigma {highlighted_sigma} Highlighted)")
                plt.xlabel("Episodes")
                plt.ylabel(f"{title}")
                plt.xlim(left=0, right=self.min_length)
                plt.ylim(bottom=0)
                plt.legend(title="Sigma")
                plt.grid(True)
                if self.export_to_wandb:
                    self.send_to_wandb(fig, title)
                else:
                    plt.show()

    def plot_all_sigma_data(self):
        pertinent_headers = [
            "collection/agents/reward/episode_reward_min",
            "collection/agents/reward/reward_mean",
            "collection/agents/reward/episode_reward_max",
            "collection/agents/reward/episode_reward_mean",
            "collection/agents/reward/episode_reward_max",
            "collection/agents/social_influenced_reward/social_influenced_reward_max",
            "collection/agents/social_influenced_reward/social_influenced_reward_mean",
            "collection/agents/social_influenced_reward/social_influenced_reward_min",
            "collection/agents/taxed_return/taxed_return_mean",
            "collection/agents/taxed_return/taxed_return_min",
            "collection/agents/taxed_return/taxed_return_max",
            "collection/agents/taxed_reward/taxed_reward_max",
            "collection/agents/taxed_reward/taxed_reward_min",
            "collection/agents/taxed_reward/taxed_reward_mean",
            "collection/reward/episode_reward_mean",
            "collection/reward/episode_reward_min",
            "collection/reward/episode_reward_max",
            "eval/agents/reward/episode_reward_max",
            "eval/agents/reward/episode_reward_min",
            "eval/agents/reward/episode_reward_mean",
            "eval/reward/episode_reward_min",
            "eval/reward/episode_reward_max",
            "eval/reward/episode_reward_mean",
        ]
        for header in pertinent_headers:
            data = self.fetch_and_process_sigma_data(header)
            self.visualize_individual_sigma_data(data, header)