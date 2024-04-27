
from collections import defaultdict
import os
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
from dotenv import load_dotenv

class DataAnalyzer:
    def __init__(self, wandb_entity, wandb_project, min_length = 100, color_scheme = "viridis", export_to_wandb=False):
        load_dotenv()
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.api_key = os.environ['WANDB_API_KEY']
        self.api = wandb.Api()
        self.runs = self.api.runs(f'{wandb_entity}/{wandb_project}')
        self.histories = {}
        self.min_length = min_length 
        self.color_scheme = color_scheme
        self.export_to_wandb = export_to_wandb
        self.get_histories()
        if self.export_to_wandb and self.api_key:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def send_to_wandb(self, fig, title):
        if self.export_to_wandb:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                fig_path = tmpfile.name
                fig.savefig(fig_path)
                wandb.log({f"{title}": wandb.Image(fig_path)})
                os.unlink(fig_path)

    def get_histories(self):
        for run in self.runs:
            self.histories[run.id] = run.history()

    def fetch_and_process_sigma_data(self, data_header):
        desired_group = defaultdict(list)
        for run in self.runs:
            if data_header in self.histories[run.id].columns and len(self.histories[run.id][data_header]) > self.min_length:
                desired_group[run.config["sigma_vals"]].append(self.histories[run.id][data_header].iloc[1:self.min_length].tolist())

        desired_data = {}
        for sigma, runs in desired_group.items():
            transposed_runs = list(zip(*runs))
            desired_data[sigma] = transposed_runs
    
        records = []
        for sigma, episode in desired_data.items():
            for episode_index, values in enumerate(episode):
                for value in values:
                    records.append({'Episode': episode_index, 'Sigma': sigma, data_header: value})
        return pd.DataFrame(records)

    def visualize_all_sigma_data(self, data, title):
       
        sigma_groups = data['Sigma'].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups)) 
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12,8))
            sns.lineplot(data=data, x="Episode", y=f'{title}', hue="Sigma", dashes=False, palette=color_dict, err_style="band", errorbar="se")
            plt.title(f'{title} across all Sigma groups')
            plt.xlabel('Episodes')
            plt.ylabel(f'{title}')
            plt.xlim(left=0, right=self.min_length)
            plt.ylim(bottom=0)
            plt.grid(True)
            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.show()

    def visualize_individual_sigma_data(self, data, title):
        sigma_groups = data['Sigma'].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups))
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        self.visualize_all_sigma_data(data, title)

        with sns.axes_style("darkgrid"):
        # Iterate through each sigma group to create individual plots
            for highlighted_sigma in sigma_groups:
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot each sigma group
                for sigma in sigma_groups:
                    subset = data[data['Sigma'] == sigma]
                    if sigma == highlighted_sigma:
                        # Highlight the selected sigma group
                        sns.lineplot(data=subset, x="Episode", y=f'{title}', color=color_dict[sigma], label=f'{sigma}', linewidth=2.5, errorbar='se')
                    else:
                        # Dim other sigma groups
                        sns.lineplot(data=subset, x="Episode", y=f'{title}', color=color_dict[sigma], label=f'{sigma}', linewidth=1, errorbar=None, alpha=.4)

                plt.title(f'{title} (Sigma {highlighted_sigma} Highlighted)')
                plt.xlabel('Episodes')
                plt.ylabel(f'{title}')
                plt.xlim(left=0, right=self.min_length)
                plt.ylim(bottom=0)
                plt.legend(title='Sigma')
                plt.grid(True)
                if self.export_to_wandb:
                    self.send_to_wandb(fig, title)
                else:
                    plt.show()

    def plot_all_sigma_data(self):
        pertinent_headers = ['charts/voted_p_value', 'charts/mean_taxed_return', 'charts/mean_episodic_return', 'charts/mean_raw_return', 'charts/p_mean_taxed_return', 'charts/p_mean_raw_return']
        for header in pertinent_headers:
            data = self.fetch_and_process_sigma_data(header)
            self.visualize_individual_sigma_data(data, header)
