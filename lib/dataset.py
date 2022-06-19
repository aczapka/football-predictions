import pandas as pd
import numpy as np


class Dataset:
    """Parent class that is used to manage the dataset."""

    def __init__(self):
        self.complete_dataset = None
        self.dataset = None
        self.dataset_teams = None
        self.dataset_num_teams = None

    def download_dataset_from_football_data_uk(self):
        """Download the data of the first Spanish division from football-data.co.uk"""
        print("Downloading dataset from football-data.co.uk")

        df = pd.DataFrame()
        for i in range(0, 22):
            if i < 9:
                year = f'0{i}0{i + 1}'
            elif i == 9:
                year = f'0{i}{i + 1}'
            else:
                year = f'{i}{i + 1}'
            print(f"\tDownloading season {year}")

            df_read = pd.read_csv(f'https://www.football-data.co.uk/mmz4281/{year}/SP1.csv', on_bad_lines='warn')
            df_read = df_read[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]

            try:
                df_read['Date'] = pd.to_datetime(df_read['Date'], format="%d/%m/%y")
            except ValueError:
                df_read['Date'] = pd.to_datetime(df_read['Date'], format="%d/%m/%Y")

            df = pd.concat([df, df_read])

        df = df.rename(columns={
            'Date': 'date',
            'HomeTeam': 'team_home',
            'AwayTeam': 'team_away',
            'FTHG': 'goals_home',
            'FTAG': 'goals_away'
        })
        self.complete_dataset = df
        return None

    def save_complete_dataset(self):
        """Save the complete dataset to a CSV file."""
        print("Saving complete dataset")

        self.complete_dataset.to_csv('./input/dataset.csv', index=False, encoding='UTF-8', sep=';')
        return None

    def load_complete_dataset(self):
        """Read the complete dataset from CSV file."""
        print("Loading complete dataset")

        self.complete_dataset = pd.read_csv('./input/dataset.csv', encoding='UTF-8', sep=';', parse_dates=['date'])
        return None

    def clip_complete_dataset(self, date_start, date_end):
        """Clip the complete dataset between two dates (both inclusive)."""
        print(f"Clipping the complete dataset between {date_start} and {date_end}")

        dataset_clipped = self.complete_dataset.copy()
        dataset_clipped = dataset_clipped.loc[(dataset_clipped['date'] >= date_start) &
                                              (dataset_clipped['date'] <= date_end)]

        self.dataset = dataset_clipped
        self.dataset_teams = np.sort(dataset_clipped['team_home'].unique())
        self.dataset_num_teams = len(self.dataset_teams)
        return None
