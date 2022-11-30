from dataset import Dataset
from model import Model
from backtesting import Backtesting
import pandas as pd

"""
Examples of usage:
"""
"""Download / update training dataset."""
dataset = Dataset()
dataset.download_dataset_from_football_data_uk()
dataset.save_complete_dataset()
"""Predict the results of one past season manually."""
# load dataset
model = Model()
model.load_complete_dataset()

# train model
model.clip_complete_dataset(date_start='2020-09-12', date_end='2021-05-23')
model.train(max_iterations=400)
model.save_log_parameters(file_suffix='train_' + '2020-2021')
model.save_parameters(file_suffix='train_' + '2020-2021')

# # predict past
model.clip_complete_dataset(date_start='2021-08-13', date_end='2022-05-31')
model.predict_past()
model.save_prediction_past(file_suffix='train_' + '2020-2021' + '_predict_' + '2021-2022')
"""Predict multiple past seasons iteratively"""
backtesting = Backtesting()
backtesting.load_complete_dataset()
backtesting.walk_forward(max_iterations=400)
"""Predict the future"""
# load dataset
model = Model()
model.load_complete_dataset()

# train model
model.clip_complete_dataset(date_start='2020-09-12', date_end='2021-05-23')
model.train(max_iterations=400)

# predict future (to be developed in the future)
# df_predictors = pd.read_csv('./input/future_predictors.csv', sep=';')
# model.predict_future(df_predictors)
# model.save_prediction_future(file_suffix='train_' + '2020-2021' + '_predict_' + '2021-2022')
