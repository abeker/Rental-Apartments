import munich_handler
from algorithms import regression_impl
from algorithms.neural_network import neural_networks_impl
from algorithms import xgboost_impl
from algorithms import adaboost_impl
from algorithms import random_forest_impl
from statistics import outliers, corelation
import utility.enums as enum

df_munich = munich_handler.get_munich_data()
# corelation.print_df_corelation(df_munich)
# outliers.get_outliers_statistics(df_munich)

# df_berlin = berlin.get_berlin_dataset()
model = regression_impl.train(df_munich, enum.RegressionType.LINEAR)
# xgboost_impl.train(df_munich, True, True)
# adaboost_impl.train(df_munich, True, True)
# random_forest_impl.train(df_munich)
print('-------------------------------')
neural_networks_impl.train(df_munich)
