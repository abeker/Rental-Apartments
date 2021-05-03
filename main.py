import munich_handler
import statistics.outliers as outliers
import statistics.corelation as corelation
from algorithms import regression_impl
from algorithms import xgboost_impl
from algorithms import adaboost_impl
from algorithms import random_forest_impl
import utility.enums as enum

df_munich = munich_handler.get_munich_data()
# outliers.get_outliers_statistics(df_munich)
# corelation.print_df_corelation(df_munich)
# df_berlin = berlin.get_berlin_dataset()
# model = regression_impl.train(df_munich, enum.RegressionType.LINEAR)
#xgboost_impl.train(df_munich, True, True)
#adaboost_impl.train(df_munich, True, True)
random_forest_impl.train(df_munich)
