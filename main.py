import munich
import statistics.outliers as outliers
from algorithms import regression_impl
from algorithms import xgboost_impl
import utility.enums as enum

df_munich = munich.get_munich_data()
# model = regression_impl.train(df_munich, enum.RegressionType.LINEAR)
xgboost_impl.train(df_munich, True, True)
# outliers.get_outliers_statistics(df_munich)
# corelation.print_corelation(df_munich['day_of_week'].head(20).astype('int64'), df_munich['price'].head(20))
# df_berlin = berlin.get_berlin_dataset()