import munich_handler
import berlin_handler
import statistics.outliers as outliers
import statistics.corelation as corelation
from algorithms import regression_impl
from algorithms import xgboost_impl
from algorithms import adaboost_impl
from algorithms import random_forest_impl
import utility.enums as enum


#df_munich = munich_handler.get_munich_data()
#corelation.print_df_corelation(df_munich)
df_berlin = berlin_handler.get_berlin_dataset()
#outliers.get_outliers_statistics(df_berlin)
#model = regression_impl.train(df_munich, enum.RegressionType.LINEAR)
#outliers.boxplot(df_munich,'zipcode')
#xgboost_impl.train(df_munich, True, True)
xgboost_impl.train(df_berlin, True, True)
#adaboost_impl.train(df_munich, True, True)
#random_forest_impl.train(df_munich)
#random_forest_impl.train(df_berlin)

