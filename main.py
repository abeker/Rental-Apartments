import munich
import berlin
import statistics.outliers as outliers
import statistics.corelation as corelation
import regression
import utility.enums as enum

df_munich = munich.get_munich_data()
model = regression.train(df_munich, enum.RegressionType.LINEAR)
outliers.get_outliers_statistics(df_munich)
# corelation.print_corelation(df_munich['day_of_week'].head(20).astype('int64'), df_munich['price'].head(20))
# df_berlin = berlin.get_berlin_dataset()