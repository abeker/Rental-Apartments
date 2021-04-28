import utility.aminitie_mapper as mapper
import numpy as np
import collections
import pandas as pd

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def clear_array(amenities_json):
    amenities_json = amenities_json.replace("{", "")
    amenities_json = amenities_json.replace("}", "")
    return amenities_json.split(",")

def get_coefficient_for_amenitie(df, amenitie):
    idx = df[df['index'] == amenitie].index.values.astype(int)[0]
    return df.iloc[idx]['frequency']

def get_points_for_amentities(amenities_json):
    am_splitted = clear_array(amenities_json)
    summary = 0
    for amenitie in am_splitted:
        amenitie = amenitie.replace("\"", "")
        amenitie = amenitie.strip()
        summary = summary + mapper.get_points(amenitie)
        # print(amenitie, ':', mapper.get_points(amenitie))
    # print('sum: ', summary)
    # print('------------------------------------------')
    number_of_amenities = len(am_splitted)
    return number_of_amenities

def get_list_of_all_amenities(amenities_column):
    list_of_amenities = []
    for amenities in amenities_column:
        am_splitted = clear_array(amenities)
        for amenitie in am_splitted:
            amenitie = amenitie.replace("\"", "")
            amenitie = amenitie.strip()
            list_of_amenities.append(amenitie)
    return list_of_amenities

def create_amenities_tf(amenities_column):
    amenities_list = get_list_of_all_amenities(amenities_column)
    counter = collections.Counter(amenities_list)
    df = pd.DataFrame.from_dict(counter, orient='index', columns=['frequency']).reset_index()
    df.to_csv('./Datasets/cleaned/amenities.csv')
    return df