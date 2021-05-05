import pandas as pd
import numpy as np
import math
import munich_handler as mh
import utility.amentities as amenitie_utility
import utility.host_verifications as verification_utility
pd.options.mode.chained_assignment = None  # default='warn'
np.seterr(divide = 'ignore')

berlin_calendar = pd.read_csv('./Datasets/Berlin/calendar_summary.csv', low_memory=False)
berlin_listings = pd.read_csv('./Datasets/Berlin/listings_summary.csv', low_memory=False)
berlin_center = [52.52437, 13.41053]

def clean_price(column):
    just_numbers = column.str.replace('$', '', regex=True).astype(str)
    #return just_number.replace({"nan" : np.nan}, inplace=True)
    return just_numbers.astype(str).replace('[,]', '', regex=True).astype(float)

def get_mean_value(column):
    return math.floor(column.mean())

def get_mapping(column):
    return dict([(y, x + 1) for x, y in enumerate(sorted(set(column)))])

def handle_duplicates(df):
    df = df.drop_duplicates()
    df = df.groupby('listing_id').mean().reset_index()
    return df


def unbox_calendar():
    berlin_calendar['price'] = clean_price(berlin_calendar['price'].astype('str'))
    berlin_calendar['price'] = berlin_calendar['price'].fillna(get_mean_value(berlin_calendar['price']))
    df = berlin_calendar[['listing_id', 'price']]
    df = handle_duplicates(df)
    day, month, year, df['day_of_week'] = mh.transform_date(berlin_calendar, 'date')
    return df


def map_string_properties_to_numbers():
    property_type_mapping = get_mapping(berlin_listings['property_type'].unique())
    room_type_mapping = get_mapping(berlin_listings['room_type'].unique())
    bed_type_mapping = get_mapping(berlin_listings['bed_type'].unique())
    cancellation_policy_mapping = get_mapping(berlin_listings['cancellation_policy'].unique())
    berlin_listings['neighbourhood'] = berlin_listings['neighbourhood'].astype('str')
    neighbourhood_mapping = get_mapping(berlin_listings['neighbourhood'].unique())
    berlin_listings['property_type'] = berlin_listings['property_type'] \
        .replace(property_type_mapping.keys(), property_type_mapping.values())
    berlin_listings['room_type'] = berlin_listings['room_type'] \
        .replace(room_type_mapping.keys(), room_type_mapping.values())
    berlin_listings['bed_type'] = berlin_listings['bed_type'] \
        .replace(bed_type_mapping.keys(), bed_type_mapping.values())
    berlin_listings['cancellation_policy'] = berlin_listings['cancellation_policy'] \
        .replace(cancellation_policy_mapping.keys(), cancellation_policy_mapping.values())
    berlin_listings['neighbourhood'] = berlin_listings['neighbourhood'] \
        .replace(neighbourhood_mapping.keys(), neighbourhood_mapping.values())


def fill_missing_values():
    berlin_listings['bathrooms'] = berlin_listings['bathrooms'].fillna(get_mean_value(berlin_listings['bathrooms']))
    berlin_listings['bedrooms'] = berlin_listings['bedrooms'].fillna(get_mean_value(berlin_listings['bedrooms']))
    berlin_listings['beds'] = berlin_listings['beds'].fillna(get_mean_value(berlin_listings['beds']))
    berlin_listings['security_deposit'] = clean_price(berlin_listings['security_deposit'])
    berlin_listings['security_deposit'] = berlin_listings['security_deposit'].\
        fillna(get_mean_value(berlin_listings['security_deposit']))
    berlin_listings['cleaning_fee'] = clean_price(berlin_listings['cleaning_fee'])
    berlin_listings['cleaning_fee'] = berlin_listings['cleaning_fee'].\
        fillna(get_mean_value(berlin_listings['cleaning_fee']))
      #extra_people => price for aditional guest
    berlin_listings['extra_people'] = clean_price(berlin_listings['extra_people'])
    berlin_listings['review_scores_value'] = berlin_listings['review_scores_value'].\
        fillna(get_mean_value(berlin_listings['review_scores_value']))


def transform_coordinates(coordinates):
    coord_splitted = coordinates.split(',')
    distance = abs(math.sqrt((float(coord_splitted[0]) - berlin_center[0]) ** 2
                             + (float(coord_splitted[1]) - berlin_center[1]) ** 2))
    return distance*100


def handle_coordinates(df, column_lat, column_long):
    df['location'] = df[column_lat].astype(str) + ',' + df[column_long].astype(str)
    df['location'] = df['location'].apply(lambda x: transform_coordinates(x))
    df.drop(columns=column_lat, inplace=True)
    df.drop(columns=column_long, inplace=True)


def handle_na_values(df):
    df = df.apply(lambda x: x.replace(['f', 't'], [0, 1]) if x.name in ['instant_bookable',
                                                                        'require_guest_phone_verification'] else x)
    df = df.apply(lambda x: x.fillna(0) if x.name in ['instant_bookable',
                                                      'require_guest_phone_verification',
                                                      'zipcode', 'host_since'] else x)
    df = df.apply(lambda x: x.fillna(mh.get_median_value(x)) if x.dtype.kind in 'iufc' else x)
    df = df[~df['amenities'].isin([0])]
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]


def unbox_listings():
    fill_missing_values()
    map_string_properties_to_numbers()
    for index, amenities in enumerate(berlin_listings['amenities'].values):
        berlin_listings.at[index, 'amenities'] = amenitie_utility.get_points_for_amentities(amenities)
    for index, host_verifications in enumerate(berlin_listings['host_verifications'].values):
        berlin_listings.at[index, 'host_verifications'] = verification_utility.get_points_for_verification(host_verifications)
    regex_str = "(\n[0-9]*)|( [a-zA-Z]*)|(,[a-zA-Z]*)|([a-z].)"
    berlin_listings['zipcode'] = berlin_listings['zipcode'].str.replace(regex_str, "").astype(float)
    df = berlin_listings[['id', 'latitude', 'longitude',
                          'property_type', 'room_type', 'bedrooms', 'bed_type', 'amenities',
                          'guests_included', 'extra_people', 'minimum_nights', 'number_of_reviews',
                          'cancellation_policy', 'accommodates', 'zipcode',
                          'neighbourhood', 'instant_bookable',
                          'require_guest_phone_verification', 'host_verifications',
                          'summary', 'description', 'host_since']]
    df = df.drop_duplicates()
    handle_coordinates(df, 'latitude', 'longitude')
    df = mh.handle_descriptive_features(df, ['summary', 'description'])
    df = mh.handle_host_duration(df)
    df = handle_na_values(df)
    return df


def clear_outliers(dataframe):
    dataframe = dataframe[dataframe['price'] < 600]
    dataframe = dataframe[dataframe['amenities'] < 65]
    dataframe = dataframe[dataframe['extra_people'] < 40]
    dataframe = dataframe[dataframe['minimum_nights'] < 35]
    dataframe = dataframe[dataframe['number_of_reviews'] > 0]
    dataframe = dataframe[dataframe['number_of_reviews'] < 220]
    dataframe = dataframe[dataframe['zipcode'] < 12500]
    mh.scale_data(dataframe)
    return dataframe


def get_berlin_dataset():
    print("colecting berlin data")
    unboxed_calendar = unbox_calendar()
    #print(unboxed_calendar.head(5))
    unboxed_listings = unbox_listings()
    #print(unboxed_listings.head(5))
    df = mh.collect_data(unboxed_calendar, unboxed_listings)
    df = clear_outliers(df)
    df.to_csv('./Datasets/cleaned/cleaned.csv')
    df = df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    #print(df.isna().sum())
    print(df.shape)
    print(df.head(10).to_string())
    return df

# print('change calendar summary...')
# unboxed_calendar = unbox_calendar()
# print(unboxed_calendar.head(10))
# print('change listings summary')
# unbox_listings = unbox_listings()
# print(unbox_listings.head(10))
# print('clean berlin dataset')
# df = get_berlin_dataset()
# print(df.head(10))


