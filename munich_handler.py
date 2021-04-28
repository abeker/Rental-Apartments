import pandas as pd
import utility.amentities as amenitie_utility
import utility.host_verifications as verification_utility
import math
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

munich_calendar = pd.read_csv('./Datasets/Munich/calendar.csv',  low_memory=False)
munich_listings = pd.read_csv('./Datasets/Munich/listings.csv',  low_memory=False)
munich_center = [48.13743, 11.57549]

def get_median_value(column):
    mads = column.mad() * 1.4826
    return mads

def clean_price(column):
    return column.str.replace('$', '', regex=True).astype(str)

def get_mapping(column):
    return dict([(y, x + 1) for x, y in enumerate(sorted(set(column)))])

def handle_duplicates(df):
    df = df.drop_duplicates()
    df = df.groupby('listing_id').mean().reset_index()
    return df

def transform_coordinates(coordinates):
    coord_splitted = coordinates.split(',')
    distance = abs(math.sqrt((float(coord_splitted[0]) - munich_center[0]) ** 2
                             + (float(coord_splitted[1]) - munich_center[1]) ** 2))
    return distance*100

def handle_coordinates(df, column_lat, column_long):
    df['location'] = df[column_lat].astype(str) + ',' + df[column_long].astype(str)
    df['location'] = df['location'].apply(lambda x: transform_coordinates(x))
    df.drop(columns=column_lat, inplace=True)
    df.drop(columns=column_long, inplace=True)

def handle_na_values(df):
    df = df.apply(lambda x: x.replace(['f', 't'], [0, 1]) if x.name in ['host_has_profile_pic',
                                                                        'instant_bookable',
                                                                        'require_guest_phone_verification'] else x)
    df = df.apply(lambda x: x.fillna(0) if x.name in ['host_has_profile_pic', 'instant_bookable',
                                                      'require_guest_phone_verification', 'bathrooms'] else x)
    df = df.apply(lambda x: x.fillna(get_median_value(x)) if x.dtype.kind in 'iufc' else x)
    df = df[~df['amenities'].isin([0])]
    return df

def transform_date(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    days = pd.to_datetime(df[date_column]).dt.day
    months = pd.to_datetime(df[date_column]).dt.month
    years = pd.to_datetime(df[date_column]).dt.year
    day_of_week = pd.to_datetime(df[date_column]).dt.dayofweek
    return days, months, years, day_of_week

def clear_outliers(dataframe):
    dataframe = dataframe[dataframe['price'] < 700]
    dataframe = dataframe[dataframe['amenities'] < 200]
    dataframe = dataframe[dataframe['bedrooms'] < 15]
    dataframe = dataframe[dataframe['guests_included'] < 20]
    dataframe = dataframe[dataframe['minimum_nights'] < 32]
    dataframe = dataframe[dataframe['number_of_reviews'] < 320]
    dataframe = dataframe[dataframe['number_of_reviews'] > 10]
    apply_log(dataframe)
    return dataframe

def map_string_properties_to_numbers():
    bed_type_mapping = get_mapping(munich_listings['bed_type'].unique())
    property_type_mapping = get_mapping(munich_listings['property_type'].unique())
    room_type_mapping = get_mapping(munich_listings['room_type'].unique())
    cancellation_policy_mapping = get_mapping(munich_listings['cancellation_policy'].unique())
    munich_listings['neighbourhood'] = munich_listings['neighbourhood'].astype('str')
    neighbourhood_mapping = get_mapping(munich_listings['neighbourhood'].unique())
    munich_listings['bed_type'] = munich_listings['bed_type']\
        .replace(bed_type_mapping.keys(), bed_type_mapping.values())
    munich_listings['property_type'] = munich_listings['property_type'] \
        .replace(property_type_mapping.keys(), property_type_mapping.values())
    munich_listings['room_type'] = munich_listings['room_type'] \
        .replace(room_type_mapping.keys(), room_type_mapping.values())
    munich_listings['cancellation_policy'] = munich_listings['cancellation_policy'] \
        .replace(cancellation_policy_mapping.keys(), cancellation_policy_mapping.values())
    munich_listings['neighbourhood'] = munich_listings['neighbourhood'] \
        .replace(neighbourhood_mapping.keys(), neighbourhood_mapping.values())

def handle_descriptive_features(df, column_array):
    for column in column_array:
        df[column].loc[~df[column].isnull()] = 1    # not nan
        df[column].loc[df[column].isnull()] = 0     # nan
    return df

def handle_review(df):
    df[['first_review', 'last_review']] = df[['first_review', 'last_review']].apply(pd.to_datetime)
    df['review_range'] = (df['last_review'] - df['first_review']).dt.days
    df.drop(columns=['first_review', 'last_review'], inplace=True)

def handle_host_duration(df):
    df['host_since'] = pd.to_datetime(df['host_since'])
    curr_time = pd.to_datetime("now")
    df['host_since_days'] = (df['host_since'] - curr_time).dt.days.abs()
    df.drop(columns=['host_since'], inplace=True)

def apply_log(df):
    df['host_since_days'] = np.log(df['host_since_days'])
    df['number_of_reviews'] = np.log(df['number_of_reviews'])
    df['price'] = np.log(df['price'])

def unbox_listings():
    map_string_properties_to_numbers()
    for index, amenities in enumerate(munich_listings['amenities'].values):
        munich_listings.at[index, 'amenities'] = amenitie_utility.get_points_for_amentities(amenities)
    for index, host_verifications in enumerate(munich_listings['host_verifications'].values):
        munich_listings.at[index, 'host_verifications'] = verification_utility.get_points_for_verification(host_verifications)
    munich_listings['extra_people'] = clean_price(munich_listings['extra_people'])
    munich_listings['security_deposit'] = clean_price(munich_listings['security_deposit'])
    munich_listings['security_deposit'] = munich_listings['security_deposit'].astype(str).replace('[,]', '', regex=True).astype(float)

    df = munich_listings[['id', 'latitude', 'longitude',
                          'property_type', 'room_type', 'bedrooms', 'bed_type', 'amenities',
                          'guests_included', 'extra_people', 'minimum_nights', 'number_of_reviews',
                          'cancellation_policy', 'accommodates',
                          'neighbourhood', 'instant_bookable',
                          'require_guest_phone_verification', 'host_verifications',
                          'summary', 'description', 'host_since']]
    print(df.isna().sum())
    print(df.shape)
    handle_descriptive_features(df, ['summary', 'description'])
    handle_coordinates(df, 'latitude', 'longitude')
    handle_host_duration(df)

    df = handle_na_values(df)
    return df

def unbox_calendar():
    munich_calendar['price'] = clean_price(munich_calendar['price'])
    munich_calendar['price'] = munich_calendar['price'].astype(str).replace('[,]', '', regex=True).astype(float)
    df = munich_calendar[['listing_id', 'price']]
    df = handle_duplicates(df)
    day, month, year, df['day_of_week'] = transform_date(munich_calendar, 'date')
    return df

def collect_data(df_calendar, df_listings):
    print('collecting...')
    merged_df = pd.merge(df_calendar.rename(columns={'listing_id': 'id'}), df_listings, on='id', how='left')
    merged_df.drop(columns='id', inplace=True)
    return merged_df

def get_munich_data():
    unboxed_calendar = unbox_calendar()
    unboxed_listings = unbox_listings()
    df_collected = collect_data(unboxed_calendar, unboxed_listings)
    df_collected = clear_outliers(df_collected)
    df_collected.to_csv('./Datasets/cleaned/cleaned.csv')
    print(df_collected.head(10).to_string())
    return df_collected