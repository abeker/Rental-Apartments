import pandas as pd
import utility.amentities as amenitie_utility
import utility.host_verifications as verification_utility
import math

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
    df = df.apply(lambda x: x.replace(['f', 't'], [0, 1]) if x.name in ['host_identity_verified', 'host_has_profile_pic',
                                                                        'instant_bookable',
                                                                        'require_guest_phone_verification'] else x)
    df = df.apply(lambda x: x.fillna(0) if x.name in ['host_identity_verified', 'host_has_profile_pic', 'instant_bookable',
                                                      'require_guest_phone_verification', 'bathrooms'] else x)
    df = df.apply(lambda x: x.fillna(get_median_value(x)) if x.dtype.kind in 'iufc' else x)
    df = df[df['host_since'].notna()]
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
    dataframe = dataframe[dataframe['price'] < 550]
    dataframe = dataframe[dataframe['amenities'] < 200]
    dataframe = dataframe[dataframe['bathrooms'] < 6]
    dataframe = dataframe[dataframe['bedrooms'] < 15]
    dataframe = dataframe[dataframe['guests_included'] < 20]
    dataframe = dataframe[dataframe['minimum_nights'] < 60]
    dataframe = dataframe[dataframe['number_of_reviews'] < 320]
    dataframe = dataframe[dataframe['host_total_listings_count'] < 40]
    dataframe = dataframe[dataframe['security_deposit'] < 4600]
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
                          'property_type', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
                          'guests_included', 'extra_people', 'minimum_nights', 'number_of_reviews',
                          'cancellation_policy', 'accommodates', 'host_total_listings_count',
                          'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                          'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                          'review_scores_value', 'reviews_per_month', 'security_deposit', 'neighbourhood',
                          'host_identity_verified', 'host_has_profile_pic', 'instant_bookable',
                          'require_guest_phone_verification', 'host_verifications', 'host_since']]
    df = handle_na_values(df)
    df['host_since_day'], df['host_since_month'], df['host_since_year'], day_of_week = transform_date(df, 'host_since')
    df.drop(columns='host_since', inplace=True)
    handle_coordinates(df, 'latitude', 'longitude')
    return df

def unbox_calendar():
    munich_calendar['price'] = clean_price(munich_calendar['price'])
    munich_calendar['price'] = munich_calendar['price'].astype(str).replace('[,]', '', regex=True).astype(float)
    df = munich_calendar[['listing_id', 'price']]
    df = handle_duplicates(df)
    df['day'], df['month'], df['year'], df['day_of_week'] = transform_date(munich_calendar, 'date')
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