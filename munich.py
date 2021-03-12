import pandas as pd
import math
import utility.amentities as utility

munich_calendar = pd.read_csv('./Datasets/Munich/calendar.csv')
munich_listings = pd.read_csv('./Datasets/Munich/listings.csv')

def get_mean_value(column):
    return math.floor(column.mean())

def clean_price(column):
    return column.str.replace('$', '', regex=True).astype(str)

def get_mapping(column):
    return dict([(y, x + 1) for x, y in enumerate(sorted(set(column)))])

def handle_duplicates(df):
    df = df.drop_duplicates()
    df = df.groupby('listing_id').mean().reset_index()
    return df

def map_string_properties_to_numbers():
    bed_type_mapping = get_mapping(munich_listings['bed_type'].unique())
    property_type_mapping = get_mapping(munich_listings['property_type'].unique())
    room_type_mapping = get_mapping(munich_listings['room_type'].unique())
    cancellation_policy_mapping = get_mapping(munich_listings['cancellation_policy'].unique())
    munich_listings['bed_type'] = munich_listings['bed_type']\
        .replace(bed_type_mapping.keys(), bed_type_mapping.values())
    munich_listings['property_type'] = munich_listings['property_type'] \
        .replace(property_type_mapping.keys(), property_type_mapping.values())
    munich_listings['room_type'] = munich_listings['room_type'] \
        .replace(room_type_mapping.keys(), room_type_mapping.values())
    munich_listings['cancellation_policy'] = munich_listings['cancellation_policy'] \
        .replace(cancellation_policy_mapping.keys(), cancellation_policy_mapping.values())

def unbox_listings():
    munich_listings['bathrooms'] = munich_listings['bathrooms'].fillna(get_mean_value(munich_listings['bathrooms']))
    munich_listings['bedrooms'] = munich_listings['bedrooms'].fillna(get_mean_value(munich_listings['bedrooms']))
    munich_listings['beds'] = munich_listings['beds'].fillna(get_mean_value(munich_listings['beds']))
    munich_listings['extra_people'] = clean_price(munich_listings['extra_people'])
    map_string_properties_to_numbers()
    for index, amenities in enumerate(munich_listings['amenities'].values):
        munich_listings.at[index, 'amenities'] = utility.get_points_for_amentities(amenities)

    df = munich_listings[['id', 'latitude', 'longitude',
                          'property_type', 'room_type', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
                          'guests_included', 'extra_people', 'minimum_nights', 'number_of_reviews',
                          'cancellation_policy']]
    return df

def unbox_calendar():
    munich_calendar['price'] = clean_price(munich_calendar['price'])
    munich_calendar['date'] = pd.to_datetime(munich_calendar['date'])
    munich_calendar['available'] = munich_calendar['available'].replace(['f', 't'], [0, 1])
    munich_calendar.drop('adjusted_price', axis=1, inplace=True)
    munich_calendar['price'] = munich_calendar['price'].astype(str).replace('[,]', '', regex=True).astype(float)
    df = munich_calendar[['listing_id', 'price']]
    df = handle_duplicates(df)
    return df

def collect_data(df_calendar, df_listings):
    print('collecting...')
    merged_df = pd.merge(df_listings, df_calendar.rename(columns={'listing_id': 'id'}), on='id', how='left')
    return merged_df

def get_munich_data():
    unboxed_calendar = unbox_calendar()
    unboxed_listings = unbox_listings()
    df_collected = collect_data(unboxed_calendar, unboxed_listings)
    return df_collected