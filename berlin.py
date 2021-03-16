import pandas as pd
import math
import numpy as np

berlin_calendar = pd.read_csv('./Datasets/Berlin/calendar_summary.csv', low_memory=False)
berlin_listings = pd.read_csv('./Datasets/Berlin/listings_summary.csv', low_memory=False)

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
    return df


def map_string_properties_to_numbers():
    property_type_mapping = get_mapping(berlin_listings['property_type'].unique())
    room_type_mapping = get_mapping(berlin_listings['room_type'].unique())
    bed_type_mapping = get_mapping(berlin_listings['bed_type'].unique())
    berlin_listings['property_type'] = berlin_listings['property_type'] \
        .replace(property_type_mapping.keys(), property_type_mapping.values())
    berlin_listings['room_type'] = berlin_listings['room_type'] \
        .replace(room_type_mapping.keys(), room_type_mapping.values())
    berlin_listings['bed_type'] = berlin_listings['bed_type'] \
        .replace(bed_type_mapping.keys(), bed_type_mapping.values())

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


def unbox_listings():
    fill_missing_values()
    map_string_properties_to_numbers()
    df = berlin_listings [['id', 'latitude', 'longitude', 'property_type', 'room_type', 'bathrooms', 'bedrooms',
                           'bed_type', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people',
                           'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_value']]
    df = df.drop_duplicates()
    return df


def get_berlin_dataset():
    unboxed_calendar = unbox_calendar()
    unboxed_listings = unbox_listings()
    df = pd.merge(unboxed_listings, unboxed_calendar.rename(columns={'listing_id': 'id'}), on='id', how='left')
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


