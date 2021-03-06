import pandas as pd

munich_calendar = pd.read_csv('./Datasets/Munich/calendar.csv')
munich_listings = pd.read_csv('./Datasets/Munich/listings.csv')

def unbox_listings():
    munich_listings.drop(
        ['listing_url', 'scrape_id', 'last_scraped', 'summary', 'space', 'description', 'experiences_offered',
         'neighborhood_overview', 'notes', 'transit', 'interaction', 'thumbnail_url', 'medium_url', 'picture_url',
         'xl_picture_url', 'host_url', 'host_name', 'host_location', 'host_about', 'host_acceptance_rate',
         'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'street', 'neighbourhood',
         'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'market', 'smart_location',
         'country_code', 'country', 'is_location_exact', 'calendar_updated', 'has_availability', 'availability_30',
         'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped', 'requires_license',
         'license', 'jurisdiction_names'],
        axis=1, inplace=True)

def unbox_calendar():
    munich_calendar['price'] = munich_calendar['price'].str.replace('$', '', regex=True).astype(str)
    munich_calendar['adjusted_price'] = munich_calendar['adjusted_price'].str.replace('$', '', regex=True).astype(str)
    munich_calendar['date'] = pd.to_datetime(munich_calendar['date'])
    munich_calendar['available'] = munich_calendar['available'].replace(['f', 't'], [0, 1])
    munich_calendar.drop('adjusted_price', axis=1, inplace=True)

def collect_data():
    print('collecting...')
    return 'something collected'

def get_munich_data():
    unbox_calendar()
    unbox_listings()
    df_collected = collect_data()
    # print(munich_calendar.head(5).to_string())
    print(munich_listings.columns)
    return df_collected