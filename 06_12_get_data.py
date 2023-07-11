import pandas as pd
import geopandas as gpd
import numpy as np
from sodapy import Socrata
from shapely.geometry import shape
from shapely.geometry import Point
import pipeline as pl
import pipeline_models as plm
import ast
import time

start = time.time()

# Data download - Crime data
client = Socrata("data.cityofchicago.org", "WZEdPks1IM6q8xFmGAEpMjUrf")
data = client.get("6zsd-86xi", where="year >= 2010", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/crime.csv')

# Data download - Stations location
data = client.get("8pix-ypme", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/stations_location.csv')

# Data download - Transit stations
data = client.get("t2rn-p8d7", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/transit_stations.csv')

# Boundaries data
def get_coordinates(shape_df, df, lat_var, long_var):
    df[long_var] = df[long_var].astype('float')
    df[lat_var] = df[lat_var].astype('float')
    df['coordinates'] = list(zip(df[long_var], df[lat_var]))
    df['coordinates'] = df['coordinates'].apply(Point)
    geo_df = gpd.GeoDataFrame(df, geometry='coordinates')
    df_with_codes = gpd.sjoin(geo_df, gdf_boundaries, how="inner", op='intersects')
    return df_with_codes

# Download boundaries data
client = Socrata("data.cityofchicago.org", "WZEdPks1IM6q8xFmGAEpMjUrf")
data = client.get("bt9m-d2mf", limit=1000000)
df_boundaries = pd.DataFrame.from_dict(data)
df_boundaries['the_geom'] = df_boundaries['the_geom'].apply(shape)
gdf_boundaries = gpd.GeoDataFrame(df_boundaries).set_geometry('the_geom')

time_boundaries = time.time()

# Business license data
licenses = pd.read_csv("Data/Business_Licenses.csv")

# Preprocessing business license data
licenses.dropna(subset=['LOCATION', 'LICENSE TERM START DATE', 'LICENSE TERM EXPIRATION DATE'], inplace=True)
licenses['start_year'] = pd.DatetimeIndex(licenses['LICENSE TERM START DATE']).year
licenses['final_year'] = pd.DatetimeIndex(licenses['LICENSE TERM EXPIRATION DATE']).year
licenses = licenses[(licenses['start_year'] != 2020) & (licenses['start_year'] >= 2005) & (licenses['start_year'] != 2019)]
licenses['LICENSE TERM START DATE'] = pd.to_datetime(licenses['LICENSE TERM START DATE'])
licenses['LICENSE TERM EXPIRATION DATE'] = pd.to_datetime(licenses['LICENSE TERM EXPIRATION DATE'])

# Merge duration data with business features
new_licenses = licenses[licenses['APPLICATION TYPE'] == 'ISSUE']
new_licenses = new_licenses.merge(duration, how='left', on=['ACCOUNT NUMBER', 'SITE NUMBER'])
geo_licenses = get_coordinates(shape_df=gdf_boundaries, df=new_licenses, lat_var='LATITUDE', long_var='LONGITUDE')

geo_licenses.to_csv('Data/geo_licenses.csv')

time_licenses = time.time()

# CTA stations location data
stations_location = pd.read_csv("Data/stations_location.csv")
vars_to_keep = ['map_id', 'location', 'ada', 'blue', 'brn', 'g', 'o', 'p', 'pexp', 'pnk', 'red']
stations_location = stations_location[vars_to_keep]
stations_location['latitude'] = None
stations_location['longitude'] = None
for row in range(0, len(stations_location)):
    location = ast.literal_eval(stations_location['location'][row])
    stations_location.at[row, 'latitude'] = location['latitude']
    stations_location.at[row, 'longitude'] = location['longitude']
stations_location.drop(['location'], axis=1, inplace=True)
stations_location = pd.read_csv('Data/stations.csv')
aggregations = {
    'ada': 'max',
    'blue': 'max',
    'brn': 'max',
    'g': 'max',
    'o': 'max',
    'p': 'max',
    'pexp': 'max',
    'pnk': 'max',
    'red': 'max',
    'longitude': 'first',
    'latitude': 'first'
}
stations_location = stations_location.groupby(['map_id'], as_index=False).agg(aggregations)

transit_stations = pd.read_csv("Data/transit_stations.csv")
transit_stations['month_beginning'] = pd.to_datetime(transit_stations['month_beginning'])
features_stations = ['station_id', 'month_beginning', 'avg_saturday_rides', 'avg_sunday_holiday_rides', 'avg_weekday_rides', 'monthtotal']
transit_stations = transit_stations[features_stations]

# Calculate distance between license locations and stations
df = pl.open_csv_data("Data/Processed_Data.csv")
stations_location = pd.read_csv('Data/stations.csv')
df = df[['ID', 'LICENSE ID', 'LONGITUDE', 'LATITUDE']]

df['key'] = 0
stations_location['key'] = 0

radius_stations = df.merge(stations_location, how='outer', on='key')
radius_stations['km'] = haversine_np(radius_stations['LONGITUDE'], radius_stations['LATITUDE'], radius_stations['longitude'], radius_stations['latitude'])

sum_agg = {
    'ada': 'sum',
    'blue': 'sum',
    'brn': 'sum',
    'g': 'sum',
    'o': 'sum',
    'p': 'sum',
    'pexp': 'sum',
    'pnk': 'sum',
    'red': 'sum'
}

# Filter stations within a certain radius
radius_stations = radius_stations[radius_stations['km'] <= 10]

df_ids = pd.DataFrame()
df_ids['LICENSE ID'] = radius_stations['LICENSE ID'].unique()
for rad in [1, 3, 5, 10]:
    temp_db = radius_stations[radius_stations['km'] <= rad]
    temp_db = temp_db.groupby(['LICENSE ID'], as_index=False).agg(sum_agg)
    lines_names = ['ada', 'blue', 'brn', 'g', 'o', 'p', 'pexp', 'pnk', 'red']
    for var in lines_names:
        temp_db.rename(columns={var: var + str(rad)}, inplace=True)
    df_ids = df_ids.merge(temp_db, how='left', on='LICENSE ID')

df_ids.to_csv('Data/distances_radius.csv')

time_station = time.time()

# Crime data
crime = pd.read_csv('Data/crime.csv')
to_keep_crime = ['arrest', 'domestic', 'fbi_code', 'iucr', 'location_description', 'primary_type', 'year', 'latitude', 'longitude']
crime = crime[to_keep_crime]

# Merge crime data with geographical data
crime = crime.dropna(subset=['latitude', 'longitude'])
crime = get_coordinates(shape_df=gdf_boundaries, df=crime, lat_var='latitude', long_var='longitude')

# Aggregation of crime data
aggregations = {
    'arrest': 'sum',
    'domestic': 'sum',
    'fbi_code': 'count',
    'iucr': 'count',
    'location_description': 'count',
    'primary_type': 'count',
}
crime_block = crime.groupby(['year', 'tract_bloc'], as_index=False).agg(aggregations)
crime_block.reset_index()
crime_block.to_csv('Data/crime_block_level.csv')

# Merge all data
geo_licenses = pd.read_csv('Data/geo_licenses.csv')
crime_block = pd.read_csv('Data/crime_block_level.csv')
df_ids = pd.read_csv('Data/distances_radius.csv')

all_data = geo_licenses.merge(crime_block, how='left', on='tract_bloc', indicator=True)
all_data = geo_licenses.merge(df_ids, how='left', on='LICENSE ID', indicator=True)

all_data['less_1_year'] = np.where(all_data['duration'] <= 365, 1, 0)
all_data['1_2_years'] = np.where((all_data['duration'] > 365) & (all_data['duration'] <= 2 * 365), 1, 0)
all_data['2_3_years'] = np.where((all_data['duration'] > 2 * 365) & (all_data['duration'] <= 3 * 365), 1, 0)
all_data['more_3_years'] = np.where(all_data['duration'] > 3 * 365, 1, 0)

all_data['start_year'] = pd.to_datetime(all_data['start_year'], format='%Y')
all_data.to_csv('Data/Processed_Data.csv')
