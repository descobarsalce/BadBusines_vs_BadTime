
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Pre-steps. Data download - if selected:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

client = Socrata("data.cityofchicago.org", "WZEdPks1IM6q8xFmGAEpMjUrf")

data = client.get("6zsd-86xi", where="year >= 2010", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/crime.csv')

data = client.get("8pix-ypme", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/stations_location.csv')

data = client.get("t2rn-p8d7", limit=1000000)
df_temp = pd.DataFrame.from_dict(data)
df_temp.to_csv('Data/transit_stations.csv')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. Boundaries data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_coordinates(shape_df, df, lat_var, long_var):
    df[long_var] = df[long_var].astype('float')
    df[lat_var] = df[lat_var].astype('float')
    df['coordinates'] = list(zip(df[long_var], df[lat_var]))
    df['coordinates'] = df['coordinates'].apply(Point)
    geo_df = gpd.GeoDataFrame(df, geometry='coordinates')
    df_with_codes = gpd.sjoin(geo_df, gdf_boundaries, how="inner", op='intersects')
    return df_with_codes

# I have to download this one because if we save it as csv the format gets lost,
# so we need to download it every time (and it's not too heavy).
client = Socrata("data.cityofchicago.org", "WZEdPks1IM6q8xFmGAEpMjUrf")
data = client.get("bt9m-d2mf", limit=1000000)
df_boundaries = pd.DataFrame.from_dict(data)
df_boundaries['the_geom'] = df_boundaries['the_geom'].apply(shape)
gdf_boundaries = gpd.GeoDataFrame(df_boundaries).set_geometry('the_geom')

time_boundaries = time.time()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. Business license
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

licenses = pd.read_csv("Data/Business_Licenses.csv")

print("Licenses data length: ", len(licenses))
print("Licenses data nulls: ", licenses.isna().sum())

licenses = licenses.sort_values(['LICENSE ID'])

# Drop cases where we dont have essential information:
licenses.dropna(subset=['LOCATION'], inplace=True)
licenses.dropna(subset=['LICENSE TERM START DATE'], inplace=True)
licenses.dropna(subset=['LICENSE TERM EXPIRATION DATE'], inplace=True)

licenses['start_year'] = pd.DatetimeIndex(licenses['LICENSE TERM START DATE']).year
licenses['final_year'] = pd.DatetimeIndex(licenses['LICENSE TERM EXPIRATION DATE']).year
licenses['start_year'].value_counts().sort_values()
licenses['final_year'].value_counts().sort_values()
licenses = licenses[(licenses['start_year']!= 2020) & (licenses['start_year']>= 2005)
    & (licenses['start_year']!= 2019)]
licenses['LICENSE TERM START DATE']= pd.to_datetime(licenses['LICENSE TERM START DATE'])
licenses['LICENSE TERM EXPIRATION DATE']= pd.to_datetime(licenses['LICENSE TERM EXPIRATION DATE'])

duration = licenses.groupby(['ACCOUNT NUMBER','SITE NUMBER'], as_index=False).agg({'LICENSE TERM START DATE': 'min', 'LICENSE TERM EXPIRATION DATE': 'max'})
duration['duration'] = duration['LICENSE TERM EXPIRATION DATE'] - duration['LICENSE TERM START DATE']
duration['positive_duration'] = np.where(duration.duration.dt.days > 0, 'OK', 'Not OK')
duration['positive_duration'].value_counts().sort_values()
duration = duration[duration['duration'].dt.days > 0]

duration.duration.describe(percentiles = [.1,.2,.3,.4,.5,.6,.7,.8,.9])
duration.duration.describe()

duration['duration'] = duration['duration'].dt.days

licenses['APPLICATION TYPE'].value_counts()

# Now merge duration data with business features:
new_licenses = licenses[licenses['APPLICATION TYPE']=='ISSUE']

new_licenses = new_licenses.merge(duration, how='left', left_on=['ACCOUNT NUMBER','SITE NUMBER'], right_on=['ACCOUNT NUMBER','SITE NUMBER'])
geo_licenses = get_coordinates(shape_df=gdf_boundaries, df=new_licenses, lat_var='LATITUDE', long_var='LONGITUDE')

geo_licenses.to_csv('Data/geo_licenses.csv')

time_licenses = time.time()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CTA stations location data:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

stations_location = pd.read_csv("Data/stations_location.csv")
vars_to_keep = ['map_id', 'location', 'ada', 'blue', 'brn', 'g', 'o', 'p', 'pexp', 'pnk', 'red']
stations_location = stations_location[vars_to_keep]
stations_location['latitude'] = None
stations_location['longitude'] = None
for row in range(0, len(stations_location)):
    location = ast.literal_eval(stations_location['location'][row])
    stations_location['latitude'].at[row]  = location['latitude']
    stations_location['longitude'].at[row] = location['longitude']
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

#transit = transit_stations.merge(stations_location, left_on='station_id', right_on='map_id')

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

df = pl.open_csv_data("Data/Processed_Data.csv")
stations_location = pd.read_csv('Data/stations.csv')
df = df[['ID', 'LICENSE ID', 'LONGITUDE', 'LATITUDE']]

df['key'] = 0
stations_location['key'] = 0

radius_stations = df.merge(stations_location, how='outer', left_on='key', right_on='key')
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

# Now compute number of stations within a range:
radius_stations = radius_stations[radius_stations['km'] <= 10]

df_ids = pd.DataFrame()
df_ids['LICENSE ID'] = radius_stations['LICENSE ID'].unique()
for rad in [1, 3, 5, 10]:
    temp_db = radius_stations[radius_stations['km'] <= rad]
    temp_db = temp_db.groupby(['LICENSE ID'], as_index=False).agg(sum_agg)
    lines_names = ['ada', 'blue', 'brn', 'g', 'o', 'p', 'pexp', 'pnk', 'red']
    for var in lines_names:
        temp_db.rename(columns={var: var + str(rad)}, inplace=True)
    df_ids = df_ids.merge(temp_db, how='left', left_on='LICENSE ID', right_on='LICENSE ID')

df_ids.to_csv('Data/distances_radius.csv')

time_station = time.time()

#######################################################
# Crime data:
#######################################################

crime = pd.read_csv('Data/crime.csv')
to_keep_crime = ['arrest', 'domestic', 'fbi_code', 'iucr', 'location_description', 'primary_type', 'year', 'latitude', 'longitude']
crime = crime[to_keep_crime]

# Merge with geographical data - I'll keep non-missing locations only:
crime = crime[(crime.longitude.isnull()==False) & (crime.latitude.isnull()==False)]
crime = get_coordinates(shape_df=gdf_boundaries, df=crime, lat_var='latitude', long_var='longitude')

# Define the aggregation calculations
aggregations = {
 'arrest': 'sum',
 'domestic': 'sum',
 'fbi_code': 'count',
 'iucr': 'count',
 'location_description': 'count',
 'primary_type': 'count',
}

# Perform groupby aggregation by "month", but only on the rows that are of type "call"
crime_block = crime.groupby(['year','tract_bloc'], as_index=False).agg(aggregations)
crime_block.reset_index()

crime_block.to_csv('Data/crime_block_level.csv')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Merge data:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

geo_licenses = pd.read_csv('Data/geo_licenses.csv')
crime_block  = pd.read_csv('Data/crime_block_level.csv')
df_ids       = pd.read_csv('Data/distances_radius.csv')

all_data = geo_licenses.merge(crime_block, how='left', left_on='tract_bloc', right_on='tract_bloc', indicator=True)

all_data = geo_licenses.merge(df_ids, how='left', left_on='LICENSE ID', right_on='LICENSE ID', indicator=True)

all_data['less_1_year'] = np.where((all_data['duration'] <= 365), 1, 0)
all_data['1_2_years'] = np.where((all_data['duration'] > 365)   & (all_data['duration'] <= 2*365), 1, 0)
all_data['2_3_years'] = np.where((all_data['duration'] > 2*365) & (all_data['duration'] <= 3*365), 1, 0)
all_data['more_3_years'] = np.where((all_data['duration'] > 3*365), 1, 0)

all_data['start_year'] = pd.to_datetime(all_data['start_year'].astype(str), format='%Y')

all_data.to_csv('Data/Processed_Data.csv')
