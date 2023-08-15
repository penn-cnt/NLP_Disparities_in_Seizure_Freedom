import plotly.express as px
import plotly
import json
import pandas as pd
import numpy as np
import sys
import copy
import pickle
import matplotlib.pyplot as plt
import pipeline_utilities as pu
import statsmodels.api as sm
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter



def str_to_int(x):
    try:
        return int(x)
    except:
        return np.nan

#load map data
with open('map_data/de_delaware_zip_codes_geo.json', 'r') as f:
    de_geo = json.load(f)
with open('map_data/nj_new_jersey_zip_codes_geo.json', 'r') as f:
    nj_geo = json.load(f)
with open('map_data/pa_pennsylvania_zip_codes_geo.json', 'r') as f:
    pa_geo = json.load(f)   
all_geo = de_geo
all_geo['features'] += nj_geo['features'] + pa_geo['features']
zcta_codes = [zcta['properties']['ZCTA5CE10'] for zcta in all_geo['features']]
penn_coords = {'lat':39.94995021337088, 'lon':-75.19295295299449}
#====================================================================================================#
#Plot census data

#load income data. We want columns S1901_C01_001E (estimate # of housholds), S1901_C01_012E (median income), S1901_C01_012M (margin of error), GEO_ID, NAME
raw_income_data = pd.read_csv('map_data/ACSST5Y2021.S1901-Data.csv', skiprows=[1])[['GEO_ID', 'NAME', 'S1901_C01_001E', 'S1901_C01_012E', 'S1901_C01_012M']]
income_data = pd.DataFrame({'ZCTA':raw_income_data['NAME'].apply(lambda x: x.split()[1]), 
                            'median_income':raw_income_data['S1901_C01_012E'].apply(lambda x: str_to_int(x)),
                            'med_num_household':raw_income_data['S1901_C01_001E'].apply(lambda x: str_to_int(x))})
#we keep only income data that we have geo data on
income_data = income_data.loc[income_data['ZCTA'].isin(zcta_codes)]
zcta_codes = list(income_data['ZCTA'])

#load race data. We want columns B02001_001E (estimate # of people), B02001_002E (estimate # of white people), GEO_ID, NAME
raw_race_data = pd.read_csv('map_data/ACSDT5Y2021.B02001-Data.csv', skiprows=[1])[['GEO_ID', 'NAME', 'B02001_001E', 'B02001_002E']]
race_data = pd.DataFrame({'ZCTA':raw_race_data['NAME'].apply(lambda x: x.split()[1]), 
                            'num_people':raw_race_data['B02001_001E'].apply(lambda x: str_to_int(x)),
                            'num_white_people':raw_race_data['B02001_002E'].apply(lambda x: str_to_int(x))})
#we keep only race data that we have geo data on
race_data = race_data.loc[race_data['ZCTA'].isin(zcta_codes)]
#calculate proportion white
race_data['num_people'] = race_data['num_people'].replace(0, np.nan)
race_data['percent_white'] = race_data.apply(lambda x: x.num_white_people/x.num_people, axis=1)
race_data['proportion_non-white'] = 1-race_data['percent_white']

#plot median income per zip code in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(income_data, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='median_income',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                           range_color=(0,200000)
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_6.html')

#plot perecent non white per zip code in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(race_data, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='proportion_non-white',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_7.html')

#plot number of people per zip code in the tri-state area
fig = px.choropleth_mapbox(race_data, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='num_people',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
plotly.offline.plot(fig, filename='map_figures/sup_fig_8.html')

#====================================================================================================#
#plot outcomes and patient data

#load patients
with open('outcome_measures.pkl', 'rb') as f:
    all_agg_pats = pickle.load(f)['all_agg_pats']

#load demographics
#because each visit generates a new demographics report, sort by date, then drop all but the latest one
all_demographics = pd.read_pickle('demographic_data.pkl')
all_demographics = all_demographics.sort_values(by='CONTACT_DATE').drop_duplicates(subset='MRN', keep='last')
    
#we only want demographics for the patients that we have outcome measures for
mrns_with_outcomes_and_demo = set([pat.pat_id for pat in all_agg_pats]) & set(all_demographics['MRN'])
all_agg_pats = [pat for pat in all_agg_pats if pat.pat_id in mrns_with_outcomes_and_demo]
demographics = all_demographics.loc[all_demographics['MRN'].isin(mrns_with_outcomes_and_demo)]    
print(f"Total number of patients with both demographics and outcome measures: {len(demographics)}, {len(all_agg_pats)}")

#bin patients into their zcta codes
all_pat_demographics_in_zcta = {zcta:all_demographics.loc[all_demographics['ZIP'] == zcta] for zcta in zcta_codes}
patient_demographics_in_zcta = {zcta:demographics.loc[demographics['ZIP'] == zcta] for zcta in zcta_codes}
agg_pats_in_zcta = {zcta:[agg_pat for agg_pat in all_agg_pats if agg_pat.pat_id in patient_demographics_in_zcta[zcta]['MRN'].values] for zcta in zcta_codes}

#check that there is an equal number of patient demographics and agg_pats in each zcta
for zcta in patient_demographics_in_zcta:
    if len(patient_demographics_in_zcta[zcta]) != len(agg_pats_in_zcta[zcta]):
        raise
        
#patient population count per zcta, and population count normalized by # of households per zcta
#runtimewarnings are if there are 0 households in a zcta - census data error?
all_pat_count = {}
all_pat_count_norm = {}
for zcta in zcta_codes:
    all_pat_count[zcta] = len(all_pat_demographics_in_zcta[zcta]) if zcta in all_pat_demographics_in_zcta else 0
    all_pat_count_norm[zcta] = len(all_pat_demographics_in_zcta[zcta])/race_data.loc[race_data['ZCTA'] == zcta].iloc[0].num_people if zcta in all_pat_demographics_in_zcta else 0
all_pat_count = pd.DataFrame.from_dict(all_pat_count, orient='index').reset_index().rename({'index':'ZCTA', 0:'pat_count'}, axis=1)
all_pat_count_norm = pd.DataFrame.from_dict(all_pat_count_norm, orient='index').reset_index().rename({'index':'ZCTA', 0:'norm_pat_count'}, axis=1)

#for clearer visualization, replace 0's with nans.
all_pat_count['pat_count'] = all_pat_count['pat_count'].replace(0, np.nan)
all_pat_count_norm['normalized_pat_count'] = all_pat_count_norm['norm_pat_count'].replace(0, np.nan)

#plot normalized patient count per zcta in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(all_pat_count_norm, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='normalized_pat_count',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                           range_color=(0,0.01)
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_1.html')

#get the sex of all patients per zcta
all_pat_sex_per_zcta = {}
for zcta in zcta_codes:
    all_pat_sex_per_zcta[zcta] = np.sum(all_pat_demographics_in_zcta[zcta].GENDER == 'F')/len(all_pat_demographics_in_zcta[zcta]) if len(all_pat_demographics_in_zcta[zcta]) >= 3 else np.nan
all_pat_sex_per_zcta = pd.DataFrame.from_dict(all_pat_sex_per_zcta, orient='index').reset_index().rename({'index':'ZCTA', 0:'proportion_female'}, axis=1)

#plot normalized patient count per zcta in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(all_pat_sex_per_zcta, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='proportion_female',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_2.html')

#get the race of all patients per zcta
all_pat_race_per_zcta = {}
for zcta in zcta_codes:
    all_pat_race_per_zcta[zcta] = np.sum(all_pat_demographics_in_zcta[zcta].RACE != 'White')/len(all_pat_demographics_in_zcta[zcta]) if len(all_pat_demographics_in_zcta[zcta]) >= 3 else np.nan
all_pat_race_per_zcta = pd.DataFrame.from_dict(all_pat_race_per_zcta, orient='index').reset_index().rename({'index':'ZCTA', 0:'proportion_non-white'}, axis=1)

%%time
#plot normalized patient count per zcta in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(all_pat_race_per_zcta, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='proportion_non-white',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_3.html')

#get the ages of all patients per zcta
all_pat_age_per_zcta = {}
for zcta in zcta_codes:
    all_pat_age_per_zcta[zcta] = all_pat_demographics_in_zcta[zcta].AGE.mean() if len(all_pat_demographics_in_zcta[zcta]) >= 3 else np.nan
all_pat_age_per_zcta = pd.DataFrame.from_dict(all_pat_age_per_zcta, orient='index').reset_index().rename({'index':'ZCTA', 0:'mean_age'}, axis=1)

#plot normalized patient count per zcta in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(all_pat_age_per_zcta, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='mean_age',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_4.html')

#calculate the average proportion visits that are not-seizure-free in each zcta. 
hasSz_per_pat_in_zcta = {}
for zcta in agg_pats_in_zcta:
    hasSz_ct = [[visit.hasSz for visit in agg_pat.aggregate_visits if visit.hasSz != 2] for agg_pat in agg_pats_in_zcta[zcta]]
    num_vis_ct = np.sum([len(pat_vis) for pat_vis in hasSz_ct])
    hasSz_ct = np.sum([np.sum(pat_vis) for pat_vis in hasSz_ct])
    hasSz_per_pat_in_zcta[zcta] = hasSz_ct / num_vis_ct
hasSz_per_pat_in_zcta = pd.DataFrame.from_dict(hasSz_per_pat_in_zcta, orient='index').reset_index().rename({'index':'ZCTA', 0:'avg_seizure_rate'}, axis=1)

#keep only ZCTAs with at least 3 patients
hasSz_per_pat_in_zcta = hasSz_per_pat_in_zcta.loc[hasSz_per_pat_in_zcta['ZCTA'].isin(pat_count.loc[pat_count.pat_count >= 3].ZCTA)]

#plot average hasSz visits per patient per zcta in the tri-state NJ, PA, DE area
fig = px.choropleth_mapbox(hasSz_per_pat_in_zcta, 
                           geojson=all_geo,
                           locations='ZCTA', 
                           featureidkey='properties.ZCTA5CE10',
                           color='avg_seizure_rate',
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-positron",
                           center=penn_coords,
                           opacity=0.33,
                          )
fig.add_scattermapbox(fill='toself', fillcolor='red', #marker={'size':300},
                      lat=[penn_coords['lat']], lon=[penn_coords['lon']])
plotly.offline.plot(fig, filename='map_figures/sup_fig_5.html')