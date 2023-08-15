import pandas as pd
from pymer4.models import Lmer
import pickle
import os
import sys
import pipeline_utilities as pu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from collections import defaultdict
from multiprocessing import Pool

def str_to_int(x):
    try:
        return int(x)
    except:
        return np.nan

#Load outcomes
with open(r'outcome_measures.pkl', 'rb') as f:
    all_agg_pats = pickle.load(f)['all_agg_pats']

#load demographics
#because each visit generates a new demographics report. Now that we operate on the visit-level, we want duplicate entries
all_demographics = pd.read_pickle('demographic_data.pkl')
    
#what medications are rescue medications(1), which are ASMs(0), and which aren't useful to us (2)
med_classes = pd.read_csv('asm_usages.csv', index_col=0)
asm_generics = set(med_classes.loc[med_classes['class'] == 0].index)
    
#load medications   
all_meds = pd.read_pickle('medication_data.pkl')
#drop duplicated entries and keep only outpatient medications
all_meds = all_meds.drop_duplicates(subset=all_meds.columns[:-1])
all_meds = all_meds.loc[all_meds.ORDER_MODE != 'Inpatient']
#keep only the name of the drug
all_meds['DESCRIPTION'] = pu.get_all_asm_names_from_description('ASM_list_07252023.csv',
                                                      'exclusionary_ASM_lists.csv',
                                                      all_meds, 'DESCRIPTION')
#keep only drugs we care about
all_meds = all_meds.loc[all_meds['DESCRIPTION'].isin(asm_generics)]

#we only care about medications, demographics, and outcomes that we have shared data for
shared_mrns = set(all_meds.MRN) & set(all_demographics.MRN) & set([pat.pat_id for pat in all_agg_pats])
meds = all_meds.loc[all_meds.MRN.isin(shared_mrns)]
demographics = all_demographics.loc[all_demographics.MRN.isin(shared_mrns)]
agg_pats = [pat for pat in all_agg_pats if pat.pat_id in shared_mrns]

#print some basic info
print(f"Number of patients after removing patients missing from datasets: {len(agg_pats)}")
print(f"Number of visits: {np.sum([len(pat.aggregate_visits) for pat in agg_pats])}")

#for each agg_pat, add what medications they are taking, removing duplicates, and including the earliest starting date and latest ending date of a particular medication
def add_medications_to_pats(i):
    pat = agg_pats[i]
    
    pat_meds = meds.loc[meds.MRN == pat.pat_id]
    
    #pat.medications is normally a list. Change it to an empty dict
    pat.medications = {}
    
    for idx, row in pat_meds.iterrows():
        #create the medication entry
        med_start_date = row.START_DATE if not pd.isnull(row.START_DATE) else row.ORDERING_DATE
        
        #if this is a new medication, add it to the dictionary
        if row.DESCRIPTION not in pat.medications:
            pat.medications[row.DESCRIPTION] = {'name':row.DESCRIPTION, 'start_date':med_start_date, 'end_date':row.END_DATE}
        #if this medication already exists, update the entry's start and end dates
        else:
            #check if there's a nan in the start date
            if pd.isnull(pat.medications[row.DESCRIPTION]['start_date']):
                pat.medications[row.DESCRIPTION]['start_date'] = med_start_date
            elif med_start_date < pat.medications[row.DESCRIPTION]['start_date']:
                pat.medications[row.DESCRIPTION]['start_date'] = med_start_date
            
            #check if there's a nan in the end date
            if pd.isnull(pat.medications[row.DESCRIPTION]['end_date']):
                pat.medications[row.DESCRIPTION]['end_date'] = row.END_DATE
            elif row.END_DATE > pat.medications[row.DESCRIPTION]['end_date']:
                pat.medications[row.DESCRIPTION]['end_date'] = row.END_DATE
                
    return i, pat
with Pool(processes=23) as pool:
    for pool_out in pool.imap_unordered(add_medications_to_pats, range(len(agg_pats))):
        agg_pats[pool_out[0]] = pool_out[1]

#load income data. We want columns S1901_C01_001E (estimate # of housholds), S1901_C01_012E (median income), S1901_C01_012M (margin of error), GEO_ID, NAME
raw_income_data = pd.read_csv('map_data/ACSST5Y2021.S1901-Data.csv', skiprows=[1])[['GEO_ID', 'NAME', 'S1901_C01_001E', 'S1901_C01_012E', 'S1901_C01_012M']]
income_data = pd.DataFrame({'ZCTA':raw_income_data['NAME'].apply(lambda x: x.split()[1]), 
                            'med_income':raw_income_data['S1901_C01_012E'].apply(lambda x: str_to_int(x)),
                            'med_num_household':raw_income_data['S1901_C01_001E'].apply(lambda x: str_to_int(x))}).set_index('ZCTA')
zcta_codes = list(income_data.index)

# Add income data to demographics info
demographics['ZCTA_income'] = demographics.apply(lambda x: income_data.loc[x.ZIP[:5], 'med_income'] if x.ZIP[:5] in income_data.index else np.nan, axis=1)

#get insurance info
insurance_coding = pd.read_excel('payers_CE.xlsx')
insurance_coding = dict(zip(insurance_coding.insurance, insurance_coding['public=0']))
insurance_coding = defaultdict(lambda: np.nan , insurance_coding)
demographics['is_private_insurance'] = demographics['PAYOR_NAME'].apply(lambda x:insurance_coding[x])

#assign each patient in agg_pats an ID, mapping their MRN to this ID. 
mrn_to_id = {}
ct = 0
for pat in agg_pats:
    mrn_to_id[pat.pat_id] = ct
    ct += 1

#a dictionary to convert the number of medications to string, and limit it to 4+.
num_med_limiter = {0:'0', 1:'1', 2:'2', 3:'3'}
num_med_limiter = defaultdict(lambda: "4+" , num_med_limiter)

def get_patient_visit_datum(i):
    pat = agg_pats[i]
    pat_visit_info = []
    
    #skip patients with less than 3 visits
    if len(pat.aggregate_visits) < 3:
        return i, pat_visit_info
    
    sorted_visits = sorted(pat.aggregate_visits, key=lambda x:x.visit_date)    
    for i in range(1, len(sorted_visits)):
        vis = sorted_visits[i]
        
        #skip IDK visits
        if vis.hasSz == 2:
            continue
        
        #find the associated demographic info
        visit_demographics = demographics.loc[(demographics['MRN'] == pat.pat_id) & (demographics['CONTACT_DATE'] == vis.visit_date)]
        
        #skip visits where we're missing demographic info
        if len(visit_demographics) == 0:
            continue
        
        #get the most complete one - the one with the least nans
        visit_demographics = visit_demographics.iloc[np.argmin(visit_demographics.isnull().sum(axis=1))]
        
        #Count how many ASMs they're taking at this visit and cap ASMs to 0,1,2,3,4+
        num_meds = num_med_limiter[np.sum([(pat.medications[med]['start_date'] <= vis.visit_date) & (pat.medications[med]['end_date'] >= vis.visit_date) for med in pat.medications])]
        
        #categorize income
        #income groups:
            #100k+:ref
            #75-100k
            #50-75k
            #<50k
        income=None
        if visit_demographics.ZCTA_income < 50000:
            income="<50k"
        elif visit_demographics.ZCTA_income >= 50000 and visit_demographics.ZCTA_income < 75000:
            income="50k-75k"
        elif visit_demographics.ZCTA_income >= 75000 and visit_demographics.ZCTA_income < 100000:
            income="75k-100k"
        elif visit_demographics.ZCTA_income >= 100000:
            income=">100k"
            
        #categorize age
        #age groups
            #18-40:ref
            #40-65
            #65+
        age=None
        if visit_demographics.AGE >= 18 and visit_demographics.AGE < 40:
            age="18-40"
        if visit_demographics.AGE >= 40 and visit_demographics.AGE < 65:
            age="40-65"
        elif visit_demographics.AGE >= 65:
            age=">65"
            
        #categorize ethnicity. Skip if no ethnicity is provided
        ethnicity = visit_demographics.ETHNICITY
        if pd.isnull(ethnicity) or ethnicity == 'Patient Declined':
            continue
        
        #categorize race
        race=None
        if visit_demographics.RACE == 'White' or visit_demographics.RACE == 'HLW-Hispanic Latino/White':
            race='White'
        elif visit_demographics.RACE == 'Black or African American' or visit_demographics.RACE == 'HLB-Hispanic Latino/Black':
            race='Black'
        elif visit_demographics.RACE == 'Asian':
            race='Asian'
        else:
            race='Other'
        
        #gather information about this visit
        #initialize this datum
        visit_datum = {'patient_ID': mrn_to_id[pat.pat_id],
                       'gender': int(visit_demographics.GENDER != 'M'), #Men is the reference group (0)
                       'race': race, #White is the reference group (0)
                       'ethnicity':int(ethnicity != 'Not Hispanic or Latino'), #Non-Hispanic is the reference group
                       'age': age,
                       'insurance_type': 1 - visit_demographics.is_private_insurance, #Private insurance is the reference group (0)
                       'income': income,
                       'num_asms': num_meds,
                       'months_since_last_visit': (vis.visit_date - sorted_visits[i-1].visit_date).days/(365/12),
                       'hasSz': vis.hasSz}
        
        pat_visit_info.append(visit_datum)
    return i, pat_visit_info
#for each visit that we have outcome measures on, find the demographics and medication information, matching by MRN and date
visit_df = []
with Pool(processes=23) as pool:
    for pool_out in pool.imap_unordered(get_patient_visit_datum, range(len(agg_pats))):
        visit_df.append(pool_out[1])
visit_df = [item for sublist in visit_df for item in sublist]
visit_df = pd.DataFrame(visit_df).dropna()

print('\n\n')
print("Data has finished processing.")
print("\n")
print(f"Available fields: {list(visit_df.columns)}")
print(f"Total number of visits: {len(visit_df)}")
print("\n")

print("Field breakdown: ")
for col in visit_df.columns:
    print('\n')
    print(col)
    if len(visit_df[col].unique()) > 25:
        print(f"Spanning [{visit_df[col].min()}, {visit_df[col].max()}]")
    else:
        print(f"{visit_df[col].value_counts()}")
    print('\n')

print("Calculating univariate analyses")
print('\n\n')

print('\n===================================================\n')
print("Univariate Gender vs. HasSz: 'hasSz ~ gender + months_since_last_visit + (1 | patient_ID)'")
print()
uni_gender_mdl = Lmer("hasSz ~ gender + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_gender_mdl.fit())
with open('outcomes_vs_demographics_regression_outputs/uni_gender_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_gender_mdl, f)


print('\n===================================================\n')
print("Univariate Race vs. HasSz: 'hasSz ~ race + months_since_last_visit + (1 | patient_ID)'")
print()
uni_race_mdl = Lmer("hasSz ~ race + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_race_mdl.fit(factors={'race':["White", "Black", "Asian", "Other"]}))
with open('outcomes_vs_demographics_regression_outputs/uni_race_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_race_mdl, f)
    
print('\n===================================================\n')
print("Univariate Ethnicity vs. HasSz: 'hasSz ~ ethnicity + months_since_last_visit + (1 | patient_ID)'")
print()
uni_ethnicity_mdl = Lmer("hasSz ~ ethnicity + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_ethnicity_mdl.fit())
with open('outcomes_vs_demographics_regression_outputs/uni_ethnicity_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_ethnicity_mdl, f)

print('\n===================================================\n')
print("Univariate Age vs. HasSz: 'hasSz ~ age + months_since_last_visit + (1 | patient_ID)'")
print()
uni_age_mdl = Lmer("hasSz ~ age + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_age_mdl.fit(factors={'age':["18-40", "40-65", ">65"]}))
with open('outcomes_vs_demographics_regression_outputs/uni_age_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_age_mdl, f)

print('\n===================================================\n')
print("Univariate Insurance_type vs. HasSz: 'hasSz ~ insurance_type + months_since_last_visit + (1 | patient_ID)'")
print()
uni_insurance_mdl = Lmer("hasSz ~ insurance_type + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_insurance_mdl.fit())
with open('outcomes_vs_demographics_regression_outputs/uni_insurance_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_insurance_mdl, f)


print('\n===================================================\n')
print("Univariate Income vs. HasSz: 'hasSz ~ income + months_since_last_visit + (1 | patient_ID)'")
print()
uni_income_mdl = Lmer("hasSz ~ income + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_income_mdl.fit(factors={'income':[">100k", "75k-100k", "50k-75k", "<50k"]}))
with open('outcomes_vs_demographics_regression_outputs/uni_income_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_income_mdl, f)

print('\n===================================================\n')
print("Univariate Num_asms vs. HasSz: 'hasSz ~ num_asms + months_since_last_visit + (1 | patient_ID)'")
print()
uni_asms_mdl = Lmer("hasSz ~ num_asms + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(uni_asms_mdl.fit(factors={'num_asms':['0','1', '2', '3', '4+']}))
with open('outcomes_vs_demographics_regression_outputs/uni_asms_mdl_py38.pkl', 'wb') as f:
    pickle.dump(uni_asms_mdl, f)

print("\nCalculating multivariate analysis")
print('\n')

print('\n===================================================\n')
print("Multivariate vs. HasSz: 'hasSz ~ gender + race + ethnicity + age + insurance_type + income + num_asms + months_since_last_visit + (1 | patient_ID)'")
print()
multi_mdl = Lmer("hasSz ~ gender + race + ethnicity + age + insurance_type + income + num_asms + months_since_last_visit + (1 | patient_ID)", data=visit_df, family='binomial')
print(multi_mdl.fit(factors={'num_asms':['0','1', '2', '3', '4+'], 
                             'income':[">100k", "75k-100k", "50k-75k", "<50k"], 
                             'age':["18-40", "40-65", ">65"],
                             'race':["White", "Black", "Asian", "Other"]},
                    control="optimizer='bobyqa', optCtrl=list(maxfun=5e6)"))
with open('outcomes_vs_demographics_regression_outputs/multi_mdl_py38.pkl', 'wb') as f:
    pickle.dump(multi_mdl, f)
    
    
    