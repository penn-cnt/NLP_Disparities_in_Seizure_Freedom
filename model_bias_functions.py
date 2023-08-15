import pandas as pd
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import pipeline_utilities as pu
from scipy.stats import fisher_exact,kstest,uniform,mannwhitneyu
import seaborn as sns
import scipy as sc
from collections import defaultdict
from datetime import datetime

def str_to_int(x):
    try:
        return int(x)
    except:
        return np.nan
        
def score_to_probs(scores):
    scores = scores.replace('[','').replace(']','').replace(' ',',').split(',')
    scores = [sc for sc in scores if len(sc)>0]
    sum_probas = np.sum([np.exp(float(i)) for i in scores])
    all_probas = np.array([np.exp(float(i))/sum_probas for i in scores])
    return [all_probas[0],(all_probas[1]+all_probas[2])]
        
def get_aggregate_hasSz(pred_list):
    """Performs plurality voting on a list of hasSz predictions.
    Copied and adapted from get_aggregate_hasSz(self) in pipeline_utilities.py"""
    if None in pred_list:
        raise
        
    #count the votes
    votes = dict.fromkeys(set(pred_list), 0)
    for vote in pred_list:
        votes[vote] += 1

    #get the value(s) with the highest number of votes
    most_votes = -1
    most_vals = []
    for val in votes:
        if votes[val] > most_votes:
            most_votes = votes[val]
            most_vals = []
        if votes[val] >= most_votes:
            most_vals.append(val)

    #if there is only 1 value with most votes, pick it
    if len(most_vals) == 1:
        return most_vals[0]
    #otherwise, if 0,1 both have the highest number of visits, then return idk (2)
    elif (0 in most_vals) and (1 in most_vals):
        return 2
    #otherwise, it must be that either 0 and 1 are tied with idk (2). Return either the 0 or 1
    else:
        most_vals.sort() #sort, since IDK is always 2
        return most_vals[0]
        
class BiasPatient():
    def __init__(self, pat_id):
        self.pat_id = pat_id
        self.medications = {}

def add_medications_to_pat(pat):
    pat_meds = all_meds.loc[all_meds.MRN == pat.pat_id]

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
    return pat
    
def max_min_scale(x):
    return (x - x.min())/(x.max() - x.min()) 
def g_race_nan(x):
    return pd.isnull(x.RACE)
def g_black(x):
    return x['RACE'] == "Black or African American"
def g_white(x):
    return x["RACE"] == "White"
def g_asian(x):
    return x['RACE'] == 'Asian'
def g_orace(x):
    return ~(g_black(x) + g_white(x) + g_asian(x))
def g_hispanic(x):
    return x['ETHNICITY'] == 'Hispanic Latino'
def g_not_hispanic(x):
    return x["ETHNICITY"] == 'Not Hispanic or Latino'
def g_male(x):
    return x['GENDER'] == 'M'
def g_female(x):
    return x['GENDER'] == 'F'
def g_allo(x):
    return g_black(x) + g_orace(x)
def g_private(x):
    return x.is_private_insurance == 1
def g_public(x):
    return x.is_private_insurance == 0
income_bins = [0, 50000, 75000, 100000, matched_pats.median_zcta_income.max()+1]
def g_income_above(x, threshold):
    return x.median_zcta_income >= threshold
def g_income_below(x, threshold):
    return x.median_zcta_income < threshold
age_splits = np.array_split(matched_pats.AGE.sort_values(), 5)
age_bins = [age_splits[0].min()] + [split.max() for split in age_splits]
def g_asm(x, asm_num):
    return x.num_asms == asm_num
def g_income_bin(x, lower, upper):
    return (g_income_above(x, lower)) & (g_income_below(x, upper))
def g_age_bin(x, lower, upper):
    return (x.AGE >= lower) & (x.AGE < upper)

def MSCE(y, predictions):
    '''
    Function to compute K_2(f, D), also known as mean squared calibration error (MSCE)
    '''
    idx_sort = np.argsort(predictions, kind='mergesort')
    sorted_predictions = predictions[idx_sort]
    sorted_y = y[idx_sort]
    difference_array = sorted_y - sorted_predictions
    values, idx_start, count = np.unique(sorted_predictions, return_counts=True, return_index=True)
    calibration_error = 0
    for idx in range(len(values) - 1):
        calibration_error += count[idx]*((difference_array[idx_start[idx]:idx_start[idx+1]].mean()) **2)
    calibration_error += count[len(values)-1]*((difference_array[idx_start[len(values) - 1]:].mean()) **2)
    return (1/len(y)) * np.array(calibration_error)
    
def BNC(y, predictions):
    return np.mean(predictions[~y.astype(bool).squeeze()])
    
def BPC(y, predictions):
    return np.mean(predictions[y.astype(bool).squeeze()])
    
def get_bootstrap_stat(y,predictions,fun,boots=10000,n=None):
    if not n:
        n = len(y)
    straps = np.zeros((boots,))
    for i in range(boots):
        bidxs = np.random.choice(n,(n,1)).astype(int)
        straps[i] = fun(y[bidxs],predictions[bidxs])
    return straps
    
def get_perm_stat(y,predictions,fun,g,boots=10000):
    straps = np.zeros((boots,))
    n = len(y)
    for i in range(boots):
        bidxs = np.random.choice(n,(n,1),replace=False).astype(int)
        perm_y = y[bidxs]
        straps[i] = fun(perm_y[g],predictions[g])
    return straps
    
def perm_test(x,perms):
    pmean = np.nanmean(perms)
    x = x - pmean
    perms = perms-pmean
    pleft = (sum(perms <= min([x,-1*x]))+1)/(len(perms)+1)
    pright = (sum(perms >= max([x,-1*x]))+1)/(len(perms)+1)
    return pleft+pright
    
def get_stats(y,predictions,fun,gs,boots=20000,n=None,perm = False):
    true_msce = np.zeros((len(gs),))
    straps = np.zeros((boots,len(gs)))
    allgidx = gs[0] + gs[1]
    for i,g in enumerate(gs):
        gy = y[g]
        gpred = predictions[g]
        true_msce[i] = fun(gy,gpred)
        if perm:
            straps[:,i] = get_perm_stat(y[allgidx],predictions[allgidx],fun,g[allgidx],boots)
        else:
            straps[:,i] = get_bootstrap_stat(gy,gpred,fun,boots,n)
    return true_msce,straps