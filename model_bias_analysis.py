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
from tqdm import tqdm
from model_bias_functions import *
import statsmodels.api as sm

#load visit notes
with open(r'epilepsy_notes.pkl', 'rb') as f:
    all_notes = pickle.load(f)

#load demographics
#add a column to all demographics that's just whether or not the model got it correct. Leave blank for now
all_demographics = pd.read_pickle('demographic_data.pkl')
all_demographics['aggregate_pred_correct'] = np.nan
    
#load old notes
old_notes = pd.read_csv(r'old_notes.csv')
old_notes['pat_enc_csn_id'] = old_notes['pat_enc_csn_id'].apply(str)

#get insurance info
insurance = all_demographics.groupby('MRN')['PAYOR_NAME'].unique().explode().dropna().reset_index()
insurance_coding = pd.read_excel('payers.xlsx')
#categorize insurance as public or private into a dictionary
public_or_private = {row.insurance:row['public=0'] for idx, row in insurance_coding.iterrows()}
all_demographics['is_private_insurance'] = all_demographics['PAYOR_NAME'].apply(lambda x: public_or_private[x] if not pd.isnull(x) else np.nan)

#load income data. We want columns S1901_C01_001E (estimate # of housholds), S1901_C01_012E (median income), S1901_C01_012M (margin of error), GEO_ID, NAME
raw_income_data = pd.read_csv('map_data/ACSST5Y2021.S1901-Data.csv', skiprows=[1])[['GEO_ID', 'NAME', 'S1901_C01_001E', 'S1901_C01_012E', 'S1901_C01_012M']]
income_data = pd.DataFrame({'ZCTA':raw_income_data['NAME'].apply(lambda x: x.split()[1]), 
                            'med_income':raw_income_data['S1901_C01_012E'].apply(lambda x: str_to_int(x)),
                            'med_num_household':raw_income_data['S1901_C01_001E'].apply(lambda x: str_to_int(x))})
income_data = {row.ZCTA:row.med_income for idx, row in income_data.iterrows()}

#add income data to demographics
#truncate zipcodes to 5 digits
all_demographics['ZIP'] = all_demographics['ZIP'].apply(lambda x: str(x)[:5])
all_demographics['median_zcta_income'] = all_demographics.apply(lambda x: income_data[x.ZIP] if x.ZIP in income_data.keys() else np.nan, axis=1)

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

#iterate through old notes. Find where a pat_enc_csn_id is shared between old and new notes, but the MRNs are different
identical_mrns = {}
missing_mrns = []
for idx, row in old_notes.iterrows():
    #check if the MRNs match, then just add the link
    if row.pat_id in all_notes.MRN.values:
        identical_mrns[row.pat_id] = row.pat_id
    #if the MRNs do not match, check if there's a enc_csn_id
    if row.pat_enc_csn_id in all_notes.PAT_ENC_CSN_ID.values:
        #get the visit they share
        shared_vis = all_notes.loc[all_notes.PAT_ENC_CSN_ID == row.pat_enc_csn_id]
        identical_mrns[row.pat_id] = shared_vis.MRN.iloc[0]
    else:
        missing_mrns.append(row.pat_id)
        
#load the jamia classification results
agg_preds = []
hasSz_preds = [pd.read_csv(f'hasSz_epi_NOTES_MODEL_{seed}/eval_predictions.tsv', sep='\t') for seed in [2, 17, 42, 97, 136]]

#get all predictions across the seeds and do plurality voting
for j in range(len(hasSz_preds[0])):
    agg_pred= {}
    agg_pred['True_Label'] = hasSz_preds[0].iloc[j].True_Label
    agg_pred['ID'] = hasSz_preds[0].iloc[j].ID
    id_split = hasSz_preds[0].iloc[j].ID.split("_")
    agg_pred['MRN'] = id_split[0]
    agg_pred['note_author'] = id_split[1]
    agg_pred['visit_date'] = id_split[2]
    agg_pred['predictions'] = [hasSz_preds[i].iloc[j]['argmax'] if hasSz_preds[i].iloc[j].True_Label == agg_pred['True_Label'] else None for i in range(len(hasSz_preds))]
    agg_pred['probabilities'] = np.mean(np.array([score_to_probs(hasSz_preds[i].iloc[j]['Predictions']) for i in range(len(hasSz_preds))]),axis=0)
    agg_pred['agg_hasSz'] = get_aggregate_hasSz(agg_pred['predictions'])
    agg_preds.append(agg_pred)
agg_preds = pd.DataFrame(agg_preds)

#iterate through the JAMIA predictions and find their demographics
matched_pats = []
missed_ct = 0
for idx, row in agg_preds.iterrows():
    #skip all missing MRNs
    if row.MRN in missing_mrns:
        missed_ct += 1
        continue
    
    #match the patient to the prediction and update if they got it correct
    matched = all_demographics.loc[all_demographics['MRN'] == identical_mrns[row.MRN]].sort_values(by='CONTACT_DATE').drop_duplicates(subset='MRN', keep='last')
    matched.aggregate_pred_correct = row.agg_hasSz == row.True_Label
    matched['Label'] = row.True_Label
    matched['Pred'] = row.agg_hasSz
    matched["true_prob"] = row.probabilities[1]
    
    #match the patient to their medications
    this_pat = add_medications_to_pat(BiasPatient(row.MRN))
    
    #count the number of prescriptions that pass through this visit date
    this_visit_date = datetime.strptime(row.visit_date, '%Y-%m-%d')
    num_asms = 0
    for asm in this_pat.medications:
        num_asms += int(this_visit_date >= this_pat.medications[asm]['start_date'] and this_visit_date <= this_pat.medications[asm]['end_date'])
            
    #add the number of ASMs they are taking, capping it at 4+
    matched['num_asms'] = str(num_asms) if num_asms < 4 else "4+"

    matched_pats.append(matched)
matched_pats = pd.concat(matched_pats).reset_index(drop=True)
matched_pats = matched_pats[matched_pats.Label != 2]
print(len(matched_pats))

#========================================================================#
#Accuracy measures

print(f"Overall accuracy: {matched_pats.aggregate_pred_correct.sum()/len(matched_pats)}")

#see if sex has anything to do with model accuracy
gender_table = {}
for gender in set(matched_pats.GENDER.dropna()):
    gender_group = matched_pats.loc[matched_pats.GENDER == gender]
    gender_table[gender] = [gender_group.aggregate_pred_correct.sum(), np.sum(1 - gender_group.aggregate_pred_correct)]
    print(f"Accuracy for group {gender} with {len(gender_group)} notes: {gender_group.aggregate_pred_correct.sum()/len(gender_group)}")
#construct the contingency table
gender_rc = [gender_table[gender] for gender in gender_table]
#calculate fisher exact
gender_fisher = fisher_exact(gender_rc)
print(f"P-value comparing M vs. F for probability correct: {gender_fisher[1]}")

#see if enthnicity has anything to do with model accuracy
ethnicity_table = {}
for ethnicity in set(matched_pats.ETHNICITY.dropna()):
    ethnicity_group = matched_pats.loc[matched_pats.ETHNICITY == ethnicity]
    ethnicity_table[ethnicity] = [ethnicity_group.aggregate_pred_correct.sum(), np.sum(1 - ethnicity_group.aggregate_pred_correct)]
    print(f"Accuracy for group {ethnicity} with {len(ethnicity_group)} notes: {ethnicity_group.aggregate_pred_correct.sum()/len(ethnicity_group)}")
#construct the contingency table
ethnicity_rc = [ethnicity_table[ethnicity] for ethnicity in ethnicity_table]
#calculate fisher's exact
ethnicity_fisher = fisher_exact(ethnicity_rc)
print(f"P-value comparing HL vs. NHL for probability correct: {ethnicity_fisher[1]}")

#see if public/private insurance has anything to do with model accuracy
insurance_table = {}
for insurance in set(matched_pats.is_private_insurance.dropna()):
    insurance_group = matched_pats.loc[matched_pats.is_private_insurance == insurance]
    insurance_table[insurance] = [insurance_group.aggregate_pred_correct.sum(), np.sum(1 - insurance_group.aggregate_pred_correct)]
    print(f"Accuracy for group {insurance} with {len(insurance_group)} notes: {insurance_group.aggregate_pred_correct.sum()/len(insurance_group)}")
#construct contingency table
insurance_rc = [insurance_table[insurance] for insurance in insurance_table]
#calculate fisher
insurance_fisher = fisher_exact(insurance_rc)
print(f"P-value comparing Private vs. Public for probability correct: {insurance_fisher[1]}")

#see if race has anything to do with model accuracy
race_bins = ['White', 'Black or African American', 'Asian', 'Other']
race_bin_values = {}
for race in set(matched_pats.RACE.dropna()):
    race_group = matched_pats.loc[matched_pats.RACE == race]
    print(f"Accuracy for group {race} with {len(race_group)} notes: {race_group.aggregate_pred_correct.sum()/len(race_group)}")
    race_bin_values[race] = [race_group.aggregate_pred_correct.sum(), len(race_group)] 
#compare the race_bins
race_bin_accs = {}
race_bin_cts = {}
for race in race_bin_values:    
    if race in race_bins:
        if race not in race_bin_accs:
            race_bin_accs[race] = 0
            race_bin_cts[race] = 0
        race_bin_accs[race] += race_bin_values[race][0]/race_bin_values[race][1]
        race_bin_cts[race] += race_bin_values[race][1]
    else:
        if 'Other' not in race_bin_accs:
            race_bin_accs['Other'] = [0,0]
            race_bin_cts['Other'] = 0
        race_bin_accs['Other'][0] += race_bin_values[race][0]
        race_bin_accs['Other'][1] += race_bin_values[race][1]
        race_bin_cts['Other'] += race_bin_values[race][1]
race_bin_accs['Other'] = race_bin_accs['Other'][0]/race_bin_accs['Other'][1]
print(f"\nComparing Races Categorical")
print(f"Race bins edges: {race_bins}")
print(f"Age accuracy between bin edges: {race_bin_accs}")
print(f"Number of examples between bin edges: {race_bin_cts}")
race_acc_test = kstest(max_min_scale(np.array(list(race_bin_accs.values()))), 'uniform')
print(race_acc_test)

#see if age has anything to do with model accuracy.
age_bins = [18, 40, 65, 999]
age_bin_accs = []
age_bin_cts = []
for i in range(1, len(age_bins)):
    age_bin_accs.append(matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])].aggregate_pred_correct.mean())
    age_bin_cts.append(len(matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])]))
print(f"Age bin edges: {age_bins}")
print(f"Age accuracy between bin edges: {age_bin_accs}")
print(f"Number of examples between bin edges: {age_bin_cts}")
age_acc_test = kstest(max_min_scale(np.array(age_bin_accs)), 'uniform')
print(age_acc_test)

#see if Income has anything to do with model accuracy.
income_bins = [0, 50000, 75000, 100000, matched_pats.median_zcta_income.max()+1]
income_bin_accs = []
income_bin_cts = []
for i in range(1, len(income_bins)):
    income_bin_accs.append(matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])].aggregate_pred_correct.mean())
    income_bin_cts.append(len(matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])]))
print(f"Income bin edges: {income_bins}")
print(f"Income accuracy between bin edges: {income_bin_accs}")
print(f"Number of examples between bin edges: {income_bin_cts}") 
income_acc_test = kstest(max_min_scale(np.array(income_bin_accs)), 'uniform')
print(income_acc_test)

#========================================================================#
#Probability analyses

#binary variables - permutation tests
#PCB
ps = []
binary_pcb_pvals = []
y = matched_pats.Label.to_numpy()
predictions = matched_pats.true_prob.to_numpy()
for g in [(g_male,g_female),(g_private,g_public),(g_not_hispanic,g_hispanic)]:
    a,b = get_stats(y,predictions,BPC,[g[0](matched_pats),g[1](matched_pats)],perm = True,boots = 10000)
    p = perm_test(np.diff(a),np.diff(b,axis=1))
    ps.append(p)
    plt.figure()
    plt.hist(np.diff(b,axis=1));
    plt.axvline(x=np.diff(a))
    plt.title(str(g[1]) + " - " + str(g[0]) + " - p: " + str(p))
    binary_pcb_pvals.append(p[0])
#NCB
ps = []
binary_ncb_pvals = []
y = matched_pats.Label.to_numpy()
predictions = matched_pats.true_prob.to_numpy()
for g in [(g_male,g_female),(g_private,g_public),(g_not_hispanic, g_hispanic)]:
    a,b = get_stats(y,predictions,BNC,[g[0](matched_pats),g[1](matched_pats)],perm = True,boots = 10000)
    p = perm_test(np.diff(a),np.diff(b,axis=1))
    ps.append(p)
    plt.figure()
    plt.hist(np.diff(b,axis=1));
    plt.axvline(x=np.diff(a))
    plt.title(str(g[1]) + " - " + str(g[0]) + " - p: " + str(p))
    binary_ncb_pvals.append(p[0])
    
#continuous variables
cont_pcb_pvals = []
cont_ncb_pvals = []

#calculate PCB and NCB for categorical race
race_fnrs = []
race_fprs = []
for race_fct in [g_white, g_black, g_asian, g_orace]:
    race_fnrs.append(BNC(matched_pats[race_fct(matched_pats)].Label.to_numpy(), matched_pats[race_fct(matched_pats)].true_prob.to_numpy()))
    race_fprs.append(BPC(matched_pats[race_fct(matched_pats)].Label.to_numpy(), matched_pats[race_fct(matched_pats)].true_prob.to_numpy()))
cont_pcb_pvals.append(kstest(max_min_scale(np.array(race_fprs)), 'uniform')[1])
cont_ncb_pvals.append(kstest(max_min_scale(np.array(race_fnrs)), 'uniform')[1])
print(f"KS PCB: {cont_pcb_pvals[-1]}")
print(f"KS NCB: {cont_ncb_pvals[-1]}")

#Calculate PCB, NCB for age groups
age_fnrs = []
age_fprs = []
for i in range(1, len(age_bins)):
    age_fnrs.append(BNC(matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])].Label.to_numpy(),matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])].true_prob.to_numpy()))
    age_fprs.append(BPC(matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])].Label.to_numpy(),matched_pats[g_age_bin(matched_pats, age_bins[i-1], age_bins[i])].true_prob.to_numpy()))
cont_pcb_pvals.append(kstest(max_min_scale(np.array(age_fprs)), 'uniform')[1])
cont_ncb_pvals.append(kstest(max_min_scale(np.array(age_fnrs)), 'uniform')[1])
print(f"KS PCB: {cont_pcb_pvals[-1]}")
print(f"KS NCB: {cont_ncb_pvals[-1]}")

#Calculate PCB, NCB for Income groups
income_fnrs = []
income_fprs = []
for i in range(1, len(income_bins)):
    income_fnrs.append(BNC(matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])].Label, matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])].Pred))
    income_fprs.append(BPC(matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])].Label, matched_pats[g_income_bin(matched_pats, income_bins[i-1], income_bins[i])].Pred))
cont_pcb_pvals.append(kstest(max_min_scale(np.array(income_fprs)), 'uniform')[1])
cont_ncb_pvals.append(kstest(max_min_scale(np.array(income_fnrs)), 'uniform')[1])
print(f"KS PCB: {cont_pcb_pvals[-1]}")
print(f"KS NCB: {cont_ncb_pvals[-1]}")

#======================================================================#
#plotting

# Setting color pallette
full_cmap = np.array([[254,235,226],
[251,180,185],
[247,104,161],
[197,27,138],
[122,1,119]])/255
binary_cmap = np.array([
[251,180,185],
[197,27,138]])/255

# Race
label = "Race"
label_list = race_bins
race_map = {label:[],"metric":[],"value":[]}
for ri,ridx in enumerate((g_white(matched_pats),g_black(matched_pats), g_asian(matched_pats), g_orace(matched_pats))):
    race_map[label].append(label_list[ri])
    race_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    race_map[label].append(label_list[ri])
    race_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    race_map[label].append(label_list[ri])
    race_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    race_map["metric"].append("Accuracy")
    race_map["metric"].append("PCB")
    race_map["metric"].append("NCB")
race_df = pd.DataFrame(race_map,columns=[label,"metric","value"])

# Ethnicity
label = "Ethnicity"
label_list = ['Not Hispanic or Latino', 'Hispanic Latino']
ethnicity_map = {label:[],"metric":[],"value":[]}
for ri,ridx in enumerate((g_hispanic(matched_pats),g_not_hispanic(matched_pats))):
    ethnicity_map[label].append(label_list[ri])
    ethnicity_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    ethnicity_map[label].append(label_list[ri])
    ethnicity_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    ethnicity_map[label].append(label_list[ri])
    ethnicity_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    ethnicity_map["metric"].append("Accuracy")
    ethnicity_map["metric"].append("PCB")
    ethnicity_map["metric"].append("NCB")
ethnicity_df = pd.DataFrame(ethnicity_map,columns=[label,"metric","value"])

# Sex
label='Sex'
label_list = ["Male","Female"]
sex_map = {label:[],"metric":[],"value":[]}
for ri,ridx in enumerate((g_male(matched_pats),g_female(matched_pats))):
    sex_map[label].append(label_list[ri])
    sex_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    sex_map[label].append(label_list[ri])
    sex_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    sex_map[label].append(label_list[ri])
    sex_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    sex_map["metric"].append("Accuracy")
    sex_map["metric"].append("PCB")
    sex_map["metric"].append("NCB")
sex_df = pd.DataFrame(sex_map,columns=sex_map.keys())

# Insurance
label = "Insurance"
label_list = ["Private","Public"]
data_map = {label:[],"metric":[],"value":[]}
for ri,ridx in enumerate((g_private(matched_pats),g_public(matched_pats))):
    data_map[label].append(label_list[ri])
    data_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map["metric"].append("Accuracy")
    data_map["metric"].append("PCB")
    data_map["metric"].append("NCB")
insurance_df = pd.DataFrame(data_map,columns=data_map.keys())

# Age
label = "Age Ranges"
label_list = ['18-39', '40-64', '65+']
data_map = {label:[],"metric":[],"value":[]}
bin_list_to_enum = [g_age_bin(matched_pats, age_bins[i-1], age_bins[i]) for i in range(1, len(age_bins))]
for ri,ridx in enumerate(bin_list_to_enum):
    data_map[label].append(label_list[ri])
    data_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map["metric"].append("Accuracy")
    data_map["metric"].append("PCB")
    data_map["metric"].append("NCB")
age_df = pd.DataFrame(data_map,columns=data_map.keys())

# Age
label = "Age Ranges"
label_list = ['18-39', '40-64', '65+']
data_map = {label:[],"metric":[],"value":[]}
bin_list_to_enum = [g_age_bin(matched_pats, age_bins[i-1], age_bins[i]) for i in range(1, len(age_bins))]
for ri,ridx in enumerate(bin_list_to_enum):
    data_map[label].append(label_list[ri])
    data_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map["metric"].append("Accuracy")
    data_map["metric"].append("PCB")
    data_map["metric"].append("NCB")
age_df = pd.DataFrame(data_map,columns=data_map.keys())


# Income
label = "Median Zipcode\nIncome Ranges"
label_list = [r'<\$50k', r'\$50k to <\$75k', r'\$75k to <\$100k', r'\$100k+']
data_map = {label:[],"metric":[],"value":[]}
bin_list_to_enum = [g_income_bin(matched_pats, income_bins[i-1], income_bins[i]) for i in range(1, len(income_bins))]
for ri,ridx in enumerate(bin_list_to_enum):
    data_map[label].append(label_list[ri])
    data_map["value"].append(matched_pats[ridx].aggregate_pred_correct.sum()/sum(ridx))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BPC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map[label].append(label_list[ri])
    data_map["value"].append(BNC(matched_pats[ridx].Label.to_numpy(),matched_pats[ridx].true_prob.to_numpy()))
    data_map["metric"].append("Accuracy")
    data_map["metric"].append("PCB")
    data_map["metric"].append("NCB")
income_df = pd.DataFrame(data_map,columns=data_map.keys())

# cmap_2 = np.array([[166,97,26],[1,133,113]])/255
cmap_2 = np.array([[223,194,125],[53,141,143]])/255
cmap_3 = np.array([[223,194,125],[128,205,193],[1,133,113]])/255
cmap_4 = np.array([[166,97,26],[223,194,125],[128,205,193],[1,133,113]])/255
sns.reset_defaults()

sns.color_palette(cmap_4)
fig,axs = plt.subplots(3,2,figsize=(19,10))
sns.set_palette(cmap_2)
sns.barplot(x = ethnicity_df.metric, y = ethnicity_df.value, hue = ethnicity_df.Ethnicity,ax=axs[0,0], palette=cmap_2)
axs[0,0].set_xlabel("")
axs[0,0].set_ylim([0,1])
sns.set_palette(cmap_2)
sns.barplot(x = sex_df.metric, y = sex_df.value, hue = sex_df.Sex,ax=axs[1,0], palette=cmap_2)
axs[1,0].set_xlabel("")
axs[1,0].set_ylim([0,1])
sns.set_palette(cmap_2)
sns.barplot(x = insurance_df.metric, y = insurance_df.value, hue = insurance_df["Insurance"],ax=axs[2,0], palette=cmap_2)
axs[2,0].set_xlabel("")
axs[2,0].set_ylim([0,1])
sns.set_palette(cmap_4)
sns.barplot(x = race_df.metric, y = race_df.value, hue = race_df.Race,ax=axs[0,1], palette=cmap_4)
axs[0,1].set_xlabel("")
axs[0,1].set_ylim([0,1])
sns.set_palette(cmap_3)
sns.barplot(x = age_df.metric, y = age_df.value, hue = age_df["Age Ranges"],ax=axs[1,1], palette=cmap_3)
axs[1,1].set_xlabel("")
axs[1,1].set_ylim([0,1])
sns.set_palette(cmap_4)
sns.barplot(x = income_df.metric, y = income_df.value, hue = income_df["Median Zipcode\nIncome Ranges"],ax=axs[2,1], palette=cmap_4)
axs[2,1].set_xlabel("")
axs[2,1].set_ylim([0,1])
panel_letters = ['A','B','C','D','E','F']
for i, ax in enumerate(axs.flatten()):
    ax.annotate(panel_letters[i], xy=(-.1, 1.1), xycoords='axes fraction',
                fontsize=12, fontweight='bold', va='top', ha='right')
plt.savefig("model_bias.png", dpi=600, bbox_inches='tight')
plt.savefig("model_bias.pdf", dpi=600, bbox_inches='tight')
plt.show()

#================================================================#
#p-value adjustments
acc_pvals = [gender_fisher[1], ethnicity_fisher[1], insurance_fisher[1], race_acc_test[1], age_acc_test[1], income_acc_test[1]]
pcb_pvals = binary_pcb_pvals + cont_pcb_pvals
ncb_pvals = binary_ncb_pvals + cont_ncb_pvals
all_pvals = acc_pvals + pcb_pvals + ncb_pvals
#adjust all p-values simultaneously
all_adj_pvals = sm.stats.fdrcorrection(all_pvals)[1]
{f"{all_pvals[i]}_{i}":all_adj_pvals[i] for i in range(len(all_adj_pvals))}