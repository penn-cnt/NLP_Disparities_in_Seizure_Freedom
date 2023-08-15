import string
import numpy as np
import pandas as pd
import os
import pickle
import forestplot as fp
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm

#load in the model fits
model_fits = {}
for pkl_file in os.listdir('outcomes_vs_demographics_regression_outputs'):
    if ".pkl" not in pkl_file:
        continue
    with open(f'outcomes_vs_demographics_regression_outputs/{pkl_file}', 'rb') as f:
        model_fits[pkl_file] = pickle.load(f)
        
#format the data into tables
forest_data = []

#reference male == 0
forest_data.append({
    'group':'Gender',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': 'Male (Ref.)'
})
#female  == 1
forest_data.append({
    'group':'Gender',
    'OR':model_fits['uni_gender_mdl_py38.pkl'].coefs['OR']['gender'],
    'OR_2.5_ci':model_fits['uni_gender_mdl_py38.pkl'].coefs['OR_2.5_ci']['gender'],
    'OR_97.5_ci':model_fits['uni_gender_mdl_py38.pkl'].coefs['OR_97.5_ci']['gender'],
    'P-value':model_fits['uni_gender_mdl_py38.pkl'].coefs['P-val']['gender'],
    'label': 'Female'
})

#reference white == 0
forest_data.append({
    'group':'Race',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': 'White (Ref.)'
})
#black == 1
forest_data.append({
    'group':'Race',
    'OR':model_fits['uni_race_mdl_py38.pkl'].coefs['OR']['race1'],
    'OR_2.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_2.5_ci']['race1'],
    'OR_97.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_97.5_ci']['race1'],
    'P-value':model_fits['uni_race_mdl_py38.pkl'].coefs['P-val']['race1'],
    'label': 'Black'
})
#asian == 2
forest_data.append({
    'group':'Race',
    'OR':model_fits['uni_race_mdl_py38.pkl'].coefs['OR']['race2'],
    'OR_2.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_2.5_ci']['race2'],
    'OR_97.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_97.5_ci']['race2'],
    'P-value':model_fits['uni_race_mdl_py38.pkl'].coefs['P-val']['race2'],
    'label': 'Asian'
})
#other == 3
forest_data.append({
    'group':'Race',
    'OR':model_fits['uni_race_mdl_py38.pkl'].coefs['OR']['race3'],
    'OR_2.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_2.5_ci']['race3'],
    'OR_97.5_ci':model_fits['uni_race_mdl_py38.pkl'].coefs['OR_97.5_ci']['race3'],
    'P-value':model_fits['uni_race_mdl_py38.pkl'].coefs['P-val']['race3'],
    'label': 'Other'
})

#reference not hispanic/latino == 0
forest_data.append({
    'group':'Ethnicity',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': 'Not Hispanic or Latino (Ref.)'
})
#hispanic or latino  == 1
forest_data.append({
    'group':'Ethnicity',
    'OR':model_fits['uni_ethnicity_mdl_py38.pkl'].coefs['OR']['ethnicity'],
    'OR_2.5_ci':model_fits['uni_ethnicity_mdl_py38.pkl'].coefs['OR_2.5_ci']['ethnicity'],
    'OR_97.5_ci':model_fits['uni_ethnicity_mdl_py38.pkl'].coefs['OR_97.5_ci']['ethnicity'],
    'P-value':model_fits['uni_ethnicity_mdl_py38.pkl'].coefs['P-val']['ethnicity'],
    'label': 'Hispanic or Latino'
})

#reference private == 0
forest_data.append({
    'group':'Insurance',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': 'Private (Ref.)'
})
#public == 1
forest_data.append({
    'group':'Insurance',
    'OR':model_fits['uni_insurance_mdl_py38.pkl'].coefs['OR']['insurance_type'],
    'OR_2.5_ci':model_fits['uni_insurance_mdl_py38.pkl'].coefs['OR_2.5_ci']['insurance_type'],
    'OR_97.5_ci':model_fits['uni_insurance_mdl_py38.pkl'].coefs['OR_97.5_ci']['insurance_type'],
    'P-value':model_fits['uni_insurance_mdl_py38.pkl'].coefs['P-val']['insurance_type'],
    'label': 'Public'
})

#reference income >100k
forest_data.append({
    'group':'Income',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': r'\$100k+ (Ref.)'
})
#income 75-100k
forest_data.append({
    'group':'Income',
    'OR':model_fits['uni_income_mdl_py38.pkl'].coefs['OR']['income1'],
    'OR_2.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_2.5_ci']['income1'],
    'OR_97.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_97.5_ci']['income1'],
    'P-value':model_fits['uni_income_mdl_py38.pkl'].coefs['P-val']['income1'],
    'label': r'\$75k-\$100k'
})
#income 50-75k
forest_data.append({
    'group':'Income',
    'OR':model_fits['uni_income_mdl_py38.pkl'].coefs['OR']['income2'],
    'OR_2.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_2.5_ci']['income2'],
    'OR_97.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_97.5_ci']['income2'],
    'P-value':model_fits['uni_income_mdl_py38.pkl'].coefs['P-val']['income2'],
    'label': r'\$50k-\$75k'
})
#income <50k
forest_data.append({
    'group':'Income',
    'OR':model_fits['uni_income_mdl_py38.pkl'].coefs['OR']['income3'],
    'OR_2.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_2.5_ci']['income3'],
    'OR_97.5_ci':model_fits['uni_income_mdl_py38.pkl'].coefs['OR_97.5_ci']['income3'],
    'P-value':model_fits['uni_income_mdl_py38.pkl'].coefs['P-val']['income3'],
    'label': r'<\$50k'
})

#reference age 18-40
forest_data.append({
    'group':'Age',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': '18-40 yrs (Ref.)'
})
#age 40-65
forest_data.append({
    'group':'Age',
    'OR':model_fits['uni_age_mdl_py38.pkl'].coefs['OR']['age1'],
    'OR_2.5_ci':model_fits['uni_age_mdl_py38.pkl'].coefs['OR_2.5_ci']['age1'],
    'OR_97.5_ci':model_fits['uni_age_mdl_py38.pkl'].coefs['OR_97.5_ci']['age1'],
    'P-value':model_fits['uni_age_mdl_py38.pkl'].coefs['P-val']['age1'],
    'label': '40-65 yrs'
})
#age 65+
forest_data.append({
    'group':'Age',
    'OR':model_fits['uni_age_mdl_py38.pkl'].coefs['OR']['age2'],
    'OR_2.5_ci':model_fits['uni_age_mdl_py38.pkl'].coefs['OR_2.5_ci']['age2'],
    'OR_97.5_ci':model_fits['uni_age_mdl_py38.pkl'].coefs['OR_97.5_ci']['age2'],
    'P-value':model_fits['uni_age_mdl_py38.pkl'].coefs['P-val']['age2'],
    'label': '65+ yrs'
})

#reference asm == 0
forest_data.append({
    'group':'Number of ASMs',
    'OR':1,
    'OR_2.5_ci':1,
    'OR_97.5_ci':1,
    'P-value':0,
    'label': '0 ASMs (Ref.)'
})
#asm == 1
forest_data.append({
    'group':'Number of ASMs',
    'OR':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR']['num_asms1'],
    'OR_2.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_2.5_ci']['num_asms1'],
    'OR_97.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_97.5_ci']['num_asms1'],
    'P-value':model_fits['uni_asms_mdl_py38.pkl'].coefs['P-val']['num_asms1'],
    'label': '1 ASM'
})
#asm == 2
forest_data.append({
    'group':'Number of ASMs',
    'OR':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR']['num_asms2'],
    'OR_2.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_2.5_ci']['num_asms2'],
    'OR_97.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_97.5_ci']['num_asms2'],
    'P-value':model_fits['uni_asms_mdl_py38.pkl'].coefs['P-val']['num_asms2'],
    'label': '2 ASMs'
})
#asm == 3
forest_data.append({
    'group':'Number of ASMs',
    'OR':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR']['num_asms3'],
    'OR_2.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_2.5_ci']['num_asms3'],
    'OR_97.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_97.5_ci']['num_asms3'],
    'P-value':model_fits['uni_asms_mdl_py38.pkl'].coefs['P-val']['num_asms3'],
    'label': '3 ASMs'
})
#asm == 4+
forest_data.append({
    'group':'Number of ASMs',
    'OR':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR']['num_asms4'],
    'OR_2.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_2.5_ci']['num_asms4'],
    'OR_97.5_ci':model_fits['uni_asms_mdl_py38.pkl'].coefs['OR_97.5_ci']['num_asms4'],
    'P-value':model_fits['uni_asms_mdl_py38.pkl'].coefs['P-val']['num_asms4'],
    'label': '4+ ASMs'
})

#plot the forest plot
forest_data = pd.DataFrame(forest_data)
ax = fp.forestplot(forest_data,  
                   estimate="OR", 
                   ll="OR_2.5_ci", 
                   hl="OR_97.5_ci", 
                   varlabel="label", 
                   groupvar="group", 
                   pval="P-value", 
                   ylabel="OR (95% CI)",
                   xlabel="$\leftarrow$" + " Less Sz. | More Sz. " + r"$\rightarrow$",
                   starpval=False,
              )
plt.vlines(1.0, -1, len(forest_data)+forest_data.group.nunique()-1, colors='k', linestyles=':', linewidth=1.0)
ax.set_xscale('log')
ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(f"outcomes_vs_demographics_regression_outputs/univariate_forest_plot.png", dpi=600, bbox_inches='tight')
plt.savefig(f"outcomes_vs_demographics_regression_outputs/univariate_forest_plot.pdf", dpi=600, bbox_inches='tight')
plt.show()

#P-value corrections
forest_data_no_ref = forest_data.loc[forest_data['P-value'] > 0]
adjusted_p = sm.stats.fdrcorrection(forest_data_no_ref['P-value'].to_numpy())
forest_data_no_ref['adjusted_p-value'] = adjusted_p[1]