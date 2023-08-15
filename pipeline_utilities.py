from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import re
import string
import json
import os
import sys

class run_silently():
    """A helper function to disable print statements. 
    Copied from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Patient:
    """Patient object. Contains information about the patient's identifiers, visits and medications"""
    def __init__(self, pat_id):
        self.pat_id = pat_id
        self.visits = []
        self.medications = []
        
    def add_visit(self, visit):
        if visit not in self.visits:
            self.visits.append(visit)
        else:
            print("Warning: This visit already exists for this patient. Visit was not added to Patient's visit list")
            
    def add_medication(self, medication):
        self.medications.append(medication)
        
    def __eq__(self, other):
        if isinstance(other, Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
    def __str__(self):
        return self.pat_id
    
    
class Aggregate_Patient(Patient):
    """Aggregate Patient Object. Contains information about a patient's identifiers, medications, and aggregate visits."""
    def __init__(self, pat_id):
        super().__init__(pat_id)
        self.aggregate_visits = []
        self.visits = None #aggregate patients must use aggregate_visits.
        
    def add_aggregate_visit(self, aggregate_visit):
        if aggregate_visit not in self.aggregate_visits:
            self.aggregate_visits.append(aggregate_visit)
        else:
            print("Warning: This aggregate visit already exists for this patient. Aggregate Visit was not added to Patient's aggregate visit list")
        
    def __eq__(self, other):
        if isinstance(other, Aggregate_Patient):
            return self.pat_id == other.pat_id
        else:
            return False
        
class Visit:
    """
    Visit Object. Generated from information from a single medical note
        Patient: The Patient with this Visit
        note_id: The note's ID
        pat_enc_id: The visit's patient encounter ID
        author: The name of the provider
        visit_date: The date of the visit
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
        context: The note text of the visit
        full_text: The full text of the visit
        visit_type: The type of the visit (new patient, return patient)
    """
    def __init__(self, patient, note_id, pat_enc_id, 
                 author, visit_date, visit_type,
                 hasSz, pqf, elo, 
                 context, full_text):
        
        self.Patient = patient
        self.note_id = note_id
        self.pat_enc_id = pat_enc_id
        self.author = author
        self.visit_date = visit_date
        self.visit_type = visit_type
        
        self.hasSz = hasSz
        self.pqf = pqf
        self.elo = elo
        
        self.context = context
        self.full_text = full_text            
        
    def __str__(self):
        """Prints information for this visit"""
        return f"Visit for patient {self.Patient.pat_id} on {self.visit_date}, written by {self.author}: HasSz = {self.hasSz}; pqf_per_month = {self.pqf}; elo = {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Visit):
            return (self.Patient == other.Patient) and (self.note_id == other.note_id) and (self.visit_date == other.visit_date) and (self.author == other.author) and (self.pat_enc_id == other.pat_enc_id)
        else:
            return False
        
class Aggregate_Visit:
    """
    Class for a visit that combines multiple of the same visit (if multiple models with different seeds make predictions for the same visit)
        Aggregate_Patient: The Aggregate Patient with this Visit
        note_id: The visit's ID
        pat_enc_id: The visit's patient encounter ID
        author: The name of the provider
        visit_date: The date of the visit
        all_visits: A list of the (same) visits that make up this single aggregate visit. 
        hasSz: The seizure freedom classification
        pqf: The seizure frequency value
        elo: The date of last seizure value
        context: The note text of the visit
        full_text: The full text of the visit
        visit_type: The type of the visit (new patient, return patient)
    """
    def __init__(self, aggregate_patient, all_visits):
        #first, check if the visits are all the same
        if all_visits.count(all_visits[0]) != len(all_visits):
            raise ValueError(f"Not all visits are the same")
            
        #get the basic info for the visit
        self.Aggregate_Patient = aggregate_patient
        self.all_visits = all_visits
        
        #get information from the visits
        self.note_id = all_visits[0].note_id
        self.pat_enc_id = all_visits[0].pat_enc_id
        self.author = all_visits[0].author
        self.visit_type = all_visits[0].visit_type
        self.visit_date = all_visits[0].visit_date
        
        self.context = all_visits[0].context
        self.full_text = all_visits[0].full_text
    
        #get the hasSz, pqf and elo for each visit
        self.all_hasSz = [vis.hasSz for vis in all_visits]
        self.all_pqf = [vis.pqf if not (pd.isnull(vis.pqf) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an placemarker arbitrary value for aggregate functions (below)
        self.all_elo = [vis.elo if not (pd.isnull(vis.elo) or isinstance(vis.pqf, str)) else -299.0645 for vis in all_visits] #convert nan or strings to an arbitrary placemarker value for aggregate functions (below)
        
        #calculate plurality voting
        self.hasSz = self.__get_aggregate_hasSz()
        self.pqf = self.__get_aggregate_pqf()
        self.elo = self.__get_aggregate_elo()

        
    def __get_aggregate_hasSz(self):
        """ 
        Gets the seizure freedom value for the aggregate visit by (plurality) voting.
        If there is a tie at the highest number of votes,
            If yes and no have the same number of votes, then default to IDK
            If Yes or No has the same number votes as IDK, then default to either Yes or No
        """        
        #count the votes
        votes = dict.fromkeys(set(self.all_hasSz), 0)
        for vote in self.all_hasSz:
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
        
    def __get_aggregate_pqf(self):
        """
        Calculate the seizure frequency with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_pqf), 0)
        for vote in self.all_pqf:
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
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def __get_aggregate_elo(self):
        """
        Calculate the date of last seizure with plurality voting
        If there is a tie at the highest number of votes, there must be either 2 values with 2 votes, or 5 values with 1 vote.
            Return nan if there are two valid values. Otherwise, there must be at least 1 nan. Return the other value
            If there are 5 potential values, then return nan,
        """
        #count the votes
        votes = dict.fromkeys(set(self.all_elo), 0)
        for vote in self.all_elo:
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
        most_vals = np.array(most_vals)
                
        #if there is only 1 value with most votes, pick it
        if len(most_vals) == 1:
            return most_vals[0] if most_vals[0] != -299.0645 else np.nan
        #otherwise, if there are two values with the most votes.
        elif len(most_vals) == 2:
            #if one of the values is nan, return the other
            if np.sum(most_vals == -299.0645) == 1:
                return most_vals[~(most_vals == -299.0645)][0]
            #if both values are not nan, return nan
            else:
                return np.nan
        #otherwise, it must be that there are multiple possible answers, each with a vote
        #thus, return nan
        else:
            return np.nan
    
    def __str__(self):
        return f"Aggregate Visit Object for {self.Aggregate_Patient.pat_id} on {self.visit_date}, written by {self.author}. hasSz: {self.hasSz}, pqf: {self.pqf}, elo: {self.elo}"
    
    def __eq__(self, other):
        if isinstance(other, Aggregate_Visit):
            return (self.Aggregate_Patient == other.Aggregate_Patient) and (self.visit_date == other.visit_date) and (self.author == other.author) and (self.all_visits == other.all_visits)
        else:
            return False
        
def aggregate_patients_and_visits(all_pats):
    """Aggregates patients and visits from dictionary of array of patients all_pats, where each key is a different seed"""

    #initialize the array of Aggregate_Patients
    agg_pats = []
    
    #for simplicity, get the first key
    k = list(all_pats.keys())[0]
    
    #create Aggregate_Patients and fill in their Aggregate_Visits
    for i in range(len(all_pats[k])):
        new_Agg_Pat = Aggregate_Patient(all_pats[k][i].pat_id)
        
        #get aggregate visits
        for j in range(len(all_pats[k][i].visits)):
            new_Agg_visit = Aggregate_Visit(aggregate_patient=new_Agg_Pat,
                                            all_visits = [all_pats[seed][i].visits[j] for seed in all_pats.keys()]
                                           )
            new_Agg_Pat.add_aggregate_visit(new_Agg_visit)
        
        agg_pats.append(new_Agg_Pat)
            
    return agg_pats

def get_asm_name(description, name_dict, exc_names, letter_regex):
    """
    Extracts the name of an ASM from the description provided.
    """
    desc_no_sym = re.sub(letter_regex, ' ', description.lower()).strip()
    desc_split = desc_no_sym.split()

    #iterate through the word-bigrams of the split text
    for i in range(1, len(desc_split)):
        test_str = f"{desc_split[i-1]} {desc_split[i]}"
        if not pd.isnull(name_dict[test_str]):
            return name_dict[test_str]

    #iterate through the word unigrams of the split text
    for text in desc_split:
        if text in exc_names:
            return np.nan
        if not pd.isnull(name_dict[text]):
            return name_dict[text]
    
    return np.nan

def get_all_asm_names_from_description(path_to_asm_names, path_to_exclusion_names, medications, desc_column_name, return_name_dict=False):
    """
    Extracts the name of an ASM from the description provided in the prescription table.
    """
    letter_regex = re.compile(r'[^a-zA-Z]+')

    asm_names = pd.read_csv(path_to_asm_names)
    #remove symbols and numbers
    asm_names['Brand'] = asm_names['Brand'].apply(lambda x: re.sub(letter_regex, ' ', x.lower()).strip())
    asm_names['Generic'] = asm_names['Generic'].apply(lambda x: re.sub(letter_regex, ' ', x.lower()).strip())
    #unify 'extended release' to 'xr'
    asm_names['Brand'] = asm_names['Brand'].apply(lambda x: re.sub(r'extended release', 'xr', x.lower()).strip())
    
    #load in the exclusion criteria
    exc_names = pd.read_csv('/project/nlp/cnt_epilepsydata/kevinxie_nlp_projects_code/data_pull_processing/exclusionary_ASM_lists.csv', header=None)[0].str.lower().to_numpy()
    
    #dictionary to convert the names of brand names to generic drugs
    brand_to_generic = {row.Brand.lower():row.Generic.lower() for idx, row in asm_names.iterrows()}
    #for code simplicity, we'll also map generic to generic
    brand_to_generic.update({row.Generic.lower():row.Generic.lower() for idx, row in asm_names.iterrows()})
    brand_to_generic = defaultdict(lambda: np.nan , brand_to_generic)
    
    if return_name_dict:
        return medications[desc_column_name].apply(lambda x: get_asm_name(x, brand_to_generic, exc_names, letter_regex)), brand_to_generic
    else:
        return medications[desc_column_name].apply(lambda x: get_asm_name(x, brand_to_generic, exc_names, letter_regex))