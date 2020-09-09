'''
# This script collects all the candidate perovskites generated (after applying Gradient Boosting classification)
# The compound system was tested for novely by comparing the quaternary system with existing ICSD
# This script integrates many other scripts and would take extensive amount of time to run as it is. Therefore it is recommended to run it part by part sequentially

@Achintha_Ihalage
@08_Jun_2020
'''
import numpy as np
import pandas as pd
import math
import re
import pathlib
from collections import Counter
from pymatgen import Composition, Element
from ML.classification import PVClassifier
from autoencoder import AutoEncoder
from ionic_radi import ElementSym

path = str(pathlib.Path(__file__).parent.absolute())

##############################################################################################################################################################

clf = PVClassifier()
clf.get_perovskite_candidates(prob_threshold=0.98)	# must run this to create new_perovskite_candidates.csv

###############################################################################################################################################################






###############################################################################################################################################################

## generated compounds classified as perovskites by GB classifier
df_new_pv = pd.read_csv(path+'/ML/new_perovskite_candidates.csv', sep='\t')

# all generated compounds // to merge with df_new_pv and compare with all ICSD data to check if the quaternary system has been studied befores
df_all_gen = pd.read_csv(path+'/ICSD_data/all_generated_compounds.csv', sep='\t')

# all ICSD experimental compositions
all_experimental_comps = pd.read_csv(path+'/ICSD_data/all_compositions_withDup.csv', sep='\t')


## merge dataframes 
df_new_pv_quaternary = df_all_gen.merge(df_new_pv, on='StructuredFormula')

def quaternary_exists(row):	# we must iterate through rows because the place of the quaternary system element does not matter. 
							# For example, (Na0.4K0.6)NbO3 and (K0.2Na0.8)NbO3 have the same quaternary system  /// Hence we can't use pandas merge function
	quaternary_system = [row.A1, row.A2, row.B1, row.B2]
	exp_quaternary = np.array(all_experimental_comps[['A1', 'A2', 'B1', 'B2']])
	exist_list = [Counter(quaternary_system) == Counter(system) for system in exp_quaternary]
	if sum(exist_list)>0:
		return 1
	else:
		return 0

df_new_pv_quaternary['Exist'] = df_new_pv_quaternary.apply(quaternary_exists, axis=1)

df_new_quaternary = df_new_pv_quaternary[df_new_pv_quaternary['Exist']==0]	# take the candidate perovskite systems that do not exist in ICSD

df_new_quaternary.to_csv(path+'/ML/candidates_not_in_ICSD.csv', sep='\t', index=False)	# new_quaternary_candidates.csv

###############################################################################################################################################################






###############################################################################################################################################################

## To better ensure synthesizability, we will get the compounds whose elements exhibit thier respective most common oxidation states
df_new_quaternary = pd.read_csv(path+'/ML/candidates_not_in_ICSD.csv', sep='\t')	# new_quaternary_candidates.csv

def has_common_oxi_states(row):

	comp = Composition(row.StructuredFormula)
	int_comp = Composition(comp.get_integer_formula_and_factor()[0]) # get integer formular

	###############################################################################################
	# according to the documentation of pymatgen's Composition class' oxi_state_guesses(), the default
	# value of the argument all_oxi_states is False, which means that it should only use most common
	# oxidation states of elements to make the guesses. However, it seems to be using all oxidation states.
	#  E.g. Try the compound - KCa9V10O30, This can't be solved only with Vanadium's common oxidation state (+5)
	# However, this is solved by oxi_state_guesses() function, which means it's using other oxidation states as well.
	# Therefore, we are going to overide the elements' oxidation states with the most common oxidation states
	# by setting the argument oxi_states_override in oxi_state_guesses()
	################################################################################################

	## create a common oxidation states dictonary to overide
	dic = {}
	for element in ElementSym('H').ox_st_dict.keys():	# dummy element to instantiate the ElementSym class
		if element=='Eu':	# common and most stable ox. st. of Eu is +3, pymatgen provides both (+2 & +3). However, we select only +3
			common_ox_st = (3,)
		else:
			common_ox_st = Element(element).common_oxidation_states
		dic[element] = common_ox_st

	# return true if it could be solved using common oxi states
	if len(int_comp.oxi_state_guesses(oxi_states_override=dic))>0:
		return 1
	else:
		return 0

df_new_quaternary['has_common_oxi_states'] = df_new_quaternary.apply(has_common_oxi_states, axis=1)

best_new_quaternary = df_new_quaternary[df_new_quaternary['has_common_oxi_states']==1]
best_new_quaternary.to_csv(path+'/ML/common_ox_st_candidates.csv', sep='\t', index=False)	# best_new_quaternary_candidates.csv

##############################################################################################################################################################






###############################################################################################################################################################

best_new_quaternary = pd.read_csv(path+'/ML/common_ox_st_candidates.csv', sep='\t')	# best_new_quaternary_candidates.csv
cand_quaternary_systems = np.array(best_new_quaternary[['A1', 'A2', 'B1', 'B2']]).astype('U')
unique_quaternary = np.unique(np.sort(cand_quaternary_systems), axis=0)

## get unique quaternary systems
q_systems = []
for q_system in unique_quaternary.tolist():
	q_system.remove('_')
	q_system.append('O')
	q_systems.append('-'.join(q_system))

quaternary_systems = pd.DataFrame({'Quaternary System': q_systems})
quaternary_systems.to_csv(path+'/ML/quaternary_systems.csv', sep='\t', index=False)


#### Arrange chemical elements in A,B-sites in standard order and remove additional '0's to have widely accepted notation
def arrange_elements(row):
	site_list = re.split(r'[()]',row.StructuredFormula)	# get what's inside parantheses and others separately
	doped_site = max(site_list, key=len)	# get doped site re.sub(r"(?<=\d)0+", "", d)
	site_rearranged = '(' + re.sub(r"(?<=\d)0+", "", Composition(doped_site).formula.replace(' ','')) + ')'
	site_list[site_list.index(doped_site)] = site_rearranged	# replace with processed site
	compound = str(''.join(site_list))
	return compound

best_new_quaternary['StructuredFormula'] = best_new_quaternary.apply(arrange_elements, axis=1)


# Find 5 most similar compounds
ae = AutoEncoder()
VAE = ae.build_AE(vae=True)
VAE.load_weights(path+'/saved_models/best_model_VAE.h5')

def get_similar_compounds(row):
	most_similar_df = ae.most_similar(VAE, row.StructuredFormula, n=5, vae=True)
	return most_similar_df['CollectionCode'].to_list(), most_similar_df['StructuredFormula'].to_list(),\
			most_similar_df['CrystalClass'].to_list(), most_similar_df['Euclidean Distance'].to_list()

best_new_quaternary['Most Similar ICSD IDs'], best_new_quaternary['Most Similar Compounds'], best_new_quaternary['Most Similar Crystal Classes'], \
									best_new_quaternary['Euclidean Distances'] = zip(*best_new_quaternary.apply(get_similar_compounds, axis=1))

best_new_quaternary_sorted = best_new_quaternary.sort_values(by='Mean_classification_prob',ascending=False)
best_new_quaternary_sorted.to_csv(path+'/ML/novel_perovskite_candidates.csv', sep='\t', index=False)

###############################################################################################################################################################






###############################################################################################################################################################
## We select compounds which has elements that has only one common oxidation state. This is to better ensure the stability of the compound i.e. no redox reactions
novel_perovskites = pd.read_csv(path+'/ML/novel_perovskite_candidates.csv', sep='\t')

def get_nonredox(row):
	elements = [row.A1, row.A2, row.B1, row.B2]
	elements.remove('_')	# remove undoped dummy site
	non_redox = all(len(Element(elem).common_oxidation_states)==1 for elem in elements)	# check if all the elements have only 1 common oxidation state respectively
	return non_redox

novel_perovskites['is_non_redox'] = novel_perovskites.apply(get_nonredox, axis=1)

non_redox_perovskite_candidates = novel_perovskites[novel_perovskites['is_non_redox']==True]
non_redox_perovskite_candidates.to_csv(path+'/ML/non_redox_perovskite_candidates.csv', sep='\t', index=False)

###############################################################################################################################################################






###############################################################################################################################################################

## The next stage is automated web scraping of these compounds to further ensure novelty
## Please see 'web_scraper/bing_scraper.py'. This should be run separately because it is a python2 script

###############################################################################################################################################################
