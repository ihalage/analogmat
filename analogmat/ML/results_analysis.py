'''
This script processes classification results and creates matrix data to plot perovskite probability distribution of some A-site and B-site element combinations

@Achintha_Ihalage
@05_Jul_2020

'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import pandas as pd
import numpy as np
import itertools
import sys
sys.path.append("..")
from generate_all_perovskites import ABSites
from pymatgen import Composition, Element
from ionic_radi import ElementSym

path = str(pathlib.Path(__file__).parent.absolute().parent)

class ResultsAnalysis():

	def __init__(self):
		self.generated_comps = pd.read_csv(path+'/ICSD_data/all_generated_compounds.csv', sep='\t')
		self.results = pd.read_csv(path+'/ML/classification_results.csv', sep='\t')		# must run get_novel_perovskites.py to obtain this
		self.AB = ABSites()

		# Assign numerical values to elements for plotting purposes
		self.A_elem_num = {'Li': 1, 'Na':2, 'K':3, 'Rb':4, 'Cs':5, 'Be':6, 'Mg':7, 'Ca':8, 'Sr':9, 'Ba':10, 'Sc':11, 'Ti':12, 'Zn':13, 'Y':14, 'Zr':15, 'Nb':16, 'Ag':17, 'Cd':18, 
				'Hf':19, 'Ta':20, 'Hg':21, 'B':22, 'Al':23, 'Ga':24, 'In':25, 'Tl':26, 'Si':27, 'Ge':28, 'Sn':29, 'Pb':30, 'As':31, 'Sb':32, 'Bi':33, 'La':34, 'Ce':35, 'Pr':36,
				'Nd':37, 'Sm':38, 'Eu':39, 'Gd':40, 'Tb':41, 'Dy':42, 'Ho':43, 'Er':44, 'Tm':45, 'Yb':46, 'Lu':47, 'Pu':48, 'Am':49}

		self.B_elem_num = {'Li':1, 'Na':2, 'Be':3, 'Mg':4, 'Ca':5, 'Sr':6, 'Sc':7, 'Ti':8, 'V':9, 'Cr':10, 'Mn':11, 'Fe':12, 'Co':13, 'Ni':14, 'Cu':15, 'Zn':16, 'Y':17, 'Zr':18, 
				'Nb':19, 'Mo':20, 'Tc':21, 'Ru':22, 'Pd':23, 'Ag':24, 'Cd':25, 'Hf':26, 'Ta':27, 'W':28, 'Re':29, 'Os':30, 'Ir':31, 'Au':32, 'Hg':33, 'B':34, 'Al':35, 'Ga':36, 
				'In':37, 'Tl':38, 'Si':39, 'Ge':40, 'Sn':41, 'Pb':42, 'Sb':43, 'Bi':44, 'Se':45, 'Te':46, 'La':47, 'Ce':48, 'Sm':49, 'Eu':50, 'Dy':51, 'Ho':52, 'Er':53, 'Tm':54,
				'Yb':55, 'Lu':56, 'Th':57, 'Pa':58, 'U':59, 'Np':60, 'Pu':61, 'Am':62}

	

	def A_site_analysis(self):
		df = self.results.merge(self.generated_comps, on='StructuredFormula')
		print(df.info(verbose=True, null_counts=True))

		A1_A2_B1_prob = []	# list to track the mean classification probability of each system. E.g. what's the mean classification probability of the system Ba(1-x)SrxTiO3 over all x? elements are represented by number as above
		try:
			for a in itertools.combinations(self.AB.A_sites, 2):
				for b in self.AB.B_sites:
					current_system = df[(df.A1==a[0]) & (df.A2==a[1]) & (df.B1==b)]				
					if current_system.shape[0]>0:	# is system exists, or not null
						mean_prob = np.mean(np.array(current_system['Mean_classification_prob']))
						A1_A2_B1_prob.append([self.A_elem_num[a[0]], self.A_elem_num[a[1]], self.B_elem_num[b], mean_prob])
						print (a[0], a[1], b, mean_prob)
			prob_df = pd.DataFrame(A1_A2_B1_prob, columns=['A1', 'A2', 'B1', 'Mean_classification_prob'])
			prob_df.to_csv(path+'/plot_data/A_doped_clf_probs.csv', sep='\t', index=False)
		except KeyboardInterrupt:
			prob_df = pd.DataFrame(A1_A2_B1_prob, columns=['A1', 'A2', 'B1', 'Mean_classification_prob'])
			prob_df.to_csv(path+'/plot_data/A_doped_clf_probs.csv', sep='\t', index=False)

	def B_site_analysis(self):
		df = self.results.merge(self.generated_comps, on='StructuredFormula')
		A1_B1_B2_prob = []	# list to track the mean classification probability of each system. E.g. what's the mean classification probability of the system PbZr(1-x)TixO3 over all x? elements are represented by number as above
		try:
			for b in itertools.combinations(self.AB.B_sites, 2):
				for a in self.AB.A_sites:
					current_system = df[(df.A1==a) & (df.B1==b[0]) & (df.B2==b[1])]

					if current_system.shape[0]>0:
						mean_prob = np.mean(np.array(current_system['Mean_classification_prob']))
						A1_B1_B2_prob.append([self.A_elem_num[a], self.B_elem_num[b[0]], self.B_elem_num[b[1]], mean_prob])
						print (a, b[0], b[1], mean_prob)
			prob_df = pd.DataFrame(A1_B1_B2_prob, columns=['A1', 'B1', 'B2', 'Mean_classification_prob'])
			prob_df.to_csv(path+'/plot_data/B_doped_clf_probs.csv', sep='\t', index=False)
		except KeyboardInterrupt:
			prob_df = pd.DataFrame(A1_B1_B2_prob, columns=['A1', 'B1', 'B2', 'Mean_classification_prob'])
			prob_df.to_csv(path+'/plot_data/B_doped_clf_probs.csv', sep='\t', index=False)


	def arrange_A_to_upper_triangle(self):	# arrange data into a lower triangle
		df_A = pd.read_csv(path+'/plot_data/A_doped_clf_probs.csv', sep='\t')

		def swap_cols(row):
			if row.A1>row.A2:
				return row.A2, row.A1
			return row.A1, row.A2
		df_A['A1'], df_A['A2'] = zip(*df_A.apply(swap_cols, axis=1))
		df_A.to_csv(path+'/plot_data/A_doped_clf_probs.csv', sep='\t', index=False)

	def arrange_B_to_upper_triangle(self):	# arrange data into a lower triangle
		df_B = pd.read_csv(path+'/plot_data/B_doped_clf_probs.csv', sep='\t')

		def swap_cols(row):
			if row.B1>row.B2:
				return row.B2, row.B1
			return row.B1, row.B2
		df_B['B1'], df_B['B2'] = zip(*df_B.apply(swap_cols, axis=1))
		df_B.to_csv(path+'/plot_data/B_doped_clf_probs.csv', sep='\t', index=False)

	def A_doped_compounds(self, elem='Na'):	# get classification results of all (A0.5A'0.5)BO3 compounds for a given A (e.g. A={Na, K, Rb, Cs, Mg, Ca, Sr, Ba, ..})
		df = self.results.merge(self.generated_comps, on='StructuredFormula')
		df_elem = df[((df.A1==elem) | (df.A2==elem)) & (df.A1_frac==0.5) & (df.B2=='_')]

		def has_common_oxi_states(row):	# this snippet is borrowed from get_novel_perovskites.py
			comp = Composition(row.StructuredFormula)
			int_comp = Composition(comp.get_integer_formula_and_factor()[0]) # get integer formular

			## create a common oxidation states dictonary to overide
			dic = {}
			for element in ElementSym('H').ox_st_dict.keys():	# dummy element to instantiate the ElementSym class
				if element=='Eu':	
					common_ox_st = (3,)
				else:
					common_ox_st = Element(element).common_oxidation_states
				dic[element] = common_ox_st

			# return true if it could be solved using common oxi states
			if len(int_comp.oxi_state_guesses(oxi_states_override=dic))>0:
				return 1
			else:
				return 0

		df_elem['has_common_oxi_states'] = df_elem.apply(has_common_oxi_states, axis=1)
		df_elem = df_elem[df_elem['has_common_oxi_states']==1]

		unique_A = np.unique(df_elem[['A1', 'A2']].values)
		A_num = {k:self.A_elem_num[k] for k in unique_A}
		A_num = {k: v for k, v in sorted(A_num.items(), key=lambda item: item[1])}
		A_num = dict(zip(list(A_num.keys()), np.arange(1, len(unique_A)+1) ))
		print(A_num)

		unique_B = np.unique(df_elem[['B1']].values)
		B_num = {k:self.B_elem_num[k] for k in unique_B}
		B_num = {k: v for k, v in sorted(B_num.items(), key=lambda item: item[1])}
		B_num = dict(zip(list(B_num.keys()), np.arange(1, len(unique_B)+1) ))

		def elem_to_num(row):
			A1Num = A_num[elem]
			A2Num = A_num[row.A2] if row.A1==elem else A_num[row.A1]
			B1Num = B_num[row.B1]

			return A1Num, A2Num, B1Num
		
		df_elem['A1Num'], df_elem['A2Num'], df_elem['B1Num'] = zip(*df_elem.apply(elem_to_num, axis=1))
		df_elem = df_elem.sort_values(by=['Mean_classification_prob'], ascending=False)
		df_elem[['A1Num', 'A2Num', 'B1Num', 'Mean_classification_prob']].to_csv(path+'/plot_data/'+elem+'_clf_results.csv', sep='\t', index=False)
		print(df_elem.shape[0])


	def B_doped_compounds(self, elem='V'):	# get classification results of all A(B0.5B'0.5)O3 compounds for a given B (e.g. B={V, Nb, Ta, ..})
		df = self.results.merge(self.generated_comps, on='StructuredFormula')
		df_elem = df[((df.B1==elem) | (df.B2==elem)) & (df.B1_frac==0.5) & (df.A2=='_')]

		def has_common_oxi_states(row):	# this snippet is borrowed from get_novel_perovskites.py
			comp = Composition(row.StructuredFormula)
			int_comp = Composition(comp.get_integer_formula_and_factor()[0]) # get integer formular

			## create a common oxidation states dictonary to overide
			dic = {}
			for element in ElementSym('H').ox_st_dict.keys():	# dummy element to instantiate the ElementSym class
				if element=='Eu':	
					common_ox_st = (3,)
				else:
					common_ox_st = Element(element).common_oxidation_states
				dic[element] = common_ox_st

			# return true if it could be solved using common oxi states
			if len(int_comp.oxi_state_guesses(oxi_states_override=dic))>0:
				return 1
			else:
				return 0

		df_elem['has_common_oxi_states'] = df_elem.apply(has_common_oxi_states, axis=1)
		df_elem = df_elem[df_elem['has_common_oxi_states']==1]

		unique_A = np.unique(df_elem[['A1']].values)
		A_num = {k:self.A_elem_num[k] for k in unique_A}
		A_num = {k: v for k, v in sorted(A_num.items(), key=lambda item: item[1])}
		A_num = dict(zip(list(A_num.keys()), np.arange(1, len(unique_A)+1) ))
		# A_num = {'Li': 1, 'Na': 2, 'K': 3, 'Rb': 4, 'Cs': 5, 'Be': 6, 'Mg': 7, 'Ca': 8, 'Sr': 9, 'Ba': 10, 'Zn': 11, 'Ag': 12, 'Cd': 13, 'Hg': 14, 'Tl': 15, 'Pb': 16, 'Sm': 17, 'Eu': 18, 'Yb': 19}
		# A_num = {'Li': 1, 'Na': 2, 'K': 3, 'Rb': 4, 'Cs': 5, 'Be': 6, 'Mg': 7, 'Ca': 8, 'Sr': 9, 'Ba': 10, 'Zn': 11, 'Ag': 12, 'Cd': 13, 'Hg': 14, 'Tl': 15, 'Ge': 16, 'Pb': 17, 'La': 18, 'Nd': 19, 'Sm': 20, 'Eu': 21, 'Yb': 22}
		# A_num = dict(zip(unique_A, np.arange(1, len(unique_A)+1)))

		unique_B = np.unique(df_elem[['B1', 'B2']].values)
		B_num = {k:self.B_elem_num[k] for k in unique_B}
		B_num = {k: v for k, v in sorted(B_num.items(), key=lambda item: item[1])}
		B_num = dict(zip(list(B_num.keys()), np.arange(1, len(unique_B)+1) ))
		# B_num = {'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 6, 'Co': 7, 'Y': 8, 'Zr': 9, 'Nb': 10, 'Mo': 11, 'Ru': 12, 'Pd': 13, 'Hf': 14, 'Ta': 15, 'Ir': 16, 'Au': 17, 'Hg': 18, 'B': 19, 'Al': 20, 'Ga': 21, 'In': 22, 'Tl': 23, 'Si': 24, 'Ge': 25, 'Sn': 26, 'Pb': 27, 'Sb': 28, 'Bi': 29, 'Se': 30, 'Te': 31, 'La': 32, 'Ce': 33, 'Sm': 34, 'Eu': 35, 'Dy': 36, 'Ho': 37, 'Er': 38, 'Tm': 39, 'Yb': 40, 'Lu': 41, 'Th': 42, 'Pa': 43, 'Np': 44}
		# B_num = {'Ti': 1, 'V': 2, 'Cr': 3, 'Mn': 4, 'Fe': 5, 'Co': 6, 'Ni': 7, 'Zr': 8, 'Nb': 9, 'Mo': 10, 'Tc': 11, 'Ru': 12, 'Pd': 13, 'Hf': 14, 'Ta': 15, 'W': 16, 'Re': 17, 'Os': 18, 'Ir': 19, 'Si': 20, 'Ge': 21, 'Sn': 22, 'Pb': 23, 'Sb': 24, 'Se': 25, 'Te': 26, 'Ce': 27, 'Th': 28, 'Pa': 29, 'U': 30, 'Np': 31}
		# B_num = dict(zip(unique_B, np.arange(1, len(unique_B)+1)))

		def elem_to_num(row):
			A1Num = A_num[row.A1]
			B1Num = B_num[elem]
			B2Num = B_num[row.B2] if row.B1==elem else B_num[row.B1]
			return A1Num, B1Num, B2Num
		
		df_elem['A1Num'], df_elem['B1Num'], df_elem['B2Num'] = zip(*df_elem.apply(elem_to_num, axis=1))
		df_elem = df_elem.sort_values(by=['Mean_classification_prob'], ascending=False)
		df_elem[['A1Num', 'B1Num', 'B2Num', 'Mean_classification_prob']].to_csv(path+'/plot_data/'+elem+'_clf_results.csv', sep='\t', index=False)
