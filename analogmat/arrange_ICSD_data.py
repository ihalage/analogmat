'''
Data processing
@Achintha_Ihalage
'''


import pandas as pd
import numpy as np
import os
import re
import math
import pathlib
import itertools
from pymatgen.core.composition import Composition, Element
from pymatgen.symmetry.groups import SpaceGroup, PointGroup
from ionic_radi import ElementSym	# to get ionic radius
from mendeleev import element	# get properties that are not available in pymatgen

path = str(pathlib.Path(__file__).parent.absolute())


relative_dir = path+'/ICSD_data/Perovskites_Non_perovskites_txt/'
data_dir = path+'/ICSD_data/'

def txt_to_csv(relative_dir, data_dir):
	df = pd.DataFrame(columns=['CollectionCode', 'HMS',	'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 
							'CellVolume', 'FormulaWeight', 'Temperature', 'PublicationYear', 'Quality'])
	columns=['CollectionCode', 'HMS',	'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 
							'CellVolume', 'FormulaWeight', 'Temperature', 'PublicationYear', 'Quality']
	for filename in os.listdir(relative_dir):
		if filename.endswith(".txt"):
			data = pd.read_csv(relative_dir+filename, sep='\t')
			data = data.iloc[:, :-1]
			df = pd.concat([df,data], axis=0, ignore_index=True)
	# df.to_csv(data_dir+'ICSD_perovskites.csv', sep='\t', index=False)
	df.to_csv('test.csv', sep='\t', index=False)
	# df.to_csv(data_dir+'ICSD_all_data.csv', sep='\t', index=False)
	return df



class Perovskites():

	def __init__(self, df):
		_ = self.parse_formula(df)



	# 1. check whether the material is A(B'B'')O3 or (A'A'')BO3 typed
			# get the elements
	# 2. Get Atomic fractions in the (same)fromat of A(A')B(B')O3
	def parse_formula(self, df):

		def get_no_elements(row):
			digit_removed = ''.join([i for i in row if not i.isdigit()])
			return (len(digit_removed.split('.')) + 1)	#of elements

		def parse_sites(row):
			comp = Composition(row)
			# get elements in the order of the compound // pymatgen's comp.formula gives elements sorted by electronegativity,
			# we don't want that.  We want to preserve A(A')B(B')O3 order
			elements = [''.join([i for i in elem if not i.isdigit()]).replace(".", "") for elem in comp.formula.split()]	
			indices = {e: row.find(e) for e in elements}
			A1, A2, B1, B2 = '_', '_', '_', '_'
			A1_frac, A2_frac, B1_frac, B2_frac, O_frac = 0, 0, 0, 0, 0
			if (len(row.split("(")[0])==0 or len(row.split("(")[0])>8):	# type =1   ### value '8' is hard coded and it works. But check
				A1, A2, B1, O = sorted(elements, key=indices.get)
				elem_fracs = [comp.get_atomic_fraction(Element(i))*comp.num_atoms for i in [A1, A2, B1, O]]
				# now formulate A(A')B(B')O3 format
				A1_frac, A2_frac, B1_frac, O_frac = [round(i*3.0/elem_fracs[-1], 3) for i in elem_fracs]
			else:	# type = 2
				A1, B1, B2, O = sorted(elements, key=indices.get)
				elem_fracs = [comp.get_atomic_fraction(Element(i))*comp.num_atoms for i in [A1, B1, B2, O]]
				A1_frac, B1_frac, B2_frac, O_frac = [round(i*3.0/elem_fracs[-1], 3) for i in elem_fracs]
				
			return A1, A1_frac, A2, A2_frac, B1, B1_frac, B2, B2_frac, O, O_frac
		self.get_no_elements = get_no_elements
		self.parse_sites = parse_sites

		
		df['StructuredFormula'] = df['StructuredFormula'].str.replace(' ', '')	# remove all white spaces
		df = df[df['StructuredFormula'].apply(get_no_elements) <= 4]	# remove materials that have more than 4 elements and fractional oxygens
		df = df[(df.StructuredFormula.str.contains("\(")) & (df.StructuredFormula.str.contains("\)")) ]#& 
							 # ((df.StructureType.str.contains(pat="perovskite", case=False)) | (df.StructureType.str.contains(pat="elpasolite", case=False)) )]#| (df.StructureType.str.contains(pat="linbo3", case=False)))]

		df['A1'], df['A1_frac'], df['A2'], df['A2_frac'], df['B1'], df['B1_frac'], \
						df['B2'], df['B2_frac'], df['O'], df['O_frac'] = zip(*df['StructuredFormula'].map(parse_sites))
		# delete others represented in miscellaneous formats
		df = df[(df.A1_frac <= 1) & (df.A2_frac <= 1) & (df.B1_frac <= 1) & (df.B2_frac <= 1) & (df.O_frac <= 3)]
		df = df[abs(df.A1_frac + df.A2_frac + df.B1_frac + df.B2_frac -2 )<0.002]	# impose 0.002 threshold mol fraction difference // sum of mole fractions is within 1 +- (0.002)
		return df


	def parse_lattice_param(self, df):
		def get_param(row):
			a, b, c, alpha, beta, gamma = [re.sub("[\(\[].*?[\)\]]", "", i) for i in row.split(' ')]
			return a, b, c, alpha, beta, gamma

		df['a'], df['b'], df['c'], df['alpha'], df['beta'], df['gamma'] = zip(*df['CellParameter'].map(get_param))
		return df

	def parse_space_group(self, df):
		def space_group_num(row):
			try:
				if len(row)>2 and row[-1] in ['H', 'R']:	# remove conventions
					return SpaceGroup(row[:-1]).int_number, SpaceGroup(row[:-1]).crystal_system
				else:
					return SpaceGroup(row).int_number, SpaceGroup(row).crystal_system
			except ValueError:	# miscellaneous/incorrect space-group. pymatgen wasn't able to identify these
				return None, None

		df['SpaceGroupNum'], df['CrystalClass'] = zip(*df['HMS'].map(space_group_num))
		return df

	# add additional features from python's mendeleev/pymatgen packages // element related
	def add_features(self, df):

		###############################################################################################################################
		# Unfortunately, pymatgen's Element().ionic_radii does not provide ionic radii for some of the well known oxidation states of 
		# some elements (e.g. Cr's ionic radius is only given for +2 oxydation state whereas most common oxidation state is +3)
		# Therefore a python script containing the "crystal" ionic radii is written (ionic_radii.py). Shanon's work indicates that 
		# crystal ionic radii best represent the physical ionic radii of solids
		###############################################################################################################################
		def get_avg_ionic_radius(element, total_atoms, fractional_charge):	# calculate average ionic radius of a fractional oxi_state site
			elem = Element(element)
			ox_states_element = elem.common_oxidation_states
			# ionic_radii_dict = elem.ionic_radii
			total_charge = round(fractional_charge*total_atoms)
			def get_possibilities(ox_states_element, total_atoms, total_charge):	# calculate oxidation numbers of each individual ions that has an average fractional oxi_state
				coef_possibilities = itertools.product(np.arange(total_atoms), repeat=len(ox_states_element))
				for i in coef_possibilities:
					if sum(i)==total_atoms and np.array(i).dot(np.array(ox_states_element).T)==total_charge:
						yield i	# outputs how many ions have oxi_state of [ox_states_elements]
			if len(list(get_possibilities(ox_states_element, total_atoms, total_charge)))>0: # try to solve with most common oxidation states
				no_possible_ions = list(get_possibilities(ox_states_element, total_atoms, total_charge))[0]
			else:
				ox_states_element = ElementSym(element).ox_st_dict[element]	# if no solutions with most common oxidation states, consider other oxi_states that we know ionic radii
				try:	# check if this has a solution/ if not return -1
					no_possible_ions = list(get_possibilities(ox_states_element, total_atoms, total_charge))[0]
				except IndexError:
					return -1
			try:
				ox_state_no_ions = dict(zip(ox_states_element, no_possible_ions))

				avg_ionic_radius = sum([ElementSym(element).ionic_radii(st)*ions for st,ions in ox_state_no_ions.items()])/sum(ox_state_no_ions.values())
			except KeyError:	# the ionic radius for this oxidation state is not available 
				return -1

			return avg_ionic_radius

		def get_ionic_properties(row):
			# getting oxidation state of fractional formula is not implemented yet in pymatgen
			# round the fractional formular to 1 decimal place in order to speed up guessed_oxidation_state calculation in pymatgen
			# sum of fractions is preserved (=1)
			elem_frac = {row.A1:row.A1_frac, row.A2:row.A2_frac, row.B1:row.B1_frac, row.B2:row.B2_frac, row.O:row.O_frac}
			# _=elem_frac.pop('_', None)
			elem_frac_red = { k:v for k, v in elem_frac.items() if (v<1 and v>0)}	# remove empty cation and oxygen anion
			# ceil and floor to 1 decimal places and create list
			frac_list=[(k,math.ceil(v*10)/10.0) if v==min(list(elem_frac_red.values())) else (k,math.floor(v*10)/10.0) for k, v in elem_frac_red.items()]
			frac_dict = {k:v for k,v in frac_list}
			elem_frac_copy = elem_frac.copy()	# make a copy so that original element fractions are not updated
			elem_frac_copy.update(frac_dict)	# update the dictionary with 1 decimal precision {Element: fraction} where fraction ceiled/floored for 1 decimal place

			# get fractional formula with 1 decimal point rounded fractions
			l=[]
			[l.append(k+str(v)) for k,v in elem_frac_copy.items() if k!='_'][0]
			formula_1deci = ''.join(l)
			
			comp = Composition(formula_1deci) # create pymatgen Composition object
			int_comp = Composition(comp.get_integer_formula_and_factor()[0]) # get integer formular
			elem_ionic_radius = {}
			# try:
			
			if len(int_comp.oxi_state_guesses())>0:
				ox_state_dict = int_comp.oxi_state_guesses()[0] # get the best oxidation state guess / pymatgen outputs fractional oxidation states where necessary
				for (elem,ox_st) in list(ox_state_dict.items()):
					if not ox_st.is_integer() or ox_st not in ElementSym(elem).ox_st_dict[elem]:	
						avg_ionic_radius = get_avg_ionic_radius(elem, int(int_comp.get(elem)), ox_st)
						if avg_ionic_radius == -1:	# oxidation states cannot be solved with all available oxidation states 
							break

						elem_ionic_radius[elem] = avg_ionic_radius
					else:
						ionic_radius = ElementSym(elem).ionic_radii(ox_st)
						elem_ionic_radius[elem] = ionic_radius
				if len(elem_ionic_radius)==4:
					# now update the first elem_frac dict with the found ionic radius values (some oinic radii may be averages because of fractional oxidation state)
					elem_radii = elem_frac.copy() # for clarity
					elem_radii.update(elem_ionic_radius)

					rA_avg = (elem_frac[row.A1]*elem_radii[row.A1] + elem_frac[row.A2]*elem_radii[row.A2])
					rB_avg = (elem_frac[row.B1]*elem_radii[row.B1] + elem_frac[row.B2]*elem_radii[row.B2])
					rO = ElementSym('O').ionic_radii(-2)	# oxygen's oxidation state is always -2

					ox_st_dict_copy = elem_frac.copy()	# to find nA, nB, nO
					ox_st_dict_copy.update(ox_state_dict)

					nA = elem_frac[row.A1]*ox_st_dict_copy[row.A1] + elem_frac[row.A2]*ox_st_dict_copy[row.A2]
					nB = elem_frac[row.B1]*ox_st_dict_copy[row.B1] + elem_frac[row.B2]*ox_st_dict_copy[row.B2]
					nO = -2 # Oxygent oxidation state

				# to make it easy to understand that these materials are discarded, the else statements are added below // not necessary if Ra_avg, Rb_avg etc. were initialized with -1 at the top
				else:
					rA_avg = -1	# discard these materials / oxidation states could not be solved by the algorithm
					rB_avg = -1
					rO = ElementSym('O').ionic_radii(-2)

					nA = -1
					nB = -1
					nO = -2 # Oxygen oxidation state

			else:
				rA_avg = -1	# discard these materials where the ions show unknown oxidation states/ typically higher that the max oxi_state of a particular element
				rB_avg = -1	# pymatgen's oxi_state_guesses() couldn't solve these even with fractional oxidation states
				rO = ElementSym('O').ionic_radii(-2)

				nA = -1	# discard
				nB = -1	# discard
				nO = -2 # Oxygent oxidation state


			return nA, nB, nO, rA_avg, rB_avg, rO

		###################### features extracted from pymatgen ##########################
		def get_atomic_number(row):
			atom_numA1, atom_numA2, atom_numB1, atom_numB2, atom_numO = [Element(i).number if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return atom_numA1, atom_numA2, atom_numB1, atom_numB2, atom_numO
		def get_mendeleev_number(row):
			mend_numA1, mend_numA2, mend_numB1, mend_numB2, mend_numO = [Element(i).mendeleev_no if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return mend_numA1, mend_numA2, mend_numB1, mend_numB2, mend_numO
		def get_atomic_radius(row):
			atomic_rA1, atomic_rA2, atomic_rB1, atomic_rB2, atomic_rO = [Element(i).atomic_radius  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return atomic_rA1, atomic_rA2, atomic_rB1, atomic_rB2, atomic_rO
		def get_atom_electronegativity(row):
			A1_X, A2_X, B1_X, B2_X, O_X = [Element(i).X  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return A1_X, A2_X, B1_X, B2_X, O_X
		def get_atomic_mass(row):
			M_A1, M_A2, M_B1, M_B2, M_O = [Element(i).atomic_mass  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return M_A1, M_A2, M_B1, M_B2, M_O
		def get_atomic_volume(row):
			V_A1, V_A2, V_B1, V_B2, V_O = [Element(i).molar_volume  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return V_A1, V_A2, V_B1, V_B2, V_O
		def get_thermal_conductivity(row):
			therm_con_A1, therm_con_A2, therm_con_B1, therm_con_B2, therm_con_O = [Element(i).thermal_conductivity  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return therm_con_A1, therm_con_A2, therm_con_B1, therm_con_B2, therm_con_O
		def get_row(row):
			Row_A1, Row_A2, Row_B1, Row_B2, Row_O = [Element(i).row  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return Row_A1, Row_A2, Row_B1, Row_B2, Row_O
		def get_group(row):	# get periodic table group number of the elements
			Group_A1, Group_A2, Group_B1, Group_B2, Group_O = [Element(i).group  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return Group_A1, Group_A2, Group_B1, Group_B2, Group_O

		########################## features extracted from mendeleev package ###############################
		def get_dipole_polarizability(row):
			polarizability_A1, polarizability_A2, polarizability_B1, polarizability_B2, polarizability_O = [element(i).dipole_polarizability  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return polarizability_A1, polarizability_A2, polarizability_B1, polarizability_B2, polarizability_O
		def get_lattice_constant(row):
			lattice_const_A1, lattice_const_A2, lattice_const_B1, lattice_const_B2, lattice_const_O = [element(i).lattice_constant  if i!='_' else 0 for i in [row.A1, row.A2, row.B1, row.B2, row.O]]
			return lattice_const_A1, lattice_const_A2, lattice_const_B1, lattice_const_B2, lattice_const_O

		def get_r_combinations(row):
			return row.rA_avg/row.rB_avg, row.rA_avg/row.rO, row.rB_avg/row.rO, row.rA_avg - row.rB_avg
		def get_Goldschmidt_TF(row):
			return (row.rA_avg + row.rO)/(math.sqrt(2)*(row.rB_avg + row.rO))



		df['atom_numA1'], df['atom_numA2'], df['atom_numB1'], df['atom_numB2'], df['atom_numO'] = zip(*df.apply(get_atomic_number, axis=1))
		df['mend_numA1'], df['mend_numA2'], df['mend_numB1'], df['mend_numB2'], df['mend_numO'] = zip(*df.apply(get_mendeleev_number, axis=1))
		df['atomic_rA1'], df['atomic_rA2'], df['atomic_rB1'], df['atomic_rB2'], df['atomic_rO'] = zip(*df.apply(get_atomic_radius, axis=1))
		df['A1_X'], df['A2_X'], df['B1_X'], df['B2_X'], df['O_X'] = zip(*df.apply(get_atom_electronegativity, axis=1))
		df['M_A1'], df['M_A2'], df['M_B1'], df['M_B2'], df['M_O'] = zip(*df.apply(get_atomic_mass, axis=1))
		df['V_A1'], df['V_A2'], df['V_B1'], df['V_B2'], df['V_O'] = zip(*df.apply(get_atomic_volume, axis=1))
		df['therm_con_A1'], df['therm_con_A2'], df['therm_con_B1'], df['therm_con_B2'], df['therm_con_O'] = zip(*df.apply(get_thermal_conductivity, axis=1))
		df['polarizability_A1'], df['polarizability_A2'], df['polarizability_B1'], df['polarizability_B2'], df['polarizability_O'] = zip(*df.apply(get_dipole_polarizability, axis=1)) # mendeleev # slow
		df['lattice_const_A1'], df['lattice_const_A2'], df['lattice_const_B1'], df['lattice_const_B2'], df['lattice_const_O'] = zip(*df.apply(get_lattice_constant, axis=1)) # mendeleev # slow
		df['Row_A1'], df['Row_A2'], df['Row_B1'], df['Row_B2'], df['Row_O'] = zip(*df.apply(get_row, axis=1))
		df['Group_A1'], df['Group_A2'], df['Group_B1'], df['Group_B2'], df['Group_O'] = zip(*df.apply(get_group, axis=1))
		df['nA'], df['nB'], df['nO'], df['rA_avg'], df['rB_avg'], df['rO'] = zip(*df.apply(get_ionic_properties, axis=1))
		df['rA/rB'], df['rA/rO'], df['rB/rO'], df['rA-rB'] = zip(*df.apply(get_r_combinations, axis=1))
		df['Goldschmidt_TF'] = df.apply(get_Goldschmidt_TF, axis=1)
		df = df[(df['A1_frac']+df['A2_frac'] == 1) & (df['B1_frac']+df['B2_frac'] == 1)]	# materials in the format of A(A')B(B')O3
		df = df[df.rA_avg != -1]	# discard the materials that we couldn't solve the oxidation states
		df = df[df.rA_avg > df.rB_avg]	# discard incorrectly identified A-B sites due to miscellaneous chemical formula formatting
		
		return df

	def SISSO_features(self, df):	# add 10 best features as obtained by SISSO algorithm // see the file SISSO/calculations/perovskite_descriptors/feature_space/space_001d.name

		def get_sisso_features(row):
			f1 = abs((row.atomic_rA1**3)-abs(row.B1_X - row.polarizability_B1))
			f2 = abs((row.V_A1 + row.atomic_rA1) - abs(row.Group_B1 - row.polarizability_B1))
			f3 = (row.Row_A1 + row.atomic_rA1) - abs(row.A1_X - row.B1_X)
			f4 = (abs(row.atomic_rA1 - row.therm_con_B2)/(row.mend_numA1)**3)
			f5 = abs((row.V_A1 + row.atomic_rA1) - abs(row.B1_X - row.polarizability_B1))
			f6 = (row.Row_A1 + row.polarizability_A1) - abs(row.rA_avg - row.rB_avg - row.rA_avg/row.rB_avg)
			f7 = (row.Row_A1 + row.polarizability_A1) - abs(row.rA_avg/row.rB_avg - row.Goldschmidt_TF)
			f8 = math.sin(row.atomic_rB1) + (row.Row_A1 + row.polarizability_A1)
			f9 = (row.mend_numA1)**-1 - math.cos(row.Goldschmidt_TF)
			f10 = (row.mend_numA1)**-1 + abs(row.rA_avg - row.rB_avg - row.rA_avg/row.rB_avg)

			return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
		
		df['SISSO_1'], df['SISSO_2'], df['SISSO_3'], df['SISSO_4'], df['SISSO_5'], df['SISSO_6'], df['SISSO_7'], df['SISSO_8'], df['SISSO_9'], df['SISSO_10'] = zip(*df.apply(get_sisso_features, axis=1))
		return df


class NonPerovskites():

	def __init__(self, perovskites_cls):
		self.perovskites = perovskites_cls

	def process_npv(self, df):
		
		def is_ABO3_struc(row):
			comp = Composition(row)
			if round(comp.get('O')%3)==0:	# get Oxygen fraction, this should be a multiple of 3 to have a final formula in the format of A(A')B(B')O3
				return True
			return False

		df['StructuredFormula'] = df['StructuredFormula'].str.replace(' ', '')	# remove all white spaces
		df = df[df['StructuredFormula'].apply(self.perovskites.get_no_elements) <= 4]	# remove materials that have more than 4 elements
		df = df[(df.StructuredFormula.str.contains("\(")) & (df.StructuredFormula.str.contains("\)"))]

		df = df[df['StructureType'].str.contains(pat="perovskite", case=False) == False]	# get all non-perovskites
		df = df[df['StructureType'].str.contains(pat="Elpasolite", case=False) == False]	# Elpasolites are also double perovskites

		df = df[df['StructuredFormula'].apply(is_ABO3_struc) == True]

		df['A1'], df['A1_frac'], df['A2'], df['A2_frac'], df['B1'], df['B1_frac'], \
						df['B2'], df['B2_frac'], df['O'], df['O_frac'] = zip(*df['StructuredFormula'].map(self.perovskites.parse_sites))
		df = df[(df["HMS"].str.contains('P n -3 Z') | df['HMS'].str.contains('P 42/n Z') | df['HMS'].str.contains('F d -3 m Z')) == False]	# remove miscallaneous space group

		df = df[(df['A1_frac']+df['A2_frac'] == 1) & (df['B1_frac']+df['B2_frac'] == 1)]	# materials in the format of A(A')B(B')O3

		df = self.perovskites.parse_lattice_param(df)	# get lattice parameters
		df = self.perovskites.parse_space_group(df)		# get space group number 
		df = self.perovskites.add_features(df)			# add atomic properties, such as oxidation states, ionic radii
		df = self.perovskites.SISSO_features(df)	# add SISSO calculated features
		df['is_perovskite'] = 0

		return df
