'''
	1. This python script generates all theoretically possible (A(1-x)A'x)BO3 and A(B(1-x)B'x)O3 perovskite oxides
	2. The generated new compounds are also subject to charge neutrality condition and pauling's valence rule
@Achintha_Ihalage
'''

import numpy as np
import pandas as pd
import itertools
import pathlib
from pymatgen.core.composition import Composition, Element
from arrange_ICSD_data import Perovskites

path = str(pathlib.Path(__file__).parent.absolute())

class ABSites():

	def __init__(self):

		self.A_sites = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'Zn', 'Ga', 'Ge', 'As',
						'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'Hg', 'Tl', 'Pb', 'Bi',
						'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Pu', 'Am']
		self.B_sites = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'Se',
						'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
						'Ce', 'Sm', 'Eu', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am']

AB = ABSites()

class CompGen():

	def __init__(self, AB_sites):
		self.AB_sites = AB_sites
		self.d = 0.05	# molar fraction interval
		self.fractions = np.arange(self.d, 0.999, self.d)

	def type1_comps(self):	# generate all combinations in the format of (A(1-x)A'x)BO3
		for a in itertools.combinations(self.AB_sites.A_sites, 2):
			for f in self.fractions:
				for b in self.AB_sites.B_sites:
					if b != a[0] and b!=a[1]:	# we assume same element cannot appear at both A and B sites
						yield ((a[0], a[1], b, '_', 'O'), (f, 1-f, 1, 0, 3), '(%s%s%s%s)%s%s'%(a[0], str(round(f, 2)), a[1], str(round(1-f, 2)), b, 'O3'))	# generator object yielding {elem: frac} dict and comp formula

	def type2_comps(self):	# generate all combinations in the format of A(B(1-x)B'x)O3
		for b in itertools.combinations(self.AB_sites.B_sites, 2):
			for f in self.fractions:
				for a in self.AB_sites.A_sites:
					if a!=b[0] and a!=b[1]:		# we assume same element cannot appear at both A and B sites
						yield ((a, '_', b[0], b[1], 'O'), (1, 0, f, 1-f, 3), '%s(%s%s%s%s)%s'%(a, b[0], str(round(f, 2)), b[1], str(round(1-f, 2)), 'O3'))

	def unzip(self, b):
		xs, ys, zs = zip(*b)
		return xs, ys, zs

	def create_df(self):
		elems1, fracs1, comps1 = self.unzip(list(self.type1_comps()))
		elems2, fracs2, comps2 = self.unzip(list(self.type2_comps()))

		# add chemical formula
		df = pd.DataFrame(comps1, columns=['StructuredFormula'])
		df = df.append(pd.DataFrame(comps2, columns=['StructuredFormula']), ignore_index=True)

		# add elements and corresponding molar fractions
		df['A1'], df['A1_frac'] = np.concatenate([np.array(elems1)[:,0], np.array(elems2)[:,0]]), np.concatenate([np.array(fracs1)[:,0], np.array(fracs2)[:,0]])
		df['A2'], df['A2_frac'] = np.concatenate([np.array(elems1)[:,1], np.array(elems2)[:,1]]), np.concatenate([np.array(fracs1)[:,1], np.array(fracs2)[:,1]])
		df['B1'], df['B1_frac'] = np.concatenate([np.array(elems1)[:,2], np.array(elems2)[:,2]]), np.concatenate([np.array(fracs1)[:,2], np.array(fracs2)[:,2]])
		df['B2'], df['B2_frac'] = np.concatenate([np.array(elems1)[:,3], np.array(elems2)[:,3]]), np.concatenate([np.array(fracs1)[:,3], np.array(fracs2)[:,3]])
		df['O'], df['O_frac'] = np.concatenate([np.array(elems1)[:,4], np.array(elems2)[:,4]]), np.concatenate([np.array(fracs1)[:,4], np.array(fracs2)[:,4]])

		d=df.iloc[:]
		df1 = Perovskites().add_features(d)
		df1 = df1[(df1['nA'] <= df1['nB'])]
		df1.to_csv(path+'/all_comps.csv', sep='\t', index=False)
