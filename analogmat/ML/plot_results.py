
'''
Plot classification probabilities of selected A(BB')O3 compositions
@Achintha_Ihalage
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
# from generate_all_perovskites import ABSites
from pymatgen import Composition, Element
from ionic_radi import ElementSym
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import pickle

rcParams.update({'figure.autolayout': True})
rcParams['axes.linewidth'] = 2
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['axes.labelweight'] = 'bold'


path = str(pathlib.Path(__file__).parent.absolute().parent)


class ABBO3_Viz():

	def __init__(self):
		self.clf_results = pd.read_csv(path+'/ML/data/clf_results_plot.csv', sep='\t')
		self.selected_A_elems = {'Na':1, 'K':2, 'Cs':3, 'Ca':4, 'Sr':5, 'Ba':6, 'Pb':7}
		self.selected_B_elems = {'Ta':1, 'Nb':2, 'Zr':3, 'Co':4, 'Fe':5, 'Mn':6, 'Ti':7, 'Sc':8}

	def plot_Bdoped(self):
		cm = plt.cm.get_cmap('RdYlBu_r')
		h=8
		w=7
		fig = plt.figure(figsize=(12, 10))
		gs = gridspec.GridSpec(h, w,
		 wspace=0.2, hspace=0.5, 
		 top=1.-0.5/(h+1), bottom=0.5/(h+1), 
		 left=0.5/(w+1), right=1-0.5/(w+1)) 

		def plot(ax,a, b, B2_elems, nelems, x, probs, verts):
			
			for i,elm in enumerate(B2_elems[:nelems]):
				sc = ax.scatter([elm]*len(x[i]), x[i], c=probs[i], marker=verts, vmin=0, vmax=1, s=50, cmap=cm)
			if a!=0:
				ax.set_yticks([])
			else:
				ax.set_yticks(np.arange(0.1, 1, 0.4))
				ax.set_ylabel({v:k for k, v in self.selected_B_elems.items()}[b+1], fontsize=16)
			if b==len(self.selected_B_elems.keys())-1:
				ax.set_xlabel({v:k for k, v in self.selected_A_elems.items()}[a+1], fontsize=16)
			ax.tick_params(axis='x', rotation=-45)
			return sc

		for a,kA in enumerate(self.selected_A_elems.keys()):
			for b,kB in enumerate(self.selected_B_elems.keys()):
				df = self.clf_results[(self.clf_results.A1==kA) & (self.clf_results.B1==kB) & (self.clf_results.A2=='_')]
				usual_elms = ['Os', 'Ce', 'Ir', 'Nd', 'Eu', 'Gd', 'Dy', 'Ho', 'U', 'Np', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', # these are some of well known B-site elements
							'Nb', 'Mo', 'In', 'Sn', 'Hf', 'Ta', 'W', 'Bi']
				modes = df['B2'].mode().tolist()
				mode_usual_elms = list(set(usual_elms) & set(modes))
				B2_elems = df['B2'].value_counts().index.tolist()
				for el in mode_usual_elms:
					B2_elems.remove(el)
					B2_elems.insert(0, el)
				probs = []
				x = []
				for e in B2_elems:
					probs.append(df[df.B2 == e]['Mean_classification_prob'].tolist())
					x.append(df[df.B2 == e]['B2_frac'].tolist())

				if len(B2_elems)>=8:
					nelems = 8
				else:
					nelems = len(B2_elems)	

				width = 60
				height = 25
				ax= plt.subplot(gs[b,a])
				verts = list(zip([-width,width,width,-width],[-height,-height,height,height]))
				sc = plot(ax, a, b, B2_elems, nelems, x, probs, verts)
		
		# fig.subplots_adjust(right=0.5)
		cbar_ax = fig.add_axes([0.955, 0.15, 0.0125, 0.7])
		fig.colorbar(sc, cax=cbar_ax)
		plt.savefig(path+'/ML/clf_results_Bdoped.png', format='png', dpi=800)
		plt.show()

	def plot_individual(self):
		cm = plt.cm.get_cmap('RdYlBu_r')
		for a,kA in enumerate(self.selected_A_elems.keys()):
			for b,kB in enumerate(self.selected_B_elems.keys()):
				df = self.clf_results[(self.clf_results.A1==kA) & (self.clf_results.B1==kB) & (self.clf_results.A2=='_')]
				usual_elms = ['Os', 'Ce', 'Ir', 'Nd', 'Eu', 'Gd', 'Dy', 'Ho', 'U', 'Np', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', # these are some of well known B-site elements
							'Nb', 'Mo', 'In', 'Sn', 'Hf', 'Ta', 'W', 'Bi']
				modes = df['B2'].mode().tolist()
				mode_usual_elms = list(set(usual_elms) & set(modes))
				B2_elems = df['B2'].value_counts().index.tolist()
				for el in mode_usual_elms:
					B2_elems.remove(el)
					B2_elems.insert(0, el)
				probs = []
				x = []
				for e in B2_elems:
					probs.append(df[df.B2 == e]['Mean_classification_prob'].tolist())
					x.append(df[df.B2 == e]['B2_frac'].tolist())

				if len(B2_elems)>=8:
					nelems = 8
				else:
					nelems = len(B2_elems)	
				fig = plt.figure()
				ax = fig.add_subplot(111)
				width = 500
				height = 200
				verts = list(zip([-width,width,width,-width],[-height,-height,height,height]))
				for i,elm in enumerate(B2_elems[:nelems]):
					sc = ax.scatter([elm]*len(x[i]), x[i], c=probs[i], marker=verts, vmin=0, vmax=1, s=500, cmap=cm)

				ax.margins(x=0.1, y=0.05)
				plt.gcf().set_size_inches(5,5)
				plt.tight_layout()
				cbar = fig.colorbar(sc)
				cbar.ax.tick_params(labelsize=16)

				plt.savefig(path+'/ML/plots/%s_%s.png'%(kA, kB), dpi=800)
				plt.close()
				# plt.show()

