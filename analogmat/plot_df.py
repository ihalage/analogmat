'''

The fingerprint representations of perovskite and non-perovskite materials along with their crystal system are investigated
Other algorithms such as t-SNE and PCA are also implemented for visualization

@Achintha_Ihalage
@02_Jul_2020
'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import pathlib
from keras.models import load_model
from autoencoder import AutoEncoder
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import rcParams


np.random.seed(15)
path = str(pathlib.Path(__file__).parent.absolute())

# plotting parameters
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12 
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'

class Fingerprints():

	def __init__(self, autoencoder_cls):
		self.AE = autoencoder_cls


	def get_fprint_dfs(self, model='vae'):
		if model == 'ae':
			exp_fprints = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_AE_trained.pkl')

		else: 
			exp_fprints = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_VAE_trained.pkl')
		
		pv_df = self.AE.read_csv(path+'/ICSD_data/perovskites.csv').sample(1750)
		npv_df = self.AE.read_csv(path+'/ICSD_data/non_perovskites.csv').sample(200)
		pv_npv_df = pd.concat([pv_df, npv_df], ignore_index=True)
		pv_npv_df_fprint = pv_npv_df.merge(exp_fprints[['CollectionCode', 'Fingerprint']], on='CollectionCode')

		def process_df(row):	# convert categorical crystal system name to a numerical value for plotting purposes // split fingeprint list in to x,y coordinate values
			system_val = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3, 'tetragonal': 4, 'cubic': 5, 'trigonal': 6, 'hexagonal': 7}
			return row.Fingerprint[0], row.Fingerprint[1], system_val[row.CrystalClass]

		pv_fprint_df = pv_df.merge(exp_fprints[['CollectionCode', 'Fingerprint']], on='CollectionCode')
		pv_fprint_df['Fingerprint_x'], pv_fprint_df['Fingerprint_y'], pv_fprint_df['CrystalSystemNum'] = zip(*pv_fprint_df.apply(process_df, axis=1))	# get crystal system as a numerical value
		npv_fprint_df = npv_df.merge(exp_fprints[['CollectionCode', 'Fingerprint']], on='CollectionCode')
		npv_fprint_df['Fingerprint_x'], npv_fprint_df['Fingerprint_y'], npv_fprint_df['CrystalSystemNum'] = zip(*npv_fprint_df.apply(process_df, axis=1))	# get crystal system as a numerical value

		
		pv_A = pv_fprint_df[pv_fprint_df['A2_frac']>0]	# A-site doped perovskites
		pv_B = pv_fprint_df[pv_fprint_df['A2_frac']==0]	# B-site doped perovskites
		npv_A = npv_fprint_df[npv_fprint_df['A2_frac']>0]	# A-site doped non-perovskites
		npv_B = npv_fprint_df[npv_fprint_df['A2_frac']==0]	# B-site doped non-perovskites

		# save csv to be imported to OriginLab for plotting
		pv_A[['CollectionCode', 'StructuredFormula', 'Fingerprint_x', 'Fingerprint_y', 'CrystalSystemNum']].to_csv(path+'/plot_data/pv_A-doped.csv', sep='\t', index=False)
		pv_B[['CollectionCode', 'StructuredFormula', 'Fingerprint_x', 'Fingerprint_y', 'CrystalSystemNum']].to_csv(path+'/plot_data/pv_B-doped.csv', sep='\t', index=False)
		npv_A[['CollectionCode', 'StructuredFormula', 'Fingerprint_x', 'Fingerprint_y', 'CrystalSystemNum']].to_csv(path+'/plot_data/npv_A-doped.csv', sep='\t', index=False)
		npv_B[['CollectionCode', 'StructuredFormula', 'Fingerprint_x', 'Fingerprint_y', 'CrystalSystemNum']].to_csv(path+'/plot_data/npv_B-doped.csv', sep='\t', index=False)

		return pv_A, pv_B, npv_A, npv_B

	def plot_fingerprints(self, model='vae'):
		pv_A, pv_B, npv_A, npv_B = self.get_fprint_dfs(model=model)

		ax = sns.scatterplot(x="Fingerprint_x", y="Fingerprint_y", data=pv_A, label='Perovskites (A-doped)')
		ax = sns.scatterplot(x="Fingerprint_x", y="Fingerprint_y", data=pv_B, color='purple', label='Perovskites (B-doped)')
		ax = sns.scatterplot(x="Fingerprint_x", y="Fingerprint_y", data=npv_A, label='Non-perovskites (A-doped)')
		ax = sns.scatterplot(x="Fingerprint_x", y="Fingerprint_y", data=npv_B, label='Non-perovskites (B-doped)')
		if model == 'ae':
			plt.xlim(-2, 45)
			plt.legend(loc="best")
		else:
			plt.xlim(-4, 4)
			plt.ylim(-4, 4)
			plt.legend(loc="lower left")
		plt.xlabel('Fingerprint x-component')
		plt.ylabel('Fingerprint y-component')
		plt.savefig(model+'_fingerprints.png', dpi=800)
		plt.show()

	def plot_pca_tsne(self, algo='tsne'):
		exp_fprints = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_VAE.pkl')
		pv_df = self.AE.read_csv(path+'/ICSD_data/perovskites.csv').sample(1750)
		npv_df = self.AE.read_csv(path+'/ICSD_data/non_perovskites.csv').sample(200)
		pv_npv_df = pd.concat([pv_df, npv_df], ignore_index=True)
		df = pv_npv_df.drop(['CollectionCode', 'HMS', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight', 'Temperature',
 					'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'SpaceGroupNum', 'CrystalClass', 'is_perovskite'], axis=1)

		if algo=='pca':
			tsne_coordinates = PCA(n_components=2).fit_transform(df)
		else:
			tsne_coordinates = TSNE(n_components=2, perplexity=80).fit_transform(df)

		df_with_tsne = pd.concat([pv_npv_df, pd.DataFrame({'t-SNE_Coordinates': list(tsne_coordinates)}, columns=['t-SNE_Coordinates'])], axis=1)

		def split_tsne_coordinates(row):
			return row['t-SNE_Coordinates'].tolist()[0], row['t-SNE_Coordinates'].tolist()[1]

		pv_dimred_df = pv_df.merge(df_with_tsne[['CollectionCode', 't-SNE_Coordinates']], on='CollectionCode')
		pv_dimred_df['t-SNE_x'], pv_dimred_df['t-SNE_y'] = zip(*pv_dimred_df.apply(split_tsne_coordinates, axis=1)) 
		npv_dimred_df = npv_df.merge(df_with_tsne[['CollectionCode', 't-SNE_Coordinates']], on='CollectionCode')
		npv_dimred_df['t-SNE_x'], npv_dimred_df['t-SNE_y'] = zip(*npv_dimred_df.apply(split_tsne_coordinates, axis=1))

		pv_A = pv_dimred_df[pv_dimred_df['A2_frac']>0]	# A-site doped perovskites
		pv_B = pv_dimred_df[pv_dimred_df['A2_frac']==0]	# B-site doped perovskites
		npv_A = npv_dimred_df[npv_dimred_df['A2_frac']>0]	# A-site doped non-perovskites
		npv_B = npv_dimred_df[npv_dimred_df['A2_frac']==0]	# B-site doped non-perovskites

		ax = sns.scatterplot(x="t-SNE_x", y="t-SNE_y", data=pv_A, label='Perovskites_A-doped')
		ax = sns.scatterplot(x="t-SNE_x", y="t-SNE_y", data=pv_B, color='purple', label='Perovskites_B-doped')
		ax = sns.scatterplot(x="t-SNE_x", y="t-SNE_y", data=npv_A, label='Non-perovskites_A-doped')
		ax = sns.scatterplot(x="t-SNE_x", y="t-SNE_y", data=npv_B, label='Non-perovskites_B-doped')

		plt.legend(loc="best")
		plt.xlabel('x-component')
		plt.ylabel('y-component')
		plt.tight_layout()
		plt.savefig(algo+'_visualisation.png', dpi=800)
		plt.show()

