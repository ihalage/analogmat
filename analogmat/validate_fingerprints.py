'''
Find the N-nearest neighbours (N=5) of each experimental compound in the fingerprint space. 
The highest voted crystal class by the neighbors should ideally be the actual crystal class of the considered compound.
Voted crystal class and the real crystal class are compared and the crystal class identification accuracy of fingerprinting model is reported

@Achintha_Ihalage
@29_Jun_2020

'''
# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from autoencoder import AutoEncoder
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from matplotlib import rcParams
from sklearn.metrics import classification_report

path = str(pathlib.Path(__file__).parent.absolute())
exp_data_file = path+'/ICSD_data/ICSD_all_data.csv'
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'

class CrystalSystem():

	def __init__(self):
		self.exp_df = pd.read_csv(exp_data_file, sep='\t')
		self.ae = AutoEncoder()
		self.VAE = self.ae.build_AE(vae=True)
		self.VAE.load_weights(path+'/saved_models/best_model_VAE.h5')

	def validate(self):
		def get_similar_compounds(row):
			most_similar_df = self.ae.most_similar(self.VAE, row.StructuredFormula, n=6, experimental=1, vae=True)	# get 6 most similar crystal systems, 1st would refer to the material itself. Discard it and get the rest
			real_crystal_class = row.CrystalClass	# crystal system
			real_space_group = row.HMS	# space group
			similar_crystal_systems = most_similar_df['CrystalSystem'].to_list()[1:]		
			similar_space_groups = most_similar_df['HMS'].to_list()[1:]
			most_voted_class = max(similar_crystal_systems, key=similar_crystal_systems.count)	# if 2 crystal systems have 2 votes each, the one with the lowest euclidean distance is selected
			most_voted_space_group = max(similar_space_groups, key=similar_space_groups.count)
			if real_crystal_class == most_voted_class:	
				correctly_identified_cc = 1
			else:
				correctly_identified_cc = 0
			if real_space_group == most_voted_space_group:
				correctly_identified_spg = 1
			else:
				correctly_identified_spg = 0

			print (row.StructuredFormula, correctly_identified_cc)

			return most_similar_df['CollectionCode'].to_list()[1:], most_similar_df['StructuredFormula'].to_list()[1:],\
					most_similar_df['CrystalSystem'].to_list()[1:], most_voted_class, most_voted_space_group, most_similar_df['Euclidean Distance'].to_list()[1:], correctly_identified_cc, correctly_identified_spg

		self.exp_df['Most Similar ICSD IDs'], self.exp_df['Most Similar Compounds'], self.exp_df['Most Similar Crystal Systems'], self.exp_df['predicted_crystal_system'], self.exp_df['predicted_space_group'],\
			self.exp_df['Euclidean Distances'], self.exp_df['crystal_system_correctly_identified'], self.exp_df['space_group_correctly_identified'] = zip(*self.exp_df.apply(get_similar_compounds, axis=1))
		
		corrects_cc =  self.exp_df['crystal_system_correctly_identified'].sum()
		correct_percentage_cc = corrects_cc*100.0/self.exp_df[self.exp_df['CrystalClass'].notna()].shape[0]
		corrects_spg = self.exp_df['space_group_correctly_identified'].sum()
		correct_percentage_spg = corrects_spg*100.0/self.exp_df[self.exp_df['HMS'].notna()].shape[0]
		print ('%.2f%% of crystal systems have been correctly identified by the VAE fingerprinting model'%(correct_percentage_cc))
		print ('%.2f%% of space groups have been correctly identified by the VAE fingerprinting model'%(correct_percentage_spg))
		self.exp_df[['StructuredFormula', 'HMS', 'CrystalClass', 'Most Similar ICSD IDs', 'Most Similar Compounds', 'Most Similar Crystal Systems', 'predicted_space_group','predicted_crystal_system', 'crystal_system_correctly_identified', 'space_group_correctly_identified']].to_csv(path+'/fingerprint_validation.csv', sep='\t', index=False)
		return self.exp_df[['StructuredFormula', 'HMS', 'CrystalClass', 'Most Similar ICSD IDs', 'Most Similar Compounds', 'Most Similar Crystal Systems', 'predicted_space_group','predicted_crystal_system', 'crystal_system_correctly_identified', 'space_group_correctly_identified']]

	

	def get_confusion_matrix(self):		# confusion matrix for crystal class classification
		system_val = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3, 'tetragonal': 4, 'cubic': 5, 'trigonal': 6, 'hexagonal': 7}
		validate_df = pd.read_csv(path+'/fingerprint_validation.csv', sep='\t')
		true_lbl_str = validate_df['CrystalClass'].tolist()
		predicted_lbl_str = validate_df['predicted_crystal_system']
		cm = np.zeros((7, 7))	# empty confusion matrix
		for t, p in zip(true_lbl_str, predicted_lbl_str):
			if not pd.isnull(t):
				print (t, p)
				cm[system_val[p]-1][system_val[t]-1] +=1
		print(cm)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		title = 'Confusion Matrix'
		classes = np.array(['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Cubic', 'Trigonal', 'Hexagonal'])

				# classification report
		y_true = []
		y_pred = []
		for t, p in zip(true_lbl_str, predicted_lbl_str):
			if not pd.isnull(t):
				y_true.append(system_val[t]-1)
				y_pred.append(system_val[p]-1)

		print(classification_report(y_true, y_pred, target_names=classes, digits=4))

		fig, ax = plt.subplots()
		im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
		ax.figure.colorbar(im, ax=ax)
		# We want to show all ticks...
		ax.set(xticks=np.arange(cm.shape[1]),
		       yticks=np.arange(cm.shape[0]),
		       # ... and label them with the respective list entries
		       xticklabels=classes, yticklabels=classes,
		       #title=title,
		       ylabel='True label',
		       xlabel='Predicted label')

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12,
		         rotation_mode="anchor")
		plt.setp(ax.get_yticklabels(), fontsize=12)
		ax.set_xlabel('Predicted label', fontsize=14)
		ax.set_ylabel('True label', fontsize=14)
		normalize=True
		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
		    for j in range(cm.shape[1]):
		        ax.text(j, i, format(cm[i, j], fmt),
		                ha="center", va="center",
		                color="white" if cm[i, j] > thresh else "black")
		fig.tight_layout()
		fig.savefig(path+'/figures/fingerprint_conf_mat.png', dpi=800, bbox_inches='tight')
		plt.show()

	def get_spg_conf_mat(self): # confusion matrix for space group classification
		validate_df = pd.read_csv(path+'/fingerprint_validation.csv', sep='\t')
		true_lbl_str = validate_df['HMS'].tolist()
		predicted_lbl_str = validate_df['predicted_space_group'].tolist()
		unique_spgs = list(sorted(set(true_lbl_str + predicted_lbl_str)))
		dim = len(unique_spgs)
		spg_dict = {k:v for (v,k) in enumerate(unique_spgs)}

		cm = np.zeros((dim, dim))	# empty confusion matrix
		for t, p in zip(true_lbl_str, predicted_lbl_str):
			if not pd.isnull(t):
				# print (t, p)
				cm[spg_dict[p]-1][spg_dict[t]-1] +=1
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		title = 'Confusion Matrix'
		classes = np.array(unique_spgs)

		# classification report
		y_true = []
		y_pred = []
		for t, p in zip(true_lbl_str, predicted_lbl_str):
			if not pd.isnull(t):
				y_true.append(spg_dict[t]-1)
				y_pred.append(spg_dict[p]-1)

		print(classification_report(y_true, y_pred, target_names=np.array(unique_spgs), digits=4))



		fig, ax = plt.subplots()
		im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
		ax.figure.colorbar(im, ax=ax)
		# We want to show all ticks...
		ax.set(xticks=np.arange(cm.shape[1]),
		       yticks=np.arange(cm.shape[0]),
		       # ... and label them with the respective list entries
		       xticklabels=classes, yticklabels=classes,
		       #title=title,
		       ylabel='True label',
		       xlabel='Predicted label')

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=4,
		         rotation_mode="anchor")
		plt.setp(ax.get_yticklabels(), fontsize=4)
		ax.set_xlabel('Predicted label')
		ax.set_ylabel('True label')
		normalize=True
		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.

		fig.tight_layout()
		fig.savefig(path+'/figures/fingerprint_spg_conf_mat.png', dpi=800, bbox_inches='tight')
		plt.show()
