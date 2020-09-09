'''

This script implements a gradient boosing classification model to identify the crystal system of materials
Other algorithms such as KNN, random forest, decision tree and SVM are implemented for comparison
Specifically, it is interesting to investigate how KNN (N=5) performs in comparison to the VAE fingerprinting model when identifying crystal system of compounds

@Achintha_Ihalage
@03_Jul_2020
'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams

np.random.seed(1)	# for reproducability

path = str(pathlib.Path(__file__).parent.absolute().parent)
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'

class StructureClf():

	def __init__(self):
		self.exp_df = pd.read_csv(path+'/ICSD_data/ICSD_all_data.csv', sep='\t')
		self.exp_df = self.exp_df[self.exp_df['CrystalClass'].notna()]	# get rows that have non-null crystal system
		self.df = self.exp_df.drop(['CollectionCode', 'HMS', 'SpaceGroupNum', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight','Temperature', 
								'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
								'O_frac', 'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O',
								'Row_O', 'Group_O', 'nO', 'rO'], axis=1)
		self.scaler = StandardScaler()

	def crystal_system_clf(self, algo='gradient_boosting'):

		def system_to_num(row):	# convert categorical crystal system name to a numerical value
			system_val = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3, 'tetragonal': 4, 'cubic': 5, 'trigonal': 6, 'hexagonal': 7}
			return system_val[row.CrystalClass]

		df = self.df.copy()
		df['CrystalSystemNum'] = df.apply(system_to_num, axis=1)
		df_x = df.drop(['CrystalSystemNum', 'CrystalClass'], axis=1)
		df_x = self.scaler.fit_transform(df_x)
		df_y = df[['CrystalSystemNum']]

		X_train, X_test, y_train, y_test = train_test_split(df_x, df_y.values.ravel(), test_size=0.2, random_state=10)

		if algo=='gradient_boosting':
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
		elif algo=='knn':
			clf = KNeighborsClassifier(n_neighbors=5)
		elif algo=='random_forest':
			clf = RandomForestClassifier(max_depth=3, random_state=10)
		elif algo=='decision_tree':
			clf = DecisionTreeClassifier(random_state=10)
		elif algo=='svm':
			clf = svm.SVC()
		else:
			print('The selected algorithm is not available. Using Gradient Boosting Classifier ...')
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
		# print(np.all(np.isfinite(y_train)))
		# print(max(y_train))
		# clf.fit(X_train, y_train)

		cv_scores = cross_val_score(clf, df_x, df_y.values.ravel(), cv=10)

		print ('Mean accuracy: %.3f +/- (%.3f)'%(np.mean(cv_scores), np.std(cv_scores)))

	def cross_val_conf_mat(self, algo='gradient_boosting'):
		def system_to_num(row):	# convert categorical crystal system name to a numerical value
			system_val = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3, 'tetragonal': 4, 'cubic': 5, 'trigonal': 6, 'hexagonal': 7}
			return system_val[row.CrystalClass]

		df = self.df.copy()
		df['CrystalSystemNum'] = df.apply(system_to_num, axis=1)
		df_x = df.drop(['CrystalSystemNum', 'CrystalClass'], axis=1)
		df_x = self.scaler.fit_transform(df_x)
		df_y = df[['CrystalSystemNum']]

		if algo=='gradient_boosting':
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
		elif algo=='knn':
			clf = KNeighborsClassifier(n_neighbors=5)
		elif algo=='random_forest':
			clf = RandomForestClassifier(max_depth=3, random_state=10)
		elif algo=='decision_tree':
			clf = DecisionTreeClassifier(random_state=10)
		elif algo=='svm':
			clf = svm.SVC()
		else:
			print('The selected algorithm is not available. Using Gradient Boosting Classifier ...')
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
		# print(np.all(np.isfinite(y_train)))
		# print(max(y_train))
		# clf.fit(X_train, y_train)

		y_pred = cross_val_predict(clf, df_x, df_y.values.ravel(), cv=10)
		cm = confusion_matrix(np.array(df_y), y_pred)
		# print(cm)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		title = 'Confusion Matrix'
		classes = np.array(['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Cubic', 'Trigonal', 'Hexagonal'])

		# find precision, recall and F1 score
		precision = 1.0*np.diag(cm) / np.sum(cm, axis = 0)
		recall = 1.0*np.diag(cm) / np.sum(cm, axis = 1)
		f1_score = 2.0*((precision*recall)/(precision+recall))

		avg_recall = np.mean(recall)
		avg_precision = np.mean(precision)
		avg_f1_score = np.mean(f1_score)

		print("Average Precision: %.2f (+/- %.2f)" % (np.mean(precision), np.std(precision)))
		print("Average Recall: %.2f (+/- %.2f)" % (np.mean(recall), np.std(recall)))
		print("Average F1 Score: %.2f (+/- %.2f)" % (np.mean(f1_score), np.std(f1_score)))

		print ('Average Precision: ', avg_precision)
		print ('Average Recall: ', avg_recall)
		print ('Average F1 Score: ', avg_f1_score)

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
		fig.savefig(path+'/figures/fingerprint_conf_mat_'+algo+'.png', dpi=800, bbox_inches='tight')
		plt.show()

	def spg_clf(self, algo='gradient_boosting'):
		df = pd.read_csv(path+'/ICSD_data/ICSD_all_data.csv', sep='\t')
		df = df[df['HMS'].notna()]
		df = df.drop(['CollectionCode', 'CrystalClass', 'SpaceGroupNum', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight','Temperature', 
								'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
								'O_frac', 'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O',
								'Row_O', 'Group_O', 'nO', 'rO'], axis=1)

		unique_spgs = sorted(set(df['HMS'].tolist()))
		print(unique_spgs)
		spg_dict = {k:v for (v,k) in enumerate(unique_spgs)}

		def spg_to_num(row):
			return spg_dict[row.HMS]


		df['HMS_label'] = df.apply(spg_to_num, axis=1)
		df_x = df.drop(['HMS_label', 'HMS'], axis=1)
		df_x = self.scaler.fit_transform(df_x)
		df_y = df[['HMS_label']]

		X_train, X_test, y_train, y_test = train_test_split(df_x, df_y.values.ravel(), test_size=0.2, random_state=10)

		if algo=='gradient_boosting':
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
		elif algo=='knn':
			clf = KNeighborsClassifier(n_neighbors=5)
		elif algo=='random_forest':
			clf = RandomForestClassifier(max_depth=3, random_state=10)
		elif algo=='decision_tree':
			clf = DecisionTreeClassifier(random_state=10)
		elif algo=='svm':
			clf = svm.SVC()
		else:
			print('The selected algorithm is not available. Using Gradient Boosting Classifier ...')
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)

		cv_scores = cross_val_score(clf, df_x, df_y.values.ravel(), cv=10)

		print ('Mean accuracy: %.3f +/- (%.3f)'%(np.mean(cv_scores), np.std(cv_scores)))
