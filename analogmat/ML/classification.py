'''
ML models
@Achintha_Ihalage
'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
from matplotlib import pyplot as plt
from matplotlib import rcParams, colors
import seaborn as sns
import pandas as pd
import math
import numpy as np
from tqdm import tqdm # progress bar
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
# import seaborn as sns
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

np.random.seed(1)	# for reproducability

path = str(pathlib.Path(__file__).parent.absolute().parent)
perovskite_file = path+'/ICSD_data/perovskites.csv'
non_perovskite_file = path+'/ICSD_data/non_perovskites.csv'
hypo_compounds_file = path+'/ICSD_data/all_generated_compounds.csv'
clf_results_file = path+'/ML/classification_results.csv'
clf_new_comps_file = path+'/ML/new_perovskite_candidates.csv'


# set tick width
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

class PVClassifier():

	def __init__(self):
		self.perovskites = self.read_csv(perovskite_file)
		self.non_perovskites = self.read_csv(non_perovskite_file)
		self.perovskite_ft = self.perovskites.drop(['CollectionCode', 'HMS', 'SpaceGroupNum', 'CrystalClass', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight','Temperature', 
								'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
								'O_frac', 'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O',
								'Row_O', 'Group_O', 'nO', 'rO'], axis=1)
								#'SISSO_1', 'SISSO_2', 'SISSO_3', 'SISSO_4', 'SISSO_5', 'SISSO_6', 'SISSO_7', 'SISSO_8', 'SISSO_9', 'SISSO_10'], axis=1)
		self.perovskite_ft.drop_duplicates(keep='first', inplace=True)	# remove if duplicates occur after dropping lattice parameters etc.
		self.non_perovskites_ft = self.non_perovskites.drop(['CollectionCode', 'HMS', 'SpaceGroupNum', 'CrystalClass', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight','Temperature', 
								'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
								'O_frac', 'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O',
								'Row_O', 'Group_O', 'nO', 'rO'], axis=1)
								#'SISSO_1', 'SISSO_2', 'SISSO_3', 'SISSO_4', 'SISSO_5', 'SISSO_6', 'SISSO_7', 'SISSO_8', 'SISSO_9', 'SISSO_10'], axis=1)
		self.non_perovskites_ft.drop_duplicates(keep='first', inplace=True)
		self.scaler = StandardScaler()

	def read_csv(self, file):
		return pd.read_csv(file, sep='\t')

	def train_and_test(self, algo='gradient_boosting', no_iterations=100):
		test_scores_arr = []	# array to store the scores of each classifier
		cross_val_mean = []	# cross validation mean accuracies of each classifier
		cross_val_std = []	# cross validation std of each classifier 
		
		for i in range(no_iterations):	# iterate N times to avoid the effect of random/unbalanced data splitting
			perovskite_sample = self.perovskite_ft.sample(250, random_state=i)	# select N no of samples to avoid unbalanced data problem
			non_perovskite_sample = self.non_perovskites_ft
			df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))	# join 2 dfs and shuffle to have a good data distribution
			df_x = df.drop(['is_perovskite'], axis=1)
			df_x = self.scaler.fit_transform(df_x)	# scale x features (optional) // very similar mean accuracy (0.9371) is obtained even without scaling. No data leaking
			df_y = df[['is_perovskite']]


			X_train, X_test, y_train, y_test = train_test_split(df_x, df_y.values.ravel(),test_size=0.2, random_state=i)
			if algo=='gradient_boosting':
				clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)
			elif algo=='random_forest':
				clf = RandomForestClassifier(max_depth=3, random_state=10)
			elif algo=='decision_tree':
				clf = DecisionTreeClassifier(random_state=10)
			elif algo=='svm':
				clf = svm.SVC()
			else:
				print('The selected algorithm is not available. Using Gradient Boosting Classifier ...')
				clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)

			clf.fit(X_train, y_train)

			test_scores_arr.append(clf.score(X_test, y_test))

			# 10-fold cross validation
			cv_scores = cross_val_score(clf, df_x, df_y.values.ravel(), cv=10)
			cross_val_mean.append(np.mean(cv_scores))
			cross_val_std.append(np.std(cv_scores))
			print('Iteration Number: %d/%d'%(i+1, no_iterations))
			print('Current Test Accuracy: %.4f' %(clf.score(X_test, y_test)))
			print('Current 10-Fold CV accuracy: %.4f ± (%.4f)\n ' %(np.mean(cv_scores), np.std(cv_scores)))

		print('Final Mean Test Accuracy: %.4f ± (%f)' %(np.mean(test_scores_arr), np.std(test_scores_arr)))
		print('Final Mean 10-Fold CV Accuracy: %.4f ± (%.4f)' %(np.mean(cross_val_mean), np.std(cross_val_mean)))

	def tune_hypeparameters(self, algo='gradient_boosting', no_iterations=100):

		params = {'learning_rate':[0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.15], 'n_estimators':[100, 250, 400, 500, 1000]}
		mean_test_scores = []
		for i in range(no_iterations):
			perovskite_sample = self.perovskite_ft.sample(250, random_state=np.random.randint(100))
			non_perovskite_sample = self.non_perovskites_ft	
			df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))
			df_x = df.drop(['is_perovskite'], axis=1)
			df_x = self.scaler.fit_transform(df_x)	# scale x features
			df_y = df[['is_perovskite']]

			gridcv = GridSearchCV(estimator = GradientBoostingClassifier(random_state=10), param_grid = params, scoring='accuracy', n_jobs=4, iid=False, cv=10)
			X_train, X_test, y_train, y_test = train_test_split(df_x, df_y.values.ravel(),test_size=0.01, random_state=i)	# literally no test set needed here
			gridcv.fit(X_train,y_train)
			mean_test_scores.append(gridcv.cv_results_['mean_test_score'])
			print(gridcv.cv_results_['mean_test_score'], gridcv.best_params_, gridcv.best_score_)
		mean_accuracies = np.mean(np.array(mean_test_scores), axis=0)
		best_model_idx = np.argmax(mean_accuracies)
		l_rate_n_estimators = list(itertools.product(params['learning_rate'], params['n_estimators']))
		best_model_tuple = l_rate_n_estimators[best_model_idx]
		best_lrate = best_model_tuple[0]
		best_nestimators = best_model_tuple[1]
		print('\nBest Hyperparameters: ',{'learning_rate': best_lrate, 'n_estimators': best_nestimators})
		print('Best Mean Accuracy: ',max(mean_accuracies))

	def feature_importance(self, no_iterations=10):
		f_importance_list = []
		for i in range(no_iterations):
			perovskite_sample = self.perovskite_ft.sample(250, random_state=np.random.randint(100))	# select N no of samples to avoid unbalanced data problem
			non_perovskite_sample = self.non_perovskites_ft		
			df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))	# join 2 dfs and shuffle to have a good data distribution
			df_x = df.drop(['is_perovskite'], axis=1)
			df_y = df[['is_perovskite']]

			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)
			clf.fit(df_x, df_y.values.ravel())

			importances = clf.feature_importances_
			feature_importance = {k:v for k,v in zip(df_x.columns, importances)}
			sorted_fimportance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
			f_importance_list.append(feature_importance)


		imp_list = np.array(list(zip(*[f.values() for f in f_importance_list])))
		f_list = list(df_x.columns)
		mean_importances = np.mean(imp_list, axis=1)
		std_importances = np.std(imp_list, axis=1)
		f_mean_imp = {k:v for k,v in zip(f_list, mean_importances)}
		sorted_f_mean_imp = {k: v for k, v in sorted(f_mean_imp.items(), key=lambda item: item[1], reverse=True)}
		f_mean_std = {k:v for k,v in zip(f_list, std_importances)}
		sorted_f_mean_std = {**sorted_f_mean_imp, **f_mean_std}

		rcParams["errorbar.capsize"] = 2.5
		fig, ax = plt.subplots()
		ax.margins(0,0)
		ax.barh(np.arange(len(list(sorted_f_mean_imp.keys())[:20])), np.array(list(sorted_f_mean_imp.values())[:20])*100, xerr=np.array(list(sorted_f_mean_std.values())[:20])*100, align='center')
		ax.set_yticks(np.arange(len(list(sorted_f_mean_imp.keys())[:20])))
		ax.set_yticklabels(list(sorted_f_mean_imp.keys())[:20])
		ax.invert_yaxis()  # labels read top-to-bottom
		ax.set_xlabel('Feature importance (%)')
		plt.tight_layout()
		plt.show()



	def plot_confusion_matrix(self, algo='gradient_boosting'):
		perovskite_sample = self.perovskite_ft.sample(250, random_state=10)
		non_perovskite_sample = self.non_perovskites_ft		
		df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))
		df_x = df.drop(['is_perovskite'], axis=1)
		df_x = self.scaler.fit_transform(df_x)	# scale x features
		df_y = df[['is_perovskite']]

		if algo=='gradient_boosting':
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)
		elif algo=='random_forest':
			clf = RandomForestClassifier(max_depth=2, random_state=10)
		elif algo=='decision_tree':
			clf = DecisionTreeClassifier(random_state=10)
		elif algo=='svm':
			clf = svm.SVC()
		else:
			print('The selected algorithm is not available. Using Gradient Boosting Classifier ...')
			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)

		# clf.fit(X_train, y_train)
		# plot_confusion_matrix(clf, X_test, y_test, display_labels=['Non-perovskites', 'Perovskites'], cmap='Blues')


		X = np.array(df_x)
		y = df_y.values.ravel()


		y_pred = cross_val_predict(clf, df_x, df_y.values.ravel(), cv=10)
		cm = confusion_matrix(np.array(df_y), y_pred)

		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		title = 'Confusion Matrix'
		classes = np.array(['Non-perovskite', 'Perovskite'])

		# find precision, recall and F1 score
		precision = 1.0*np.diag(cm) / np.sum(cm, axis = 0)
		recall = 1.0*np.diag(cm) / np.sum(cm, axis = 1)
		f1_score = 2.0*((precision*recall)/(precision+recall))

		avg_recall = np.mean(recall)
		avg_precision = np.mean(precision)
		avg_f1_score = np.mean(f1_score)

		print("Average Precision: %.2f (+/- %.4f)" % (np.mean(precision), np.std(precision)))
		print("Average Recall: %.2f (+/- %.4f)" % (np.mean(recall), np.std(recall)))
		print("Average F1 Score: %.2f (+/- %.4f)" % (np.mean(f1_score), np.std(f1_score)))

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
		fmt = '.3f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i in range(cm.shape[0]):
		    for j in range(cm.shape[1]):
		        ax.text(j, i, format(cm[i, j], fmt),
		                ha="center", va="center", fontsize=15,
		                color="white" if cm[i, j] > thresh else "black")
		fig.tight_layout()
		fig.savefig(path+'/figures/'+algo+'_conf_mat.png', dpi=800, bbox_inches='tight')

		plt.show()

	def plot_roc_curve(self):
		perovskite_sample = self.perovskite_ft.sample(250, random_state=9)	# select 250 samples to avoid unbalanced data problem
		non_perovskite_sample = self.non_perovskites_ft		
		df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))	# join 2 dfs and shuffle to have a good data distribution
		df_x = df.drop(['is_perovskite'], axis=1)
		df_x = self.scaler.fit_transform(df_x)	
		df_y = df[['is_perovskite']]

		X = np.array(df_x)
		y = df_y.values.ravel()

		# ROC with cross validation
		# #############################################################################
		# Classification and ROC analysis

		# Run classifier with cross-validation and plot ROC curves
		cv = StratifiedKFold(n_splits=10)
		classifier = GradientBoostingClassifier(learning_rate=0.1, n_estimators=250, random_state=10)

		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)

		fig, ax = plt.subplots()
		for i, (train, test) in enumerate(cv.split(X, y)):
		    classifier.fit(X[train], y[train])
		    viz = plot_roc_curve(classifier, X[test], y[test],
		                         name='ROC fold {}'.format(i+1),
		                         alpha=0.8, lw=2, ax=ax)
		    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
		    interp_tpr[0] = 0.0
		    tprs.append(interp_tpr)
		    aucs.append(viz.roc_auc)

		ax.plot([0, 1], [0, 1], linestyle='--', lw=2.5, color='r', alpha=1)

		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		ax.plot(mean_fpr, mean_tpr, color='b',

		         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		        lw=2.5, alpha=1)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.3)

		ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
		ax.set_xlabel('False positive rate', fontsize=15)
		ax.set_ylabel('True positive rate', fontsize=15)
		ax.legend(loc="lower right", fontsize=4, prop={'weight':'bold', 'size':10})
		ax.minorticks_on()
		plt.tight_layout()
		fig.savefig(path+'/figures/roc_curve.png', dpi=800)
		plt.show()



	def get_tolerance_factors(self):
		def GoldschmidtTF(row):
			t = (row.rA_avg + row.rO)/(math.sqrt(2)*(row.rB_avg + row.rO))
			return t

		def TauFactor(row):
			t = (row.rO/row.rB_avg) - row.nA*(row.nA - (row.rA_avg/row.rB_avg)/math.log(row.rA_avg/row.rB_avg))
			return t

		# calculate Goldschmidt Tolerance Factor for experimental perovskites and non-perovskites
		self.perovskites['GoldschmidtTF'] = self.perovskites.apply(GoldschmidtTF, axis=1)
		self.non_perovskites['GoldschmidtTF'] = self.non_perovskites.apply(GoldschmidtTF, axis=1)

		# calculate Tau Factor as proposed in Science Advances paper (DOI: 10.1126/sciadv.aav0693)
		self.perovskites['TauFactor'] = self.perovskites.apply(TauFactor, axis=1)
		self.non_perovskites['TauFactor'] = self.non_perovskites.apply(TauFactor, axis=1)

		return self.perovskites[['StructuredFormula', 'GoldschmidtTF', 'TauFactor']], self.non_perovskites[['StructuredFormula', 'GoldschmidtTF', 'TauFactor']]

	def evaluate_tolerance_factors(self):
		df_pv, df_npv = self.get_tolerance_factors()
		pv_GTF_accuracy = df_pv[(df_pv.GoldschmidtTF >= 0.85) & (df_pv.GoldschmidtTF<=1.1)].shape[0]*100.0/df_pv.shape[0]
		npv_GTF_accuracy = df_npv[(df_npv.GoldschmidtTF < 0.85) | (df_npv.GoldschmidtTF>1.1)].shape[0]*100.0/df_npv.shape[0]
		pv_TauF_accuracy = df_pv[df_pv.TauFactor <= 4.18].shape[0]*100.0/df_pv.shape[0]
		npv_TauF_accuracy = df_npv[df_npv.TauFactor > 4.18].shape[0]*100.0/df_npv.shape[0]

		print ('Goldschmidt TF accuracy for perovskites: %.4f' %(pv_GTF_accuracy))
		print ('Goldschmidt TF accuracy for non-perovskites: %.4f' %(npv_GTF_accuracy))

		print ('Tau Factor accuracy for perovskites: %.4f' %(pv_TauF_accuracy))
		print ('Tau Factor accuracy for non-perovskites: %.4f' %(npv_TauF_accuracy))

	def get_perovskite_candidates(self, prob_threshold=0.98, no_iterations=100):	# get the compounds that were classified 98 or more times out of 100 as perovskites (Similar to this Nat. Comms paper, https://doi.org/10.1038/s41467-018-03821-9)
		total_hypothetical_compounds = self.read_csv(hypo_compounds_file)		
		hypothetical_compounds = total_hypothetical_compounds.dropna()	 # drop rows with Nan, mendeleev does not include polarizability and lattice constant of some elements
		hypo_df = hypothetical_compounds.drop(['StructuredFormula', 'A1', 'A2', 'B1', 'B2', 'O',
												'O_frac', 'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O',
												'Row_O', 'Group_O', 'nO', 'rO'], axis=1)
		# print (hypo_df.info())
		hypo_df_results = hypothetical_compounds[['StructuredFormula', 'Goldschmidt_TF', 'rB/rO']]	# new database to store classification results
		hypo_df_results['PV_classifications_out_of_100'] = 0
		hypo_df_results['is_predicted_perovskite'] = 0
		clf_prob = []	# to tract classification probability
		for i in tqdm(range(no_iterations)):
			perovskite_sample = self.perovskite_ft.sample(250, random_state=i)	# select 300 samples to avoid unbalanced data problem
			non_perovskite_sample = self.non_perovskites_ft		# use full non_perovskite database (283 samples), because these are limited
			df = shuffle(pd.concat([perovskite_sample, non_perovskite_sample], ignore_index=True))	# join 2 dfs and shuffle to have a good data distribution
			df_x = df.drop(['is_perovskite'], axis=1)
			# df_x = self.scaler.fit_transform(df_x)	# optional
			df_y = df[['is_perovskite']]

			clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=10)
			clf.fit(df_x, df_y.values.ravel())	
			y_pred = clf.predict(hypo_df)
			y_pred_proba = clf.predict_proba(hypo_df)[:,1]
			clf_prob.append(y_pred_proba)
			hypo_df_results['PV_classifications_out_of_100']+=y_pred	

		hypo_df_results.loc[hypo_df_results.PV_classifications_out_of_100 == no_iterations, 'is_predicted_perovskite'] = 1		# compounds that were classified 100% as perovskites
		mean_clf_probs = np.mean(np.array(clf_prob), axis=0)	# mean classification probability of each hypothetical compound
		hypo_df_results['Mean_classification_prob'] = mean_clf_probs
		new_predicted_perovskites = hypo_df_results[(hypo_df_results['is_predicted_perovskite']==1) & (hypo_df_results['Mean_classification_prob']>prob_threshold)]
		print('\n##################### Classification Results ###############################')
		print ('\n%d new perovskite candidates were found out of %d hypothetical compounds!'%(new_predicted_perovskites.shape[0], total_hypothetical_compounds.shape[0]))
		print ('%.2f %% of total compounds were discarded!\n'%((total_hypothetical_compounds.shape[0]-new_predicted_perovskites.shape[0])*100.0/total_hypothetical_compounds.shape[0]))
		print('############################################################################')
		hypo_df_results.to_csv(clf_results_file, sep='\t', index=False)
		new_predicted_perovskites.to_csv(clf_new_comps_file, sep='\t', index=False)
