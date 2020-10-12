'''
Predict possible phase transitions and bandgaps of (AA')BO3 A(BB')O3 compositions
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
import re
import sys
sys.path.append("..")
from autoencoder import AutoEncoder
from arrange_ICSD_data import Perovskites
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib
import matplotlib.patches as patches
import numpy.polynomial.polynomial as poly
from scipy.interpolate import make_interp_spline, BSpline
import yaml
import matplotlib.patches as mpatches


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

class BandgapPhase():
	def __init__(self):
		self.edft = pd.read_csv(path+'/ML/data/pero_dft.csv', sep='\t')
		self.comps_wdup = pd.read_csv(path+'/ICSD_data/ICSD_all_data_with_all_phases.csv', sep='\t')	# compositions with duplicates for phase prediction
		self.ae = AutoEncoder()
		self.VAE = self.ae.build_AE(vae=True)
		self.VAE.load_weights(path+'/saved_models/best_model_VAE.h5')
		self.scaler = StandardScaler()


	def arrange_comp(self):	# arrange chemical formula in (AA')BO3 and A(BB')O3 order
		def arrange_formula(row):
			formula_lst = re.findall('[A-Z][^A-Z]*', row.SimulatedComposition.strip())
			B_doped = 0
			if len(formula_lst) == 4:
				r = re.compile("([a-zA-Z]+)([0-9]+)")
				m = r.match(formula_lst[0])
				elem1 = m.group(1)
				frac1 = m.group(2)
				if int(frac1)==8:
					B_doped = 1
				if B_doped==1:	# A(BB')O3 type
					A1 = elem1
					A1_frac = ''
					B1 = r.match(formula_lst[1]).group(1)
					B1_frac = str(int(r.match(formula_lst[1]).group(2))/8.0)
					B2 = r.match(formula_lst[2]).group(1)
					B2_frac = str(int(r.match(formula_lst[2]).group(2))/8.0)
					StructuredForm = "{}({}{}){}".format(A1,B1+B1_frac,B2+B2_frac,'O3')
				else:	# (AA')BO3 type
					A1 = elem1
					A1_frac = str(int(frac1)/8.0)
					A2 = r.match(formula_lst[1]).group(1)
					A2_frac = str(int(r.match(formula_lst[1]).group(2))/8.0)
					B1 = r.match(formula_lst[2]).group(1)
					B1_frac = ''
					StructuredForm = "({}{}){}{}".format(A1+A1_frac,A2+A2_frac,B1,'O3')
				return StructuredForm
			else:
				return 0



		self.edft['StructuredFormula'] = self.edft.apply(arrange_formula, axis=1)
		df = self.edft[self.edft.StructuredFormula!=0]

		def get_fprint(row):	# getting material fingerprint for each composition
			print(row.StructuredFormula)
			try:
				fprint = self.ae.get_fingerprint(self.VAE, row.StructuredFormula)
				print (fprint)
				return fprint[0], fprint[1]
			except:
				return None, None
			return 

		df['Fingerprint_x'], df['Fingerprint_y'] = zip(*df.apply(get_fprint, axis=1))
		df = df.dropna()
		df.to_pickle(path+'/ML/data/processed_dft_data.pkl')
		print(df.info())

	def get_all_features(self):
		df = pd.read_pickle(path+'/ML/data/processed_dft_data.pkl')
		pv = Perovskites(df)
		df = pv.parse_formula(df)
		df = pv.add_features(df)
		df = pv.SISSO_features(df)
		df = df.drop(['SimulatedComposition', 'O p-band center (eV)', 'Predicted Log k* (cm/s)', 'Charge transfer gap (eV)', 'Formation energy', 'Fingerprint_x', 'Fingerprint_y'], axis=1)
		df.to_csv(path+'/ML/data/dft_data_with_features.csv', sep='\t', index=False)

	def bgap_pred_all_features(slef):
		df = pd.read_csv(path+'/ML/data/dft_data_with_features.csv', sep='\t')
		df = df.drop(['StructuredFormula', 'A1',	'A1_frac',	'A2',	'A2_frac',	'B1',	'B1_frac',	'B2',	'B2_frac',	'O',	'O_frac',
			'atom_numO', 'mend_numO', 'atomic_rO', 'O_X', 'M_O', 'V_O', 'therm_con_O', 'polarizability_O', 'lattice_const_O', 'Row_O', 'Group_O', 'nO', 'rO'], axis=1)
		df_x = df.drop(['Ehull', 'Bandgap'], axis=1)
		df_y = df[['Bandgap']]
		algo_dict_mse = {'SVR':[], 'PLS':[], 'EN':[], 'KNN':[], 'RAND':[], 'GBR':[]}
		algo_dict_mae = {'SVR':[], 'PLS':[], 'EN':[], 'KNN':[], 'RAND':[], 'GBR':[]}
		for i in range(20):
			X_train, X_test, y_train, y_test = train_test_split(df_x, df_y.values.ravel(),test_size=0.2, random_state=i)
			pipelines = []
			pipelines.append(('SVR', Pipeline([('Scaler', StandardScaler()), ('SVR',SVR())]))) #
			pipelines.append(('PLS', Pipeline([('Scaler', StandardScaler()), ('PLS', PLSRegression())])))
			pipelines.append(('EN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
			pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
			pipelines.append(('RAND', Pipeline([('Scaler', StandardScaler()),('RAND', RandomForestRegressor())])))
			pipelines.append(('GBR', Pipeline([('Scaler', StandardScaler()),('GBR', GradientBoostingRegressor())])))

			results = []
			names = []
			for name, model in pipelines:
			    kfold = KFold(n_splits=10, random_state=10)
			    cv_results_mse = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
			    cv_results_mae = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
			    # results.append(cv_results)
			    # names.append(name)
			    msg_mse = "%s: MSE %f (%f)" % (name, cv_results_mse.mean(), cv_results_mse.std())
			    msg_mae = "%s: MAE %f (%f)" % (name, cv_results_mae.mean(), cv_results_mae.std())
			    print(msg_mse)
			    print(msg_mae)
			    algo_dict_mse[name].append(np.sqrt(-1*cv_results_mse.mean()))
			    algo_dict_mae[name].append(-1*cv_results_mae.mean())
			print ('\n')
		print('SVR 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['SVR']).mean(), np.array(algo_dict_mae['SVR']).mean(), np.array(algo_dict_mae['SVR']).std()))
		print('PLS 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['PLS']).mean(), np.array(algo_dict_mae['PLS']).mean(), np.array(algo_dict_mae['PLS']).std()))
		print('EN 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['EN']).mean(), np.array(algo_dict_mae['EN']).mean(), np.array(algo_dict_mae['EN']).std()))
		print('KNN 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['KNN']).mean(), np.array(algo_dict_mae['KNN']).mean(), np.array(algo_dict_mae['KNN']).std()))
		print('RAND 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['RAND']).mean(), np.array(algo_dict_mae['RAND']).mean(), np.array(algo_dict_mae['RAND']).std()))
		print('GBR 10-fold CV RMSE: %.3f  MAE: %.3f (%.3f)'%(np.array(algo_dict_mse['GBR']).mean(), np.array(algo_dict_mae['GBR']).mean(), np.array(algo_dict_mae['GBR']).std()))


	def plot_bandgap_fprints(self):
		df = pd.read_pickle(path+'/ML/data/processed_dft_data.pkl')

		cm = plt.cm.get_cmap('RdYlBu_r')
		fig = plt.figure()
		ax = fig.add_subplot(111)
		sc = ax.scatter(np.array(df['Fingerprint_x']), np.array(df['Fingerprint_y']), c=np.array(df['Bandgap']), marker='o', s=10, cmap=cm)
		plt.tight_layout()
		plt.colorbar(sc)
		plt.show()

	def most_similar_dft(self, compound, n=6):
		df = pd.read_pickle(path+'/ML/data/processed_dft_data.pkl')
		df['Fingerprint'] = df[['Fingerprint_x','Fingerprint_y']].values.tolist()
		dft_fingerprints = np.stack(df['Fingerprint'])
		fingerprint = self.ae.get_fingerprint(self.VAE, compound, vae=True)

		euc_dis_list = np.array([])
		for i in dft_fingerprints:
			euc_dis = self.ae.get_euclidean_distance(fingerprint, i)
			euc_dis_list = np.append(euc_dis_list, euc_dis)
		ind = np.argsort(euc_dis_list)[:n]
		eucledian_distance = euc_dis_list[ind]
		result_df = pd.concat([df.iloc[ind][['StructuredFormula', 'Bandgap', 'Ehull', 'Formation energy']].reset_index(drop=True), 
							pd.DataFrame(list(eucledian_distance), columns=['Euclidean Distance'])], axis=1)	# pretty print
		# print (result_df)
		return result_df

	def predict_bandgap(self, n=5):
		df = pd.read_pickle(path+'/ML/data/processed_dft_data.pkl')
		df = df.drop_duplicates(keep='first')	# there were 2 duplicates
		def pred_prop(row):
			most_similar = self.most_similar_dft(row.StructuredFormula, n=n+1)
			predicted_bandgap = np.array(most_similar['Bandgap'].to_list()[1:]).mean()	# discard 1st element because it's the material being considered

			return predicted_bandgap


		df['Predicted_Bandgap'] = df.apply(pred_prop,axis=1)
		df[['StructuredFormula', 'Bandgap', 'Predicted_Bandgap']].to_csv(path+'/ML/data/Bgap_predictions.csv', sep='\t', index=False)
		
		bandgap_mse = mean_squared_error(np.array(df['Bandgap']), np.array(df['Predicted_Bandgap']))
		bandgap_mae = mean_absolute_error(np.array(df['Bandgap']), np.array(df['Predicted_Bandgap']))

		print('Bandgap RMSE: ', np.sqrt(bandgap_mse))
		print('Bandgap MAE: ', bandgap_mae)


	def find_phases(self):
		df = self.comps_wdup[['StructuredFormula', 'A1', 'A1_frac', 'A2', 'A2_frac', 'B1', 'B1_frac', 'B2', 'B2_frac', 'CrystalClass']]
		df_noformula = df.drop(['StructuredFormula'], axis=1).drop_duplicates(keep='first')
		print(df_noformula.info())

		def get_phases(row):
			phases = ((df_noformula['A1'] == row.A1) & (df_noformula['A1_frac'] == row.A1_frac) & (df_noformula['A2'] == row.A2) & (df_noformula['A2_frac'] == row.A2_frac) & 
				(df_noformula['B1'] == row.B1) & (df_noformula['B1_frac'] == row.B1_frac) & (df_noformula['B2'] == row.B2) & (df_noformula['B2_frac'] == row.B2_frac))
			competing_phase_idx = phases[phases == True].index.tolist()
			other_phases = []
			for idx in competing_phase_idx:
				other_phases.append(df.iloc[idx]['CrystalClass'])
			multiple_phases = 1 if len(other_phases)>1 else 0
			return other_phases, multiple_phases, len(other_phases)

		df_noformula['All_phases'], df_noformula['Multiple_phases'], df_noformula['How_many_phases'] = zip(*df_noformula.apply(get_phases, axis=1))
		df_noformula.drop_duplicates(subset=['A1', 'A1_frac', 'A2', 'A2_frac', 'B1', 'B1_frac', 'B2', 'B2_frac',
										 'Multiple_phases', 'How_many_phases'], keep='first', inplace=True)
		df_noformula['StructuredFormula'] = df.iloc[df_noformula.index]['StructuredFormula']
		df_noformula.to_csv(path+'/ML/data/all_phases_new.csv', sep='\t')
		print(df_noformula['Multiple_phases'].sum())

	def predict_phases(self):
		df = pd.read_csv(path+'/ML/data/all_phases_new.csv', sep='\t')
		df = df[df.Multiple_phases == 1]
		def get_nearest_crystal_systems(row):
			most_similar_df = self.ae.most_similar(self.VAE, compound=row.StructuredFormula, experimental=1, n=6)	# consider the composition being considered and 5 other neighbours
			similar_crystal_systems = most_similar_df['CrystalSystem'].to_list()[1:]	# discard 1st composition, the one being considered
			all_phase_lst = yaml.load(row.All_phases)
			predicted_phases = list(set(all_phase_lst) & set(similar_crystal_systems))
			# print(predicted_phases)
			return similar_crystal_systems, predicted_phases, len(predicted_phases)

		df['5_most_similar_crystal_systems'], df['overlapping_predictions'], df['Num_overlapping_predictions'] = zip(*df.apply(get_nearest_crystal_systems, axis=1))
		df.to_csv(path+'/ML/data/phase_pred_results_new.csv', sep='\t')
		df_pr = df[(df.Num_overlapping_predictions == df.How_many_phases)]
		# print(df_pr.shape)
		df_3_more = df[df.How_many_phases >= 3]
		# print(df_3_more.shape)
		df_pred_2_more = df_3_more[df_3_more.Num_overlapping_predictions >=2]
		df_pred_2_more.to_csv(path+'/ML/data/phase_pred_plotdata_new.csv', sep='\t')
		# print(df_pred_2_more.shape)
		# df_correct = df[df.Num_overlapping_predictions >1]
		# print(df_correct.shape)

	def plot_phase_prediction(self, ax=None):
		df = pd.read_csv(path+'/ML/data/phase_pred_plotdata_new.csv', sep='\t')
		crystal_systems = {1: 'Triclinic', 2:'Monoclinic', 3:'Orthorhombic', 4:'Tetragonal', 5:'Cubic', 6:'Trigonal', 7:'Hexagonal'}
		# cm = plt.cm.get_cmap('RdYlBu_r')
		if ax==None:
			ax = plt.gca()
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		width = 200
		height = 500
		verts = list(zip([-width,width,width,-width],[-height,-height,height,height]))
		def plot_individual_comp(row):
			for i in yaml.load(row.All_phases):
				c = 1 if i in yaml.load(row.overlapping_predictions) else 0
				sc = ax.scatter(row.StructuredFormula, i.title(), c=c, marker=verts, vmin=0, vmax=1, s=500, cmap=matplotlib.colors.ListedColormap(['blue', 'red']))

		SC = df.apply(plot_individual_comp, axis=1)
		ax.tick_params(axis='x',  labelsize=10)
		ax.tick_params(axis='y',  rotation=60)
		red_patch = mpatches.Patch(color='red', label='Predicted')
		blue_patch = mpatches.Patch(color='blue', label='Not-predicted')
		ax.legend(handles=[red_patch, blue_patch], prop={'weight':'bold', 'size': 11})
		ax.set_xlabel('Composition', fontsize=14)
		plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=10,
		         rotation_mode="anchor")
		ax.set_ylabel('Displayed phases', fontsize=14)
		return SC
		# plt.show()
