'''
Autoencoder class (VAE & vanilla)
@Achintha_Ihalage
'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import re
from pymatgen import Composition, Element
from arrange_ICSD_data import Perovskites
import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import mse
# import keras
np.random.seed(0)	# for reproducibility

path = str(pathlib.Path(__file__).parent.absolute())

class AutoEncoder():

	def __init__(self):
		self.all_compounds_file = path+'/ICSD_data/ICSD_all_data.csv'
		self.cand_perovskites_file = path+'/ML/data_pkl/cand_pero.pkl'
		self.all_df = self.read_csv(self.all_compounds_file)
		self.df = self.all_df.drop(['CollectionCode', 'HMS', 'SpaceGroupNum', 'CrystalClass', 'StructuredFormula', 'StructureType', 'Authors', 'CellParameter', 'CellVolume', 'FormulaWeight','Temperature', 
								'PublicationYear', 'Quality', 'A1', 'A2', 'B1', 'B2', 'O', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'], axis=1)
		self.no_features = len(self.df.columns)
		self.latent_dim = 2
		self.scaler = StandardScaler()
		self.data = self.scaler.fit_transform(self.df)

		self.X_train, self.X_test = train_test_split(self.data, test_size=0.01, random_state=12)	# for final testing we need a small sample, validation split is defined when training

	def read_csv(self, file):
		return pd.read_csv(file, sep='\t')

	def build_AE(self, vae=True):
		def sampling(args):
			"""Reparameterization trick by sampling from an isotropic unit Gaussian.
			# Arguments
				args (tensor): mean and log of variance of Q(z|X)
			# Returns
				z (tensor): sampled latent vector
			"""

			z_mean, z_log_var = args
			batch = K.shape(z_mean)[0]
			dim = K.int_shape(z_mean)[1]
			# by default, random_normal has mean = 0 and std = 1.0
			epsilon = K.random_normal(shape=(batch, dim))
			return z_mean + K.exp(0.5 * z_log_var) * epsilon

		if vae:
			inputs = Input(shape=(self.no_features,))
			x = Dense(128, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(inputs)
			x = Dense(64, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
			z_mean = Dense(self.latent_dim, name='z_mean')(x)
			z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
			# use reparameterization trick to push the sampling out as input
			# note that "output_shape" isn't necessary with the TensorFlow backend
			z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
			# instantiate encoder model
			encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')


			# build decoder model
			latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
			x = Dense(32, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(latent_inputs)
			x = Dense(64, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
			outputs = Dense(self.no_features, activation='linear')(x)

			# instantiate decoder model
			decoder = Model(latent_inputs, outputs, name='decoder')

			# instantiate VAE model
			outputs = decoder(encoder(inputs)[2])
			VAE = Model(inputs, outputs, name='vae_mlp')
			reconstruction_loss = mse(inputs, outputs)
			reconstruction_loss *= self.no_features
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			vae_loss = K.mean(reconstruction_loss + kl_loss)
			VAE.add_loss(vae_loss)
			VAE.compile(optimizer='adam')
			return VAE

		else:
			in_features = Input(shape=(self.no_features,))
			encoded = Dense(128, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(in_features)
			encoded = Dense(64, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(encoded)
			encoded = Dense(self.latent_dim, activation='relu', name='Fingerprint')(encoded)	# this is the learned embedding of the compound, named as the fingerprint

			decoded = Dense(32, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(encoded)
			decoded = Dense(64, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(decoded)
			decoded = Dense(self.no_features, activation='linear')(decoded)

			autoencoder = Model(in_features, decoded)
			return autoencoder

	def train(self, vae=True):
		if vae:
			VAE = self.build_AE(vae=True)

			
			VAE.summary()
			filepath=path+'/saved_models/best_model_VAE.h5'
			checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True, save_best_only=True)
			VAE.fit(self.X_train,
				epochs=1500,
				batch_size=32,
				validation_split=0.2, callbacks=[checkpointer])

		else:
			model = self.build_AE(vae=False)
			model.compile(optimizer='adam', loss='mse')

			filepath=path+'/saved_models/best_model_AE.h5'
			checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

			# train
			model.fit(self.X_train, self.X_train, epochs=1000,batch_size=32, shuffle=True, validation_split=0.2, callbacks=[checkpointer])

	def get_fingerprint(self, model, compound, experimental=0, vae=True):

		if vae:

			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])

		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder

		if experimental==0:	# generated materials in the correct format
			# arrange composition in electronegativity order in A-B sites separately
			site_list = re.split(r'[()]',compound)	# get what's inside parantheses and others separately
			doped_site = max(site_list, key=len)	# get doped site re.sub(r"(?<=\d)0+", "", d)
			site_rearranged = '(' + re.sub(r"(?<=\d)0+", "", Composition(doped_site).formula.replace(' ','')) + ')'	#arrange elements in electronegativity order, remove '0' and club with ()
			site_list[site_list.index(doped_site)] = site_rearranged	# replace with processed site
			composition = str(''.join(site_list))
			# create a dataframe using the input compound
			df = pd.DataFrame(columns=["StructuredFormula"], data=[[composition]])
		else:	# experimental compounds have complex formula formattings
			df = pd.DataFrame(columns=["StructuredFormula"], data=[[compound]])
		
		pv = Perovskites(df)
		df = pv.parse_formula(df)
		# df = pv.parse_space_group(df)
		df = pv.add_features(df)
		df = pv.SISSO_features(df)
		df = df.drop(['StructuredFormula', 'A1', 'A2', 'B1', 'B2', 'O'], axis=1)

		df = self.scaler.transform(df)
		fingerprint = encoder.predict(df)
		return fingerprint.flatten()

	def get_cosine_similarity(self, list1, list2):
		return np.dot(list1, list2)/(np.linalg.norm(list1)*np.linalg.norm(list2))	# return cosine similarity between list1 and list2
	def get_euclidean_distance(self, list1, list2):
		return np.linalg.norm(np.array(list1)-np.array(list2))

	def create_fingerprint_df(self, model, vae=True):
		if vae:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])
		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder

		fingerprints = encoder.predict(self.data)	# scaled, experimental database
		fingerprints_df = pd.concat([self.all_df, pd.DataFrame({'Fingerprint': list(fingerprints)}, columns=['Fingerprint'])], axis=1)
		fingerprints_df = fingerprints_df[['CollectionCode', 'HMS', 'CrystalClass', 'StructuredFormula', 'StructureType', 'A1', 'A2', 'B1', 'B2', 'Fingerprint']]
		fingerprints_df.dropna(subset=['Fingerprint'],inplace=True)
		if vae:
			fingerprints_df.to_pickle(path+'/ICSD_data/exp_fingerprints_VAE.pkl')
		else:
			fingerprints_df.to_pickle(path+'/ICSD_data/exp_fingerprints_AE.pkl')

	def create_candidate_fingerprint_df(self, model, vae=True):	
		if vae:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])
		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder

		self.cand_perovskites = pd.read_pickle(self.cand_perovskites_file)
		self.cand_perovskites_df = self.cand_perovskites.drop(['StructuredFormula', 'A1', 'A2', 'B1', 'B2', 'O', 'PV_classifications_out_of_100',
														'is_predicted_perovskite', 'Mean_classification_prob', 'Exist', 'has_common_oxi_states'], axis=1, errors='ignore')
		df_scaled = self.scaler.transform(self.cand_perovskites_df)
		fingerprints = encoder.predict(df_scaled)
		fingerprints_df = pd.concat([self.cand_perovskites, pd.DataFrame({'Fingerprint': list(fingerprints)}, columns=['Fingerprint'])], axis=1)
		fingerprints_df = fingerprints_df[['StructuredFormula', 'Mean_classification_prob', 'A1', 'A2', 'B1', 'B2', 'Fingerprint']]

		if vae:
			fingerprints_df.to_pickle(path+'/ICSD_data/candidate_fingerprints_VAE.pkl')
		else:
			fingerprints_df.to_pickle(path+'/ICSD_data/candidate_fingerprints_AE.pkl')


	def most_similar(self, model, compound=None, fingerprint=False, experimental=0, n=5, vae=True):
		if vae:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])
			exp_fingerprints_df = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_VAE.pkl')
			if compound:
				try:
					fingerprint = self.get_fingerprint(model, compound, experimental=experimental, vae=True)
				except:
					print ("Cannot be charge balanced!")
					return 0

		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder
			exp_fingerprints_df = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_AE.pkl')
			if compound:
				try: 
					fingerprint = self.get_fingerprint(model, compound, vae=False)
				except:
					print ("Cannot be charge balanced!")
					return 0

		exp_fingerprints = np.stack(exp_fingerprints_df['Fingerprint'])

		euc_dis_list = np.array([])
		for i in exp_fingerprints:
			euc_dis = self.get_euclidean_distance(fingerprint, i)
			euc_dis_list = np.append(euc_dis_list, euc_dis)
		ind = np.argsort(euc_dis_list)[:n]
		eucledian_distance = euc_dis_list[ind]
		result_df = pd.concat([exp_fingerprints_df.iloc[ind][['CollectionCode', 'HMS', 'CrystalClass', 'StructuredFormula']].reset_index(drop=True), 
							pd.DataFrame(list(eucledian_distance), columns=['Euclidean Distance'])], axis=1)	# pretty print
		result_df = result_df.rename(columns={'CrystalClass': 'CrystalSystem'})
		return result_df

	def most_similar_cand_perovskites(self, model, compound=None, prob_threshold=0.98, fingerprint=False, except_elems=['elements'], n=5, vae=True):
		if vae:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])
			cand_fingerprints_df = pd.read_pickle(path+'/ICSD_data/candidate_fingerprints_VAE.pkl')
			if compound:
				try:
					fingerprint = self.get_fingerprint(model, compound, vae=True)
				except:
					print ("Cannot be charge balanced!")
					return -1
		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder
			cand_fingerprints_df = pd.read_pickle(path+'/ICSD_data/candidate_fingerprints_AE.pkl')
			if compound:
				try:
					fingerprint = self.get_fingerprint(model, compound, vae=False)
				except:
					print ("Cannot be charge balanced!")
					return -1

		cand_fingerprints_df = cand_fingerprints_df[cand_fingerprints_df.Mean_classification_prob >= prob_threshold]	# impose prob threshold
		cand_fingerprints_df = cand_fingerprints_df.reset_index(drop=True)

		cand_fingerprints = np.stack(cand_fingerprints_df['Fingerprint'])

		euc_dis_list = np.array([])
		for i in cand_fingerprints:
			euc_dis = self.get_euclidean_distance(fingerprint, i)
			euc_dis_list = np.append(euc_dis_list, euc_dis)
		if except_elems[0]!='elements':
			ind = np.argsort(euc_dis_list)[:1000]
		else:
			ind = np.argsort(euc_dis_list)[:n]
		eucledian_distance = euc_dis_list[ind]
		result_df = pd.concat([cand_fingerprints_df.iloc[ind][['StructuredFormula', 'A1', 'A2', 'B1', 'B2', 'Mean_classification_prob']].reset_index(drop=True), 
							pd.DataFrame(list(eucledian_distance), columns=['Euclidean Distance'])], axis=1)	# pretty print

		def get_filtered_comps(row):
			elem_list = [row.A1, row.A2, row.B1, row.B2]
			if any(elem in elem_list for elem in except_elems):
				return 1
			return 0
		if except_elems[0]!='elements':
			result_df['has_except_elem'] = result_df.apply(get_filtered_comps, axis=1)
			result_df = result_df[result_df['has_except_elem']==0].iloc[:n]
		return result_df[['StructuredFormula', 'Mean_classification_prob', 'Euclidean Distance']]



	def most_similar_cosine(self, model, compound=None, experimental=0, fingerprint=False, n=5, vae=True):
		if vae:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('encoder').get_output_at(0)[0])
			exp_fingerprints_df = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_VAE.pkl')
			if compound:
				try:
					fingerprint = self.get_fingerprint(model, compound, experimental=0, vae=True)
				except:
					print ("Cannot be charge balanced!")
					return -1
		else:
			encoder = Model(inputs=model.inputs, outputs=model.get_layer('Fingerprint').output)	# redefine model to output right after the encoder
			exp_fingerprints_df = pd.read_pickle(path+'/ICSD_data/exp_fingerprints_AE.pkl')
			if compound:
				try:
					fingerprint = self.get_fingerprint(model, compound, experimental=0, vae=False)
				except:
					print ("Cannot be charge balanced!")

		exp_fingerprints = np.stack(exp_fingerprints_df['Fingerprint'])

		cos_sim_list = np.array([])
		for i in exp_fingerprints:
			cos_sim = self.get_cosine_similarity(fingerprint, i)
			cos_sim_list = np.append(cos_sim_list, cos_sim)
		ind = np.argsort(-cos_sim_list)[:n]
		similarity = cos_sim_list[ind]
		result_df = pd.concat([exp_fingerprints_df.iloc[ind][['CollectionCode', 'HMS', 'CrystalClass', 'StructuredFormula']].reset_index(drop=True), 
							pd.DataFrame(list(similarity), columns=['Cosine Similarity'])], axis=1)	# pretty print
		return result_df

	def most_similar_cubic(self, model, compound=None, fingerprint=False, n=10, vae=True):
		try:
			most_similar_df = self.most_similar(model, compound=compound, n=1000, vae=vae)
			most_similar_cubic_df = most_similar_df[most_similar_df['CrystalSystem']=='cubic']
		except:
			print ("Cannot be charge balanced")
			return -1
		if n <= most_similar_cubic_df.shape[0]:
			return most_similar_cubic_df.iloc[:n]
		else:
			return most_similar_cubic_df
