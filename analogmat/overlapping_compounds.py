'''
# Find the compounds that were reported to form both perovskite and non-perovskite structures

@Achintha_Ihalage
@11_Jun_2020
'''
import numpy as np
import pandas as pd
import pathlib

path = str(pathlib.Path(__file__).parent.absolute())

df_pv = pd.read_csv(path+'/ICSD_data/perovskites.csv', sep='\t')
df_npv = pd.read_csv(path+'/ICSD_data/non_perovskites.csv', sep='\t')

# merge dataframes 
overlap_comps = df_npv.merge(df_pv, on=['A1', 'A1_frac', 'A2', 'A2_frac', 'B1', 'B1_frac', 'B2', 'B2_frac'])

# print (overlap_comps)
overlap_comps.to_csv(path+'/ICSD_data/overlapping_compounds.csv', sep='\t', index=False)