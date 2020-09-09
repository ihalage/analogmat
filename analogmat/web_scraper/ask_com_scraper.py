''' 

@Achintha_Ihalage
'''

 #!/usr/bin/env python

from bs4 import BeautifulSoup
import urllib,urllib2
import pandas as pd
import sys
import types
import re

file = '../ICSD_data/ICSD_all_data.csv'

def read_csv(file):
	return pd.read_csv(file, sep='\t')

def scrape_ask(row):

	def search(query):
		# print urllib.quote_plus(query)
		# address = "https://uk.ask.com/?o=0&l=dir&ad=dirN/web?q=%s" % (urllib.quote_plus(query))
		address = "https://uk.ask.com/web?o=0&l=dir&qo=serpSearchTopBox&q=%s" % (query)
		# print address
		getRequest = urllib2.Request(address, None, {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0'})

		urlfile = urllib2.urlopen(getRequest)
		htmlResult = urlfile.read(200000)
		results = extractSearchResults(htmlResult)
		print results
		urlfile.close()

		return 0

	compound = row.StructuredFormula
	result = search(compound)
	return result



df = read_csv(file)
df['found_on_web'] = df.iloc[:300].apply(scrape_ask, axis=1)
df[['StructuredFormula', 'found_on_web']].to_csv('google_exp_data.csv', sep='\t', index=False)


