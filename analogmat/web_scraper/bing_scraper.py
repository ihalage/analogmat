''' 
############################################################################################################################
# This script implements a web scraping tool using 'Bing' search engine and should only be used for academic purposes.
# Please note that this is a python2 script, hence should be run separately from all other python3 scripts
############################################################################################################################

# The novelty of the generated perovskite compounds is checked and only those with no records found on internet will be reported as new candidate perovskites

@Achintha_Ihalage
@07_Jun_2020
'''

# Disable warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bs4 import BeautifulSoup
import urllib,urllib2
import pandas as pd
import sys
import pathlib
import types
import re
import random
from pymatgen import Composition	# monty 1.x is required
import socks
import socket


# create a class for high-throughput web scraping and individual compound web scraping
class BingScraper():
	def __init__(self):
		path = str(pathlib.Path(__file__).parent.absolute().parent)
	
	def read_csv(self, file, sep='\t'):
		return pd.read_csv(file, sep=sep)

	def search(self, query):
		ip_port_list = ["3.11.214.31:80", "163.172.189.32:8811", "185.134.23.198:80", "163.172.180.18:8811", "178.128.228.158:80"]	# constantly updated proxy list by selecting from (https://free-proxy-list.net/)
		proxy = urllib2.ProxyHandler({"http": "3.10.145.250:80"})
		opener = urllib2.build_opener(proxy)
		urllib2.install_opener(opener)
		address = "http://www.bing.com/search?q=%s" % (urllib.quote_plus(query))

		getRequest = urllib2.Request(address, None, {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'}) # we need to make it look like the request is coming
																												#  from a browser. Other automatic scraping will be refused
		# if above User-Agent doesn't work try this  --->    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0'
		# user agents: ['Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11']
		# please see (https://stackoverflow.com/questions/802134/changing-user-agent-on-urllib2-urlopen) for more options

		urlfile = urllib2.urlopen(getRequest)
		htmlResult = urlfile.read(200000)
		urlfile.close()

		soup = BeautifulSoup(htmlResult)

		[s.extract() for s in soup('span')]
		unwantedTags = ['a', 'strong', 'cite']
		for tag in unwantedTags:
			for match in soup.findAll(tag):
				match.replaceWithChildren()

		results = soup.findAll('li', { "class" : "b_algo" })
		comp = query.replace("(","").replace(")","")	# composition without parantheses to enable better search
		# comp1 = query.replace("(","[").replace(")","]")
		for result in results:
			title = str(result.find('h2')).replace(" ", " ")
			# print('this is title',title)
			description = str(result.find('p')).replace(" ", " ")
			text = title+description	# concatenate title and description
			rep = {" ": "", "(": "", ")":"", "[":"", "]":""} #  to remove spaces, parantheses, brackets etc.
			# use these three lines to do the replacement
			rep = dict((re.escape(k), v) for k, v in rep.items()) 
			pattern = re.compile("|".join(rep.keys()))
			text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)	# replace
			if comp in text:
				print query, '1'
				return 1

		print query, '0'
		return 0

	def scrape_df(self, df=None, file=None):
		def scrape_comp(row):
			compound = row.StructuredFormula
			result = self.search(compound)
			return result
		if not df:
			df = self.read_csv(file, sep='\t')
		df['found_on_web'] = df.apply(scrape_comp, axis=1)
		df[['StructuredFormula', 'found_on_web']].to_csv(self.path+'/bing_scrape_results.csv', sep='\t', index=False)

	def scrape_generated_comps(self, df=pd.DataFrame(), file=None):
		def scrape_comp(row):
			# reformulate the doped site by electronegativity order of elements for a better web scraping,and remove '0's & parantheses. e.g.  Pb(Ti0.50Zr0.50)O3 --> PbZr0.5Ti0.5O3
			site_list = re.split(r'[()]',row.StructuredFormula)	# get what's inside parantheses and others separately
			doped_site = max(site_list, key=len)	# get doped site re.sub(r"(?<=\d)0+", "", d)
			site_rearranged = '(' + re.sub(r"(?<=\d)0+", "", Composition(doped_site).formula.replace(' ','')) + ')'	# arrange elements in electronegative order, remove '0' and club with ()
			site_list[site_list.index(doped_site)] = site_rearranged	# replace with processed site
			compound = str(''.join(site_list))
			# print (compound)
			result = self.search(compound)
			return compound, result
		if df.empty:
			df = self.read_csv(file, sep='\t')
		df['Chemical_formula'], df['found_on_web'] = zip(*df.apply(scrape_comp, axis=1))
		# df[['StructuredFormula', 'found_on_web']].to_csv(self.path+'/novel_perovskite_candidates.csv', sep='\t', index=False)
		return df[['Chemical_formula', 'found_on_web']]
		

	def scrape_compound(self, compound):
		s = socks.socksocket()
		# socks.set_default_proxy(socks.SOCKS5, "localhost")
		
		# s.set_proxy(socks.HTTP, "203.202.245.62", 80)
		# socket.socket = socks.socksocket
		proxy = urllib2.ProxyHandler({"http": "3.10.145.250:80"})	#3.11.214.31:80	178.128.228.158:80    203.202.245.62:80   188.40.183.187:1080   185.63.253.203:5836
		opener = urllib2.build_opener(proxy)
		urllib2.install_opener(opener)
		address = "http://www.bing.com/search?q=%s" % (urllib.quote_plus(compound))

		getRequest = urllib2.Request(address, None, {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'})

		urlfile = urllib2.urlopen(getRequest)
		htmlResult = urlfile.read(200000)
		urlfile.close()

		soup = BeautifulSoup(htmlResult)

		[s.extract() for s in soup('span')]
		unwantedTags = ['a', 'strong', 'cite']
		for tag in unwantedTags:
			for match in soup.findAll(tag):
				match.replaceWithChildren()

		results = soup.findAll('li', { "class" : "b_algo" })
		comp = compound.replace("(","").replace(")","")	# composition without parantheses to enable better search
		c=0
		for result in results:
			title = str(result.find('h2')).replace(" ", " ")
			description = str(result.find('p')).replace(" ", " ")
			text = title+description	# concatenate title and description
			rep = {" ": "", "(": "", ")":"", "[":"", "]":""} # define desired replacements here /// to remove spaces, parantheses, brackets etc.
			# use these three lines to do the replacement
			rep = dict((re.escape(k), v) for k, v in rep.iteritems()) 
			#Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
			pattern = re.compile("|".join(rep.keys()))
			text_trimmed = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)	# replace
			if comp in text_trimmed:
				c+=1
				if c==1:
					print '########################################'
					print compound, ' is found on web!!! \nSee below for results\n\n'
				print "# TITLE: "+ title + "\n#"
				print "# DESCRIPTION: " + description
				print "# ___________________________________________________________\n#"
				
		if c>0:
			return 1
		print ' Bing could not find ' + compound +' on web!!! \n'
		return 0
