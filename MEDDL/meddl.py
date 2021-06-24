from flair.data import Corpus
from flair.data import Sentence 

from flair.embeddings import TokenEmbeddings, WordEmbeddings, \
	 StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
	 PooledFlairEmbeddings, ELMoEmbeddings, BertEmbeddings , RoBERTaEmbeddings
from typing import List

# # 6. initialize trainer
# from flair.trainers import ModelTrainer

# 5. initialize sequence tagger
from flair.models import SequenceTagger

# # 6. initialize trainer
# from flair.training_utils import EvaluationMetric

# 9. continue trainer at later point
from pathlib import Path

import pandas as pd
import tqdm
import os, sys
import json
import re


# import preprocessor as p
# p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY)

package_directory = os.path.dirname(os.path.abspath(__file__))




class MedDLEntityExtractor:
	def __init__(self, model_name):
		"""
		possibilities are:
		-- Micromed (Twitter)
		-- AMT (Reddit)
		-- CADEC (CADEC)
		"""

		if model_name == 'AMT':
			self.SEQUENCE_LIMIT = 200
		elif model_name == 'CADEC':
			self.SEQUENCE_LIMIT = 300
		else:
			self.SEQUENCE_LIMIT = 150

		self.selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
								'bert':0, 'twitter':0, 'elmo':0, 'roberta':1}

		self.selected_embeddings_text = [key  for key in self.selected_embeddings if self.selected_embeddings[key]]
		self.selected_embeddings_text = '_'.join(self.selected_embeddings_text)
		print (self.selected_embeddings_text)

		# print (package_directory)
		self.model_dir = 'resources/model/FA_' + model_name + self.selected_embeddings_text 
		self.model_path = os.path.join(package_directory, self.model_dir, 'final-model.pt')
		print (self.model_path)

		# load the model you trained
		self.model = SequenceTagger.load(self.model_path)


	# for other social media text
	def get_clean_body(self, body):
		return " ".join(re.findall(r"[\w']+|[/().,!?;]", body))

	# for tweets
	def preprocess_body(self, body):
		body = body.encode('UTF-8').decode('UTF-8')
		body=body.encode('ascii', 'ignore').decode('ascii')
		return p.clean(body)


	def splitter(self, n, s):
		pieces = [w for w in s.split() ]
		return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))


	def extract(self, text):
		"""		
			input: model_name string -- AMT, Micromed, or CADEC
				   text -- to process
			output:
					dictionary -- {'sym': [list of symptoms],
								   'drug': [list of drugs]}
								example: {'sym': 'alopecia; ',\
										  'drug': 'Acitretin; '}
		"""

		parsed = 0
		skipped = 0

		body = str(text)

		# this is a special library for cleaning tweets invocation
		# if model_name == 'Micromed':
		# 	body = self.preprocess_body(body)

		body = self.get_clean_body(body)

		bodies = self.splitter(self.SEQUENCE_LIMIT, body)

		dis_res = ''
		drug_res = ''
		res_dict = {}

		for el in bodies:
			#print (el)
			sentence = Sentence(str(el))

			try:
				# # predict tags and print
				self.model.predict(sentence)

				res = sentence.to_dict(tag_type='ner')
				# print (res['entities'])

				for el in res['entities']:

					# print (el)
					labels = str(el['labels'])
					# print (labels)
					if 'DIS' in labels:
						conf = labels.replace("[","").replace("]", "").replace("(","").replace(")","").replace("DIS","").replace("DRUG","")
						dis_res = el['text'].replace('\n', ' ')+ '; ' + dis_res
						# print (dis_res)

					elif 'DRUG' in labels:
						conf = labels.replace("[","").replace("]", "").replace("(","").replace(")","").replace("DIS","").replace("DRUG","")
						drug_res = el['text'].replace('\n', ' ') + '; ' +  drug_res 
						# print (drug_res)
					parsed += 1

			except Exception as e:
				print ('skipped', e, el)
				skipped += 1
				if 'CUDA error' in str(e) or 'cublas runtime error' in str(e):
					print ("Total skipped posts ", skipped)
					print ("Total parsed posts ", parsed)
					print (sentence)
					return
					# sys.exit()
				continue

		if dis_res != '' or drug_res != '':
			res_dict['sym'] = dis_res
			res_dict['drug'] = drug_res

		return res_dict


	def extract_dataframe(self, df, text_field):
		"""		
			input: 
				   df dataframe -- with whatever columns, and the text to parse
				   text_field -- the name of the text column
				   df_outfile -- where to save the output
			output:
					will create 3 files:
					two .csc files with index, and one additional columns: for drugs or diseases
					one .json file with index and both diseses and drugs

			NOTE: this function has a hard-coded address to model dir
				  that could be improved for other setups but here we have only that one so ok
		"""

		res = df[text_field].apply(lambda x: self.extract(x))
		print (res)
		try:
			# this is for syms 
			df['sym'] = [x['sym'] if x != {} else None for x in res.values ]
		except Exception as e:
			print (e)
			df['sym'] = None
		try:
			# this is for drugs
			df['drug'] = [x['drug'] if x != {} else None for x in res.values ]
		except Exception as e:
			print (e)
			df['drug'] = None		
		return df
		
	