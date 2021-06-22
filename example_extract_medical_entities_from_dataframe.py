
"""
	sanja7s
	MEDDL: code to run Medical Entity Extraction

	given an input dataframe "data" with 
	a text column "column_name", and
	an index column "index_name",
	invoke the script for prediction spf.predict

	depending on the data type, you will select a different model
	e.g., for Twitter, select Micromed, and for Reddit, select AMT
	other parameters in the model are set to best, do not change them
"""

import sys, os
import pandas as pd
sys.path.append('MEDDL/')
import predict_flair as spf

script_directory = os.path.dirname(os.path.abspath(__file__))


"""
possibilities are:
-- Micromed (Twitter)
-- AMT (Reddit)
-- CADEC (CADEC)
"""
model = 'Micromed' 
selected_embeddings = {'glove':1, 'char':0, 'flair':0, 'pooled-flair':0, \
						'bert':0, 'twitter':0, 'elmo':0, 'roberta':1}


in_dir ='data/'
out_dir = 'results/'

in_dir = os.path.join(script_directory, in_dir)
out_dir = os.path.join(script_directory, out_dir)

file_name = "example_input.csv"

def predict_file(f_in, f_out):

	print ("predicting", f_in)

	column_name = "cons"
	index_name = "index"

	data = pd.read_csv(f_in)

	# model, selected_embeddings, data, column_name, index_name, output_file
	# the prediction function will save 3 files in the out_dir:
	# on with extracted diseases, other with drugs, and the third, with both
	# in each out file, you have the index column you provided here also saved
	spf.predict(model, selected_embeddings, data, column_name, index_name, f_out)



predict_file(f_in=os.path.join(in_dir, file_name), \
	f_out=os.path.join(out_dir, file_name))
