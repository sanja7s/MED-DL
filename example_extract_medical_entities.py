"""
	sanja7s
	MED-DL: code to run Medical Entity Extraction

	given an input post or dataframe "data" with 
	a text column, extract medical entities.

	depending on the data type, you will select a different model
	e.g., for Twitter, select Micromed, and for Reddit, select AMT
	other parameters in the model are set to the best, do not change them
"""
import pandas as pd
import sys, os, csv
from MEDDL import meddl

"""
possibilities are:
-- Micromed (Twitter)
-- AMT (Reddit)
"""

def predict_file(f_in, text_column, sep, f_out, drop_empty=True):
	"""
		customise this function as needed
		for your data
	"""

	print ("predicting", f_in)
	data = pd.read_csv(f_in, sep=sep, nrows=1000)

	res_df = extractor.extract_dataframe(data, text_column)
	if drop_empty == True:
		res_df = res_df.dropna(subset=['sym', 'drug'])
	print (res_df)

	if f_out is not None:
		if sep == ';':
			res_df.to_csv(f_out, encoding='utf-8', index=False,\
				sep=sep, quoting=csv.QUOTE_ALL)
		else:
			res_df.to_csv(f_out, encoding='utf-8', index=False,\
				sep=sep)

### initialize
model_name = 'AMT' 
extractor = meddl.MedDLEntityExtractor(model_name)

### from post
text = 'Acitretin caused me alopecia.'
res = extractor.extract(text)
print (res)

### from dataframe
file_name = "example_data_reddit.csv"
in_dir ='data/'
out_dir = 'results/'
f_in = os.path.join(in_dir, file_name)
f_out = os.path.join(out_dir, file_name)
predict_file(f_in, 'text', '|', f_out)