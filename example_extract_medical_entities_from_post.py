
"""
	sanja7s
	MED-DL: code to run Medical Entity Extraction

	given an input dataframe "data" with 
	a text column "column_name", and
	an index column "index_name",
	invoke the script for prediction spf.predict

	depending on the data type, you will select a different model
	e.g., for Twitter, select Micromed, and for Reddit, select AMT
	other parameters in the model are set to best, do not change them
"""

import sys, os
from MEDDL import meddl

"""
possibilities are:
-- Micromed (Twitter)
-- AMT (Reddit)
"""

model_name = 'AMT' 
extractor = meddl.MedDLEntityExtractor(model_name)

text = 'Acitretin caused me alopecia.'
res = extractor.extract(text)
print (res)