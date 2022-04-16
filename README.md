# MedDL: Medical Entity Extractor from Social Media

This is a repository for the model described in the paper [Extracting Medical Entities from Social Media](https://dl.acm.org/doi/abs/10.1145/3368555.3384467). [Another repository](https://github.com/sanja7s/MedRed) contains all the sources and data related to the paper. This repository is for those who are only interested in *applying the pretrained models*. 



## Structure

`MED-DL` contains the library for extraction. Inside this folder,`resources` requires the pretrained models, which can be downloaded from [FigShare models](https://doi.org/10.6084/m9.figshare.12039933.v1). Download the 3 folders with pretrained models and place them all under `resources/model`. 

`data` contains a sample dataset, and `results` contains a sample extractor output on the given dataset.



`example_extract_medical_entities.py` shows how to run the extraction on a sample post (textual input) or on a sample dataframe with a text column.


## Requirements

* flair**
* pandas
* tqdm
* typing

** for the models to work, the [flair](https://github.com/flairNLP/flair) library requires you to download the [RoBERTa embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md).


### If you use this repostiory, please consider citing our paper
[Extracting Medical Entities from Social Media](https://dl.acm.org/doi/abs/10.1145/3368555.3384467)
