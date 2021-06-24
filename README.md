# MED-DL Medical Entity Extractor from Social Media


## Structure

`MED-DL` contains the library for extraction. Inside this folder,`resources` requires the pretrained models, which can be downloaded from [FigShare models](https://doi.org/10.6084/m9.figshare.12039933.v1). Download the 3 folders with pretrained models and place them all under `resources/model`. 

`data` contains a sample dataset, and `results` contains a sample extractor output on the given dataset.



`example_extract_medical_entities.py` shows how to run the extraction on a sample post (textual input) or on a sample dataframe with a text column.



## Requirements

* flair**
* pandas
* tqdm
* typing

** for the models to work, for the [flair](https://github.com/flairNLP/flair) library, you also need to download the [RoBERTa embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md).


### If you use this repostiory, please consider citing our paper
[Extracting Medical Entities from Social Media](https://dl.acm.org/doi/abs/10.1145/3368555.3384467)