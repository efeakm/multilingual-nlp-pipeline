# multilingual-nlp-pipeline

## Language Translation and Identification Pipeline  

### Overview  
This repository contains a pipeline designed for identifying the language of a given text and translating it to English. It leverages state-of-the-art models and libraries primarily developed by Facebook for robust NLP tasks. Specifically, we use Facebook's FastText for language identification and Facebook's NLLB ("No Language Left Behind") model for translation, capable of handling 200+ languages.

### Requirements  
fasttext  
transformers  
pandas  
sentencepiece  
torch  

### Installation   
Install the necessary packages and models by running the following commands:

```bash
!wget https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin
!pip install fasttext
!pip install -U pip transformers
!pip install pandas
!pip install sentencepiece
!pip install torch
```

### Usage  
#### Language Identification with Facebook's FastText  
FastText is an open-source, free, lightweight library developed by Facebook, designed to perform robust and efficient text classification and language identification tasks. In this pipeline, we use FastText for the initial identification of the text's language.

```python
import fasttext
pretrained_lang_model = "/content/lid218e.bin"  # Path of model file
lang_detection_model = fasttext.load_model(pretrained_lang_model)
```

#### Translation with Facebook's NLLB ("No Language Left Behind") Model  
The NLLB model by Facebook is capable of translating more than 200 languages. It aims to provide comprehensive language coverage, living up to its name—No Language Left Behind. The pipeline uses Huggingface's Transformers library to implement the translation.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

checkpoint = 'facebook/nllb-200-distilled-600M'
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir='/models')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='/models')
```
#### Data Operations  
A pandas DataFrame is used to store comments, and the pipee function translates these comments to English.

```python
import pandas as pd
df = pd.read_csv('your_dataset.csv')
df = pipee(df)
```

### Known Limitations
The code is optimized for GPU usage. Running it on a CPU may be much slower.
Be cautious of memory usage when working with large datasets.

### Contributing
Contributions are welcome! Feel free to open an issue or make a pull request.
