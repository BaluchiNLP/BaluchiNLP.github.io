# Introduction
## Overview
Implementing a Transformer-based NMT model for Baluchi is a complex task that requires expertise in machine learning, natural language processing, and understanding the Baluchi language. This repository aims to provide comprehensive information and resources necessary for building an efficient NMT system tailored for the Baluchi language.

# Part 1: Preparation and Planning
## 1. Language Documentation
### Language Resources Development
Compilation of dictionaries, grammar guides, and language learning resources to support the NMT model development.

## 2. Technology Development

### Neural Machine Translation
Transformer-based NMT model for Baluchi to English and back translation.

# Part 2: Data Management
## 1. Data Collection
  Need parallel text data in both Baluchi and English. This can include news articles, books, websites, or any other source with aligned sentences in both languages.
## 2. Data Preprocessing:
  Cleaning the text, removing noise and irrelevant characters, tokenizing words, and converting everything to lowercase. Might also need to handle specific challenges like diacritics and named entities in Baluchi text.
#### Split the text: 
  Separate the Baluchi and English sentences into individual files or datasets.
#### Clean the text: 
  Remove irrelevant characters, punctuation, and special symbols that might interfere with the model's learning. This might include HTML tags, line breaks, and extra spaces.
#### Convert to lowercase: 
Most NLP models work better with lowercase text.
#### Handle diacritics: 
If Baluchi text contains diacritics, decide whether to remove them, normalize them, or keep them. Consider the linguistic importance of diacritics in Baluchi and the potential impact on translation quality.
#### Tokenize the text: 
Break the sentences into individual words or subword units (e.g., characters) based on chosen framework and model requirements.
#### Aligning Sentences:
Ensure that each sentence in the Baluchi file corresponds to its exact translation in the English file. This is crucial for parallel training of the NMT model.
If data doesn't have automatic alignment information, might need to manually align the sentences using tools like [Hunalign](https://github.com/danielvarga/hunalign) or manually line them up in a spreadsheet.


#### Filtering and Sampling:
Remove sentences with missing information, inconsistencies, or excessive noise.
If dataset is imbalanced (e.g., many short sentences and few long ones), consider sampling or oversampling techniques to ensure the model is exposed to diverse sentence lengths.
Depending on computational resources, might need to limit the dataset size for training. Consider starting with a smaller subset and gradually increasing as model improves.

#### Saving and Formatting:
Choose a format compatible with chosen NMT framework. Common formats include plain text files with aligned sentences, tab-separated value (TSV) files, or JSON files.
Keep track of the preprocessing steps, filtering criteria, and any specific characteristics of the data for future reference.

#### Data augmentation: 
Techniques like back-translation (generating synthetic training data by translating English text back to Baluchi) can help improve model performance with limited data.

# Part 3: Model Development and Training
### Tokenizer
Tokenization is a crucial preprocessing step in Natural Language Processing (NLP) and Machine Translation (MT), where text is split into meaningful units such as words, phrases, or symbols. This step is essential for transforming natural language into a form that a machine learning model can understand and process.

Pre-existing tools [Wordpiece](https://blog.research.google/2021/12/a-fast-wordpiece-tokenization-system.html) or [SentencePiece](https://github.com/google/sentencepiece) for preprocessing and tokenization tasks.

[Tokenizer Tutorial](https://huggingface.co/transformers/v3.4.0/tokenizer_summary.html)

### Framework and Model:
- Opensource NMT include:
  - [Marian NMT](https://marian-nmt.github.io/)
  - [FairSeq](https://github.com/facebookresearch/fairseq) includes pretrained models for Neural Machine Translation and Neural Language Modeling.

  - [OpenNMT](https://github.com/OpenNMT)
 
- Transformer model:
  There are various Transformer architectures like:
  - [The base Transformer](https://huggingface.co/docs/transformers/en/index)
  - [Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)
  - [T5](https://paperswithcode.com/method/t5)
 
### Training the Model:
#### Fine-tune the pre-trained model: 
Most frameworks provide pre-trained Transformer models on large datasets like English-French or English-German. Fine-tune these models on Baluchi-English dataset to adapt them to specific translation task.
#### Hyperparameter tuning: 
Experiment with different hyperparameters like learning rate, batch size, and optimizer to optimize model's performance. 
#### Monitor training: 
Metrics like BLEU score, ROUGE score, and translation quality to assess model's progress.

# Part 4: Evaluation and Refinement
Test model on a held-out set of Baluchi-English sentences that it hasn't seen during training. This will give a more realistic picture of its generalizability.
Identify common errors made by model and try to understand the reasons behind them. Use this information to refine training data, model architecture, or hyperparameters.





