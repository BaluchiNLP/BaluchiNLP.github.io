## Introduction
**Overview**: Implementing a Transformer-based NMT model for Baluchi is a complex task requiring expertise in machine learning, natural language processing, and the Baluchi language. This repository provides comprehensive information and resources necessary for building an efficient NMT system tailored for the Baluchi language.

## Part 1: Preparation and Planning
### 1. Language Documentation
- **Language Resources Development**: Compilation of dictionaries, grammar guides, and language learning resources to support NMT model development.

### 2. Technology Development
- **Neural Machine Translation**: Introduction to the Transformer-based NMT model for Baluchi, including back translation techniques.

## Part 2: Data Management
### 1. Data Collection
- **Sources**: Identifying and gathering parallel text data in both Baluchi and English from various sources like news articles, books, and websites.

### 2. Data Preprocessing
- **Text Preparation**: Cleaning text, removing noise, tokenizing, handling diacritics, and aligning sentences.
- **Formatting and Saving**: Selecting appropriate formats for the NMT framework and documenting preprocessing steps.

## Part 3: Model Development and Training
### 1. Tokenization
Tokenization is a crucial preprocessing step in Natural Language Processing (NLP) and Machine Translation (MT), where text is split into meaningful units such as words, phrases, or symbols. This step is essential for transforming natural language into a form that a machine learning model can understand and process.
There are many tokenizer from LLM (Large Language Model) devlopers such as OpenAI and Google. OpenAI's opensource version of Tokenizer named [Tiktoken](https://github.com/openai/tiktoken), primarily designed for use with OpenAI's large language models like GPT-4 and ChatGPT. It uses Byte pair encoding (BPE); is a way of converting text into tokens.
Another popular tokenizer [Sentencepiece](https://github.com/google/sentencepiece) developed by Google and is opensource. It implements subword units ( byte-pair-encoding (BPE) and unigram language model). SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation and Neural Machine Translation system.


- **Tools and Tutorials**: Utilizing pre-existing tools like Wordpiece or SentencePiece for preprocessing and tokenization tasks. [Tokenizer Tutorial](https://huggingface.co/transformers/v3.4.0/tokenizer_summary.html)

### 2. Framework and Model Selection
- **Open-source Options**: Exploration of [Marian NMT](https://marian-nmt.github.io/), [FairSeq](https://github.com/facebookresearch/fairseq), [OpenNMT](https://github.com/OpenNMT), and various Transformer architectures like [The base Transformer](https://huggingface.co/docs/transformers/en/index), [Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl), and [T5](https://paperswithcode.com/method/t5).

### 3. Training the Model
- **Techniques**: Fine-tuning pre-trained models, hyperparameter tuning, and monitoring training progress with metrics like BLEU and ROUGE scores.

## Part 4: Evaluation and Refinement
- **Testing and Refinement**: Evaluating the model on a held-out set of Baluchi-English sentences and refining based on performance and error analysis.

## Conclusion
**Summary and Future Directions**: A recap of the key points and insights into potential areas for further research and development.
