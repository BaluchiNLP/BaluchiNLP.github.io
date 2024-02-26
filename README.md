**Authors:**<br>
Mazhar Mohsin 
and 
Mukhtar Hussain

## Introduction
**Overview**: Implementing a Transformer-based Neural Machine Translation (NMT) model for Baluchi is a complex task requiring expertise in machine learning, natural language processing, and the Baluchi language. This repository provides comprehensive information and resources necessary for building an efficient NMT system for Baluchi language.

## Part 1: Preparation and Planning
### 1. Language Documentation
- **Language Resources Development**: Numerous organizations, academic institutions, academies, and publishers are actively engaged in developing resources for the Baluchi language, including books, literature, and dictionaries. A substantial amount of Baluchi text data is available from these sources. However, one of the primary challenges in data collection is securing reliable texts. A significant issue is the lack of standardization in Baluchi text; not all authors adhere to a uniform set of rules for writing, indicating the absence of a standardized method for Baluchi written text. This diversity in writing styles complicates efforts to compile and utilize Baluchi language resources effectively.
Building on these efforts, individual initiatives have also been undertaken to translate texts between Baluchi and English, further enriching the language's resources. Notably, the Department of Linguistics and Philology at Uppsala University has achieved significant progress in developing a standardized written form of Baluchi. This endeavor marks a crucial step towards resolving the standardization issue and facilitates a more unified approach to writing and translating the Baluchi language.
Uppsala University has conducted extensive Baluchi-English translation work, resulting in the publication of books featuring Baluchi-English translated texts, as well as Baluchi-English dictionaries. Such resources are invaluable for developing Neural Machine Translation (NMT) systems, which require parallel texts like those translating between Baluchi and English. The translated texts sourced from Uppsala, along with contributions from other organizations and individuals, will significantly bolster the progress of this project, facilitating the creation of a more robust and accurate NMT system for the Baluchi language.

- **Dictionaries**: A dictionary from a Baluchi to English can be leveraged at the data preparation or preprocessing stage of training a Transformer for NMT. Specifically, it can be used to augment the training data by generating synthetic parallel sentences or enhancing the vocabulary. Dictionaries can assist in the fine-tuning phase, where they can help adjust the model's translations for specific terms or phrases, ensuring better alignment with human translations. This approach is particularly useful for addressing the challenges posed by the scarcity of parallel corpora in Baluchi language translation tasks. [Balochi Dictionary](https://www.webonary.org/balochidictionary/)

- **Grammar guides**: Grammar guides are crucial in the development of Neural Machine Translation (NMT) systems for Baluchi, offering deep insights into syntax, morphology, and language structure beyond the lexical information provided by dictionaries. They enhance NMT systems by informing the design and adjustment process to better capture linguistic nuances during data preparation, aiding in text normalization and the handling of complex grammatical constructions for more effective model training. These are pivotal in error analysis and model refinement, helping identify systematic issues based on the model's output, thereby guiding the generation of synthetic data that adheres to grammatical rules and improving overall translation quality. By leveraging grammar guides, developers can ensure the input data accurately reflects Baluchi's unique features, systematically address translation errors, and implement post-processing adjustments to align the model's output with standard grammatical conventions, boosting the performance and reliability of NMT system for the Baluchi language. [A Grammar of Modern Standard Balochi](https://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1372275&dswid=6611)

### 2. Technology Development
- **Neural Machine Translation**: Introduction to the Transformer-based NMT model for Baluchi, including back translation techniques.

#### An example of base Transformer model: 
An example to understand how a Transformer model works in Neural Machine Translation (NMT) from English to Baluchi. The Transformer model consists of several key components: the embedding layer, positional encoding, the encoder, the decoder, and finally, the output layer.
In this revised example, the English sentence "Two people came to our home and had dinner with us last night" is translated into Baluchi as "Do mardom dóshi may lógá átk o gón má shámesh kort". Here's how the process is adapted:

##### 1. Input Embedding
- **Action**: The input sentence is split into tokens (words or subwords), and each token is converted into a high-dimensional vector using an embedding layer. 
- **Example**: `["Two", "people", "came", "to", "our", "home", "and", "had", "dinner", "with", "us", "last", "night"]` -> `[Vector(Two), Vector(people), Vector(came), Vector(to), Vector(our), Vector(home), Vector(and), Vector(had), Vector(dinner), Vector(with), Vector(us), Vector(last), Vector(night)]`

##### 2. Positional Encoding
- **Action**: Since Transformers do not have a recurrence mechanism like RNNs, positional encodings are added to the embeddings to give the model information about the order of the words.
- **Example**: The vectors from the previous step are modified to encode the position of each word in the sentence.

##### 3. Encoder
- **Action**: The encoder processes the sentence in parallel. It consists of multiple layers, each containing two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. 
    - **Self-Attention**: Helps the encoder look at other words in the input sentence while encoding a specific word.
    - **Feed-Forward Network**: Processes the output from the attention mechanism.
- **Example**: Each word's embedding, now with positional information, is transformed, considering the context provided by the rest of the sentence.

##### 4. Decoder
- **Action**: The decoder also has multiple layers but with an additional sub-layer for attention over the encoder's output. During training, the decoder is given the target sentence up to the current word and predicts the next word.
    - **Masked Self-Attention**: Prevents the decoder from peeking at future tokens in the target sentence.
    - **Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the input sentence.
- **Example**: Starts with the start-of-sentence token and predicts the first word in Baluchi, then uses the predicted words to predict the next word until the end-of-sentence token is predicted.

##### 5. Output Layer
- **Action**: The decoder's output is transformed into a vector of scores for each word in the target vocabulary. A softmax function is then applied to get probabilities for each word being the next correct word in the translation.
- **Example**: The output vector for the first predicted word is converted into probabilities, and the word with the highest probability is chosen as the first word of the translated sentence.

##### 6. Generation of the Translation
- **Action**: This process is repeated for each word in the target sentence until the model predicts the end-of-sentence token or reaches a maximum length.
- **Example**: `["Do", "mardom", "dóshi", "may", "lógá", "átk", "o", "gón", "má", "shámesh", "kort"]`

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
While OpenAI's tokenizer has potential, SentencePiece remains a strong contender due to its flexibility in handling diacritics, pre-trained models, and ease of integration.

The integration of tokenization into the pipeline of transformer-based language models, such as BERT, GPT (Generative Pretrained Transformer), and T5 (Text-to-Text Transfer Transformer), has marked a significant milestone in NLP. These models leverage advanced tokenization techniques to preprocess text, which is then fed into deep neural networks to perform a wide range of tasks, including translation, text generation and text summarization.

Languages with high morphological complexity benefit from efficient tokenization strategies. Experiment with different subword tokenization methods (e.g., Byte Pair Encoding (BPE), SentencePiece) that can better capture the morphology of the language. Adjusting the tokenization can significantly impact the model’s ability to understand and translate text in morphologically rich languages.

Tokenization can be done at the word level, character level, or subword level:
- **Word-level:** This is simple and interpretable, but might miss nuances of morphology and diacritics.
- **Character-level:** Captures all information, including diacritics, but can be less efficient for large models and might not handle complex morphology well.
- **Subword-level:** Offers a balance between granularity and efficiency, handling diacritics and morphology while being suitable for large models.

- **Some Considerations about Tokenizers:**
-  Does the tokenizer offer pre-trained models for languages similar to Baluchi?
-  Does it have good documentation and support for handling diacritics?
-  Is the tokenizer easy to use with the chosen NMT framework?





- **Tools and Tutorials**: Utilizing pre-existing tools like Wordpiece or SentencePiece for preprocessing and tokenization tasks. [Tokenizer Tutorial](https://huggingface.co/transformers/v3.4.0/tokenizer_summary.html)

### 2. Framework and Model Selection
- **Open-source Options**: Exploration of [Marian NMT](https://marian-nmt.github.io/), [FairSeq](https://github.com/facebookresearch/fairseq), [OpenNMT](https://github.com/OpenNMT), and various Transformer architectures like [The base Transformer](https://huggingface.co/docs/transformers/en/index), [Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl), and [T5](https://paperswithcode.com/method/t5).

### 3. Training the Model
- **Techniques**: Fine-tuning pre-trained models, hyperparameter tuning, and monitoring training progress with metrics like BLEU and ROUGE scores.

## Part 4: Evaluation and Refinement
- **Testing and Refinement**: Evaluating the model on a held-out set of Baluchi-English sentences and refining based on performance and error analysis.

## Conclusion
**Summary and Future Directions**: A recap of the key points and insights into potential areas for further research and development.
