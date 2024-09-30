import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-9\s]')

# Text Normalization
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(' ', lower)
        no_non_ascii = NON_ASCII.sub('', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

# Tokenize and Pad Sequences
def tokenize_and_pad(texts, max_features, max_length):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_texts, tokenizer
