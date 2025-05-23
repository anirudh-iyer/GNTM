import sys
sys.path.append("../")
from settings import *  # INSTAGRAM_ADDR, etc.
import pandas as pd
import numpy as np
import os
import re
import string

np.random.seed(6)

import re
import string

def load_stopwords(file_path):
    """Load stop words from a file into a Python set."""
    stop_words = set()
    with open(os.path.join(file_path, "EN_gensim_stopword.txt"), encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stop_words.add(word)
    return stop_words

def clean_text(text, stop_words=None):
    """Clean text by lowercasing, removing URLs, punctuation, extra whitespace,
    and filtering out stop words if provided."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    if stop_words is not None:
        words = [w for w in words if w not in stop_words]
    # Remove extra whitespace (if any remains) and join back into a string.
    return " ".join(words)

def process_dataset(dataset='Instagram', STOPWORD=False):
    if dataset != 'Instagram':
        raise ValueError("Only the Instagram dataset is supported in this version.")
    
    # Use the Instagram data path from settings
    path = INSTAGRAM_ADDR
#    data_file = os.path.join(path, "ig_desc.csv")
    data_file = os.path.join(path, "pre_oct7.csv")
#    data_file = os.path.join(path, "post_oct7.csv")

    print("read data")
    data = pd.read_csv(data_file, header=0)
    data = data.reset_index(drop=True)
    data['idx'] = data.index
    
    # Load stop words from EN_STOP_WORDS if you want to filter them out.
    stop_words = load_stopwords(EN_STOP_WORDS)
    
    # Clean the content and filter out stop words.
    data['content'] = data['content'].fillna("").astype(str).str.strip()
    data['content'] = data['content'].apply(lambda x: clean_text(x, stop_words=stop_words))
    
    # Optionally, drop rows that are empty after cleaning.
    data = data[data['content'] != '']

    np.random.seed(42)
    rand = np.random.rand(len(data))
    data.loc[rand < 0.70, 'train'] = 1
    data.loc[(rand >= 0.70) & (rand < 0.85), 'train'] = -1
    data.loc[rand >= 0.85, 'train'] = 0
    # Print unique values in 'column_name'
    unique_values = data['train'].unique()
    print(unique_values)
    
    vocab = {}
    for text in data['content']:
        for word in text.split():
            vocab[word] = vocab.get(word, 0) + 1

    return data, vocab

    
# Generate vocabulary file from overall_clean.csv
def generate_vocab_file_in_preprocess(data, data_path, freq_threshold=5):
    vocab = {}
    for text in data["content"]:
        for word in text.strip().split():
            vocab[word] = vocab.get(word, 0) + 1
    vocab = {word: freq for word, freq in vocab.items() if freq > freq_threshold}
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    out_fname = os.path.join(data_path, "vocab.txt")
    with open(out_fname, "w", encoding="utf-8") as f:
        for word, freq in sorted_vocab:
            f.write(f"{word} {freq}\n")
    print("Filtered Vocab file saved to:", out_fname)

if __name__ == '__main__':
    data, vocab = process_dataset(dataset='Instagram', STOPWORD=True)
    print("Number of documents:", len(data))
    print("Unfiltered Vocabulary size:", len(vocab))
    output_file = os.path.join(INSTAGRAM_ADDR, "overall_stop.csv")
    data.to_csv(output_file, index=False, quoting=1)
    print("Processed data saved to:", output_file)
    
    generate_vocab_file_in_preprocess(data, INSTAGRAM_ADDR, freq_threshold=5)
