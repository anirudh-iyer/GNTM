# build_embeddings.py
import numpy as np
import os
import argparse
from settings import INSTAGRAM_ADDR, GLOVE_ADDR

def build_embedding_matrix(vocab_file, glove_file, output_file, embedding_dim=300):
    # Build vocabulary dictionary from vocab_file.
    word2id = {}
    with open(vocab_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                word2id[word] = len(word2id) + 1  # start index at 1
    print(f"Vocabulary size: {len(word2id)}")

    # Initialize embedding matrix with small random values
    emb_matrix = np.random.randn(len(word2id) + 1, embedding_dim) * 0.01

    # Load GloVe and fill in known words
    print("Loading GloVe embeddings?")
    found = 0
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            try:
                vec = np.asarray(vals[1:], dtype='float32')
                if vec.shape[0] != embedding_dim:
                    continue
            except ValueError:
                continue
            if word in word2id:
                emb_matrix[word2id[word]] = vec
                found += 1
    print(f"Found {found}/{len(word2id)} words in GloVe.")

    # Save
    np.save(output_file, emb_matrix)
    print("Saved embedding matrix to", output_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--STOPWORD", action="store_true", help="Whether to use the _stop suffix")
    args = p.parse_args()

    embedding_dim = 300
    stop_str = "_stop" if args.STOPWORD else ""
    vocab_file = os.path.join(INSTAGRAM_ADDR, "vocab.txt")
    glove_file = os.path.join(GLOVE_ADDR, "glove.840B.300d.txt")
    output_file = os.path.join(INSTAGRAM_ADDR, f"{embedding_dim}d_words{stop_str}.npy")

    build_embedding_matrix(vocab_file, glove_file, output_file, embedding_dim)
