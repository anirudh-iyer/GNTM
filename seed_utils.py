### seed_utils.py
# Utility to read your Excel of (token, topic) seeds and map into vocab indices
import os, re
import pandas as pd
from settings import INSTAGRAM_ADDR

def load_seeds(seed_excel_path, vocab):
    """
    Reads an Excel file with columns ['token','topic'],
    strips any trailing numbers from each token (e.g. 'colonization 347' ? 'colonization'),
    maps each distinct topic to an integer ID, and returns:
      - seed_dict: { topic_id: [vocab_index, ...] }
      - topic2id:  { topic_name: topic_id }
    """
    # load only the token (col A) and topic (col C)
    df = pd.read_excel(
        os.path.join(INSTAGRAM_ADDR, seed_excel_path),
        header= None,
        names=['token','topic']
    )
    # strip trailing numbers
    #df['token'] = df['token'].astype(str).str.replace(r'\s*\d+$', '', regex=True)

    df['token'] = df['token'].astype(str).str.replace(r'(^\d+\s+|\s+\d+$)', '', regex=True)


    # map topics ? IDs
    topic_names = df['topic'].unique().tolist()
    topic2id    = { name: idx for idx, name in enumerate(topic_names) }

    # build seed_dict
    seed_dict = { tid: [] for tid in topic2id.values() }
    for _, row in df.iterrows():
        token = row['token']
        tid   = topic2id[row['topic']]
        vidx  = vocab.get(token, -1)
        if vidx > 0:
            seed_dict[tid].append(vidx)

    return seed_dict, topic2id
