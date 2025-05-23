import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
#from dataPrepare.utils import *
from utils import *
import pandas as pd
from collections import Counter
from torch_sparse import SparseTensor
import numpy as np

class PreTextData(object):
    """docstring for MonoTextData"""

    def __init__(self, fname, ngram=3, min_length=10, max_length=None, vocab=None, edge_threshold=10):
        super(PreTextData, self).__init__()
        self.data, self.vocab, self.dropped, self.labels, self.word_count, self.train_split, self.itemids \
            = self._read_corpus(fname, vocab, max_length=max_length, min_length=min_length)
        self.ngram = ngram
        self.pairVocab(edge_threshold)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, vocab: VocabEntry, min_length=10, max_length=1000):
        labels = []
        data = []
        tran_split = []
        itemids = []
        dropped = 0
        word_count = 0
        csvdata = pd.read_csv(fname, header=0, dtype={'label': int, 'train': int})
        for i, ss in enumerate(csvdata[['label', 'content', 'train', 'idx']].values):
            lb = ss[0]
            try:
                split_line = ss[1].split()
            except:
                print(ss[1])
            if len(split_line) < min_length:
                dropped += 1
                continue
            if max_length is not None:
                if len(split_line) > max_length:
                    dropped += 1
                    continue
            idxs = [vocab[word] for word in split_line if vocab[word] > 0]
            word_num = len(idxs)
            if word_num < 3:
                dropped += 1
                continue
            labels.append(lb)
            data.append(idxs)
            itemids.append(ss[3])
            tran_split.append(int(ss[2]))
            word_count += word_num
        print('read corpus done!')
        return data, vocab, dropped, labels, word_count, tran_split, itemids

    def pairVocab(self, threshold=15):
        pair_s = []
        n = self.ngram
        for sent in self.data:
            L = len(sent)
            nl = min(n + 1, L)
            for i in range(1, nl):
                pair = np.array([sent[:-i], sent[i:]]).transpose()
                pair_s.append(pair)
        pairs = np.concatenate(pair_s, axis=0)
        tmp = [tuple(t) for t in pairs]
        coun_dct = Counter(tmp)
        self.pair_dct = {k: coun_dct[k] for k in coun_dct if coun_dct[k] > threshold and k[0] != k[1]}
        sorted_key = sorted(self.pair_dct.keys(), key=lambda x: self.pair_dct[x], reverse=True)
        for i, key in enumerate(sorted_key):
            self.pair_dct[key] = i + 1  # start from 1
        self.whole_edge = np.array([k for k in sorted_key]).transpose()  # edge: i->j
        self.whole_edge_w = np.array([coun_dct[k] for k in sorted_key])
        print('pairVocab done!')
        print(self.whole_edge.shape)

    def process_sent(self, sent):
        n = self.ngram
        L = len(sent)
        pair_s = []
        edge_ids = []
        nl = min(n + 1, L)
        for i in range(1, nl):
            pair = np.array([sent[:-i], sent[i:]]).transpose()
            pair_s.append(pair)
        pairs = np.concatenate(pair_s, axis=0)
        tmp = [tuple(t) for t in pairs]
        dct = Counter(tmp)
        keys = dct.keys()
        r, c, v = [], [], []
        for k in keys:
            try:
                edge_id = self.pair_dct[k]
            except:
                continue
            r.append(k[0])
            c.append(k[1])
            v.append(dct[k])
            edge_ids.append(edge_id)
        edge_index = np.array([r, c])  # edge: i->j
        edge_w = np.array(v)
        idxs = np.unique(edge_index.reshape(-1))
        idx_w_dict = Counter(sent)
        idx_w = []
        lens = 0
        for id in idxs:
            idx_w.append(idx_w_dict[id])
            lens += idx_w_dict[id]
        sidxs = []
        for id in sent:
            if id not in idxs and id not in sidxs:
                sidxs.append(id)
                idx_w.append(idx_w_dict[id])
                lens += idx_w_dict[id]
        if len(idxs) > 0 and len(sidxs) > 0:
            all_idxs = np.hstack([idxs, sidxs])
        elif len(idxs) == 0 and len(sidxs) > 0:
            all_idxs = np.array(sidxs)
        else:
            all_idxs = idxs
        assert lens == len(sent)
        if len(idxs) > 0:
            idxs_map = np.zeros(max(all_idxs) + 1)
            idxs_map[all_idxs] = range(len(all_idxs))
            edge_index = idxs_map[edge_index]
        else:
            edge_index = np.array([[], []])
        return all_idxs, idx_w, edge_index, edge_w, edge_ids, L


class MyData(Data):
    def __init__(self, x=None, edge_w=None, edge_index=None, x_w=None, edge_id=None, y=None):
        super(MyData, self).__init__()
        if x is not None:
            self.x = x
        if edge_w is not None:
            self.edge_w = edge_w
        if edge_index is not None:
            self.edge_index = edge_index
        if x_w is not None:
            self.x_w = x_w
        if edge_id is not None:
            self.edge_id = edge_id
        if y is not None:
            self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        elif key == 'x':
            return 0
        elif key == 'edge_id':
            return 0
        else:
            return 0


class GraphDataset(InMemoryDataset):
    def __init__(self, root, ngram=3, vocab=None, transform=None, pre_transform=None, STOPWORD=False, edge_threshold=10):
        self.rootPath = root
        self.stop_str = '_stop' if STOPWORD else ''  
        self.edge_threshold = edge_threshold
        if vocab is None:
            # Use our custom VocabEntry from utils (if available) to read overall_clean.csv
            self.vocab = VocabEntry.from_corpus(self.rootPath + '/vocab.txt', withpad=False)
        else:
            self.vocab = vocab
        self.ngram = ngram
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.whole_edge, self.word_count, self.dropped, self.whole_edge_w = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['graph_nragm%d_dataset%s.pt' % (self.ngram, self.stop_str)]

    def download(self):
        pass

    def process(self):
        # Use the overall_clean.csv file (which is our processed Instagram data)
        dataset = PreTextData(self.rootPath + '/overall_stop.csv', ngram=self.ngram,
                               vocab=self.vocab, min_length=5, max_length=None,
                               edge_threshold=self.edge_threshold)
        data_list = []
        used_list = []
        for i in range(len(dataset)):
            sent = dataset.data[i]
            label = dataset.labels[i]
            train = dataset.train_split[i]
            idxs, idx_w, edge_index, edge_w, edge_id, L = dataset.process_sent(sent)
            if edge_index.shape[1] >= 0:
                used_list.append(dataset.itemids[i])
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                x = torch.tensor(idxs, dtype=torch.long)
                edge_w = torch.tensor(edge_w, dtype=torch.float)
                y = torch.tensor(label, dtype=torch.long).unsqueeze(0)
                train = torch.tensor(train, dtype=torch.long).unsqueeze(0)
                idx_w = torch.tensor(idx_w, dtype=torch.float)
                edge_id = torch.tensor(edge_id, dtype=torch.long)
                d = MyData(x=x, edge_w=edge_w, edge_index=edge_index,
                           x_w=idx_w, edge_id=edge_id, y=y)
                d.train = train
                d.graphy = y
                data_list.append(d)
        np.save(self.rootPath + '/used_list', used_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices, dataset.whole_edge, dataset.word_count, dataset.dropped, dataset.whole_edge_w),
                   self.processed_paths[0])

if __name__ == "__main__":
    import argparse
    from settings import INSTAGRAM_ADDR

    p = argparse.ArgumentParser()
    p.add_argument("--root", default=INSTAGRAM_ADDR)
    p.add_argument("--ngram", type=int, default=5)
    p.add_argument("--edge_threshold", type=int, default=10)
    p.add_argument("--STOPWORD", action="store_true")
    args = p.parse_args()

    ds = GraphDataset(
        root=args.root,
        ngram=args.ngram,
        STOPWORD=args.STOPWORD,
        edge_threshold=args.edge_threshold
    )
    print("→ Processed dataset:")
    print("  documents:", len(ds))
    print("  edges:", ds.whole_edge.shape)
