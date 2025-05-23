import sys
import os
import time
import argparse
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.cuda
from torch_geometric.data import DataLoader
from modules import *  # contains GDGNNModel
from dataPrepare import GraphDataset, MyData
from settings import *  # ROOTPATH, DATAPATH, etc.
from logger import Logger
from utils import *  # eval_topic, eval_top_doctopic
from seed_utils import load_seeds
import matplotlib.pyplot as plt
import optuna  # for trial.prune and reporting

clip_grad = 20.0
decay_epoch = 5
lr_decay = 0.8
max_decay = 5
ANNEAL_RATE = 0.00003


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instagram')
    parser.add_argument('--model_type', type=str, default='GDGNNMODEL')
    parser.add_argument('--prior_type', type=str, default='Dir2')
    parser.add_argument('--enc_nh', type=int, default=128)
    parser.add_argument('--num_topic', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--init_mult', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--ni', type=int, default=300)
    parser.add_argument('--nw', type=int, default=300)
    parser.add_argument('--fixing', action='store_true', default=True)
    parser.add_argument('--STOPWORD', action='store_true', default=True)
    parser.add_argument('--nwindow', type=int, default=5)
    parser.add_argument('--prior', type=float, default=0.5)
    parser.add_argument('--num_samp', type=int, default=1)
    parser.add_argument('--MIN_TEMP', type=float, default=0.3)
    parser.add_argument('--INITIAL_TEMP', type=float, default=1.0)
    parser.add_argument('--maskrate', type=float, default=0.5)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--word', action='store_true', default=True)
    parser.add_argument('--variance', type=float, default=0.995)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience in epochs')
    parser.add_argument('--seed_excel', type=str, default=None,
                        help="Path to Excel file of (token,topic) seeds")
    parser.add_argument('--seed_weight', type=float, default=1.0,
                        help="Strength of seed-guided regularization")

    args = parser.parse_args()
    # Ensure temp is initialized
    args.temp = args.INITIAL_TEMP

    # Build save dir
    save_dir = os.path.join(ROOTPATH, 'models', args.dataset, args.dataset + '_' + args.model_type)
    opt_str = f"_{args.optimizer}_m{args.momentum:.2f}_lr{args.learning_rate:.4f}"
    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]

    model_str = f"_{args.model_type}_ns{args.num_samp}_ench{args.enc_nh}_ni{args.ni}_nw{args.nw}_ngram{args.nwindow}_temp{args.INITIAL_TEMP:.2f}-{args.MIN_TEMP:.2f}"
    id_ = f"{args.dataset}_topic{args.num_topic}{model_str}_prior_type{args.prior_type}_{args.prior:.2f}{opt_str}_{args.taskid}_{args.seed}_stop{args.STOPWORD}_fix{args.fixing}_word{args.word}"
    save_dir = os.path.join(save_dir, id_)
    os.makedirs(save_dir, exist_ok=True)

    args.save_dir = save_dir
    args.save_path = os.path.join(save_dir, 'model.pt')
    args.log_path = os.path.join(save_dir, 'log.txt')

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.cuda = 'cuda' in args.device
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return args


def test(model, test_loader, mode='VAL', verbose=True):
    model.eval()
    num_sent = total_loss = loss_count = 0
    for batch in test_loader:
        batch = batch.to(model.args.device)
        bs = batch.y.size(0)
        outputs = model.loss(batch)
        if 'loss' in outputs:
            total_loss += outputs['loss'].item() * bs
            loss_count += bs
        num_sent += bs
    if verbose and loss_count > 0:
        avg = total_loss / loss_count
        print(f"{mode} avg loss: {avg:.4f}")
        return avg
    return None


def learn_feature(model, loader):
    model.eval()
    thetas, labels = [], []
    for batch in loader:
        batch = batch.to(model.args.device)
        theta = model.get_doctopic(batch)
        thetas.append(theta)
        labels.append(batch.y)
    return torch.cat(thetas).detach(), torch.cat(labels).detach()

def count_posts_per_seed_topic(thetas: torch.Tensor, seed_dict: dict) -> dict:
    """
    Count how many documents are most associated with each seed topic.

    Args:
        thetas (Tensor): shape [num_docs, num_topics], document-topic distributions
        seed_dict (dict): {topic_id: [seed_word_ids]} from the seed file

    Returns:
        dict {seed_topic_id: count of documents assigned to that topic}
    """
    # Step 1: Find the dominant topic for each document
    dominant_topics = thetas.argmax(dim=1).cpu().numpy()  # shape [num_docs]

    # Step 2: Count how many documents are assigned to each seed topic
    seed_topic_ids = seed_dict.keys()
    topic_counts = {k: 0 for k in seed_topic_ids}
    for t in dominant_topics:
        if t in topic_counts:
            topic_counts[t] += 1

    return topic_counts

def plot_seed_topic_counts(seed_topic_counts: dict, save_dir: str):
    import matplotlib.pyplot as plt
    keys, values = zip(*sorted(seed_topic_counts.items()))
    plt.figure(figsize=(8, 4))
    plt.bar(keys, values, color='lightblue', edgecolor='black')
    plt.xlabel("Seed Topic ID")
    plt.ylabel("Document Count")
    plt.title("Documents per Seed Topic")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot_seed_topic_counts.png"))
    plt.close()


def average_confidence_per_seed_topic(thetas: torch.Tensor, seed_dict: dict) -> dict:
    """
    Compute average confidence (theta_k) for documents assigned to each seed topic.

    Args:
        thetas (Tensor): [num_docs, num_topics]
        seed_dict (dict): {topic_k: [seed word ids]}

    Returns:
        dict {topic_k: average theta_k for docs assigned to k}
    """
    dominant_topics = thetas.argmax(dim=1).cpu()  # [num_docs]
    confidences = thetas.cpu()  # [num_docs, num_topics]

    topic_conf = {k: [] for k in seed_dict}
    for i, k in enumerate(dominant_topics):
        if k.item() in topic_conf:
            topic_conf[k.item()].append(confidences[i, k.item()].item())

    # Now compute the average
    avg_confidence = {
        k: (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
        for k, vals in topic_conf.items()
    }
    return avg_confidence

def plot_avg_confidence(avg_conf: dict, save_dir: str):
    keys, values = zip(*sorted(avg_conf.items()))
    plt.figure(figsize=(8, 4))
    plt.bar(keys, values, color='orange', edgecolor='black')
    plt.xlabel("Seed Topic ID")
    plt.ylabel("Average Î¸_k")
    plt.title("Average Confidence per Seed Topic")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot_avg_confidence.png"))
    plt.close()


def compute_topic_entropy(thetas: torch.Tensor) -> torch.Tensor:
    """
    Computes entropy of topic distribution for each document.

    Args:
        thetas (Tensor): shape [num_docs, num_topics]

    Returns:
        Tensor: shape [num_docs], entropy per document
    """
    epsilon = 1e-10  # To avoid log(0)
    entropies = - (thetas * (thetas + epsilon).log()).sum(dim=1)
    return entropies.cpu()  # [num_docs]

def plot_entropy_histogram(entropies: torch.Tensor, save_dir: str):
    plt.figure(figsize=(8, 4))
    plt.hist(entropies.numpy(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Entropy')
    plt.ylabel('Number of Documents')
    plt.title('Topic Entropy Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot_topic_entropy_histogram.png"))
    plt.close()


def count_multi_seed_overlap(thetas: torch.Tensor, seed_dict: dict, threshold: float = 0.2) -> int:
    """
    Count how many documents are associated with more than one seed topic (above a threshold).

    Args:
        thetas (Tensor): [num_docs, num_topics]
        seed_dict (dict): {topic_id: [seed_word_ids]}
        threshold (float): minimum theta_k to count as topic match

    Returns:
        int: number of documents with >1 seed topic above threshold
    """
    seed_topics = list(seed_dict.keys())
    relevant_thetas = thetas[:, seed_topics]  # [num_docs, #seed_topics]
    topic_mask = relevant_thetas > threshold  # [num_docs, #seed_topics]
    overlap_counts = topic_mask.sum(dim=1)    # [num_docs], how many seed topics each doc exceeds threshold
    multi_topic_docs = (overlap_counts > 1).sum().item()
    return multi_topic_docs

def plot_overlap_across_thresholds(thetas: torch.Tensor, seed_dict: dict, save_dir: str):
    thresholds = [0.1, 0.2, 0.3, 0.4]
    overlap_counts = [
        count_multi_seed_overlap(thetas, seed_dict, threshold=t) for t in thresholds
    ]
    plt.figure(figsize=(8, 4))
    plt.bar([str(t) for t in thresholds], overlap_counts, color='purple')
    plt.xlabel("Threshold")
    plt.ylabel("# Docs with >1 Seed Topic")
    plt.title("Multi-Seed Topic Overlap by Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot_multi_seed_overlap.png"))
    plt.close()


def topic_hate_label_distribution(thetas: torch.Tensor, labels: torch.Tensor, seed_dict: dict) -> dict:
    """
    For each seed topic, compute how many of its assigned documents are labeled as hate.

    Args:
        thetas (Tensor): [num_docs, num_topics]
        labels (Tensor): [num_docs], 0 = non-hate, 1 = hate
        seed_dict (dict): {topic_k: [seed_word_ids]}

    Returns:
        dict {topic_k: {'total': int, 'hate': int, 'pct_hate': float}}
    """
    dominant_topics = thetas.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    topic_stats = {}

    for k in seed_dict:
        indices = np.where(dominant_topics == k)[0]
        if len(indices) == 0:
            topic_stats[k] = {'total': 0, 'hate': 0, 'pct_hate': 0.0}
            continue
        hate_count = labels[indices].sum()
        total = len(indices)
        pct = hate_count / total
        topic_stats[k] = {'total': total, 'hate': int(hate_count), 'pct_hate': pct}

    return topic_stats

def plot_topic_hate_ratio(topic_hate_stats: dict, save_dir: str):
    import matplotlib.pyplot as plt
    topic_ids = sorted(topic_hate_stats.keys())
    hate_counts = [topic_hate_stats[k]['hate'] for k in topic_ids]
    non_hate_counts = [topic_hate_stats[k]['total'] - topic_hate_stats[k]['hate'] for k in topic_ids]

    plt.figure(figsize=(8, 5))
    plt.barh(topic_ids, non_hate_counts, color='green', label='Non-Hate')
    plt.barh(topic_ids, hate_counts, left=non_hate_counts, color='red', label='Hate')
    plt.xlabel("Document Count")
    plt.ylabel("Seed Topic ID")
    plt.title("Hate vs Non-Hate Distribution per Seed Topic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plot_topic_hate_ratio.png"))
    plt.close()


def eval_doctopic(model, loader):
    thetas, labels = learn_feature(model, loader)
    eval_top_doctopic(thetas.cpu().numpy(), labels.cpu().numpy())


def main(args, trial=None):
    print(args)
    device = torch.device(args.device)
    path = todatapath(args.dataset)
    stop_str = '_stop' if args.STOPWORD else ''

    # Load dataset
    dataset = GraphDataset(path, ngram=args.nwindow, STOPWORD=args.STOPWORD)

    args.vocab = dataset.vocab
    args.vocab_size = len(dataset.vocab)

    train_idxs = [i for i in range(len(dataset)) if dataset[i].train == 1]
    val_idxs = [i for i in range(len(dataset)) if dataset[i].train == -1]
    test_idxs = [i for i in range(len(dataset)) if dataset[i].train == 0]
    train_data, val_data, test_data = dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]

    # Build loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              follow_batch=['x','edge_id','y'])
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False,
                              follow_batch=['x','edge_id','y'])
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False,
                              follow_batch=['x','edge_id','y'])

    #initialize counters for the 'warm up' phase inside GDGNNModel

    args.iter_ = 0
    args.iter_threahold = max(30 * len(train_loader), 2000)

    # Seed-guided topics
    seed_dict = {}
    if args.seed_excel:
        seed_dict, topic2id = load_seeds(args.seed_excel, dataset.vocab.word2id)
        print(f"Loaded seeds for topics: {topic2id}")
    else:
        seed_dict = {}
    # Word embeddings and edges
    whole_edge = torch.tensor(dataset.whole_edge, dtype=torch.long, device=device)
    word_vec = torch.from_numpy(np.load(os.path.join(path, f"{args.nw}d_words{stop_str}.npy"))).float()

    # Instantiate model
    model = GDGNNModel(args, word_vec=word_vec, whole_edge=whole_edge, seed_dict = seed_dict, seed_weight=args.seed_weight).to(device)

    # Evaluation mode
    if args.eval:
        if args.load_path:
            model.load_state_dict(torch.load(args.load_path, map_location=device))
        else:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
        test(model, test_loader, 'TEST')
        # print topics
        try:
            data = pd.read_csv(os.path.join(path, f"overall{stop_str}.csv"), dtype={'label':int,'train':int})
        except FileNotFoundError:
            data = pd.read_csv(os.path.join(path, 'overall_clean.csv'), dtype={'label':int,'train':int})
        common_texts = data['content'].tolist()
        beta = model.get_beta().detach().cpu().numpy()
        print("---------------Printing the Topics------------------")
        eval_topic(beta, [dataset.vocab.id2word(i) for i in range(len(dataset.vocab))], common_texts=common_texts)
        print("---------------End of Topics------------------")

        # Step: Analyze document-topic assignments
        thetas, labels = learn_feature(model, test_loader)
        seed_topic_counts = count_posts_per_seed_topic(thetas, model.seed_dict)

        print("\n--- Document Counts for Each Seed Topic ---")
        for k, count in seed_topic_counts.items():
            print(f"Seed Topic {k}: {count} documents")

        avg_conf = average_confidence_per_seed_topic(thetas, model.seed_dict)

        print("\n--- Average Confidence (theta_k) for Each Seed Topic ---")
        for k, conf in avg_conf.items():
            print(f"Seed Topic {k}: {conf:.4f}")

        entropies = compute_topic_entropy(thetas)

        print("\n--- Topic Entropy Statistics ---")
        print(f"Mean Entropy: {entropies.mean():.4f}")
        print(f"Max Entropy:  {entropies.max():.4f}")
        print(f"Min Entropy:  {entropies.min():.4f}")

        plt.figure(figsize=(8, 4))
        plt.hist(entropies.numpy(), bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Topic Entropy')
        plt.ylabel('Number of Documents')
        plt.title('Distribution of Topic Entropy per Document')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'topic_entropy_histogram.png'))


        pd.DataFrame.from_dict(seed_topic_counts, orient='index', columns=['doc_count']).to_csv(os.path.join(args.save_dir, 'seed_topic_doc_counts.csv'))

#        multi_seed_overlap = count_multi_seed_overlap(thetas, model.seed_dict, threshold=0.2)

        for thresh in [0.1, 0.2, 0.3, 0.4]:
            overlap = count_multi_seed_overlap(thetas, model.seed_dict, threshold=thresh)
            print(f"Threshold {thresh:.1f}: {overlap} documents")

        topic_hate_stats = topic_hate_label_distribution(thetas, labels, model.seed_dict)

        print("\n--- Hate Label Distribution by Seed Topic ---")
        for k, stats in topic_hate_stats.items():
            print(f"Seed Topic {k}: {stats['hate']}/{stats['total']} = {stats['pct_hate']:.2%} hate")

        pd.DataFrame(topic_hate_stats).T.to_csv(os.path.join(args.save_dir, 'seed_topic_hate_label_distribution.csv'))

        plot_seed_topic_counts(seed_topic_counts, args.save_dir)
        plot_avg_confidence(avg_conf, args.save_dir)
        plot_entropy_histogram(entropies, args.save_dir)
        plot_overlap_across_thresholds(thetas, model.seed_dict, args.save_dir)
        plot_topic_hate_ratio(topic_hate_stats, args.save_dir)
        # Optional:
#        plot_venn_for_seed_pairs(thetas, model.seed_dict, threshold=0.2, save_dir=args.save_dir)

        return
    
    

    # Optimizers
    enc_opt = torch.optim.Adam(model.enc_params, args.learning_rate,
                               betas=(args.momentum,0.999), weight_decay=args.wdecay)
    dec_opt = torch.optim.Adam(model.dec_params, args.learning_rate,
                               betas=(args.momentum,0.999), weight_decay=args.wdecay)

    best_loss = float('inf')
    no_improve = 0
    epoch_val_losses = []

    # Training loop
    for epoch in range(args.num_epoch):
        model.train()
        iter_count = 0
        for batch in train_loader:
            args.iter_ += 1
            batch = batch.to(device)
            outputs = model.loss(batch)
            loss = outputs['loss']
            enc_opt.zero_grad(); dec_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            # alternate train
            if epoch % 2 == 0:
                dec_opt.step()
            else:
                enc_opt.step()
            iter_count += 1
        # validation
        model.eval()
        val_loss = test(model, val_loader, 'VAL') or float('inf')
        epoch_val_losses.append(val_loss)
        print(f"After epoch {epoch}, validation loss: {val_loss:.4f}")
        # display topics
        if epoch % 5 == 0:
            try:
                data = pd.read_csv(os.path.join(path, f"overall{stop_str}.csv"), dtype={'label':int,'train':int})
            except FileNotFoundError:
                data = pd.read_csv(os.path.join(path, 'overall_clean.csv'), dtype={'label':int,'train':int})
            common_texts = data['content'].tolist()
            beta = model.get_beta().detach().cpu().numpy()
            print("---------------Printing the Topics------------------")
            eval_topic(beta, [dataset.vocab.id2word(i) for i in range(len(dataset.vocab))], common_texts=common_texts)
            print("---------------End of Topics------------------")
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break
        # optuna pruning
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # finalize
    print(f"Best validation loss: {best_loss}")
    # plot curve
    plt.figure(); plt.plot(epoch_val_losses, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss'); plt.title('Validation Loss Curve')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'validation_loss_curve.png'))

    return best_loss, epoch_val_losses


if __name__ == '__main__':
    args = init_config()
    if not args.eval:
        sys.stdout = Logger(args.log_path)
    result = main(args)
    if not args.eval and isinstance(result, tuple):
        best_loss, _ = result
        print(f"Final best validation loss: {best_loss}")
