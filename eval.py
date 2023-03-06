import pickle5 as pickle
import numpy as np
import re
import logging
import pandas as pd
import hydra
import logging
from tqdm import tqdm
from sklearn.metrics import euclidean_distances
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from omegaconf import DictConfig
from rich.console import Console



log = logging.getLogger(__name__)

# def data_formatter(x):
#     x[0] = 'summarize: ' + x[0]
#     return pd.Series([x[0], x[1]], index=['source_text', 'target_text'])


def evaluate_by_correspondance_euclid_dist_score(eval_model, texts, labels, label_to_id_map, id_to_label_map):
    console = Console()
    console.rule('[green] Doing base labels based GZS predictions')
    spacer_lambda  = lambda x: ' '.join(re.findall('[A-Z][^A-Z]*', x))
    y_hat = []
    y_true = []
    base_labels_embeddings = eval_model.encode(list(id_to_label_map.values()))
    for idx, (t, l) in tqdm(enumerate(zip(texts, labels)), total=len(texts)):
        p_emb = eval_model.encode(t)
        scores = euclidean_distances(p_emb.reshape(1, -1), base_labels_embeddings)[0].tolist()
        # scores = util.cos_sim(p_emb, base_labels_embeddings)[0].tolist()
        # scores = util.dot_score(p_emb, base_labels_embeddings)[0].tolist()
        try:
            # y_true.append(label_to_id_map[''.join(l.split())])
            y_true.append(label_to_id_map[l])
            y_hat_index = np.argmin(scores)
            y_hat.append(y_hat_index)
        except KeyError:
            continue
    from sklearn.metrics import f1_score
    log.info(f'[green bold] f1_score (macro): {f1_score(y_true=y_true, y_pred=y_hat, average="macro")}')
    log.info(f'[green bold] f1_score (micro): {f1_score(y_true=y_true, y_pred=y_hat, average="micro")}')
    log.info(f'[green bold] f1_score (weighted): {f1_score(y_true=y_true, y_pred=y_hat, average="weighted")}')

def evaluate_by_correspondance_cos_sim_score(eval_model, texts, labels, label_to_id_map, id_to_label_map):
    console = Console()
    console.rule('[green] Doing base labels based GZS predictions')
    spacer_lambda  = lambda x: ' '.join(re.findall('[A-Z][^A-Z]*', x))
    y_hat = []
    y_true = []
    base_labels_embeddings = eval_model.encode(list(id_to_label_map.values()))
    for idx, (t, l) in tqdm(enumerate(zip(texts, labels)), total=len(texts)):
        p_emb = eval_model.encode(t)
        #scores = euclidean_distances(p_emb, base_labels_embeddings).tolist()
        scores = util.cos_sim(p_emb, base_labels_embeddings)[0].tolist()
        # scores = util.dot_score(p_emb, base_labels_embeddings)[0].tolist()
        try:
            # y_true.append(label_to_id_map[''.join(l.split())])
            y_true.append(label_to_id_map[l])
            y_hat_index = np.argmax(scores)
            y_hat.append(y_hat_index)
        except KeyError:
            continue
    from sklearn.metrics import f1_score
    log.info(f'[green bold] f1_score (macro): {f1_score(y_true=y_true, y_pred=y_hat, average="macro")}')
    log.info(f'[green bold] f1_score (micro): {f1_score(y_true=y_true, y_pred=y_hat, average="micro")}')
    log.info(f'[green bold] f1_score (weighted): {f1_score(y_true=y_true, y_pred=y_hat, average="weighted")}')




def get_data_frames(data_dir, test_file_name, log, debug=False):
    csv_file_name = test_file_name
    columns = ['source_text', 'target_text']
    console = Console()
    console.rule('[green] Reading test data')
    test_data = pd.read_csv(Path(data_dir)/f'{csv_file_name}', names=columns, header=None, skiprows=[0, 1])
    test_data = pd.DataFrame(test_data[['source_text', 'target_text']])
    # test_data = test_data.apply(data_formatter, axis=1)
    log.info(f'There are {len(test_data)} samples in teh evaluation set. ')
    if debug is True:
        return test_data.sample(10)
    else:
        return test_data


@hydra.main(config_path="", config_name="")
def main(cfg: DictConfig):
    import pdb
    pdb.set_trace()
    import sys
    console = Console()
    log = logging.getLogger(f'{cfg.dataset_name}_baseline_log')
    log.info(sys.argv)
    log.info(cfg)
    # load data
    test_df = get_data_frames(cfg.data_dir, cfg.test_file_name, log, cfg.debug)
    console.rule('[green]Starting evaluation')
    correspondance_model = SentenceTransformer(cfg.st_model_name)
    texts = test_df['source_text']
    labels = test_df['target_text']
    log.info(f'evaluating baseline on {cfg.dataset_name} dataset')

    with open(cfg.labels_map, 'rb') as handle:
        base_labels_dict = pickle.load(handle)
        log.info(f'There are {len(base_labels_dict["id_to_label_map"].values())} unique labels in {cfg.dataset_name}')
        log.info(f'Similarity metric is {cfg.similarity_metric}')
        if cfg.similarity_metric == 'euclid_dist':
            evaluate_by_correspondance_euclid_dist_score(correspondance_model,texts, labels, base_labels_dict['label_to_id_map'], base_labels_dict['id_to_label_map'])
        elif cfg.similarity_metric == 'cos_sim':
            evaluate_by_correspondance_cos_sim_score(correspondance_model,texts, labels, base_labels_dict['label_to_id_map'], base_labels_dict['id_to_label_map'])


if __name__ == "__main__":
    main()