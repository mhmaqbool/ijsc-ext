
import sys
from pathlib import Path
import pandas as pd
import hydra
import logging
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# global patience
# patience = 5
# global cur_patience
# cur_patience = 5
# global best_score
# best_score = 0.0



def get_examples(data_dir, train_data_file_name, debug=False):
    examples = []
    data_file_path = Path(data_dir, train_data_file_name)
    data_frame = pd.read_csv(data_file_path)
    for _, _, text_1, text_2, label in data_frame.itertuples():
            i_ex = InputExample(texts=[text_1, text_2], label=label)
            examples.append(i_ex)
    if debug is True:
        return examples[:100]
    log.info(f'There are {len(examples)} samples in {data_file_path}')
    return examples



def train(model, train_dataloader, evaluator, train_loss, epochs, warmup_steps, cp_dir, evaluation_steps, patience, save_best_model, steps_per_epoch):

    model.fit([(train_dataloader, train_loss)], evaluator=evaluator, warmup_steps=warmup_steps, epochs=epochs,  output_path=cp_dir, evaluation_steps=evaluation_steps, save_best_model=save_best_model, steps_per_epoch=steps_per_epoch, show_progress_bar=True)


@hydra.main(config_path="", config_name="")
def main(cfg: DictConfig):
    log.info(f'train command: python {" ".join(sys.argv)} ')
    log.info(cfg)
    if cfg.model_load_dir is not None:
        # this is few-shot training, we should load the model from this cfg.cp_dir
        model = SentenceTransformer(cfg.model_load_dir)
    else:
        
        model = SentenceTransformer(cfg.st_model_name)
    train_data_dir = Path(cfg.data_dir, cfg.train_dataset_name)
    dev_data_dir = Path(cfg.dev_data_dir)/Path(cfg.dev_dataset_name)
    train_examples = get_examples(train_data_dir, cfg.train_data_file_name, cfg.debug)
    dev_examples = get_examples(dev_data_dir, cfg.dev_data_file_name, cfg.debug)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='dev')
    if cfg.train_loss == 'contrastive_loss':
        train_loss = losses.ContrastiveLoss(model=model)
    elif cfg.train_loss == 'consine_sim_loss':
        train_loss = losses.CosineSimilarityLoss(model=model)
    log.info(f'Intiating contrastive training on {cfg.train_dataset_name} with ST model {cfg.st_model_name}, validating on {cfg.dev_dataset_name}')
    train(model, train_dataloader, evaluator, train_loss, cfg.epochs, cfg.warmup_steps, cfg.cp_dir, cfg.evaluation_steps, cfg.patience, cfg.save_best_model, cfg.steps_per_epoch)

if __name__ == "__main__":
    main()