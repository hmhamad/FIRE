from abc import ABC, abstractmethod
import os
from utils import adjust_for_random_iterations, RE_logger

class ModelWrapper(ABC):
    def __init__(self, exp_cfgs):
        self.exp_cfgs = exp_cfgs
        self.re_logger = RE_logger(exp_cfgs)

    @abstractmethod
    def train(self, model_path, train_path, valid_path, output_path):
        """
        Implementaion should:
        * Load the model from model_path in train mode
        * Train the model on data from train_path
        * Perform validation at end of each epoch on data from valid path
        * Save the following results into output_path directory:
            ** Save the model with best validation f1_with_ner score into best_model/ directory (includes at least: pytorch_model.bin, config.json and tokenizer_config.json)
            ** Keep track of at least the following four metrics across epochs and save them in file F1_epochs.csv: epoch_num, NER_Micro_F1, Best_NER_Micro_F1, Rel_Valid_Micro_F1, Best_Rel_Valid_Micro_F1, Rel+_Valid_Micro_F1, Best_Rel+_Valid_Micro_F1
        """
        pass

    @abstractmethod
    def eval(self, model_path, dataset_path, output_path):
        """
        Implementation should:
        * Load model from model_path in eval mode
        * Evaluate model on data from dataset_path
        * Compute the following three performance metrics and save them in file F1_{data_label}.json inside directory output_path: ner_f1, rel_f1, rel_f1_with_ner
        * Save the predicted data in file {data_label}_predictions.json inside directory output_path
            * data format should follow the standard format
            * append to each predicted entity: logit score, probability score and top2 score as following: {'score': {'logit': logit_score, 'prob': prob_score, 'top2': top2_score}}
            * append to each predicted relation: logit score and probability score as following: {'score': {'logit': logit_score, 'prob': prob_score}}
            * append to each predicted relation a unique 'embedding_id' field
        """
        pass

    @abstractmethod
    def predict(self, model_path, dataset_path, output_path):
        """
        Implementation should:
        * Load model from model_path in eval mode
        * Pass data from dataset_path through model
        * Save the predicted data in file {data_label}_predictions.json inside directory output_path
            * data format should follow the standard format
            * append to each predicted entity: logit score, probability score and top2 score as following: {'score': {'logit': logit_score, 'prob': prob_score, 'top2': top2_score}}
            * append to each predicted relation: logit score and probability score as following: {'score': {'logit': logit_score, 'prob': prob_score}}
            * append to each predicted relation a unique embedding_id field
        """
        pass

    def execute(self):
        if self.exp_cfgs.mode == 'train':
            # Loop over each random initialization
            model_path = self.exp_cfgs.model_path
            for it in range(self.exp_cfgs.n_random_iter):
                self.exp_cfgs = adjust_for_random_iterations(
                    it, self.exp_cfgs)
                self.train(model_path, self.exp_cfgs.train_path,
                           self.exp_cfgs.valid_path, self.exp_cfgs.log_path)
                trained_model_path = os.path.join(
                    self.exp_cfgs.log_path, 'best_model')
                self.eval(trained_model_path, self.exp_cfgs.test_path,
                          self.exp_cfgs.log_path)
                self.re_logger.summarize_one_iteration(it, data_label='test')
                # clean
                os.remove(os.path.join(trained_model_path, 'pytorch_model.bin'))
                
            self.re_logger.summarize_all_iterations()

        elif self.exp_cfgs.mode == 'eval':
            self.eval(
                model_path=self.exp_cfgs.model_path,
                dataset_path=self.exp_cfgs.dataset_path,
                output_path=self.exp_cfgs.log_path,
            )

        elif self.exp_cfgs.mode == 'predict':
            self.predict(self.exp_cfgs.model_path,
                         self.exp_cfgs.dataset_path, self.exp_cfgs.log_path)

        return True
