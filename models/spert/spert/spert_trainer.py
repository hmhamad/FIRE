import argparse
import math
import os
from typing import OrderedDict, Type
import csv
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from models.spert.spert import models, prediction
from models.spert.spert import sampling
from models.spert.spert import util
from models.spert.spert.entities import Dataset
from models.spert.spert.evaluator import Evaluator
from models.spert.spert.input_reader import BaseInputReader
from models.spert.spert.loss import SpERTLoss, Loss
from tqdm import tqdm
from models.spert.spert.trainer import BaseTrainer

CSV_DELIMETER = ';'

class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace, log_path ,save_path, seed):
        super().__init__(args,log_path,save_path, seed)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = 'train', 'eval'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        
        train_dataset = input_reader.read(train_path, train_label)
        validation_dataset = input_reader.read(valid_path, valid_label)
        self._log_datasets(input_reader)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # load model
        model = self._load_model(input_reader)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm, train_dataset._rel_types)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        best_train_micro_f1 = 0; best_eval_micro_f1 = 0; best_f1 = 0
        best_train_macro_f1 = 0; best_eval_macro_f1 = 0; best_ner_f1 = 0
        F1_path = os.path.join(self._log_path, 'F1_epochs.csv')
        with open(F1_path, 'a', newline='') as csv_file:
            header_row = ['{:<10}'.format('Epoch'),'{:<30}'.format('NER_Valid_Micro_F1'), '{:<30}'.format('Best_NER_Valid_Micro_F1'), '{:<30}'.format('Rel_Valid_Micro_F1'), '{:<30}'.format('Best_Rel_Valid_Micro_F1'), '{:<30}'.format('Rel+_Valid_Micro_F1'), '{:<30}'.format('Best_Rel+_Valid_Micro_F1')]
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header_row)
        
        # train
        for epoch in range(args.epochs):            
            # train epoch
            self._train_epoch(model, train_label, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval training sets
            ner_train, rel_train, rel_nec_train, rel_nec_per_type_train = self._eval(model, train_dataset, input_reader, epoch + 1, updates_epoch)
            train_micro_f1 = rel_nec_train[2]; train_macro_f1 = rel_nec_train[5]
            best_train_micro_f1 = max(best_train_micro_f1,train_micro_f1); best_train_macro_f1 = max(best_train_macro_f1,train_macro_f1)

            # eval validation sets
            ner_eval, rel_eval, rel_nec_eval, rel_nec_per_type_eval = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
            eval_micro_f1 = rel_nec_eval[2]; eval_macro_f1 = rel_nec_eval[5]
            if eval_micro_f1 > best_eval_micro_f1:
                # save best model according to validation micro f1
                if args.save_model:
                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    global_iteration = args.epochs * updates_epoch
                    self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                                    optimizer=optimizer if self._args.save_optimizer else None, extra=extra,
                                    include_iteration=False, name='best_model')
                best_eval_micro_f1 = max(best_eval_micro_f1,eval_micro_f1); best_eval_macro_f1 = max(best_eval_macro_f1,eval_macro_f1)
                best_f1 = max(best_f1,rel_eval[2])
                best_ner_f1 = max(best_ner_f1,ner_eval[2])
            print('*****************************************')
            print(f'Epoch {epoch+1}/{args.epochs}:  Best Eval Micro-F1 = {best_eval_micro_f1:.2f}, Best Train Micro-F1 = {best_train_micro_f1:.2f}')
            print(f'           : Best Eval Macro-F1 = {best_eval_macro_f1:.2f}, Best Train Macro-F1 = {best_train_macro_f1:.2f}')
            with open(F1_path, 'a', newline='') as csv_file:
                ner_f1 = ner_eval[2]; f1 = rel_eval[2]
                row = ['{:<10}'.format(f'{epoch+1}/'+str(int(args.epochs))),'{:<30}'.format(f'{ner_f1:.4f}'), '{:<30}'.format(f'{best_ner_f1:.4f}'), '{:<30}'.format(f'{f1:.4f}'), '{:<30}'.format(f'{best_f1:.4f}'), '{:<30}'.format(f'{eval_micro_f1:.4f}'), '{:<30}'.format(f'{best_eval_micro_f1:.4f}')]
                writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)
                    
            print('*****************************************')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

        return model

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model
        model = self._load_model(input_reader)
        model.to(self._device)

        # evaluate
        ner_eval, rel_eval, rel_nec_eval, rel_nec_per_type_eval  = self._eval(model, test_dataset, input_reader)

        F1_path = os.path.join(self._log_path, f'F1_{dataset_label}.csv')
        ner_f1 = ner_eval[2]; f1 = rel_eval[2]; f1_with_nec = rel_nec_eval[2]
        with open(F1_path, 'w') as csv_file:
            row1= ['{:<30}'.format('NER_Micro_F1'), '{:<30}'.format('Rel_Micro_F1'), '{:<30}'.format('Rel+_Micro_F1')]
            row2 = ['{:<30}'.format(f'{ner_f1:.4f}'), '{:<30}'.format(f'{f1:.4f}'), '{:<30}'.format(f'{f1_with_nec:.4f}')]
            row3 = [t[1] for t in rel_nec_per_type_eval]
            row4 = [str(t[0])[:5] for t in rel_nec_per_type_eval]
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row1); writer.writerow(row2); writer.writerow(row3); writer.writerow(row4) 
        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size,
                                        spacy_model=args.spacy_model)
        dataset = input_reader.read(dataset_path, 'dataset')

        model = self._load_model(input_reader)
        model.to(self._device)

        self._predict(model, dataset, input_reader)

    def _load_model(self, input_reader):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=self._args.cache_path)
        util.check_version(config, model_class, self._args.model_path)

        config.spert_version = model_class.VERSION
        model = model_class.from_pretrained(self._args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            num_relation_types=input_reader.relation_type_count - 1,
                                            num_entity_types=input_reader.entity_type_count,
                                            max_pairs=self._args.max_pairs,
                                            prop_drop=self._args.prop_drop,
                                            size_embedding=self._args.size_embedding,
                                            cache_dir=self._args.cache_path,
                                            )

        return model

    def _train_epoch(self, model: torch.nn.Module, train_label, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        rel_type_counter = torch.zeros((len(dataset._rel_types)),dtype=torch.int64); encodings_counter = {}; encodings_set = set()
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            
            ######
            for b in batch['encodings']:
                if b.sum().item() in encodings_set:
                    encodings_counter[b.sum().item()]+=1
                else:
                    encodings_set.add(b.sum().item())
                    encodings_counter[b.sum().item()]=1
            rel_type_counter[0]+=len([t for b in range(self._args.train_batch_size) for t,m in zip(batch['rel_types'][b],batch['rel_sample_masks'][b]) if (t.sum()==0 and m)])
            for i in range(len(dataset._rel_types)-1):
                rel_type_counter[i+1]+=len([t for b in range(self._args.train_batch_size) for t,m in zip(batch['rel_types'][b],batch['rel_sample_masks'][b]) if (t[i] and m)])
            
            ######
            
            # forward step
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'],
                                              )

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}.json')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self._args.rel_filter_threshold, self._args.no_overlapping, predictions_path,
                              examples_path, self._args.example_count)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)


        model.eval()

        # iterate batches
        total = math.ceil(dataset.document_count / self._args.eval_batch_size)
        for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
            # move batch to selected device
            batch = util.to_device(batch, self._device)

            # run model (forward pass)
            result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                        entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                        entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                        inference=True)
            ent_logits, entity_clf, rel_clf, rel_logits, rels, rel_repr = result

            # evaluate batch
            evaluator.eval_batch(ent_logits, entity_clf, rel_logits, rel_clf, rels, batch)

                                                                 
        prediction.store_predictions(dataset.documents, evaluator._pred_entities, evaluator._pred_relations, os.path.join(self._args.log_path, dataset.label + '_predictions.json'))
        
        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval, rel_nec_per_type_eval = evaluator.compute_scores()
        
        if dataset.label == 'test':
            self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                        epoch, iteration, global_iteration, dataset.label)

            if self._args.store_predictions and not self._args.no_overlapping:
                evaluator.store_predictions()

            if self._args.store_examples:
                evaluator.store_examples()
        
        return ner_eval, rel_eval, rel_nec_eval, rel_nec_per_type_eval
        #return rel_nec_eval[2], rel_nec_eval[5]  # micro F1

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,  store=True):
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []
        
        # No data for prediction
        # TODO: Handle in better way
        if not len(dataset):
            if store:
                prediction.store_predictions(dataset.documents, pred_entities, pred_relations, self._args.predictions_path)
            
            return None, None, None, None

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                ent_logits, entity_clf, rel_clf, rel_logits, rels, _ = result

                # convert predictions
                predictions = prediction.convert_predictions(ent_logits,entity_clf,rel_logits, rel_clf, rels,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        if store:
            prediction.store_predictions(dataset.documents, pred_entities, pred_relations, os.path.join(self._args.predictions_path, 'predictions.json'))

        return entity_clf, rel_clf, pred_entities, pred_relations

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
