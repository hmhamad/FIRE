import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

from models.rebel.src.pl_data_modules import BasePLDataModule
from models.rebel.src.pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from models.rebel.src.generate_samples import GenerateTextSamplesCallback
import json
import os
import csv
CSV_DELIMETER = ';'

class MyTrainLogger(Callback):
    def __init__(self,logdir):
        super().__init__()
        self.logdir = logdir
        self.F1_path = os.path.join(self.logdir,'F1_epochs.csv')
        with open(self.F1_path, 'a', newline='') as csv_file:
            header_row = ['{:<10}'.format('Epoch'),'{:<30}'.format('NER_Valid_Micro_F1'), '{:<30}'.format('Best_NER_Valid_Micro_F1'), '{:<30}'.format('Rel_Valid_Micro_F1'), '{:<30}'.format('Best_Rel_Valid_Micro_F1'), '{:<30}'.format('Rel+_Valid_Micro_F1'), '{:<30}'.format('Best_Rel+_Valid_Micro_F1')]
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header_row)
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.running_sanity_check:
            with open(self.F1_path, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in reader:  # Goto last line in file
                    row = [r.strip() for r in row]
                best_eval_rel_micro_f1 = float(row[6]) if trainer.logged_metrics['epoch'] > 0 else 0
                best_eval_rel_plus_micro_f1 = float(row[4]) if trainer.logged_metrics['epoch'] > 0 else 0
            with open(self.F1_path, 'a', newline='') as csv_file:
                epoch = trainer.logged_metrics['epoch']
                eval_rel_micro_f1 = trainer.logged_metrics['val_rel_F1_micro']
                best_eval_rel_micro_f1 = max(best_eval_rel_micro_f1,trainer.logged_metrics['val_rel_F1_micro'])
                eval_rel_plus_micro_f1 = trainer.logged_metrics['val_rel_plus_F1_micro']
                best_eval_rel_plus_micro_f1 = max(trainer.logged_metrics['val_rel_plus_F1_micro'],best_eval_rel_plus_micro_f1)
                ner_f1 = 0; best_ner_f1 = 0
                row = ['{:<10}'.format(f'{epoch+1}/'+str(int(pl_module.hparams['num_train_epochs']))),'{:<30}'.format(f'{ner_f1:.4f}'), '{:<30}'.format(f'{best_ner_f1:.4f}'), '{:<30}'.format(f'{eval_rel_micro_f1:.4f}'), '{:<30}'.format(f'{best_eval_rel_micro_f1:.4f}'), '{:<30}'.format(f'{eval_rel_plus_micro_f1:.4f}'), '{:<30}'.format(f'{best_eval_rel_plus_micro_f1:.4f}')]
                writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)

class MyTestLogger(Callback):
    def __init__(self, logdir, data_label):
        super().__init__()
        self.logdir = logdir
        self.F1_path = os.path.join(self.logdir,f'F1_{data_label}.csv')
        with open(self.F1_path, 'a', newline='') as csv_file:
            header_row= ['{:<30}'.format('NER_Micro_F1'), '{:<30}'.format('Rel_Micro_F1'), '{:<30}'.format('Rel+_Micro_F1')]
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header_row)
    def on_test_epoch_end(self, trainer, pl_module):
        with open(self.F1_path, 'a', newline='') as csv_file:
            test_rel_micro_f1 = trainer.logged_metrics['test_rel_micro_f1']
            test_rel_plus_micro_f1 = trainer.logged_metrics['test_rel_plus_micro_f1']
            ner_f1 = 0
            row = ['{:<30}'.format(f'{ner_f1:.4f}'), '{:<30}'.format(f'{test_rel_micro_f1:.4f}'), '{:<30}'.format(f'{test_rel_plus_micro_f1:.4f}')]
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)

def template_to_standard(dataset_path,output,log_path,data_label):
    data_gt = json.load(open(dataset_path))
    for item in data_gt:
        item['entities'] = []; item['relations'] = []
    idx = 0
    for i,batch in enumerate(output):
        for j,preds in enumerate(batch['predictions']):
            tokens = data_gt[idx]['tokens']
            tokens = [t.lower() for t in tokens]
            ent_idx = 0
            for r, rel in enumerate(preds):
                found = True
                for ent in ['head','tail']:
                    ent_tokens = rel[ent].split(' '); ent_tokens  = [t.lower() for t in ent_tokens]; ent_len = len(ent_tokens)
                    idx_ent = [[i,i+ent_len] for i in range(len(tokens)-ent_len+1) if tokens[i:i+ent_len] == ent_tokens]
                    if not idx_ent:
                        if ent_tokens[-1][-1] =='.':
                            ent_tokens = ent_tokens[:-1] + [ent_tokens[-1][:-1]] + ['.']
                            ent_len+=1
                            idx_ent = [[i,i+ent_len] for i in range(len(tokens)-ent_len+1) if tokens[i:i+ent_len] == ent_tokens]
                        if not idx_ent:
                            found = False
                            continue
                    ent_start = idx_ent[0][0]; ent_end = idx_ent[0][1]
                    data_gt[idx]['entities'].append({'start': ent_start, 'end': ent_end, 'type': rel[f'{ent}_type']})
                    ent_idx += 1
                if found:
                    data_gt[idx]['relations'].append( {'head': ent_idx-2, 'tail': ent_idx-1, 'type': rel['type']} )
            idx+=1

    with open(os.path.join(log_path,data_label+'_predictions.json'),'w') as outfile:
        json.dump(data_gt,outfile)

    
def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)
    
    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>'], # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
#         "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    data_types = json.load(open(conf.types_file,'r'))
    special_tokens_list = []
    for ent_type in data_types['entities']:
        special_tokens_list.append('<' + ent_type.lower() + '>')
    tokenizer.add_tokens(special_tokens_list, special_tokens = True)


    # if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
    #     tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    pl_module = BasePLModule(conf, config, tokenizer, model)

    callbacks_store = []
    
    mylogger =  MyTrainLogger(logdir=conf.log_path)
    callbacks_store.append(mylogger)


    # if conf.apply_early_stopping:
    #     callbacks_store.append(
    #         EarlyStopping(
    #             monitor=conf.monitor_var,
    #             mode=conf.monitor_var_mode,
    #             patience=conf.patience
    #         )
    #     )

    checkpoint_callback = ModelCheckpoint(
            monitor=conf.monitor_var,
            dirpath=conf.log_path + '/best_model',
            filename = 'pytorch_model',
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=False,
            mode=conf.monitor_var_mode,
            period = 1,
            save_weights_only=False
        )
    checkpoint_callback.FILE_EXTENSION = '.bin'
    callbacks_store.append(checkpoint_callback)

    #callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
    #callbacks_store.append(LearningRateMonitor(logging_interval='step'))
    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_epochs=conf.num_train_epochs,
        precision=conf.precision,
        amp_level=conf.amp_level,
        logger = None,
        resume_from_checkpoint=None,
        limit_val_batches=conf.val_percent_check
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

def test_or_predict(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        # cache_dir=conf.cache_dir,
        # revision=conf.model_revision,
        # use_auth_token=True if conf.use_auth_token else None,
    )
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    data_types = json.load(open(conf.types_file,'r'))
    special_tokens_list = []
    for ent_type in data_types['entities']:
        special_tokens_list.append('<' + ent_type.lower() + '>')
    tokenizer.add_tokens(special_tokens_list, special_tokens = True)

    # if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
    #     tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'scierc_typed.py':
    #     tokenizer.add_tokens(['<task>', '<method>', '<metric>', '<material>', '<other>', '<generic>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    # if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
    #     tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_path,
        config=config,
    )
    # if not conf.finetune:
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    pl_module = BasePLModule(conf, config, tokenizer, model)
    # main module declaration
    if 'pytorch_model.bin' in os.listdir(conf.model_path):
        conf.model_path = os.path.join(conf.model_path,'pytorch_model.bin')
        pl_module = pl_module.load_from_checkpoint(checkpoint_path = conf.model_path, config = config, tokenizer = tokenizer, model = model)

    # pl_module.hparams.predict_with_generate = True
    pl_module.hparams.test_file = pl_data_module.conf.test_file

    callbacks_store = []
    if conf.do_eval:
        mylogger =  MyTestLogger(logdir=conf.log_path, data_label=conf.data_label)
        callbacks_store.append(mylogger)

    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        callbacks=callbacks_store,
        logger = None
    )
    # Manually run prep methods on DataModule
    pl_data_module.prepare_data()
    pl_data_module.setup()

    trainer.test(pl_module, test_dataloaders=pl_data_module.test_dataloader())

    # save predictions on dataset to file
    dataset_path = conf.test_file
    template_to_standard(dataset_path,trainer.logged_metrics['output'],log_path = conf.log_path, data_label= conf.data_label)


def call_rebel(conf):
    conf = omegaconf.OmegaConf.create(conf.configs)
    if conf.do_train:
        train(conf)
    elif conf.do_eval or conf.do_predict:
        test_or_predict(conf)
    else:
        raise Exception(f"Unrecognized mode, make sure one of the following flags is set to true [conf.do_train, conf.do_eval, conf.do_predict]")

