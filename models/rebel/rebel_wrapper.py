import os

from models.wrapper import ModelWrapper

import os
import json
from models.rebel.src.rebel_main import call_rebel
import shutil

TRAIN_ARGS_LIST = ["seed","device_id","epochs","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","lr","save_model","train_batch_size","eval_batch_size","lr_warmup","weight_decay","max_grad_norm","neg_relation_count","neg_entity_count","size_embedding","prop_drop","save_optimizer"]
EVAL_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]
PREDICT_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]


class RebelWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path):
        
        self.exp_cfgs.edit('do_train',True) 
        self.exp_cfgs.edit('do_eval',True) 
        self.exp_cfgs.edit('do_predict',False) 

        self.exp_cfgs.edit("model_path",model_path)
        self.exp_cfgs.add("types_file",self.exp_cfgs.types_path)
        self.exp_cfgs.add("train_file",train_path)
        self.exp_cfgs.add("validation_file",valid_path)
        self.exp_cfgs.add("log_path",output_path)

        self.exp_cfgs.add("model_name_or_path",self.exp_cfgs.tokenizer_name)


        for dataset_path in [train_path,valid_path]:
            data = json.load(open(dataset_path))
            for i,sentence in enumerate(data):
                data[i]['orig_id'] = i
            with open(dataset_path,'w') as out_file:
               json.dump(data,out_file)
        
        call_rebel(self.exp_cfgs)

        return True

    def eval(self, model_path, dataset_path, output_path, data_label='test'):

        self.exp_cfgs.edit('do_train',False) 
        self.exp_cfgs.edit('do_eval',True) 
        self.exp_cfgs.edit('do_predict',False) 
        self.exp_cfgs.edit("model_path",model_path)

        # data conf
        self.exp_cfgs.add("types_file",self.exp_cfgs.types_path)
        self.exp_cfgs.add("test_file",dataset_path)
        self.exp_cfgs.add("log_path",output_path)
        self.exp_cfgs.add("data_label",data_label)

        # model conf
        self.exp_cfgs.add("model_name_or_path",self.exp_cfgs.tokenizer_name)

        data = json.load(open(dataset_path))
        for i,sentence in enumerate(data):
            data[i]['orig_id'] = i
        with open(dataset_path,'w') as out_file:
            json.dump(data,out_file)
        
        call_rebel(self.exp_cfgs)

        return True

    def predict(self, model_path, dataset_path, output_path, data_label='predict'):

        self.exp_cfgs.edit('do_train',False) 
        self.exp_cfgs.edit('do_eval',False) 
        self.exp_cfgs.edit('do_predict',True) 
        self.exp_cfgs.edit("model_path",model_path)

        # data conf
        self.exp_cfgs.add("types_file",self.exp_cfgs.types_path)
        self.exp_cfgs.add("test_file",dataset_path)
        self.exp_cfgs.add("log_path",output_path)
        self.exp_cfgs.add("data_label",data_label)

        # model conf
        self.exp_cfgs.add("model_name_or_path",self.exp_cfgs.tokenizer_name)

        data = json.load(open(dataset_path))
        for i,sentence in enumerate(data):
            data[i]['orig_id'] = i
        with open(dataset_path,'w') as out_file:
            json.dump(data,out_file)
        
        call_rebel(self.exp_cfgs)

        return True