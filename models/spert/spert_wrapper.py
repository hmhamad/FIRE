import os
from models.wrapper import ModelWrapper
from models.spert.spert_main import call_spert

TRAIN_ARGS_LIST = ["seed","device_id","epochs","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","lr","save_model","train_batch_size","eval_batch_size","lr_warmup","weight_decay","max_grad_norm","neg_relation_count","neg_entity_count","size_embedding","prop_drop","save_optimizer"]
EVAL_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]
PREDICT_ARGS_LIST = ["seed","device_id","rel_filter_threshold","max_pairs","max_span_size","tokenizer_path","eval_batch_size","size_embedding","prop_drop"]

class SpertWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path):

        config_path = os.path.join(self.exp_cfgs.log_path,'spert_config.conf')
        with open(config_path,'w') as f:
            for key,val in self.exp_cfgs.configs.items():
                if key in TRAIN_ARGS_LIST:
                    f.write(str(key) + ' = ' + str(val) + '\n')
            f.write('model_path = ' + model_path + '\n')
            f.write('train_path = ' + train_path + '\n')
            f.write('valid_path = ' + valid_path + '\n')
            f.write('log_path = ' + output_path + '\n')
            f.write('save_path = ' + output_path + '\n')
            f.write('types_path = ' + self.exp_cfgs.types_path + '\n')

        call_spert('train',config_path)

        return True

    def eval(self, model_path, dataset_path, output_path):
        config_path = os.path.join(self.exp_cfgs.log_path,'spert_config.conf')
        with open(config_path,'w') as f:
            for key,val in self.exp_cfgs.configs.items():
                if key in EVAL_ARGS_LIST:
                    f.write(str(key) + ' = ' + str(val) + '\n')
            f.write('model_path = ' + model_path + '\n')
            f.write('dataset_path = ' + dataset_path + '\n')
            f.write('log_path = ' + output_path + '\n')
            f.write('types_path = ' + self.exp_cfgs.types_path + '\n')

        call_spert('eval',config_path)

    def predict(self, model_path, dataset_path, output_path):
        
        config_path = os.path.join(self.exp_cfgs.log_path,'spert_config.conf')
        with open(config_path,'w') as f:
            for key,val in self.exp_cfgs.model_args.configs.items():
                if key in PREDICT_ARGS_LIST:
                    f.write(str(key) + ' = ' + str(val) + '\n')
            f.write('model_path = ' + model_path + '\n')
            f.write('dataset_path = ' + dataset_path + '\n')
            f.write('types_path = ' + self.exp_cfgs.model_args.types_path + '\n')
            f.write('predictions_path = ' + output_path  + '\n')

        call_spert('predict',config_path)
