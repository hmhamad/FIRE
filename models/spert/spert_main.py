import os
from models.spert.args import train_argparser, eval_argparser, predict_argparser
from models.spert.config_reader import _yield_configs
from models.spert.spert import input_reader
from models.spert.spert.spert_trainer import SpERTTrainer

CSV_DELIMETER = ';'


def train(run_args, train_path, valid_path, log_path, save_path, seed):
    trainer = SpERTTrainer(run_args, log_path, save_path, seed)
    trainer.train(train_path=train_path, valid_path=valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader,
                  )
    return True


def eval(run_args, data_path, log_path, save_path, seed):
    trainer = SpERTTrainer(run_args,log_path,save_path,seed)
    trainer.eval(dataset_path=data_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)

def predict(run_args, data_path, predictions_path, save_path, seed):
    trainer = SpERTTrainer(run_args,predictions_path,save_path,seed)
    trainer.predict(dataset_path=data_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonPredictionInputReader)


def call_spert(mode,config_path):

    if mode == 'train':
        arg_parser = train_argparser()
        args = arg_parser.parse_args(['--config',config_path])

        run_args, _, _ = next(_yield_configs(arg_parser, args, verbose=True))
        run_args.types_path = run_args.types_path

        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=str(run_args.device_id)

        train(run_args, run_args.train_path, run_args.valid_path, run_args.log_path, run_args.save_path, run_args.seed)

        return True

    elif mode == 'eval':
        arg_parser = eval_argparser()
        args = arg_parser.parse_args(['--config',config_path])

        run_args, _, _ = next(_yield_configs(arg_parser, args, verbose=True))
        run_args.types_path = run_args.types_path

        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=str(run_args.device_id)

        eval(run_args, run_args.dataset_path, run_args.log_path, '', run_args.seed)
    
    elif mode == 'predict':
        arg_parser = predict_argparser()
        args = arg_parser.parse_args(['--config',config_path])

        run_args, _, _ = next(_yield_configs(arg_parser, args, verbose=True))
        run_args.types_path = run_args.types_path

        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=str(run_args.device_id)

        predict(run_args, run_args.dataset_path, run_args.predictions_path, '', run_args.seed)
    
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict']")


