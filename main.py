import argparse
from configs.config_reader import read_json_configs
import os
import datetime
import shutil
import argparse
from mapping import map_dataset

def get_model_wrapper(exp_cfgs):
    if exp_cfgs.model == 'spert':
        from models.spert.spert_wrapper import SpertWrapper
        model_wrapper = SpertWrapper(exp_cfgs)
    elif exp_cfgs.model == 'pl_marker':
        from models.pl_marker.pl_marker_wrapper import PLMarkerWrapper
        model_wrapper = PLMarkerWrapper(exp_cfgs)
    elif exp_cfgs.model == 'rebel':
        from models.rebel.rebel_wrapper import RebelWrapper
        model_wrapper = RebelWrapper(exp_cfgs)
    
    return model_wrapper

def execute_main(exp_cfgs):
    model_wrapper = get_model_wrapper(exp_cfgs)
    model_wrapper.execute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FinancialRE entrypoint')
    parser.add_argument('--mode', default='train', type=str, choices=['train','eval','predict','calibrate','ssl', 'acl-ssl', 'curriculum'], help='Experiment Mode')
    parser.add_argument('--model', default='spert', type=str, choices=['spert','pl_marker','rebel'], help='model name')
    args = parser.parse_args()

    # get absolute paths
    dir_path = os.path.abspath(os.getcwd())

    # Get path of experiment config file
    config_path = os.path.join(dir_path,'configs', args.model + '_config.json')

    # Read Experiment config
    exp_cfgs = read_json_configs(config_path)
    exp_cfgs.add('mode',args.mode)
    exp_cfgs.add('model',args.model)

    # Setup Logging Directory
    run_key = str(datetime.datetime.now()).replace(' ', '_') + f'_{exp_cfgs.model}'

    log_dir_path = os.path.join(dir_path,'log',str(run_key))
    exp_cfgs.add('log_dir',log_dir_path)
    exp_cfgs.add('log_path',log_dir_path)
    os.makedirs(log_dir_path)
    shutil.copy(config_path,log_dir_path) 
    
    # Load Dataset
    map_dataset(exp_cfgs)

    # Execute main function
    execute_main(exp_cfgs)