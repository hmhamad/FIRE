import os
import csv
import numpy as np


CSV_DELIMETER = ';'

def adjust_for_random_iterations(it,exp_cfgs):
    # Update seed, log_path 
    exp_cfgs.edit('seed', (it+1) * exp_cfgs.seed)
    exp_cfgs.edit('log_path',os.path.join(exp_cfgs.log_dir,'iter_'+str(it)))
    create_directoy(exp_cfgs.log_path)
    return exp_cfgs

def adjust_for_CV(k,model_cfgs,iter_log_path,iter_save_path,dataset_paths):
    # Do not save models during Cross Validation
    model_cfgs.edit('save_model',False)
    # Update, log_path and save_path
    model_cfgs.edit('log_path',os.path.join(iter_log_path,'fold_'+str(k)))
    model_cfgs.edit('save_path',os.path.join(iter_save_path,'fold_'+str(k)))
    create_directoy(model_cfgs.log_path); create_directoy(model_cfgs.save_path)
    # Update dataset path
    train_dir = os.sep.join(dataset_paths['train_path'].split(os.sep)[:-1]); train_file = dataset_paths['train_path'].split(os.sep)[-1][:-5]
    dev_dir = os.sep.join(dataset_paths['dev_path'].split(os.sep)[:-1]); dev_file = dataset_paths['dev_path'].split(os.sep)[-1][:-5]
    dataset_Kfold_paths = {
        'train_path': os.path.join(train_dir,'folds',train_file + f'_fold_{k}.json'),
        'dev_path': os.path.join(dev_dir,'folds',dev_file + f'_fold_{k}.json'),
        'types_path': dataset_paths['types_path']
        }
    return model_cfgs, dataset_Kfold_paths

def create_directoy(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class RE_logger():
    
    def __init__(self,exp_cfgs):
        self.exp_cfgs = exp_cfgs
        if exp_cfgs.mode == 'train' or exp_cfgs.mode == 'curriculum':
            self.n_iter = exp_cfgs.n_random_iter
            self.log_dir = exp_cfgs.log_dir
            self.scores = {
                'iter_best_valid_ner_micro_f1':      [[] for i in range(exp_cfgs.n_random_iter)],
                'iter_best_valid_rel+_micro_f1':      [[] for i in range(exp_cfgs.n_random_iter)],
                'iter_best_valid_rel_micro_f1':      [[] for i in range(exp_cfgs.n_random_iter)],
                'iter_test_ner_micro_f1':            [[] for i in range(exp_cfgs.n_random_iter)],
                'iter_test_rel+_micro_f1':            [[] for i in range(exp_cfgs.n_random_iter)],
                'iter_test_rel_micro_f1':            [[] for i in range(exp_cfgs.n_random_iter)],
                }

    def summarize_one_iteration(self,it, data_label = 'test'):
        
        iter_log_path = os.path.join(self.log_dir,'iter_'+str(it))
        iter_mean_std_f1_path = os.path.join(iter_log_path, 'iter_mean_std_f1.csv')
        
        train_valid_F1_path = os.path.join(iter_log_path, 'F1_epochs.csv')
        with open(train_valid_F1_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:  # Goto last line in file
                row = [r.strip() for r in row]
        self.scores['iter_best_valid_ner_micro_f1'][it] = float(row[2])
        self.scores['iter_best_valid_rel_micro_f1'][it] = float(row[4])
        self.scores['iter_best_valid_rel+_micro_f1'][it] = float(row[6])  
        
        test_F1_path = os.path.join(iter_log_path, f'F1_{data_label}.csv')
        with open(test_F1_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            row = next(reader); row = next(reader) # Goto to second line in file
            row = [r.strip() for r in row]
        self.scores['iter_test_ner_micro_f1'][it] = float(row[0])
        self.scores['iter_test_rel_micro_f1'][it] = float(row[1])
        self.scores['iter_test_rel+_micro_f1'][it] = float(row[2])    

        # Save F1 scores of each iteration
        with open(iter_mean_std_f1_path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            row = ['********Test******']
            writer.writerow(row)
            row = ['NER Micro F1', 'Rel Micro F1', 'Rel+ Micro F1']
            writer.writerow(row)
            row = [round(self.scores['iter_test_ner_micro_f1'][it], 2), round(self.scores['iter_test_rel_micro_f1'][it], 2), round(self.scores['iter_test_rel+_micro_f1'][it], 2)]
            writer.writerow(row)
            row = ['********Validation******']
            writer.writerow(row)
            row = ['Best NER Micro F1', 'Best Rel Micro F1', 'Best Rel+ Micro F1']
            writer.writerow(row)
            row = [round(self.scores['iter_best_valid_ner_micro_f1'][it], 2), round(self.scores['iter_best_valid_rel_micro_f1'][it], 2), round(self.scores['iter_best_valid_rel+_micro_f1'][it], 2)]
            writer.writerow(row)
                

        return self.scores

    def summarize_all_iterations(self):
        # Save F1 scores over entire training (iterations)
        final_mean_std_f1_path = os.path.join(self.log_dir, 'mean_std_f1.csv')

        # 1) Mean of best F1 scores over entire training (iterations)
        self.scores['overall_valid_ner_mean_micro_f1'] = np.mean(self.scores['iter_best_valid_ner_micro_f1'])
        self.scores['overall_valid_rel+_mean_micro_f1'] = np.mean(self.scores['iter_best_valid_rel+_micro_f1'])
        self.scores['overall_valid_rel_mean_micro_f1'] = np.mean(self.scores['iter_best_valid_rel_micro_f1'])
        self.scores['overall_test_ner_mean_micro_f1'] = np.mean(self.scores['iter_test_ner_micro_f1'])
        self.scores['overall_test_rel+_mean_micro_f1'] = np.mean(self.scores['iter_test_rel+_micro_f1'])
        self.scores['overall_test_rel_mean_micro_f1'] = np.mean(self.scores['iter_test_rel_micro_f1'])


        # 2) Standard Deviation of best F1 scores over random initialization
        self.scores['overall_valid_ner_std_micro_f1'] = np.std(self.scores['iter_best_valid_ner_micro_f1'])
        self.scores['overall_valid_rel+_std_micro_f1'] = np.std(self.scores['iter_best_valid_rel+_micro_f1'])
        self.scores['overall_valid_rel_std_micro_f1'] = np.std(self.scores['iter_best_valid_rel_micro_f1'])
        self.scores['overall_test_ner_std_micro_f1'] = np.std(self.scores['iter_test_ner_micro_f1'])
        self.scores['overall_test_rel+_std_micro_f1'] = np.std(self.scores['iter_test_rel+_micro_f1'])
        self.scores['overall_test_rel_std_micro_f1'] = np.std(self.scores['iter_test_rel_micro_f1'])

        # 3) Best of best F1 scores over entire training (all iterations)
        self.scores['overall_valid_ner_best_micro_f1'] = np.max(self.scores['iter_best_valid_ner_micro_f1'])
        self.scores['overall_valid_rel+_best_micro_f1'] = np.max(self.scores['iter_best_valid_rel+_micro_f1'])
        self.scores['overall_valid_rel_best_micro_f1'] = np.max(self.scores['iter_best_valid_rel_micro_f1'])
        self.scores['overall_test_ner_best_micro_f1'] = np.max(self.scores['iter_test_ner_micro_f1'])
        self.scores['overall_test_rel+_best_micro_f1'] = np.max(self.scores['iter_test_rel+_micro_f1'])
        self.scores['overall_test_rel_best_micro_f1'] = np.max(self.scores['iter_test_rel_micro_f1'])
        
        # Write To File
        with open(final_mean_std_f1_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            row = ['******** Test Micro F1 Over Random Iterations******']; writer.writerow(row)
            row = ['Best NER Micro F1', 'Mean NER Micro F1', 'Std NER Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_test_ner_best_micro_f1'], 2), round(self.scores['overall_test_ner_mean_micro_f1'], 2),round(self.scores['overall_test_ner_std_micro_f1'], 2)]; writer.writerow(row)
            row = ['Best Rel Micro F1', 'Mean Rel Micro F1', 'Std Rel Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_test_rel_best_micro_f1'], 2), round(self.scores['overall_test_rel_mean_micro_f1'], 2),round(self.scores['overall_test_rel_std_micro_f1'], 2)]; writer.writerow(row)
            row = ['Best Rel+ Micro F1', 'Mean Rel+ Micro F1', 'Std Rel+ Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_test_rel+_best_micro_f1'], 2), round(self.scores['overall_test_rel+_mean_micro_f1'], 2),round(self.scores['overall_test_rel+_std_micro_f1'], 2)]; writer.writerow(row)


            row = ['******** Validation Micro F1 Over Random Iterations******']; writer.writerow(row)
            row = ['Best NER Micro F1', 'Mean NER Micro F1', 'Std NER Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_valid_ner_best_micro_f1'], 2), round(self.scores['overall_valid_ner_mean_micro_f1'], 2),round(self.scores['overall_valid_ner_std_micro_f1'], 2)]; writer.writerow(row)
            row = ['Best Rel Micro F1', 'Mean Rel Micro F1', 'Std Rel Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_valid_rel_best_micro_f1'], 2), round(self.scores['overall_valid_rel_mean_micro_f1'], 2),round(self.scores['overall_valid_rel_std_micro_f1'], 2)]; writer.writerow(row)
            row = ['Best Rel+ Micro F1', 'Mean Rel+ Micro F1', 'Std Rel+ Micro F1']; writer.writerow(row)
            row = [round(self.scores['overall_valid_rel+_best_micro_f1'], 2), round(self.scores['overall_valid_rel+_mean_micro_f1'], 2),round(self.scores['overall_valid_rel+_std_micro_f1'], 2)]; writer.writerow(row)

    