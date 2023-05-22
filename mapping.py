import json
import os
import shutil

"""
Possible Data Formats:
    * standard: ["tokens": [...], "entities":[...], "relations":[...]]
    * cluster:  ["clusters":[...], "sentences":[...], "ner":[...], "relations":[...]]
    * cluster_jsonl:  Same as Cluster format but file type is jsonl (json lines) not json
Models and Datasets:
    * Models and Format: Spert and Rebel models take the standard format while pl_marker takes the cluster_jsonl format
    * Datasets and Format: Fire and Conll04 are originally in the standard format while SciERC is originally in the the cluster format
"""

def standard_to_cluster(in_example):
    ner = [[[ent["start"],ent["end"]-1,ent["type"]] for ent in in_example["entities"]]]
    relations = [[[in_example["entities"][rel["head"]]["start"],in_example["entities"][rel["head"]]["end"]-1,in_example["entities"][rel["tail"]]["start"],in_example["entities"][rel["tail"]]["end"]-1,rel["type"]] for rel in in_example["relations"]]]
    out_example = {"clusters": [], "sentences": [in_example["tokens"]],"ner":ner,"relations":relations} 
    return out_example
    
def cluster_to_standard(in_example):
    out_examples = []; shift_idx = 0
    for idx in range(len(in_example['sentences'])):
        tokens = in_example['sentences'][idx]
        entities = []
        for ent in in_example['ner'][idx]:
            if len(ent)>3: # contains scores
                ent_dict = {'type':ent[-1], 'start':ent[3]-shift_idx, 'end':ent[4]+1-shift_idx}
                ent_dict['score'] = {'logit':ent[0], 'prob':ent[1], 'top2':ent[2]}
                entities.append(ent_dict)
            else:
                ent_dict = {'type':ent[-1], 'start':ent[0]-shift_idx, 'end':ent[1]+1-shift_idx}
                entities.append(ent_dict)

        relations = []
        for rel in in_example['relations'][idx]:
            if len(rel)>6: # contains scores and embedding id
                head_idx = [i for i in range(len(entities)) if entities[i]['start']==(rel[2][0]-shift_idx) and (entities[i]['end']-1)==(rel[2][1]-shift_idx)][0]
                tail_idx = [i for i in range(len(entities)) if entities[i]['start']==(rel[3][0]-shift_idx) and (entities[i]['end']-1)==(rel[3][1]-shift_idx)][0]
                rel_dict = {'type':rel[4], 'head':head_idx, 'tail':tail_idx}
                rel_dict['score'] = {'logit':rel[0], 'prob':rel[1]}
                rel_dict['embedding_id'] = rel[-1]
            else:
                head_idx = [i for i in range(len(entities)) if entities[i]['start']==(rel[0]-shift_idx) and (entities[i]['end']-1)==(rel[1]-shift_idx)][0]
                tail_idx = [i for i in range(len(entities)) if entities[i]['start']==(rel[2]-shift_idx) and (entities[i]['end']-1)==(rel[3]-shift_idx)][0] 
                rel_dict = {'type':rel[-1], 'head':head_idx, 'tail':tail_idx}
            relations.append(rel_dict)
        
        out_examples.append({'tokens': tokens, 'entities': entities, 'relations': relations})
        shift_idx += len(tokens)
    return out_examples

def create_directoy(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def map_datafile(in_path, out_path, from_format, to_format):
    if from_format == 'cluster_jsonl':
        with open(in_path,'r') as infile:
            data_in = []
            for  line in infile:
                data_in.append(json.loads(line))
        if to_format =='standard':
            data_out = [single_ex for line in data_in for single_ex in cluster_to_standard(line)]
        elif to_format =='cluster':
            data_out = data_in
        with open(out_path, 'w') as outfile:
            json.dump(data_out, outfile)

    elif from_format == 'standard': 
        with open(in_path,'r') as infile:
            data_in = json.load(infile)
        data_out = [standard_to_cluster(ex) for ex in data_in]
        if to_format =='cluster_jsonl':
            with open(out_path, 'w') as outfile:
                for entry in data_out:
                    json.dump(entry, outfile)
                    outfile.write('\n')
        elif to_format =='cluster':
            with open(out_path, 'w') as outfile:
                json.dump(data_out, outfile)

    elif from_format == 'cluster':
        with open(in_path,'r') as infile:
            data_in = json.load(infile)
        if to_format =='standard':
            data_out = [single_ex for line in data_in for single_ex in cluster_to_standard(line)]
            with open(out_path, 'w') as outfile:
                json.dump(data_out, outfile)
        elif to_format =='cluster_jsonl':
            data_out = data_in
            with open(out_path, 'w') as outfile:
                for entry in data_out:
                    json.dump(entry, outfile)
                    outfile.write('\n')

def map_dataset(exp_cfgs):
    dataset  = 'fire'
    dataset_dir = 'fire'
    model = exp_cfgs.model
    temp_data_dir = os.path.join(exp_cfgs.log_dir,'data')
    
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)

    create_directoy(temp_data_dir)
    shutil.copy(os.path.join(dataset_dir,f'{dataset}_types.json'),temp_data_dir)

    if model == 'spert' or model == 'rebel':
        # already in standard format
        if exp_cfgs.mode in ['train']:
            shutil.copy(exp_cfgs.train_path,temp_data_dir)
            shutil.copy(exp_cfgs.valid_path,temp_data_dir)
            shutil.copy(exp_cfgs.test_path,temp_data_dir)
        elif exp_cfgs.mode in ['eval','predict']:
            shutil.copy(exp_cfgs.dataset_path,temp_data_dir)

    elif model == 'pl_marker':
        # Standard to cluster_jsonl
        if exp_cfgs.mode in ['train']:
            map_datafile(exp_cfgs.train_path, os.path.join(temp_data_dir,exp_cfgs.train_path.split('/')[-1]), 'standard', 'cluster_jsonl')
            map_datafile(exp_cfgs.valid_path, os.path.join(temp_data_dir,exp_cfgs.valid_path.split('/')[-1]), 'standard', 'cluster_jsonl')
            map_datafile(exp_cfgs.test_path, os.path.join(temp_data_dir,exp_cfgs.test_path.split('/')[-1]), 'standard', 'cluster_jsonl')
        elif exp_cfgs.mode in ['eval','predict']:
            map_datafile(exp_cfgs.dataset_path, os.path.join(temp_data_dir,exp_cfgs.dataset_path.split('/')[-1]), 'standard', 'cluster_jsonl')
    
    if exp_cfgs.mode in ['train','acl-ssl','curriculum']:
        exp_cfgs.edit('train_path',os.path.join(temp_data_dir,exp_cfgs.train_path.split('/')[-1]))
        exp_cfgs.edit('valid_path',os.path.join(temp_data_dir,exp_cfgs.valid_path.split('/')[-1]))
        exp_cfgs.edit('test_path',os.path.join(temp_data_dir,exp_cfgs.test_path.split('/')[-1]))
        exp_cfgs.edit('types_path',os.path.join(temp_data_dir,dataset+'_types.json'))
    elif exp_cfgs.mode in ['eval','predict','calibrate']:
        exp_cfgs.edit('dataset_path',os.path.join(temp_data_dir,exp_cfgs.dataset_path.split('/')[-1]))
        exp_cfgs.edit('types_path',os.path.join(temp_data_dir,dataset+'_types.json'))
    
    return True
