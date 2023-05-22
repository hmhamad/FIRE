from os.path import isfile, join
from os import listdir
import argparse
import os
import json
import spacy
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate prediction dataset')
parser.add_argument('--input', type=str, required=True, help='Input directory')
parser.add_argument('--output', type=str, required=True,
                    help='Output directory')
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")

raw_files_dir = args.input
prediction_files_dir = args.output

if not os.path.exists(prediction_files_dir):
    os.mkdir(prediction_files_dir)


raw_files = [int(f.strip('.txt')) for f in listdir(raw_files_dir) if isfile(join(raw_files_dir, f))]
prediction_files = [int(f.strip('.json')) for f in listdir(prediction_files_dir) if isfile(join(prediction_files_dir, f))]

files_to_process = sorted(list(set(raw_files).difference(set(prediction_files))))

for file in tqdm(files_to_process):
    line_tokens = []

    try:
        with open(os.path.join(raw_files_dir, f'{file}.txt'), 'r') as f:
            text = f.read()

        doc = nlp(text)
        for line in doc.sents:
            document = line.text
            tokens = [t.text for t in line]
            if len(tokens) < 10 or len(tokens) > 200:
                continue

            line_tokens.append({'tokens': tokens})

    except Exception as e:
        print(f'Error processing file {file}: {e}')

    with open(os.path.join(prediction_files_dir, f'{file}.json'), 'w') as f:
        json.dump(line_tokens, f)
