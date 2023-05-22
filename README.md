# FIRE Dataset ![Logo](logo.png)

The FIRE Dataset is a dataset of named entities and relations in the financial domain. Our paper "FIRE: A Dataset for FInancial Relation Extraction" is available on [arXiv](https://arxiv.org/abs/XXXX.XXXX).

## Description

FIRE is a novel dataset focused on the extraction of financial relations in business and financial documents. It features 13 types of entities and 15 types of relations and can be used to train and evaluate machine learning models in the task of financial joint named entity recognition and relation extraction.

Here is an example instance from the dataset and how it is represented in json format:

```json
{
    "tokens": ["Shares","of","Tesla","dropped","14%","over","the","last","quarter"],
    "entities": [
        {"type": "FinancialEntity", "start": 0, "end": 1},
        {"type": "Company", "start": 2, "end": 3},
        {"type": "Quantity", "start": 4, "end": 5},
        {"type": "Date", "start": 7, "end": 8},
    ],
    "relations": [
        {"type": "propertyof", "head": 0, "tail": 1},
        {"type": "ValueChangeDecreaseby", "head": 0, "tail": 2},
        {"type": "Valuein", "head": 2, "tail": 3},
    ],
    "duration": 42,
}
```

## Dataset Statistics

| Split  | # of Instances | # of Entity Mentions | # of Relation Mentions |
| ------ | -------------- | -------------------- | ---------------------- |
| Train  | xxx            | xxx                  | xxx                    |
| Dev    | xxx            | xxx                  | xxx                    |
| Test   | xxx            | xxx                  | xxx                    |

## Setup Instructions

To set up your Python environment to run the baselines, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/hmhamad/FIRE.git
cd FIRE
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Reproducing Results

You can reproduce the results from our paper by running the provided scripts. For example, to run SpERT model:

```bash
python main.py --mode train --model spert --exp baseline
```

## Citing Our Work

If you use the FIRE Dataset in your research, please cite our paper:

```bibtex
@article{Hassan2023fire,
  title={FIRE: A Dataset for FInancial Relation Extraction},
  author={Hassan Hamad, Abhinav Thakur, Keith M. Chugg, Sujith Pulikodan and Nijil Kolleri},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2023}
}
```