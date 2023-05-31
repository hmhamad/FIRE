# FIRE Dataset ![Logo](logo.png)

The FIRE Dataset is a dataset of named entities and relations in the financial domain. Our paper "FIRE: A Dataset for FInancial Relation Extraction" is available on [arXiv](https://arxiv.org/abs/XXXX.XXXX).

## Description

FIRE is a dataset focused on the extraction of financial relations in business and financial documents. It features 13 types of entities and 15 types of relations and can be used to train and evaluate machine learning models in the task of financial joint named entity recognition and relation extraction.

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
| Train  | 1,995          | 10,566               | 5,812                  |
| Dev    | 427            | 2,233                | 1,213                  |
| Test   | 427            | 2,297                | 1,296                  |

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
python main.py --mode train --model spert
```

For Rebel, you might face a bug when loading model to evaluate on test set due to versioning conflicts. See this issue https://github.com/Babelscape/rebel/issues/55.
The quick fix proposed from the author is to comment out the line in the pytorch_lighting source code:
File "python3.8/site-packages/pytorch_lightning/core/saving.py", line 157, in load_from_checkpoint
checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

## Citing Our Work

If you use the FIRE Dataset in your research, please cite our paper:

```bibtex
@article{Hassan2023fire,
  title={FIRE: A Dataset for FInancial Relation Extraction},
  author={Hassan Hamad, Abhinav Thakur, Sujith Pulikodan, Nijil Kolleri and Keith M. Chugg},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2023}
}
```