# Corporate Reorganization (dataset + retriever)

This folder contains the dataset preparation pipeline and (later) the ModernBERT retriever training code for the **corporate reorganization** task.

## Data layout

Raw Label Studio exports and processed artifacts live under:

`corporate_reorganization/data/final_annotations_gold/`

See `corporate_reorganization/modernbert/data_prep/` for the script that builds the processed files from the raw exports.

## Training

SageMaker/Deepspeed entry point:

`corporate_reorganization/modernbert/train_sm.py`

Notebook template:

`corporate_reorganization/notebooks/sagemaker_retriever_training.ipynb`

## Evaluation

Entry point:

`corporate_reorganization/modernbert/eval_retriever.py`
