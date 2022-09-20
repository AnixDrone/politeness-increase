# Politeness increase using transformers

Project where we attempt to increase the politeness of a sentence using deep learning ,transformers more specifically.

The training was done on the data that is outputed from [Tag and Generate](https://github.com/tag-and-generate) paper.
The data is already pre split and that same split has been used here.

To train the transformer yourself run these commands:

1. pip install requirements (to install the requirements)
2. Modify the hyperparameters.json file to set up hyperparamaters
3. python main_training.py 

After running main_training.py script there will some files that were generated such as:
- [timestamp]_hyperparameters.json
- [timestamp]_model.pt
- [timestamp]_results.csv
- [timestamp]_translations.csv

Further testing has been done in notebooks/prediciton_eval.ipynb and notebooks/model_evaluating.ipynb.

I have trained two different models with different minimal frequencies in the vocab.

# Hyperparameters

    "emb_size": 512,
    
    "n_head": 8,
    
    "ffn_hid_dim": 512,
    
    "batch_size": 128,
   
    "num_encoder_layers": 3,
    
    "num_decoder_layers": 3,
    
    "num_epochs": 200



# Results

## Scores for the model with vocab with minimal frequency 3 are:

BLEU score: 36.1

ROUGE score: 64.5

## Scores for the model with vocab with minimal frequency 4 are:

BLEU score: 35.4

ROUGE score: 64.3

# Samples

## Model with minimal frequency 3
Good example: I need the documents || I need the documents as soon as possible

Bad Example: I want to return the files right away || I want to return the files right away

No Context: Answer the phoneright away || Could you please phone the answerright away

## Model with minimal frequency 4

Good example: Give me the documents || Could you give me the documents

Bad Example: I want to return the files right away || I want to return the files right away

No Context: I need the documents || I need to prepare the documents for execution


