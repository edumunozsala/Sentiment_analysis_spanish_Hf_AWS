---
language: es
tags:
- sagemaker
- roberta
- ruperta
- TextClassification
license: apache-2.0
datasets:
- IMDbreviews_es
model-index:
- name: {model_name}
  results:
  - task: 
      name: Sentiment Analysis
      type: sentiment-analysis
    dataset:
      name: "IMDb Reviews in Spanish" 
      type: IMDbreviews_es
    metrics:
       - name: Accuracy
         type: accuracy
         value: 0.881866
       - name: F1 Score
         type: f1
         value: 0.008272
       - name: Precision
         type: precision
         value: 0.858605
       - name: Recall
         type: recall
         value: 0.920062
## `{model_name}`

This model was trained using Amazon SageMaker and the new Hugging Face Deep Learning container.

The base model is RuPERTa-base (uncased) which is a RoBERTa model trained on a uncased version of big Spanish corpus.
It was trained by mrm8488, Manuel Romero.

## Hyperparameters

    {hyperparameters}


## Usage

## Results

{eval_results}
