# Sentiment Analysis on spanish IMDbreviews using a Ruperta model

This repository is inspired in some Hugginface tutorials and contains notebooks to explore the text dataset and to train the model. There is only a notebook on how to upload the model to the Gugginface Hub.

In the main notebook we will describe the most relevant steps to train a Hugginface model in AWS SageMaker, showing how to deal with experiments and solving some of the problems when facing with custom models when using SageMaker script mode on. Some basics concepts on SageMaker will not be detailed in order to focus on the relevant concepts.

Following steps will be explained: 
 
1. Create an Experiment and Trial to keep track of our experiments

2. Load the training data to our training instance and create train, validation and test dataset and upload to S3

3. Create the scripts to train our Huggingface model, a RoBERTa based model pretrained in a spanish corpus: RuPERTa.

4. Create an Estimator to train our model in a huggingface container in script mode

4. Download and deploy the trained model to make predictions

5. Create a Batch Transform job to make predictions for the test dataset 

## Finetune the RuPERTa model for a Sentiment Analysis task


**On development**
*Different models are under development*

## Problem description
**On development**

## The data set
**On development**

## Content

**On development**

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License.