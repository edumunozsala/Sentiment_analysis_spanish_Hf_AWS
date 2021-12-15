from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
from distutils.dir_util import copy_tree

from transformers.trainer_utils import get_last_checkpoint

from dataclass import IMDbReviews

def copy_model_checkpoint(model_folder, ckpt_folder, dest_folder):
  ''' Copy the checkpoint folder of a HFmodel from the model folder to dest folder
      Input:
      - model_folder: folder containing the checkpoint saved
      - ckpt_dir: name or substring of the nameof the checkpoint folder to copy
      - dest_folder: folder where the checkpoint folder will be saved
  '''
  # Check if the model folder is a dir folder
  if os.path.isdir(model_folder):
    # Extract all checkpoint folders in the model folder
    ckpt_dir =[d for d in os.listdir(model_folder) if ckpt_folder in d]
    # Check if there is any checkpoint folder
    if ckpt_dir:
      # Sort checkpoint folder descending
      ckpt_dir.sort()
      print("Checkpoint folder to copy: ",ckpt_dir[0])
      # Copy the checkpoint folder to the destination folder with the same name
      copy_tree(os.path.join(model_folder,ckpt_dir[0]), os.path.join(dest_folder,ckpt_dir[0]))
      print(" Checkpoint folder copied to ",dest_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--train_filename", type=str, default="train_data.pt")
    parser.add_argument("--val_filename", type=str, default="val_data.pt")
    parser.add_argument('--checkpoint', type=str, metavar='N',
                        help='Resume training from the latest checkpoint (default: False)')
    parser.add_argument("--fp16", type=bool, default=True)

    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    args, _ = parser.parse_known_args()

    # make sure we have required parameters to push
    if args.push_to_hub:
        if args.hub_strategy is None:
            raise ValueError("--hub_strategy is required when pushing to Hub")
        if args.hub_token is None:
            raise ValueError("--hub_token is required when pushing to Hub")

    # sets hub id if not provided
    if args.hub_model_id is None:
        args.hub_model_id = args.model_name.replace("/", "--")

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    #train_dataset = load_from_disk(args.training_dir)
    #test_dataset = load_from_disk(args.test_dir)
    train_file=os.path.abspath(os.path.join(args.training_dir, args.train_filename))
    train_dataset=torch.load(train_file)
    val_file=os.path.abspath(os.path.join(args.val_dir, args.val_filename))
    val_dataset=torch.load(val_file)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(val_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        save_total_limit=1,
        overwrite_output_dir=True if get_last_checkpoint(args.model_dir) is not None else False,
        fp16=args.fp16, 
        load_best_model_at_end=True,
        #metric_for_best_model="accuracy",
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # train model
    if args.checkpoint is not None:
        print('Checkpoint: ', args.checkpoint)
        # Copy the checkpoint saved in the modelfolder
        copy_model_checkpoint('/opt/ml/checkpoints/', args.checkpoint, args.model_dir)
        print('Checkpoint copied: ', args.checkpoint)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=val_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    # Check if there is a checkpoint and move it to the checkpoint folder to upload it to S3
    copy_model_checkpoint(args.model_dir, 'checkpoint-','/opt/ml/checkpoints/')

