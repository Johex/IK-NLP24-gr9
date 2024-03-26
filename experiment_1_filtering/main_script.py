
# 🤗 stuff:
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,      # Model stuff
    Trainer, TrainingArguments, DataCollatorWithPadding     # Trainer stuff
)
import evaluate

# Other stuff:
import argparse
import os
import numpy as np
import wandb



def filter_dataset(dataset: DatasetDict, args: argparse.Namespace):
    """Filter the training dataset"""
    print("Filtering the training dataset.")

    original_len = len(dataset["train"])
    
    if args.filter_cols is None and args.filter_thv is None:
        print("No filters specified. Using full dataset!")
        return dataset
    
    # Get the entries from the command line input:
    filter_cols = args.filter_cols.split(",")
    filter_thv = [float(val) for val in args.filter_thv.split(",")]
    if len(filter_cols) != len(filter_thv):
        raise RuntimeError(f"--filter_cols specified {len(filter_cols)} values"
                           f", while --filter_thv specified {len(filter_thv)}."
                           " Make sure these have the same length!")
    
    # Filter out based on the thresholds
    for col, thv in zip(filter_cols, filter_thv):
        dataset = dataset.filter(lambda x: float(x[col]) > thv)
    
    # Showing length to user :-D
    new_len = len(dataset["train"])
    print(f"Finished filtering. New size of training set is {new_len} "
          f"({100 * new_len / original_len:.2f}%) of original size.")

    return dataset

def preprocess_dataset(dataset: DatasetDict,
                       args: argparse.Namespace,
                       is_train: bool) -> Dataset | DatasetDict:
    """Tokenize a given dataset such that it is ready for use."""
    print(f"Tokenizing dataset for {'train' if is_train else 'test'}ing.")
    
    # Get the colums we want as input:
    cols = args.train_inp_cols if is_train else args.test_inp_cols
    cols = cols.split(',')
    
    # Actual tokenization:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_data(data):
        inp = [data[col] for col in cols]
        return tokenizer(*inp, truncation=True)
        
    return dataset.map(tokenize_data, batched=True)

metric_acc = evaluate.load("accuracy")
def compute_metrics(predictions):
    """Compute metrics we want to evaluate our model (during eval and test)"""
    # TODO: extend with other relevant metrics ❗
    logits, labels = predictions
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

def train_model(dataset: DatasetDict, args: argparse.Namespace):
    """Function for the actual training loop and configuration. Configuration is
    fixed/hardcoded, as it should be the same for all experiments. Thats the
    whole point 🙂🤷🏻‍♂️
    """
    print("Training the model! Using 🤗 defaults, and batch_size=32.")

    # Getting the model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels = 3)
    
    # Setting up the trainer:
    #   · Setting it up such that it runs approximately 1h on my RTX 3060 Max-Q
    #   · Using the mostly 🤗 default parameters, as these are very similar to
    #     the ones recommended in the original BERT paper (arxiv 1810.04805).
    train_args = TrainingArguments(
        output_dir=args.experiment,
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=500,
        load_best_model_at_end=True,
        max_steps=17_000    # Approx 1 hour on my laptop ^^
    )

    if args.wandb_log:
        train_args.report_to = ["wandb"]

    train_ds = dataset["train"]
    eval_ds = dataset["validation"]
    # if len(eval_ds) > 500:
    #     eval_ds = eval_ds.shuffle(1996).select(range(500))

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=AutoTokenizer.from_pretrained(args.model), # for padding
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )

    if args.wandb_log:
        wandb.config.update({
            "pretrained_model": args.model,
            "train_cols": args.train_inp_cols,
            "test_cols": args.test_inp_cols,
            "train_ds_size": len(train_ds),
            "eval_ds_size": len(eval_ds)
        }) # TODO: add more, especially filter parameters

    trainer.train()

    return model, trainer

def main(args: argparse.Namespace):
    """❗The main function, executing the whole things ^^❗"""
    print("Running experiment with:\n"
          "    model:           " + args.model + "\n"
          "    train_cols:      " + args.train_inp_cols + "\n"
          "    test_cols:       " + args.test_inp_cols + "\n")
    # TODO: more info :-)
    
    # Prepare training set:
    print("Getting the training set.")
    training_set = load_dataset("GroNLP/ik-nlp-22_transqe")
    training_set = filter_dataset(training_set, args)
    training_set = preprocess_dataset(training_set, args, True)

    # Optionally setting up wandb logging:
    if args.wandb_log:
        wandb.init(
            project=args.experiment,
            name=f"todo") # TODO add config to name

    # Train the model:
    model, trainer = train_model(training_set, args)

    # Prepare testing set:
    print("Getting the testing set.")
    testing_set = load_dataset("maximedb/sick_nl", split="train+test+validation")
    testing_set = preprocess_dataset(testing_set, args, False)

    # Freaking idiots did the labels the other way around:
    label_remap = {0: 2, 1: 1, 2: 0}
    testing_set = testing_set.map(lambda x: {"label": label_remap[x["label"]]})

    # Test the model:
    model.eval()
    test_results = trainer.predict(testing_set).metrics
    print(test_results)
    wandb.log({"test/test_accuracy": test_results["test_accuracy"]})

    wandb.finish() # We might run this in a notebook later on so to be sure ^^

if __name__ == "__main__":
    
    # Defining all the CL arguments:
    parser = argparse.ArgumentParser(
        description=    "A simple script to quickly run different models and "
                        "different (filtered) versions of the dataset with the "
                        "exact same hyperparameters, etc."
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Name of the experiment for saving.")
    parser.add_argument("--model", type=str, required=True,
                        help=   "'🤗 transformers' model name "
                                "(e.g. 'GroNLP/bert-base-dutch-cased')")
    parser.add_argument("--train_inp_cols", type=str, required=True,
                        default="maximedb/sick_nl",
                        help="Columns for train input (comma separate them!)")
    parser.add_argument("--test_inp_cols", type=str, required=True,
                        default="maximedb/sick_nl",
                        help="Columns for test input (comma separate them!)")
    parser.add_argument("--wandb_log", action="store_true",
                        help="Log to wandb. Project name is experiment name.")
    
    # Filtering arguments:
    parser.add_argument("--filter_cols", type=str,
                        help="The columns in the training dataset to apply a "
                        "threshold filter to (comma separate them!). "
                        "IMPORTANT! --filter_thv must contain equally many "
                        "comma separated threshold values!")
    parser.add_argument("--filter_thv", type=str,
                        help="Threshold values corresponding to the colums "
                        "specified in --filter_cols (comma separate them!). "
                        "Will filter out everything below those values.")

    args = parser.parse_args()

    # Running the script:
    main(args)