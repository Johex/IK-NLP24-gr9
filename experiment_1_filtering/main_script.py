
# 🤗 stuff:
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,      # Model stuff
    Trainer, TrainingArguments, EarlyStoppingCallback       # Trainer stuff
)
import evaluate

# Other stuff:
import argparse
import numpy as np
import wandb
import time



def filter_dataset(dataset: DatasetDict, args: argparse.Namespace):
    """Filter the training dataset"""
    print("Filtering the training dataset.")

    original_len = len(dataset["train"])
    
    if args.filter_cols is None and args.filter_thv is None:
        print("No filters specified. Using full dataset!")
        return dataset, dict()
    
    # Dict specifying thresholds (values) for columns (keys)
    thv_dict = dict()

    # Get the entries from the command line input:
    filter_cols = args.filter_cols.split(",")
    filter_thv = [float(val) for val in args.filter_thv.split(",")]
    if len(filter_cols) != len(filter_thv):
        raise RuntimeError(f"--filter_cols specified {len(filter_cols)} values"
                           f", while --filter_thv specified {len(filter_thv)}."
                           " Make sure these have the same length!")
    
    # Filter out based on the thresholds
    for col, thv in zip(filter_cols, filter_thv):
        thv_dict[col] = thv
        dataset = dataset.filter(lambda x: float(x[col]) > thv)
    
    # Showing length to user :-D
    new_len = len(dataset["train"])
    print(f"Finished filtering. New size of training set is {new_len} "
          f"({100 * new_len / original_len:.2f}%) of original size.")

    return dataset, thv_dict

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

    out_dir = args.experiment + time.strftime("_%Y%m%d_%H%M%S")

    train_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
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

    trainer.save_model(out_dir = "/best/")

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
    training_set, thv_dict = filter_dataset(training_set, args)
    training_set = preprocess_dataset(training_set, args, True)

    # Optionally setting up wandb logging:
    if args.wandb_log:
        
        # Creating name of run:
        mn = args.model.replace("/", "_")
        trc = args.train_inp_cols
        tec = args.test_inp_cols
        thvs = "_".join([f"{key}={val}" for key, val in thv_dict.items()])
        run_name = f"{mn}_TRC={trc}_TEC={tec}_{thvs}"

        wandb.init(project=args.experiment, name=run_name)

    # Train the model:
    model, trainer = train_model(training_set, args)

    # Prepare testing set:
    print("Getting the testing set.")
    testing_set = load_dataset("maximedb/sick_nl", split="train+test+validation")
    testing_set = preprocess_dataset(testing_set, args, False)

    # Using entailment_AB as labels. This means A is premise, B is hypothesis:
    label_remap = {
        "A_entails_B": 0,
        "A_neutral_B": 1,
        "A_contradicts_B": 2
    }
    testing_set = testing_set.map(
        lambda x: {"label": label_remap[x["entailment_AB"]]})

    # Test the model:
    model.eval()
    test_results = trainer.predict(testing_set).metrics
    print(test_results)
    if args.wandb_log:
        wandb.log({"test/test_accuracy": test_results["test_accuracy"]})
        wandb.config.update(thv_dict) # Bit ugly to do here, but fuck it..

    with open("test_results.txt", "a") as out_file:
        out_file.write(
            f"{test_results['test_accuracy']},{args.model},"
            f"{args.train_inp_cols.replace(',', '&')},"
            f"{args.test_inp_cols.replace(',', '&')},"
            f"\"{' '.join([f'{key}={val}' for key, val in thv_dict.items()])}\""
            "\n"
        )

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