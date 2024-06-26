from main_script import main
import argparse

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

    run_args = [
        # 100% of the dataset:
        # "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log"
        # "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log",

        # 50% da:
        "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.2624,0.2624",
        "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.2624,0.2624",

        # 25% da:
        "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.4861,0.4861",
        "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.4861,0.4861",

        # 10% da:
        # "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.6651,0.6651",
        # "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis --filter_thv 0.6651,0.6651"

        # 50% by pruning both da and mqm:
        "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.201648,0.201648,0.0986,0.0986"
        # "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.201648,0.201648,0.0986,0.0986"

        # 25% by pruning both da and mqm:
        "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.3926,0.3926,0.111188,0.111188"
        # "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.3926,0.3926,0.111188,0.111188"

        # 10% by pruning both da and mqm:
        # "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.555,0.555,0.1219,0.1219"
        # "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols da_premise,da_hypothesis,mqm_premise,mqm_hypothesis --filter_thv 0.555,0.555,0.1219,0.1219"
    ]

    # These runs were already done, except the 10% ones
    # run_args = [
    #     # 100% of the dataset:
    #     "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log",
    #     "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log",
        
    #     # 50% mqm:
    #     "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.107,0.107",
    #     "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.107,0.107",
        
    #     # 25% mqm:
    #     "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.1183,0.1183",
    #     "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.1183,0.1183",
        
    #     # 10% mqm:
    #     "--experiment ik-nlp-mt-quality-filter --model GroNLP/bert-base-dutch-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.127,0.127",
    #     "--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-cased --train_inp_cols premise_en,hypothesis_en --test_inp_cols sentence_A_original,sentence_B_original --wandb_log --filter_cols mqm_premise,mqm_hypothesis --filter_thv 0.127,0.127",
    # ]
    
    for run in run_args:
        args = parser.parse_args(run.split())
        main(args)
