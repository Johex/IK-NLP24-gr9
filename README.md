# Lost in Translation: Evaluating the Efficacy of Automated Translations for Multilingual NLP

We explored the impact of the quality of the machine-translated dataset on the performance of the model. We investigated how automatically generated quality estimates can 
improve the overall performance of models trained for Natural Language Inference on datasets translated from English to Dutch

## Table of Contents

- [Lost in Translation](#about)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## About <a id="about"></a>

Provide a more detailed description of your project. Explain what it does, why it's useful, and any other relevant information.

We investigated how important is the quality of the translated data to train models in other languages. We explored how automatic estimation of translation quality can improve the performance of models trained on translated data.

We assessed how the estimated quality of translated instances affects the performance of models trained on them, notably considering a subset of high-quality translated training instances. 
We later tested the performance of the trained models on the Dutch SICK-NL corpus and its English counterpart, SICK, as these have been manually created by human annotators. 

### Datasets Used

- **Training Dataset**: [GroNLP/ik-nlp-22_transqe](https://huggingface.co/datasets/GroNLP/ik-nlp-22_transqe)

- **Testing Dataset**: [maximedb/sick_nl](https://huggingface.co/datasets/maximedb/sick_nl)

### Project Tasks

1. **In-depth Analysis of Model Performance**: We conducted an in-depth analysis to examine how the performance of selected models is influenced by filtering examples based on Quality Estimation (QE) scores at various thresholds.

2. **Evaluation of Dutch Model Performance**: We evaluated the performance of a Dutch model on translated data and compared it with a multilingual model such as mBERT.

<!-- 3. **Effect of QE-based Data Selection on NLI Models**: We investigated how Quality Estimation (QE)-based data selection impacts the performance of Natural Language Inference (NLI)-trained models on sentences. -->

3. **Weighted Loss Modification**: We modified the classification model to utilize a weighted loss during fine-tuning, where Quality Estimation (QE) scores were used as weights to assign lower weight to examples with poor translation quality.


### Project Files Overview

1. **experiment_0_data_analysis/quality_distribution.ipynb**:
   - Description: This notebook contains data visualization to understand the distribution of data in terms of `da_hypothesis`, `da_premise`, `mqm_hypothesis`, and `mqm_premise` columns of the dataset. Additionally, it defines the `value_from_percentage` function to find the threshold value of the scores when provided with the percentage of total data.

2. **experiment_1_filtering/main_script.py**:
   - Description: The `main_script.py` file contains the code for all experiments.

#### Functionality Breakdown in main_script.py:

**main():**

* Takes arguments for the experiment configuration.
* Prints details about the chosen model and input/test columns.
* Calls functions for data filtering, preprocessing, training, and evaluation.

**filter_dataset():**

* Filters the training dataset based on provided arguments (columns and thresholds).
* If no arguments are given, uses the full dataset for training.
* Prints messages about the filtering process.

**preprocess_dataset():**

* Preprocesses the input columns (train and test) using tokenization based on the specified model.
* Returns the preprocessed dataset.

**wandb_log (Optional):**

* Logs experiment details and tracks resource usage (CPU/GPU) using W&B (Weights & Biases).

**train_model():**

* Trains the model on the provided training dataset.
* Logs training information to W&B (if specified).
* Returns the trained model and trainer object.

### Training Arguments

The following training arguments were used for all experiments:

```python
train_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    max_steps=17_000
)
```

**main_script.py Workflow:**

1. Fetches the GroNLP/ik-nlp-22_transqe dataset.
2. Filters the dataset based on threshold values (if provided).
3. Preprocesses the training dataset (tokenization).
4. Trains the model on the preprocessed training dataset.
5. Fetches the maximedb/sick_nl dataset.
6. Preprocesses the test dataset (tokenization).
7. Evaluates the trained model on the preprocessed test dataset.
8. Appends test results to a text file ("text_results.txt").

**Output:**

* Trained Model.
* Evaluation results on the test dataset (written to "text_results.txt").

**Notes:**

* The script requires additional libraries like `wandb` for W&B logging.
* Ensure the script runs successfully even without the optional `--wandb_log` argument.


## Installation <a id="installation"></a>

Provide instructions on how to install your project. Include any prerequisites or dependencies that need to be installed separately.

To install **[IK-NLP24-gr9]**, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Johex/IK-NLP24-gr9.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd IK-NLP24-gr9
    ```

3. **Install dependencies:**

    Use a package manager such as pip to install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

---

By following these steps, you should have **[IK-NLP24-gr9]** installed and ready to use on your system. If you encounter any issues during the installation process, please contact us for assistance.

--- 

## Usage <a id="usage"></a>

The file **experiment_1_filtering/batch_main_script.py** contains the code for all experiments runs. It can be used to run specific experiments.

### Command Line Arguments

The following command line arguments are available for running the project:

- `--experiment`: Name of the experiment for saving.
- `--model`: Hugging Face Transformers model name.
- `--train_inp_cols`: Columns for train input (comma-separated).
- `--test_inp_cols`: Columns for test input (comma-separated).
- `--wandb_log`: Flag to enable logging/tracking of experiments.
- `--filter_cols`: Columns in the training dataset to apply a threshold filter to (comma-separated).
- `--filter_thv`: Threshold values corresponding to the columns specified in `--filter_cols` (comma-separated). This will filter out everything below those values.

#### Example Command

To run the experiment, use the `main` function with the following command:

```bash
main("--experiment ik-nlp-mt-quality-filter --model google-bert/bert-base-multilingual-cased --train_inp_cols premise_nl,hypothesis_nl --test_inp_cols sentence_A,sentence_B --wandb_log")
```




