***

# June 2024 — [A new and improved implementation of Semantic Uncertainty is available]([url](https://github.com/jlko/semantic_uncertainty)), this repo is deprecated
We're excited to share a [new implementation of semantic uncertainty](https://github.com/jlko/semantic_uncertainty) which corresponds to our [2024 Nature paper](https://www.nature.com/articles/s41586-024-07421-0) _Detecting Hallucinations in Large Language Models Using Semantic Entropy_. Please use the new and improved version, we are deprecating this repository. Thank you for your interest!

This repository contains the code for our 2023 ICLR paper _Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation_.

***


### Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation

![image](https://user-images.githubusercontent.com/9898136/223775961-7f9525fc-9674-4bf4-b15f-d49487daddca.png)

# Overview

This repository contains the code used in Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation ([arXiv](https://arxiv.org/abs/2302.09664))

`run_pipeline.sh` is a slurm batch script that executes all steps of our pipeline.   `sbatch run_pipeline.sh` submits the batch script.

### Preprocessing & Config

`parse_triviaqa.py` and `parse_coqa.py`  load TriviaQA and CoQA from HuggingFace, tokenize it and store the data sets. These scripts only have to be run once. 

You'll also have to set the paths where you would like to store intermediate and final results of the pipeline in `config.py`.

The `environment.yml` lists the dependencies of the conda environment we used for our experiments.

### Generating answers and computing uncertainty measures

The components of our pipeline are:

* `generate.py` generates a number of answers for a subset of questions of a given data set. This step also evaluates the question-answering accuracy of the generated answers.
* `clean_generations.py` post-processes the generations from the first step, mainly by removing any unwanted trailing text, e.g. in cases where the model first gives the answer to the given question and then generates an additional question.
* `get_semantic_similarities.py` identifies semantic clusters in the generated set of answers from the previous step.
* `get_prompting_based_uncertainty.py` computes the p(True) baseline.
* `compute_likelihoods.py` computes the likelihoods of the generated answers under the generating model.
* `compute_confidence_measure.py` computes a range of different conficence/uncertainty measures such as the semantice entropy predictive entropy, lexical similarity, and p(True).

### Analyzing results

After running the pipeline, use `analyze_result.py` to compute performance metrics, such as the AUROC.

### Hardware requirements

Most model runs should run with at most 40GB of GPU memory. An exception are the experiments on OPT-30B which we run on two 80GB A100s.

### Dependencies

Our implemenetation uses PyTorch and HuggingFace. We use `wandb` to track our runs. environment
