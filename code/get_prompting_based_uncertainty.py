# Read generation results
import argparse
import os
import pickle
import random

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
#sns.color_palette("pastel")
from config import device_map

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

device = torch.device('cuda')

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-1.3b')
parser.add_argument('--run_id_for_few_shot_prompt', type=str, default='run_1')
parser.add_argument('--run_id_for_evaluation', type=str, default='run_1')
args = parser.parse_args()

model_name = "opt-350m"

generation_tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m", use_fast=False, cache_dir=config.data_dir)
model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda()

if model_name == 'opt-30b':
    accelerate.dispatch_model(model, device_map=device_map)
    print(model.hf_device_map)
    #device = torch.device('cuda:1')

run_name = "q+a"

with open(f'{config.output_dir} /{run_name}/{model_name}_generations.pkl', 'rb') as infile:
    sequences_for_few_shot_prompt = pickle.load(infile)

# Build few shot prompt

subset_of_sequences_for_few_shot_prompt = sequences_for_few_shot_prompt[-10:]
number_of_few_shot_samples = 5

prompt_template = 'Question: {} \n Here are some ideas that were brainstormed:{}\n Possible answer:{}\n Is the possible answer:\n (A) True\n (B) False\n The possible answer is:'
few_shot_promopt = ''
for sequence in subset_of_sequences_for_few_shot_prompt:
    question = sequence['question']
    question = question.split('Question: ')[-1].split('Answer: ')[0]
    prompt = sequence['prompt']
    generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])

    most_likely_answer = sequence['most_likely_generation']
    correct = ' True' if sequence['rougeL_to_target'] > 0.3 else ' False'
    few_shot_promopt += prompt_template.format(question, generated_texts, most_likely_answer) + correct + '\n'

# Build prompt for question
labels_across_datasets = []
p_trues_across_datasets = []

n_samples_to_use = 2000

with torch.no_grad():

    aurocs = []
    p_trues = []
    corrects = []
    for sequence in tqdm(sequences_for_few_shot_prompt[:n_samples_to_use]):

        question = sequence['question']
        if 'Question: ' in question:
            question = question.split('Question: ')[-1].split('Answer: ')[0]
        else:
            question = question.split('Q: ')[-1].split('A: ')[0]

        generated_texts = '\n'.join(sequence['cleaned_generated_texts'][:number_of_few_shot_samples])
        most_likely_answer = sequence['most_likely_generation']
        correct = 1.0 if sequence['rougeL_to_target'] > 0.3 else 0.0
        base_prompt = prompt_template.format(question, generated_texts, most_likely_answer)
        prompt_true = few_shot_promopt + prompt_template.format(question, generated_texts, most_likely_answer) + ' True'

        # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
        tokenized_base_prompt = generation_tokenizer(base_prompt)['input_ids']
        tokenized_prompt_true = torch.tensor(generation_tokenizer(prompt_true)['input_ids'], device=device)

        target_ids_true = tokenized_prompt_true.clone()
        target_ids_true[:len(tokenized_base_prompt)] = -100

        model_output_true = model(torch.reshape(tokenized_prompt_true, (1, -1)), labels=target_ids_true)
        loss_true = model_output_true.loss

        p_trues.append(loss_true.item())
        corrects.append(correct)

        labels_across_datasets += corrects
        p_trues_across_datasets += p_trues

    p_true_auroc = sklearn.metrics.roc_auc_score(1 - torch.tensor(corrects), torch.tensor(p_trues))

    # Store p_true aurocs in a pickle file
    with open(f'{config.output_dir}/{run_name}/{model_name}_p_true_aurocs.pkl', 'wb') as outfile:
        pickle.dump(p_true_auroc, outfile)
