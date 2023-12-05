import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
args = parser.parse_args()


run_name = "q+a"

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir).cuda()

if args.model == 'opt-30b':
    accelerate.dispatch_model(model, device_map=config.device_map)

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}", use_fast=False, cache_dir=config.hf_cache_dir)

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

if args.dataset == 'coqa':
    dataset = datasets.load_from_disk(f'{config.output_dir}/coqa_dataset')
    id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))
elif args.dataset == 'trivia_qa':
    dataset = datasets.load_from_disk(f'{config.output_dir}/trivia_qa')

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset


def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset


if args.dataset == 'coqa':
    questions = encode_and_format_dataset(train_dataset)
elif args.dataset == 'trivia_qa':
    questions = train_dataset

dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


def get_generations(model, dataloader, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 256
        sequences = []
        for batch in tqdm.tqdm(dataloader):

            input_ids = torch.cat(batch['input_ids']).to(device).reshape(
                1, -1) if args.dataset == 'trivia_qa' else batch['input_ids'].to(device)
            if args.decoding_method == 'beam_search':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=5,
                                                        num_return_sequences=2,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)
            elif args.decoding_method == 'greedy':
                most_likely_generation = model.generate(input_ids,
                                                        num_beams=1,
                                                        do_sample=False,
                                                        max_length=input_ids.shape[1] +
                                                        max_length_of_generated_sequence,
                                                        eos_token_id=period_token_id,
                                                        bad_words_ids=question_framing_ids)

            input_length = input_ids.shape[1] if args.dataset == 'trivia_qa' else batch['input_ids'].shape[1]
            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):

                generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_ids.shape[1] + max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=args.temperature,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):

                if args.dataset == 'coqa':
                    sequence_dict = {
                        'prompt': batch['input_ids'][i].to('cpu'),
                        'generations': generations[i].to('cpu'),
                        'id': batch['id'],
                        'question': id_to_question_mapping[batch['id'][0]]
                    }
                elif args.dataset == 'trivia_qa':
                    few_shot_question = tokenizer.decode(input_ids[0])
                    question = few_shot_question.split('Question: ')[-1].split('Answer: ')[0]
                    sequence_dict = {
                        'prompt': input_ids[0],
                        'generations': generations[i],
                        'id': batch['question_id'],
                        'few_shot_question': tokenizer.decode(input_ids[0]),
                        'question': question
                    }

                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[len(batch['input_ids'][i]):], skip_special_tokens=True))

                sequence_dict['generated_texts'] = generated_texts
                sequence_dict['most_likely_generation_ids'] = most_likely_generation[0].to('cpu')
                sequence_dict['most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[0][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['second_most_likely_generation_ids'] = most_likely_generation[1].to('cpu')
                sequence_dict['second_most_likely_generation'] = tokenizer.decode(
                    most_likely_generation[1][len(batch['input_ids'][i]):], skip_special_tokens=True)

                sequence_dict['semantic_variability_reference_answers'] = batch[
                    'semantic_variability'] if 'semantic_variability' in batch else None
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]

                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None

                    sequence_dict[rouge_type + '_to_target'] = 0.0

                sequence_dict['answer'] = batch['answer']['text'] if args.dataset == 'coqa' else batch['answer']
                sequence_dict['additional_answers'] = [x[0] for x in batch['additional_answers']
                                                      ] if args.dataset == 'coqa' else None

                sequence_dict['exact_match'] = 0.0

                reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']
                                                              ] if args.dataset == 'coqa' else batch['answer']

                for answer in reference_answers:
                    predictions = [sequence_dict['most_likely_generation'].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                         references=references,
                                                         ignore_case=True,
                                                         ignore_punctuation=True)
                    sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(rouge_results[rouge_type].mid.fmeasure,
                                                                       sequence_dict[rouge_type + '_to_target'])

                sequences.append(sequence_dict)

    return sequences


sequences = get_generations(model, dataloader, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/sequences/' + run_name).mkdir(parents=True, exist_ok=True)

with open(f'{config.output_dir}/sequences/{run_name}/{args.model}_generations.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)
