import json

import evaluate
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

with open(f'{config.data_dir}/coqa-dev-v1.0.json', 'r') as infile:
    data = json.load(infile)['data']

rouge = evaluate.load('rouge')

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")

model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

dataset = {}

dataset['story'] = []
dataset['question'] = []
dataset['answer'] = []
dataset['additional_answers'] = []
dataset['rouge1'] = []
dataset['rouge2'] = []
dataset['rougeL'] = []
dataset['semantic_variability'] = []
dataset['id'] = []

for sample_id, sample in enumerate(data):
    story = sample['story']
    questions = sample['questions']
    answers = sample['answers']
    additional_answers = sample['additional_answers']
    for question_index, question in enumerate(questions):
        dataset['story'].append(story)
        dataset['question'].append(question['input_text'])
        dataset['answer'].append({
            'text': answers[question_index]['input_text'],
            'answer_start': answers[question_index]['span_start']
        })
        dataset['id'].append(sample['id'] + '_' + str(question_index))
        additional_answers_list = []

        for i in range(3):
            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

        dataset['additional_answers'].append(additional_answers_list)
        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
        if not story[-1] == '.':
            story = story + '.'
        all_answers = [answers[question_index]['input_text']] + additional_answers_list

        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []

        # This computes the syntactic similarity across the reference answers
        for i, reference_answer in enumerate(all_answers):
            for j in range(4):
                if i != j:
                    answer_list_1.append(all_answers[i])
                    answer_list_2.append(all_answers[j])

                    qa_1 = question['input_text'] + ' ' + all_answers[i]
                    qa_2 = question['input_text'] + ' ' + all_answers[j]

                    input = qa_1 + ' [SEP] ' + qa_2

                    inputs.append(input)
                    #print(encoded_input)

        encoded_input = tokenizer.batch_encode_plus(inputs, padding=True)

        prediction = model(torch.tensor(encoded_input['input_ids'], device='cuda'))['logits']

        predicted_label = torch.argmax(prediction, dim=1)
        if 0 in predicted_label:
            has_semantically_different_answers = True

        dataset['semantic_variability'].append(has_semantically_different_answers)

        results = rouge.compute(predictions=answer_list_1, references=answer_list_2)
        dataset['rouge1'].append(results['rouge1'].mid.fmeasure)
        dataset['rouge2'].append(results['rouge2'].mid.fmeasure)
        dataset['rougeL'].append(results['rougeL'].mid.fmeasure)

dataset_df = pd.DataFrame.from_dict(dataset)

dataset = Dataset.from_pandas(dataset_df)

dataset.save_to_disk(f'{config.data_dir}/coqa_dataset')
