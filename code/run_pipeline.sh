#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="nlg_uncertainty"
``


run_id=`python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)"`

model='opt-350m'
srun python generate.py --num_generations_per_prompt='5' --model=$model --fraction_of_data_to_use='0.02' --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0'; srun python clean_generated_strings.py  --generation_model=$model --run_id=$run_id; python get_semantic_similarities.py --generation_model=$model --run_id=$run_id; python get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id; srun python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id; python compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id

