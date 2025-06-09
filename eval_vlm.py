import os
import sys
import json
import multiprocessing
from model.open_vlm import OpenVLLM

from eval.eval_clip import eval_clip_batch
from eval.eval_llm import eval_llm_batch

# Helper function to run clip evaluation and save the results
def run_clip_evaluation(model_class, model_path, image_dir, questions, output_file):
    clip_model = model_class(model_path)
    df = eval_clip_batch(clip_model, image_dir, questions)
    print(f"Finished {model_class.__name__} evaluation.")
    df.to_csv(output_file, index=False)

# Helper function to run LLM evaluation and save the results
def run_llm_evaluation(model_class, image_dir, questions, output_file):
    llm_model = model_class()
    df = eval_llm_batch(llm_model, image_dir, questions)
    print(f"Finished {model_class.__name__} evaluation.")
    df.to_csv(output_file, index=False)

# Load questions
with open('../data/question_with_prompt.json', 'r') as f:
    questions = json.load(f)
    # questions = [q for q in questions if q['id'] in [20,21,22,23]]

# image_dir = 'benchmark_paper/data/breast_patches初标校对版'
# image_dir = 'benchmark_paper/data/breast_patches初标校对版_test/'
image_dir = 'benchmark_paper/data/breast_patches_selected_onlyBPR_0304/breast_patches_selected_onlyBPR_0304'

result_dir = 'result_0306'

if not os.path.exists(f'../data/{result_dir}'):
    os.makedirs(f'../data/{result_dir}')

# Set up model parameters
models = []


i = int(sys.argv[1])
if i >= len(models):
    exit()

if len(sys.argv) > 2:
    model_name = sys.argv[2]
    save_name = model_name.replace('/', '_')
    models.append((OpenVLLM, None, f'../data/{result_dir}/{save_name}_results.csv', False))

model_class, model_path, output_file, is_clip_model = models[i]

if is_clip_model:
    run_clip_evaluation(model_class, model_path, image_dir, questions, output_file)
else:
    run_llm_evaluation(model_class, image_dir, questions, output_file)




# # Set up multiprocessing pool
# pool = multiprocessing.Pool(processes=2)

# # Run all evaluations in parallel
# for model_class, model_path, output_file, is_clip_model in models:
#     if is_clip_model:
#         pool.apply_async(run_clip_evaluation, (model_class, model_path, image_dir, questions, output_file))
#     else:
#         pool.apply_async(run_llm_evaluation, (model_class, image_dir, questions, output_file))

# # Close pool and wait for all tasks to complete
# pool.close()
# pool.join()
