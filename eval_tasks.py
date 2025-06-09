import os
import sys
import json
import multiprocessing
from model.gpt import GPT4oVisionModel

try:
    from model.januspro_local import DeepSeekJanusPro
except:
    DeepSeekJanusPro = None

from model.vir2_clip import Vir2Clip
from model.conch import Conch
from model.qwen import QwenVisionModel
from model.pathgen import PathGenClip
from model.quiltnet import QuiltNetClip
from model.plip import PLIP
from model.musk import Musk
from model.context_l import ContextL
from model.pathgen_clip_l import PathGenClipL

from model.InternVL2_5_78B import InternVL2_5_78B
from model.Llama_3_2_90B_Vision import Llama_3_2_90B_Vision
# from model.NVLM_D_72B import NVLM_D_72B
from model.Qwen25_VL_72B import Qwen25_VL_72B
from model.random_model import RandomModel

# from benchmark.model.api_vllm import OpenVLLM

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
# result_dir = 'result_0224'

if not os.path.exists(f'../data/{result_dir}'):
    os.makedirs(f'../data/{result_dir}')

# Set up model parameters
models = [
    (Vir2Clip, 'clip_ckpt/vir2_clip_epoch_5.pt', f'../data/{result_dir}/vir2_results.csv', True),
    (Conch, 'CONCH/CONCH_pytorch_model_vitb16.bin', f'../data/{result_dir}/conch_results.csv', True),
    (QuiltNetClip, '/hpc2hdd/JH_DATA/share/hlong883/PrivateShareGroup/czhangcn_mdi_dataset_2024/model/PathologyCLIPs/qulitnet_open_clip_pytorch_model.bin', f'../data/{result_dir}/quiltnet_results.csv', True),
    (PathGenClip, '/hpc2hdd/JH_DATA/share/hlong883/PrivateShareGroup/czhangcn_mdi_dataset_2024/model/PathologyCLIPs/pathgenclip.pt', f'../data/{result_dir}/pathgen_results.csv', True),
    (Musk, '', f'../data/{result_dir}/musk_results.csv', True),
    (PLIP, '', f'../data/{result_dir}/plip_results.csv', True),

    (GPT4oVisionModel, None, f'../data/{result_dir}/gpt_results.csv', False),
    (QwenVisionModel, None, f'../data/{result_dir}/qwen_results.csv', False),
    (DeepSeekJanusPro, None, f'../data/{result_dir}/januspro_results.csv', False),

    (InternVL2_5_78B, None, f'../data/{result_dir}/internvl2_5_78b_results.csv', False),
    (Llama_3_2_90B_Vision, None, f'../data/{result_dir}/llama_3_2_90b_vision_results.csv', False),
    # (NVLM_D_72B, None, f'../data/{result_dir}/nvlm_d_72b_results.csv', False),
    (Qwen25_VL_72B, None, f'../data/{result_dir}/qwen25_vl_72b_results.csv', False),
    (RandomModel, None, f'../data/{result_dir}/random_results.csv', False),

    (ContextL, 'clip_ckpt/context-l-merged-l14-336-epoch-4.bin', f'../data/{result_dir}/context_l_results.csv', True),
    (PathGenClipL, 'clip_ckpt/pathgen-clip-l.pt', f'../data/{result_dir}/pathgen_clip_l_results.csv', True),

]


# /hpc2hdd/JH_DATA/share/czhangcn/PrivateShareGroup/czhangcn_hlong_model/model/pathgen-clip-l.pt

i = int(sys.argv[1])
if i >= len(models):
    exit()

# if len(sys.argv) > 2:
#     model_name = sys.argv[2]
#     save_name = model_name.replace('/', '_')
#     models.append((OpenVLLM, None, f'../data/{result_dir}/{save_name}_results.csv', False))

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
