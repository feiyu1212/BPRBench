import os
import pandas as pd
from parser.parse_label import get_label_df

def calculate_accuracy(full_df, result_dir):
    # 1. Create unique_df with unique combinations of ['file', 'model', 'question_id', 'option_label', 'pred_option']
    unique_df = full_df.drop_duplicates(subset=['file', 'model', 'question_id', 'option_label', 'pred_option'])

    # 2. Calculate 'correct' column where prediction matches the label
    unique_df['correct'] = unique_df['option_label'] == unique_df['pred_option']

    # 2.a. Calculate total accuracy for each model group
    total_accuracy = unique_df.groupby('model')['correct'].mean().reset_index()
    total_accuracy.rename(columns={'correct': 'total_accuracy'}, inplace=True)

    # 2.b. Calculate average accuracy for each model and question_id group
    avg_accuracy_model_qid = unique_df.groupby(['model', 'question_id', 'question_text'])['correct'].mean().reset_index()
    avg_accuracy_model_qid.rename(columns={'correct': 'average_accuracy'}, inplace=True)

    # 3. Calculate 'correct' column for full_df if not already done
    # (If you prefer to keep full_df unchanged, create a copy)
    full_df['correct'] = full_df['option_label'] == full_df['pred_option']

    # 4. Calculate average accuracy for each model and tag group
    avg_accuracy_model_tag = full_df.groupby(['model', 'tag'])['correct'].mean().reset_index()
    avg_accuracy_model_tag.rename(columns={'correct': 'average_accuracy'}, inplace=True)

    # 5. Save the results to CSV files
    unique_df.to_csv(f'../data/{result_dir}/unique_df.csv', index=False)
    total_accuracy.to_csv(f'../data/{result_dir}/total_accuracy_per_model.csv', index=False)
    avg_accuracy_model_qid.to_csv(f'../data/{result_dir}/average_accuracy_per_model_question_id.csv', index=False)
    avg_accuracy_model_tag.to_csv(f'../data/{result_dir}/average_accuracy_per_model_tag.csv', index=False)

    print("All computations are complete and results have been saved to CSV files.")


# Define the base directory
# base_dir = 'benchmark_paper/data/breast_patches初标校对版_test/'
base_dir = 'benchmark_paper/data/breast_patches_selected_onlyBPR_0304/breast_patches_selected_onlyBPR_0304/'
question_file = f'../data/question_with_prompt.json'

label_df = get_label_df(base_dir, question_file)

# result_dir = 'result_0226_test_1819'
result_dir = 'result_0306'

model_file = [
    ('MUSK', f'../data/{result_dir}/musk_results.csv'),
    ('PLIP', f'../data/{result_dir}/plip_results.csv'),
    ('conch_clip', f'../data/{result_dir}/conch_results.csv'),
    ('vir2_clip', f'../data/{result_dir}/vir2_results.csv'),
    ('pathgen_clip', f'../data/{result_dir}/pathgen_results.csv'),
    ('quiltnet_clip', f'../data/{result_dir}/quiltnet_results.csv'),
    ('gpt4o', f'../data/{result_dir}/gpt_results.csv'),
    ('qwen-vl-max', f'../data/{result_dir}/qwen_results.csv'),
    ('januspro', f'../data/{result_dir}/januspro_results.csv'),
    ('InternVL2_5-26B-MPO', f'../data/{result_dir}/InternVL2_5-26B-MPO_results.csv'),
    ('llava-v1.5-7b-xtuner', f'../data/{result_dir}/llava-v1.5-7b-xtuner_results.csv'),
    ('llava-v1.5-13b-xtuner', f'../data/{result_dir}/llava-v1.5-13b-xtuner_results.csv'),
    ('cogvlm2-llama3-chat-19B', f'../data/{result_dir}/cogvlm2-llama3-chat-19B_results.csv'),
    ('InstructBLIP-7B', f'../data/{result_dir}/InstructBLIP-7B_results.csv'),
    ('InstructBLIP-13B', f'../data/{result_dir}/InstructBLIP-13B_results.csv'),

    ('InternVL2.5-78B', f'../data/{result_dir}/internvl2_5_78b_results.csv'),
    ('Llama-3.2-90B-Vision', f'../data/{result_dir}/llama_3_2_90b_vision_results.csv'),
    # (NVLM_D_72B, None, f'../data/{result_dir}/nvlm_d_72b_results.csv', False),
    ('Qwen25-VL-72B', f'../data/{result_dir}/qwen25_vl_72b_results.csv'),
    ('Random', f'../data/{result_dir}/random_results.csv'),

    ('Context-L', f'../data/{result_dir}/context_l_results.csv'),
    ('PathGen-Clip-L', f'../data/{result_dir}/pathgen_clip_l_results.csv'),

]




df_list = []

# print("Label DF columns:", label_df.columns.tolist())
# print("Pre DF columns:", pre_df.columns.tolist())

# print(label_df)

# 检查 label_df 是否为空
assert not label_df.empty, "Label DF is empty!"



for model, file in model_file:
    if not os.path.exists(file):
        continue
    print(f'### {file}')
    pre_df = pd.read_csv(file)
    pre_df = pd.merge(label_df, pre_df)
    print(pre_df)
    pre_df['model'] = model
    pre_df = pre_df[['model', 'tag', 'question_id', 'question_text', 'file', 'option_label', 'pred_option']]
    df_list.append(pre_df)


full_df = pd.concat(df_list)
calculate_accuracy(full_df, result_dir)
