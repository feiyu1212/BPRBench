import pandas as pd

data = pd.read_csv('/Users/ann/Downloads/average_senspec_per_question_id.csv')


# Define the order for the models
model_order = ['Random', 'conch_clip', 'vir2_clip', 'MUSK', 'pathgen_clip', 
               'quiltnet_clip', 'InternVL2.5-78B', 'Llama-3.2-90B-Vision', 
               'gpt4o', 'InternVL2_5-26B-MPO', 'qwen-vl-max', 'Qwen25-VL-72B', 
               'llava-v1.5-7b-xtuner', 'InstructBLIP-7B', 'llava-v1.5-13b-xtuner', 
               'InstructBLIP-13B', 'januspro']

rename_dict = {
  'Random': 'Random Choice',
  'conch_clip': 'CONCH',
  'vir2_clip': 'Vir2-CLIP',
  'MUSK': 'MUSK',
  'pathgen_clip': 'Pathgen',
  'quiltnet_clip': 'Quiltnet',
  'InternVL2.5-78B': 'InternVL2.5-78B-MPO',
  'Llama-3.2-90B-Vision': 'Llama-3.2-90B-Vision',
  'gpt4o': 'GPT4o',
  'InternVL2_5-26B-MPO': 'InternVL2.5-26B-MPO',
  'qwen-vl-max': 'Qwen-VL-MAX',
  'Qwen25-VL-72B': 'Qwen2.5-VL-72B',
  'llava-v1.5-7b-xtuner': 'llava-v1.5-7b-xtuner',
  'InstructBLIP-7B': 'InstructBLIP-7B',
  'llava-v1.5-13b-xtuner': 'llava-v1.5-13b-xtuner',
  'InstructBLIP-13B': 'InstructBLIP-13B',
  'januspro': 'januspro',
}

# Group by model and question_id, then combine sensitivity and specificity
grouped_data = data.groupby(['model', 'question_id']).agg({
    'sensitivity': 'first',
    'specificity': 'first'
}).reset_index()

# Create a new column with the formatted sensitivity/specificity string
grouped_data['sens_spec'] = grouped_data.apply(
    lambda row: f"{row['sensitivity']:.4f}/{row['specificity']:.4f}", 
    axis=1
)

# Pivot the DataFrame using the combined values
pivot_df = grouped_data.pivot(index='model', columns='question_id', values='sens_spec')

# Reset column names (to get rid of the name 'question_id')
pivot_df.columns.name = None

# Sort the index (rows) according to the specified model order
# Only include models that are actually in our data
models_in_data = [model for model in model_order if model in pivot_df.index]
pivot_df = pivot_df.loc[models_in_data]

# Convert to the desired format
result_df = pivot_df.reset_index()
result_df['model'] = result_df['model'].apply(lambda x: rename_dict[x])

# Save to CSV
result_df.to_csv('./average_senspec_per_question_id_ss.csv', index=False)


