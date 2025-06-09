import os
import glob
import json
import pandas as pd


def create_gpt_prompt(question):
    options = '\n'.join([f"{option['label']} {option['text']}" for option in question['options']])
    prompt = f"""Given the pathological image, answer the following question:

Question: {question['question']}
{options}

Please respond with the choice in the following Json format:
```Json
{{
    "option_letter": "Letter of your chosen option"
}}
```
"""
    return prompt


def parse_llm_output_to_json(output_text: str) -> dict:
    """
    Safely parse LLM output text into a Python dictionary.
    """
    start = output_text.find("{")
    end = output_text.rfind("}") + 1
    json_str = output_text[start:end]
    try:
        data = json.loads(json_str)
    except:
        data = {}
    return data

def parse_llm_response(response):
    """
    This function takes the LLM's response and extracts the chosen option letter.
    It assumes the response is a single letter.
    """
    valid_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Adjust based on your option labels
    if response.strip()[:1] in valid_options:
        return response.strip()[:1]
    data = parse_llm_output_to_json(response)
    option = data.get('option_letter', '')
    return option

def eval_llm(model, image_dir, question):
    folders = [folder for option in question['options'] for folder in option['folders']]
    image_list = []
    for folder in folders:
        image_list.extend(glob.glob(os.path.join(image_dir, folder, '*')))
    
    prompt = create_gpt_prompt(question)
    results = []
    
    for image_path in image_list:
        try:
            response = model.generate('', prompt, [image_path])
            print(response)
            option = parse_llm_response(response)
            if not option:
                print(f"Invalid response: {response}. Expected a valid option letter.")
        except Exception as e:
            print(f"Error: {e}")
            option = ''
        results.append((image_path, option))

    df = pd.DataFrame(results, columns=['file', 'pred_option'])
    return df


def eval_llm_batch(model, image_dir, questions):
    df_list = []
    for _, question in enumerate(questions):
        df = eval_llm(model, image_dir, question)
        df['question_id'] = question['id']
        df_list.append(df)
    return pd.concat(df_list)
