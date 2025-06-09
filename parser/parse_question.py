import json
import re

with open('../data/question_raw.txt', 'r') as f:
    raw_data = f.read()

def parse_question_data(raw_data):
    questions = []
    # Split the raw data into questions based on numbered question headers
    question_pattern = r'(\d+\..*?)(?=\d+\.|$)'
    matches = re.findall(question_pattern, raw_data, re.DOTALL)

    for match in matches:
        question_parts = match.strip().split('\n')
        question_text = question_parts[0].strip()
        correct_type = 'min' if '|min' in question_text else 'max'
        question_text = question_text.split('|')[0].strip()
        question_id, question_text = question_text.split('.', 1)
        
        # Now parse the options (A, B, C, D)
        options = []
        for option in question_parts[1:]:
            option_match = re.match(r"([A-Z])\.?:?\s*(.*)\s+(\{.*\})", option.strip())
            if option_match:
                option_label = option_match.group(1).strip()
                option_text = option_match.group(2).strip()
                option_data = json.loads(option_match.group(3).strip())
                folders = option_data.get('pos_list', []) + option_data.get('neg_list', [])
                folders = list(set(folders))
                options.append({
                    'label': option_label,
                    'text': option_text,
                    'classnames': option_data['prompt_list'],
                    'folders': folders
                })
        questions.append({
        	'id': int(question_id.strip()),
            'question': question_text.strip(),
            'options': options,
            'correct_type': correct_type
        })
    
    return questions

# Parse the raw data into a structured format
structured_data = parse_question_data(raw_data)

with open('../data/question_with_prompt.json', 'w+') as f:
    f.write(json.dumps(structured_data, indent=4))
