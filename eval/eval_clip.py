import traceback
import pandas as pd


def eval_clip(model, image_dir, question, i):
    correct_type = question['correct_type']
    folders = [folder for option in question['options'] for folder in option['folders']]
    classnames = [classname for option in question['options'] for classname in option['classnames']]
    classname_option = {classname: option['label']  for option in question['options'] for classname in option['classnames']}
    df = model.predict_df(image_dir, i=i, allowed_classes=folders)
    # if correct_type == 'max' use argmax, else use argmin
    if correct_type == 'max':
        df['pred'] = df[classnames].idxmax(axis=1)
    else:
        df['pred'] = df[classnames].idxmin(axis=1)
    df['pred_option'] = df['pred'].map(classname_option)
    return df


def eval_clip_batch(model, image_dir, questions):
    df_list = []
    for _, question in enumerate(questions):
        print(question['id'])
        try:
            i = question['id'] - 1
            df = eval_clip(model, image_dir, question, i)
            df = df[['file', 'pred', 'pred_option']]
            df['question_id'] = question['id']
            df_list.append(df)
        except:
            print(f'Error in {question["id"]} {question["question"]}')
            print(traceback.format_exc())

    df = pd.concat(df_list)
    return df