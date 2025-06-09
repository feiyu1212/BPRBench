import pandas as pd
import numpy as np

def calculate_metrics(group):
    """Calculate sensitivity and specificity for a group"""
    # True labels (A is positive, B is negative)
    y_true = (group['option_label'] == 'A').astype(int)
    # Predicted labels
    y_pred = (group['pred_option'] == 'A').astype(int)
    
    # Calculate confusion matrix elements
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    
    return pd.Series({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'total_samples': len(group)
    })

def process_dataframe(df):
    """Calculate metrics for the three specified groupings and save to CSV"""
    # Group by model
    model_metrics = df.groupby('model').apply(calculate_metrics).reset_index()
    model_metrics.to_csv('average_senspec_per_model_tag.csv', index=False)
    print("Saved model metrics to 'average_senspec_per_model_tag.csv'")
    
    # Group by model, tag, question_id, question_text
    detailed_metrics = df.groupby(['model', 'tag', 'question_id', 'question_text']).apply(calculate_metrics).reset_index()
    detailed_metrics.to_csv('average_senspec_per_model_tag_question_id.csv', index=False)
    print("Saved detailed metrics to 'average_senspec_per_model_tag_question_id.csv'")
    
    # Group by model, tag
    tag_metrics = df.groupby(['model', 'tag']).apply(calculate_metrics).reset_index()
    tag_metrics.to_csv('average_senspec_per_model_tag.csv', index=False)
    print("Saved tag metrics to 'average_senspec_per_model_tag.csv'")
    
    return model_metrics, detailed_metrics, tag_metrics


if __name__ == "__main__":
    result_dir = 'result_0306'
    file_path = f'../data/{result_dir}/unique_df.csv'
    df = pd.read_csv(file_path)
    print(f"Using data from {file_path}")

    # Process the DataFrame and save metrics to CSV files
    model_metrics, detailed_metrics, tag_metrics = process_dataframe(df)
    
    # Display results summary
    print("\nModel Metrics (summary):")
    print(model_metrics[['model', 'sensitivity', 'specificity', 'total_samples']])
    
    print("\nModel-Tag Metrics (summary):")
    print(tag_metrics[['model', 'tag', 'sensitivity', 'specificity', 'total_samples']])