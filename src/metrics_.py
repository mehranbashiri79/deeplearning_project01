import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import pandas as pd
import os
import seaborn as sns

# Evaluate the model and save metrics and plots
def evaluate_model(model, X_test, y_test, folder_name, model_name):
    # Make predictions
    predictions = model.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(int)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions_binary)
    plot_confusion_matrix(conf_matrix, folder_name, model_name)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plot_roc_curve(fpr, tpr, folder_name, model_name)

    # Metrics
    metrics = {
        'precision': precision_score(y_test, predictions_binary),
        'recall': recall_score(y_test, predictions_binary),
        'f1_score': f1_score(y_test, predictions_binary),
    }

    save_metrics_table(metrics, folder_name, model_name)
    return metrics

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, folder_name, model_name):
    os.makedirs(f'results/{folder_name}', exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'results/{folder_name}/{model_name}_confusion_matrix.png')
    plt.close()  

# Plot ROC Curve
def plot_roc_curve(fpr, tpr, folder_name, model_name):
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc='lower right')
    
    os.makedirs(f'results/{folder_name}', exist_ok=True)
    plt.savefig(f'results/{folder_name}/{model_name}_roc_curve.png')
    plt.close()  

# Save metrics as CSV
def save_metrics_table(metrics, folder_name, model_name):
    os.makedirs(f'results/{folder_name}', exist_ok=True)  
    df = pd.DataFrame([metrics])
    df.to_csv(f'results/{folder_name}/{model_name}_metrics_table.csv', index=False)
