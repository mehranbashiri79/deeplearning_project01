import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import get_labels_and_texts
from src.preprocessing import normalize_texts, tokenize_and_pad
from src.models import build_lstm_model
from src.train_lstm_model import train_lstm
from src.metrics_ import evaluate_model

# Load and preprocess data
train_labels, train_texts = get_labels_and_texts("Dataset/train.ft.txt", num_samples=3600000)
test_labels, test_texts = get_labels_and_texts("Dataset/test.ft.txt", num_samples=400000)

train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)

max_length = max(len(text) for text in train_texts)
train_texts, tokenizer = tokenize_and_pad(train_texts, max_features=12000, max_length=max_length)
test_texts, _ = tokenize_and_pad(test_texts, max_features=12000, max_length=max_length)

# Build and Train LSTM Model
lstm_model = build_lstm_model(12000, 128, max_length, 128, 0.1)
lstm_model = train_lstm(lstm_model, train_texts, train_labels, test_texts, test_labels, 0.000025, 60, 128, 'lstm_model')

# Evaluate LSTM Model
metrics = evaluate_model(lstm_model, test_texts, test_labels, 'lstm_results', 'lstm_model')
print(metrics)
