# Deep Learning Text Classification Project
## Project Overview

This project investigates the application of binary sentiment analysis on Amazon review texts using various neural network architectures. We have implemented four models:
1. **Long Short-Term Memory networks (LSTM)**
2. **Gated Recurrent Unit (GRU) integrated with LSTM capabilities**
3. **Convolutional Neural Networks (CNN)**
4. **Transformer-based Distil BERT model**
The dataset comprises 3.6 million Amazon reviews, providing a substantial base for analyzing and classifying sentiments as positive or negative.
Our primary aim is to evaluate and compare the performance of these models in classifying review sentiments. We employ performance metrics such as accuracy, precision, recall, F1-score, and the Area Under the Receiver Operating Characteristic Curve (AUC) to gauge their efficacy. Preliminary results indicate competitive performance across all models, with the Distil BERT model demonstrating the highest accuracy of 96.30%, showcasing its superior capability in handling sentiment analysis tasks.

Additionally, a project report detailing our methodologies, results, and analysis has been included in the repository. While the code for CNN and Distil BERT is currently not available, we plan to polish and add this code to the repository shortly. Future work will focus on addressing limitations observed in misclassification cases, particularly in detecting sarcasm and handling mixed sentiments.
## Technologies Used
- Python
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas
- NumPy
## Installation
To set up this project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. Install required packages:
   ```bash
   pip install -r lstm_gru_requirements.txt
   ```
## Usage
To run the models, execute the following scripts:
- For LSTM:
   ```bash
   python scr/run_lstm.py
   ```
- For GRU-LSTM:
   ```bash
   python scr/run_gru_lstm.py
   ```
## Data
The dataset used for this project is sourced from Kaggle:
- **[Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)**  
  *Author:* Matthias Bittlingmayer  
  *Accessed:* 2024-03-24  
  *Year:* 2017
  
 ## The dataset required for this project consists of two files that need to be downloaded from the provided link and extracted into the Dataset/ directory:
-  train.ft.txt: Contains the training data used to build the model.
-  test.ft.txt: Contains the testing data for evaluating the model’s performance.]** 
  
## Results
Results including plots and metrics are saved in the `results/` directory:
- `lstm_results/`
- `gru_lstm_results/`
## Notebooks
In the `notebooks/` directory, you will find Jupyter notebooks for further exploration and analysis:
- **eda.ipynb**: Exploratory Data Analysis.
- **lstm_sentiment_analysis.ipynb**: LSTM model training and evaluation.
- **lstm_gru_sentiment_analysis.ipynb**: GRU-LSTM model training and evaluation.
