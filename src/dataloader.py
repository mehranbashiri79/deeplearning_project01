import numpy as np

# Load Data from the text files
def get_labels_and_texts(file, num_samples=None):
    labels = []
    texts = []
    with open(file, "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            x = line.split(" ", 1)
            labels.append(int(x[0].split("__label__")[1]) - 1)
            texts.append(x[1].strip())
    return np.array(labels), texts
