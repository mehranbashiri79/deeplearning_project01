import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Function to Train GRU+LSTM Model
def train_gru_lstm(model, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size, model_name):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'results/gru_lstm_results/{model_name}_best_model.h5', monitor='val_loss', save_best_only=True)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])
    
    save_loss_accuracy_curves(history, 'gru_lstm_results', model_name)
    return model

# Save Loss and Accuracy Curves
def save_loss_accuracy_curves(history, folder_name, model_name):
    os.makedirs(f'results/{folder_name}', exist_ok=True)

    # Loss Curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'Loss Curve for {model_name}')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')    
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{folder_name}/{model_name}_loss_curve.png')
    plt.close()  

    # Accuracy Curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title(f'Accuracy Curve for {model_name}')  
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{folder_name}/{model_name}_accuracy_curve.png')
    plt.close()  
