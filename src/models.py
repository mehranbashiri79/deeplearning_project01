from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, GRU, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model

# LSTM Model
def build_lstm_model(vocab_size, embedding_dim, max_seq_length, lstm_units, dropout_rate):
    inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length)(inputs)
    spatial_dropout = SpatialDropout1D(dropout_rate)(embedding)
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(spatial_dropout)
    pooling = GlobalMaxPooling1D()(lstm)
    dense = Dense(128, activation='relu')(pooling)
    dropout = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs, outputs)
    return model

# GRU + LSTM Model
def build_gru_lstm_model(vocab_size, embedding_dim, max_seq_length, lstm_units, dropout_rate):
    inputs = Input(shape=(max_seq_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length)(inputs)
    spatial_dropout = SpatialDropout1D(dropout_rate)(embedding)
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(spatial_dropout)
    gru = Bidirectional(GRU(lstm_units, return_sequences=True))(lstm)
    pooling = GlobalMaxPooling1D()(gru)
    dense = Dense(64, activation='relu')(pooling)
    dropout = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs, outputs)
    return model
