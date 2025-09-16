from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_rnn(vocab_size, embed_dim=32, max_len=128):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        SimpleRNN(64, return_sequences=True),
        SimpleRNN(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(vocab_size, embed_dim=32, max_len=128):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru(vocab_size, embed_dim=32, max_len=128):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_dl_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history