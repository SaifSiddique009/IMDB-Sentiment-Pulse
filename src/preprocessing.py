from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import logger

def preprocess_for_ml(X_train, X_test, max_features=10000):
    """TF-IDF for ML models."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("TF-IDF vectorization complete.")
    return X_train_vec, X_test_vec, vectorizer

def preprocess_for_dl(X_train, X_test, max_len=128, vocab_size=None):
    """Tokenize and pad for DL."""
    tokenizer = Tokenizer(oov_token='<OOV>', num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=max_len)
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"DL preprocessing complete. Vocab size: {vocab_size}")
    return X_train_pad, X_test_pad, vocab_size, tokenizer