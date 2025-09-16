from src.data_loader import load_data, split_data
from src.eda import perform_eda
from src.preprocessing import preprocess_for_ml, preprocess_for_dl
from src.models.ml_models import get_baseline_models, get_advanced_ml_models
from src.models.dl_models import build_rnn, build_lstm, build_gru, train_dl_model
from src.evaluation import cross_validate, tune_model, evaluate_and_save
from src.utils import plot_history, logger
import pandas as pd

def main():
    # Load and Split
    df = load_data('data/IMDBSentimentData.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    
    # EDA
    perform_eda(df)
    
    # Preprocess
    X_train_ml, X_test_ml, _ = preprocess_for_ml(X_train, X_test)
    X_train_dl, X_test_dl, vocab_size, _ = preprocess_for_dl(X_train, X_test)
    
    # Results Dict
    results = {}
    
    # Baselines
    for name, model in get_baseline_models().items():
        acc, f1 = cross_validate(model, X_train_ml, y_train)
        results[name] = {'CV_Acc': acc, 'CV_F1': f1}
    
    # Advanced ML + Tuning
    for name, model in get_advanced_ml_models().items():
        best_params = tune_model(name, model, X_train_ml, y_train, n_trials=20)
        tuned_model = model.__class__(**best_params) if best_params else model
        acc, f1 = cross_validate(tuned_model, X_train_ml, y_train)
        results[name] = {'CV_Acc': acc, 'CV_F1': f1}
    
    # DL Models
    dl_models = {
        'RNN': build_rnn(vocab_size),
        'LSTM': build_lstm(vocab_size),
        'GRU': build_gru(vocab_size)
    }
    for name, model in dl_models.items():
        history = train_dl_model(model, X_train_dl, y_train, X_test_dl, y_test, epochs=50, batch_size=16)
        plot_history(history, name)
        acc, f1 = cross_validate(model, X_train_dl, pd.Series(y_train), is_dl=True)  # CV on DL (short epochs)
        results[name] = {'CV_Acc': acc, 'CV_F1': f1}
        # Final Test Eval
        test_loss, test_acc = model.evaluate(X_test_dl, y_test)
        logger.info(f"{name} Test Acc: {test_acc} Test Loss: {test_loss}")
    
    # Save Results
    evaluate_and_save(results)

if __name__ == "__main__":
    main()