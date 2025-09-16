import os
import logging
import matplotlib.pyplot as plt

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_history(history, model_name, output_dir='results/'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss.png'))
    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.savefig(os.path.join(output_dir, f'{model_name}_acc.png'))