import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import ngrams
from collections import Counter
from src.utils import logger
import os

def perform_eda(df: pd.DataFrame, output_dir: str = 'results/'):
    """Enhanced EDA to inform model choices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic Info
    logger.info("Data Info:")
    logger.info(df.info())
    logger.info(df.describe(include='object'))
    logger.info("Sentiment Distribution (%):")
    logger.info(df['Sentiment'].value_counts(normalize=True) * 100)
    logger.info("Missing Values:")
    logger.info(df.isnull().sum())
    
    # Review Length
    df['Review_Length'] = df['Review'].apply(lambda x: len(x.split()))
    logger.info("Review Length Stats:")
    logger.info(df['Review_Length'].describe())
    
    # Plots
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Review_Length'], bins=50, kde=True)
    plt.title('Review Length Distribution')
    plt.savefig(os.path.join(output_dir, 'review_length_dist.png'))
    
    sns.boxplot(x='Sentiment', y='Review_Length', data=df)
    plt.title('Length by Sentiment')
    plt.savefig(os.path.join(output_dir, 'length_by_sentiment.png'))
    
    # Word Clouds (inform: common words suggest BoW/TF-IDF for ML)
    positive = ' '.join(df[df['Sentiment'] == 'positive']['Review'])
    negative = ' '.join(df[df['Sentiment'] == 'negative']['Review'])
    WordCloud().generate(positive).to_file(os.path.join(output_dir, 'positive_wordcloud.png'))
    WordCloud(background_color='black').generate(negative).to_file(os.path.join(output_dir, 'negative_wordcloud.png'))
    
    # N-grams (inform: sequences suggest RNN/LSTM for context)
    def get_ngrams(text, n=2, top_k=10):
        return Counter(ngrams(text.split(), n)).most_common(top_k)
    
    logger.info("Top Bigrams Positive:")
    logger.info(get_ngrams(positive))
    logger.info("Top Bigrams Negative:")
    logger.info(get_ngrams(negative))
    
    # Observations
    observations = """
    EDA Observations:
    - Balanced sentiments: No need for heavy oversampling.
    - Short reviews dominant: Padding to ~128 words sufficient; longer ones suggest RNNs for sequence handling.
    - Common words/phrases: TF-IDF good for ML baselines; n-grams show context, favoring LSTMs over simple ML.
    - Outliers in length: May need truncation to avoid sparsity in models.
    """
    logger.info(observations)
    with open(os.path.join(output_dir, 'eda_observations.txt'), 'w') as f:
        f.write(observations)