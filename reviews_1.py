import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
tqdm.pandas()
import nltk
nltk.download('wordnet')  # скачиваем лемматизатор WordNet

from nltk.stem import WordNetLemmatizer

data = pd.read_csv('reviews.csv')


data['label'] = data['sentiment'].progress_apply(lambda label: 1 if label == 'positive' else 0)

print(data)  # выводим для просмотра результата

tqdm.pandas()


lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english')) 


def preprocess_text(text):
    # Удаляем все символы, кроме букв, цифр и пробелов
    text = re.sub(r'[^\w\s]', '', text)

    # Заменяем последовательности пробелов (включая табы и переносы строк) на один пробел
    text = re.sub(r'\s+', ' ', text)

    # Удаляем цифры
    text = re.sub(r'\d', '', text)

    
    text = word_tokenize(text, language='english')
    text = [token for token in text if token not in stopwords]


    text = [lemmatizer.lemmatize(token) for token in text]
    text = [token for token in text if token not in stopwords]

    #print(text)
    return ' '.join(text)

data['processed'] = data['review'].progress_apply(preprocess_text)
data[['processed', 'label']].to_csv('reviews_preprocessed.csv', index=False, header=True)
