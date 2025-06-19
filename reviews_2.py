import pandas as pd
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset


data = pd.read_csv('reviews_preprocessed.csv')
reviews = data.processed.values
all_words = ' '.join(reviews).split()
counter = Counter(all_words)    

vocabulary = sorted(counter, key= counter.get, reverse=True)
print(len(vocabulary))


int2word = dict(enumerate(vocabulary, 1))
int2word[0] = 'PAD'

word2int = {word: id for id, word in int2word.items()}


reviews_enc = [
    [word2int[word] for word in review.split()]
    for review in reviews                         
]

#print(reviews_enc)

sequence_len = 256
reviews_padding = np.full((len(reviews_enc), sequence_len), word2int['PAD'], dtype=int)


for i, row in enumerate(reviews_enc):
    reviews_padding[i, :len(row)] = np.array(row)[:sequence_len]


labels = data.label.to_numpy()

# Задаём долю обучающей и тестовой выборки
train_len = 0.6  
test_len = 0.5   


train_last_index = int(len(reviews_padding) * train_len)

# Разделяем данные на обучающую и остаточную часть
train_x = reviews_padding[:train_last_index]
remainder_x = reviews_padding[train_last_index:]

# То же самое делаем для меток классов
train_y = labels[:train_last_index]
remainder_y = labels[train_last_index:]


test_last_index = int(len(remainder_x)*test_len)
test_x = remainder_x[:test_last_index]
test_y = remainder_y[:test_last_index]

check_x = remainder_x[test_last_index:]
check_y = remainder_y[test_last_index:]


train_dataset = TensorDataset(
    torch.from_numpy(train_x), 
    torch.from_numpy(train_y)
)

test_dataset = TensorDataset(
    torch.from_numpy(test_x), 
    torch.from_numpy(test_y)
)

check_dataset = TensorDataset(
    torch.from_numpy(check_x), 
    torch.from_numpy(check_y)
)


from torch.utils.data import DataLoader

# Размер одного батча
batch_size = 128

# Создаём загрузчики данных для обучения, тестирования и проверки
train_loader = DataLoader(
    dataset=train_dataset, 
    shuffle=True, 
    batch_size=batch_size
)

test_loader = DataLoader(
    dataset=test_dataset, 
    shuffle=True, 
    batch_size=batch_size
)

check_loader = DataLoader(
    dataset=check_dataset, 
    shuffle=True, 
    batch_size=batch_size
)
