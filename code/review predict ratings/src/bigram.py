
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
print("0:")

df = pd.read_csv('train_preprocessed_data.csv')
print("finish_reading")
new_texts = df["X"].values
new_texts[np.argwhere(pd.isnull(new_texts))] = ''
print("1:",len(new_texts))
sentence_stream = [sent.split(' ') for sent in new_texts]
print("2")
bigram = Phraser(Phrases(sentence_stream, min_count=5, threshold=5))
print("3")
sentence_with_phrase = bigram[sentence_stream]
print("4")
result = ['']
for i in range(len(new_texts)):
    result.append(' '.join(bigram[sentence_stream[i]]))
new_texts = result[1:]
print("5")
del sentence_stream
del result
print("6:",len(sentence_with_phrase))
print("7:",len(new_texts))
df["X"] = new_texts
df.to_csv('./train_preprocessed_bi_data.csv')
print("8")
df = pd.read_csv('test_preprocessed_data.csv')
new_texts = df["X"].values
new_texts[np.argwhere(pd.isnull(new_texts))] = ''
print("1:",len(new_texts))
sentence_stream = [sent.split(' ') for sent in new_texts]
print("2")
bigram = Phraser(Phrases(sentence_stream, min_count=5, threshold=5))
print("3")
sentence_with_phrase = bigram[sentence_stream]
print("4")
result = ['']
for i in range(len(new_texts)):
    result.append(' '.join(bigram[sentence_stream[i]]))
new_texts = result[1:]
print("5")
del sentence_stream
del result
print("6:",len(sentence_with_phrase))
print("7:",len(new_texts))
df["X"] = new_texts
df.to_csv('./test_preprocessed_bi_data.csv')

