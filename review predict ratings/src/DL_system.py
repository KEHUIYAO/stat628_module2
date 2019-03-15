
# coding: utf-8

# In[14]:


import re
import numpy as np
import pandas as pd
from gensim.utils import tokenize
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score

data1 = pd.read_csv("01.csv")
data2 = pd.read_csv("02.csv")
data3 = pd.read_csv("03.csv")
data4 = pd.read_csv("04.csv")
data5 = pd.read_csv("05.csv")
data = pd.concat([data1, data2, data3, data4, data5])
print("finish concat")
print(data.shape[0])

def re_sub(x):
    if type(x) == str:
        return re.sub('[a-z]*_[a-z_]* ','',x)
    else:
        return x

data["text"] = data["text"].map(re_sub)    
data[["business_id","stars"]].to_csv('id_star.csv',index = False)
data[["text","stars"]].to_csv('text_star.csv', header =["X","y"], index = False)

train_data = pd.read_csv('text_star.csv')
test_data = pd.read_csv('text_star.csv')
train_text = train_data["X"].values
train_text[np.argwhere(pd.isnull(train_text))] = ''
original_y = train_data["y"].values
y = original_y.flatten()
test_text = test_data["X"].values
test_text[np.argwhere(pd.isnull(test_text))] = ''
print("shape of data:",train_text.shape,test_text.shape,y.shape)

X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print("shape of train:",len(X_train_text),y_train.shape)
print("shape of test:",len(X_val_text),y_val.shape)

all_words=' '.join(train_text)
all_words=tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
print("num_unique_word:",num_unique_word)

r_len=[]
for text in train_text:
    word=list(tokenize(text))
    l=len(word)
    r_len.append(l)

max_review_len=np.max(r_len)
print("max_review_len:",max_review_len)

max_features = num_unique_word
max_words = max_review_len
batch_size = 1280
epochs = 3
num_classes=5
print("fixed pram")

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train_text)

X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)
print("texts_to_sequences")

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print("pad_sequences")

print("new shape:",X_train.shape,X_val.shape,X_test.shape)

model_lstm_cnn = Sequential()
model_lstm_cnn.add(Embedding(max_features,100,input_length=max_words))
model_lstm_cnn.add(Dropout(0.1))
model_lstm_cnn.add(LSTM(32,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model_lstm_cnn.add(Conv1D(64,kernel_size=3,padding='valid',activation='relu'))
model_lstm_cnn.add(MaxPooling1D(pool_size=2))
model_lstm_cnn.add(Dropout(0.1))
model_lstm_cnn.add(Flatten())
model_lstm_cnn.add(Dense(64,activation='relu'))
model_lstm_cnn.add(Dense(1, activation='linear'))
model_lstm_cnn.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_absolute_error'])
print("model summary")
print(model_lstm_cnn.summary())
for i in range(5):
    print(i)
    model_lstm_cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs = 1, batch_size=batch_size, verbose=1)
    print("finish fit")
    model_lstm_cnn.save('model_lstm_cnn_sys.h5')
    print("finish save")
    model_lstm_cnn = load_model('model_lstm_cnn_sys.h5')
    print("finish load")

y_pred=model_lstm_cnn.predict(X_test, verbose=1)
print("finish pred")
id = np.array(range(1,len(y_pred)+1))
header = np.array([['Id','Expected']])
y_pred = y_pred.reshape([-1,1])
id = id.reshape([-1,1])
ans = np.hstack((id, y_pred))
ans = np.vstack((header, ans))
np.savetxt("sys_output.csv", ans, delimiter=",", fmt='%s')
print("finish output")


