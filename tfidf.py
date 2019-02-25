import json
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
train = pd.read_json('..//data//review_train.json',orient = 'records',lines = True,chunksize=10000)
train=next(train)
#%%
# test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=100)
# test=next(test)
#test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=100)
#test=next(test)
test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True)



#%%


def not_language(text):
    # First delete all common emoticons.
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)','',text)
    if re.sub('[\W]+','',text) == '':
        return True
    else:
        return False
not_lang_train = train[train.text.apply(not_language)].index.values
train.loc[not_lang_train,'lang_type'] = 'english'
from langdetect import detect
for i in range(train.shape[0]):
    if i in not_lang_train:
        continue
    else:
        train.loc[i,'lang_type'] = detect(train.text[i])



#%%
train_eng = train

#%%
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


wnl = WordNetLemmatizer()


def lemmatizer(text):
    tokens = word_tokenize(text)
    lemmas = []
    tagged = pos_tag(tokens)
    for tag in tagged:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return lemmas

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.pop(stop.index('but'))
stop.pop(stop.index('not'))
preposition = ['of','with','at','from','into','during',
               'including','until','till','against','among',
               'throughout','despite','towards','upon','concerning','to','in',
               'for','on','by','about','like','through','over',
               'before','between','after','since','without','under',
               'within','along','following','across','behind',
               'beyond','plus','except','but','up','out','around','down','off','above','near']
for prep in preposition:
    if prep in stop:
        stop.pop(stop.index(prep))

def no_abbreviation(text):
    text = re.sub('n\'t',' not',text)
    return text

but = ['yet','however','nonetheless','whereas','nevertheless']
although = ['although','though','notwithstanding','albeit']

def change_but(text):
    for x in but:
        text = re.sub(x,'but',text)
    return text
def change_although(text):
    for x in although:
        text = re.sub(x,'although',text)
    return text
def change_adversatives(text):
    text = change_but(text)
    text = change_although(text)
    return text

def preprocessing(text):
    # 取表情
    # emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    # 去回车
    text = re.sub('\\n',' ',text)
    # not
    # text = no_abbreviation(text)
    # 只保留字母
    text = re.sub('[\W]+',' ', text.lower())
    # 统一转折词
    # text = change_adversatives(text)
    # 词性还原
    # tokens = lemmatizer(text)
    # text = ''
    # for index, token in enumerate(tokens):
        # if token in stop:
        #     tokens[index] = ''
        # else:
        #     text = text + tokens[index] + ' '
    # return {'text':text,'emoticons':emoticons}
    return {'text':text}

from tqdm import tqdm, tqdm_pandas
tqdm.pandas()
dictionary_train = train_eng.text.progress_apply(preprocessing)
dictionary_test = test.text.progress_apply(preprocessing)
texts_train = [dictionary_train[i]['text'] for i in train_eng.index]
texts_test = [dictionary_test[i]['text'] for i in test.index]
y = train_eng.loc[dictionary_train.index]["stars"]

#%%
# 释放空间
import gc
del train_eng
del test
del train
del dictionary_train
del dictionary_test
gc.collect()


num_train = len(texts_train)
num_test = len(texts_test)

texts_train.extend(texts_test)

del texts_test
gc.collect()

texts = texts_train

del texts_train
gc.collect()

from autocorrect import spell

new_texts = ['']
for i in tqdm(range(len(texts))):
    new_texts.append([spell(j) for j in texts[i].split(' ')])

new_texts = new_texts[1:]

result = ['']
for i in range(len(new_texts)):
    result.append(' '.join(new_texts[i]))

new_texts = result[1:]

new_texts = texts

del texts
gc.collect()

from gensim.models.phrases import Phrases, Phraser

sentence_stream = [sent.split(' ') for sent in tqdm(new_texts)]

bigram = Phraser(Phrases(sentence_stream, min_count=5, threshold=5)) #mincount越小识别出来的越少，threshold higher means fewer phrases

sentence_with_phrase = bigram[sentence_stream]

result = ['']
for i in tqdm(range(len(new_texts))):
    result.append(' '.join(bigram[sentence_stream[i]]))

new_texts = result[1:]

del sentence_stream
del sentence_with_phrase
del result
gc.collect()






#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer='word', min_df = 1, lowercase = True)

response =  tf.fit_transform(new_texts)



feature_name=tf.get_feature_names()
tfidf_train = response[:num_train]
tfidf_test = response[num_train:]
#%%
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(multi_class='multinomial',solver='newton-cg')
lr.fit(tfidf_train,y)
y_pred=lr.predict(tfidf_test)

#%%
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
try:
    y=y.tolist()
    y=[x-1 for x in y]
except:
    pass
X_train, X_test, y_train, y_test = train_test_split(tfidf_train, y, test_size=0.3, random_state=0)
#加载numpy的数组到DMatrix对象
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix( X_test, label=y_test)
#1.训练模型
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 5

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 6
bst = xgb.train(param, xg_train, num_round, watchlist )

pred = bst.predict( xg_test );
print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
#%%
#2.probabilities
# do the same thing again, but output probabilities
param['objective'] = 'multi:softmax'
xg_train = xgb.DMatrix(tfidf_train, label=y)
bst = xgb.train(param, xg_train, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
#%%
xg_test = xgb.DMatrix(tfidf_test)
ylabel = bst.predict( xg_test )


ylabel=[x for x in ylabel]


#%%
ylabel=np.array(ylabel)
id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
#ans=np.vstack((header,ans))

with open("TueG1_submmit2.csv", 'wb') as f:
  f.write(b'Id,Expected\n')
  #f.write(bytes("SP,"+lists+"\n","UTF-8"))
  #Used this line for a variable list of numbers

  np.savetxt(f,ans,delimiter=",",fmt="%i,%i")


