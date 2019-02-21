#%% import packages


import pandas as pd
import os




fileDir = os.path.dirname(os.path.realpath('__file__'))
train= os.path.join(fileDir, '../data/review_train.json')


# In[10]:


train = pd.read_json(train,orient = 'records',lines = True)


# In[6]:


test = pd.read_json('data/review_test.json',orient = 'records',lines = True)


# In[1]:


import re
import numpy as np 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet


# #### Language

# In[ ]:


def not_language(text):
    # First delete all common emoticons.
    text = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)','',text)
    if re.sub('[\W]+','',text) == '':
        return True
    else:
        return False


# In[ ]:


not_lang_train = train[train.text.apply(not_language)].index.values


# In[ ]:


train.loc[not_lang_train,'lang_type'] = 'english'


# In[ ]:


from langdetect import detect
for i in range(train.shape[0]):
    if i in not_lang:
        continue
    else:
        train.loc[i,'lang_type'] = detect(train.text[i])


# In[ ]:


train_eng = train[train.lang_type == 'en']


# #### Lemmatization

# In[ ]:


porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[ ]:


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


# #### Stop-words

# In[ ]:


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


# #### Convert n't to not

# In[ ]:


def no_abbreviation(text):
    text = re.sub('n\'t',' not',text)
    return text


# #### Adversatives

# In[ ]:


but = ['yet','however','nonetheless','whereas','nevertheless']
although = ['although','though','notwithstanding','albeit']


# In[ ]:


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


# In[ ]:


def preprocessing(text):
    # 取表情
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    # 去回车
    text = re.sub('\\n',' ',text)
    # not
    text = no_abbreviation(text)
    # 只保留字母
    text = re.sub('[\W]+',' ', text.lower())
    # 统一转折词
    text = change_adversatives(text)
    # 词性还原
    tokens = lemmatizer(text)
    text = ''
    for index, token in enumerate(tokens):
        if token in stop:
            tokens[index] = ''
        else:
            text = text + tokens[index] + ' '
    return {'text':text,'emoticons':emoticons}


# In[ ]:


from tqdm import tqdm, tqdm_pandas
tqdm.pandas()
dictionary_train = train_eng.text.progress_apply(preprocessing)
dictionary_test = test.text.progress_apply(preprocessing)


# In[ ]:


y = train_eng.loc[dictionary.index]["stars"]


# In[ ]:


texts_train = [dictionary_train[i]['text'] for i in train_eng.index]
texts_test = [dictionary_test[i]['text'] for i in test.index]


# In[ ]:


texts = np.append(texts_train,texts_test)


# In[ ]:


from autocorrect import spell

new_texts = ['']
for i in tqdm(range(len(texts))):
    new_texts.append([spell(j) for j in texts[i].split(' ')])

new_texts = new_texts[1:]


# In[ ]:


result = ['']
for i in range(len(new_texts)):
    result.append(' '.join(new_texts[i]))
    
new_texts = result[1:]


# #### Bigrams for phrase

# In[ ]:


from gensim.models.phrases import Phrases, Phraser


# In[ ]:


sentence_stream = [sent.split(' ') for sent in new_texts]


# In[ ]:


bigram = Phraser(Phrases(sentence_stream, min_count=5, threshold=5)) #mincount越小识别出来的越少，threshold higher means fewer phrases


# In[ ]:


sentence_with_phrase = bigram[sentence_stream]


# In[ ]:


result = ['']
for i in range(len(new_texts)):
    result.append(' '.join(bigram[sentence_stream[i]]))
    
new_texts = result[1:]


# #### tf-idf

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tf = TfidfVectorizer(analyzer='word', min_df = 1, lowercase = False)


# In[ ]:


response =  tf.fit_transform(new_texts).toarray()


# In[ ]:


tfidf_train = response[:len(texts_train)]
tfidf_test = response[len(texts_train):]


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(multi_class='multinomial',solver='newton-cg’')
lr.fit(tfidf_train,y)
y_pred=lr.predict(tfidf_test)

