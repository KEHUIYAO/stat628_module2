#!/usr/bin/env python
# coding: utf-8

# In[72]:


# load supported packages
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
import numpy as np
np.random.seed(2018)
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


# In[90]:


trainSize=5000000
testSize=100
train = pd.read_json('..//data//review_train.json',orient = 'records',lines = True,chunksize=trainSize)
train=next(train)
#test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=testSize)
#test=next(test)
test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True)

trainText=train.iloc[:,3]
testText=test.iloc[:,3]
trainLabel=train.iloc[:,2].values
text=np.concatenate([trainText.values,testText.values])

del trainText
del testText


# In[91]:


def preprocess(text): # tokenize
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result


# In[92]:


reviewTrain = []
for i in text:
    reviewTrain.append(preprocess(i))
del text


# In[93]:


res=[]
for i in reviewTrain:
    res.append(i)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(res)]
model = Doc2Vec(documents)


# In[94]:


reviewVec=np.array([])
for i in res:
    reviewVec=np.append(reviewVec,model.infer_vector(i))
reviewVec=reviewVec.reshape(len(res),int(reviewVec.shape[0]/len(res)))


# In[95]:


lr=LogisticRegression(multi_class='multinomial',solver='newton-cg')
#
#
X_train, X_test, y_train, y_test = train_test_split(reviewVec[:trainSize,:], trainLabel, test_size=0.3, random_state=0)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print("the score is {}".format(mean_squared_error(pred,y_test)))


# In[74]:


#%%
lr.fit(reviewVec[:trainSize,:],trainLabel)
ylabel=lr.predict(reviewVec[trainSize:,:])
ylabel=np.array(ylabel)
id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
with open("TueG1_submmit4.csv", 'wb') as f:
  f.write(b'Id,Expected\n')

  np.savetxt(f,ans,delimiter=",",fmt="%i,%i")

