#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
import re
from collections import defaultdict


# In[2]:


business_train = pd.read_json('..//data//business_test.json',orient = 'records',lines = True)


# In[3]:


def nestedValue(x:str):
    """judge if a string contains nested json information"""
    return re.match(u'{.*}',x)

### first extract the column names from nested json
colnames=[]
for attribute in business_train.attributes:
    if attribute==None:
        continue
    for key,value in attribute.items():
        if not nestedValue(value):
            colnames.append(key)
        else:
            for nestKey,nestValue in ast.literal_eval(value).items():
                colnames.append(nestKey)
colnames=set(colnames)

### second extract the value from nested json file
res=defaultdict(list)
count=0
for attribute in business_train.attributes:
    if attribute==None:
        continue
    for i in colnames:
        res[i].append(None)
    for key,value in attribute.items():
        if not nestedValue(value):
            res[key][count]=value
        else:
            for nestKey,nestValue in ast.literal_eval(value).items():
                res[nestKey][count]=nestValue
            
    count+=1    


# In[5]:


pd.DataFrame(res).columns


# In[15]:


business_train = pd.read_json('business_train.json',orient = 'records',lines = True)


# In[16]:


business_train[business_train.business_id.isin([35344,150946,49924,182880,188661])]


# In[14]:


business_train.business_id

