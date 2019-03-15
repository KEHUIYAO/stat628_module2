#!/usr/bin/env python
# coding: utf-8

# ## Part 1

# In[1]:


import pandas as pd
import numpy as np
import ast
import re
from collections import defaultdict


# ### load business data and review data

# In[251]:


trainSize=10000
business_train = pd.read_json('..//data//business_train.json',orient = 'records',lines = True)
review_train=pd.read_json('..//data//review_train.json',orient='records',lines=True,chunksize=trainSize)


# In[3]:


business_train.shape


# ### first look at the data structure of business_train

# category:

# In[4]:


business_train.categories.iloc[:5]


# In[5]:


businessCategory=business_train.categories.map(lambda x: x.split(',') if x!=None else [])
res=defaultdict(int)
for i in businessCategory:
    for j in i:
        res[j]+=1

res=sorted(res.items(),key=lambda x:x[1],reverse=True)


# By term appearance frequency, we have:

# In[6]:


for i in res[:10]:
    print("{} appears {} times in the category column.".format(i[0],i[1]))
    


# In this document, we mainly focus on the Restaurants, the analysis logic can also be applied on other types of business entities.

# ### filter all the Chinese Restaurants (let's do something simple first) 

# ### we change the filter function to read zihanz's topic model's contents

# In[241]:


#business_train=business_train[business_train.categories.map(lambda x:"Restaurants" in x and "Chinese" in x if x else False)]
#business_train=business_train[business_train.categories.map(lambda x:"Restaurants" in x  if x else False)]



id=pd.read_csv("../data/id.txt",header=None)
id=id[0]
topic=pd.read_csv("../data/yaoshen.csv")
res=[]
for i in range(topic.shape[0]):
    if topic.iloc[i,2] in id:
        res.append([topic.iloc[i,2],topic.iloc[i,10],topic.iloc[i,11],topic.iloc[i,12],topic.iloc[i,13]])
topic        


# In[242]:


def extract_from_topic(colIndex,data):
    res=[]
    for i in range(len(data)):
        if data[i][colIndex]==1:
            res.append(data[i][0])
    return res


# In[246]:


fourTopicIdList=[]
for i in range(1,5):
    fourTopicIdList.append(extract_from_topic(i,res))


# In[385]:


for i in range(1,4):
    business_train=business_train[business_train.business_id.isin(fourTopicIdList[i])]




    shop_stars=pd.read_json("average_star_by_shops.json",orient="records")


    # ### Extract nested json data from business attribute data

    # In[389]:


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


        for i in colnames:
            res[i].append(None)
        if attribute==None:
            count+=1
            continue
        for key,value in attribute.items():
            if not nestedValue(value):
                res[key][count]=value
            else:
                for nestKey,nestValue in ast.literal_eval(value).items():
                    res[nestKey][count]=nestValue


        count+=1

    attributeTrain=pd.DataFrame(res)

    attributeTrain.shape

    attributeTrain['business_id']=business_train.business_id.tolist()


    # ### join business data and review rating on business_id

    # In[390]:


    attributeTrain=attributeTrain.join(shop_stars.set_index('business_id'),on="business_id",how='left')


    # ### data cleaning, drop columns with many NAs, convert string to category

    # In[391]:


    attributeTrainReduced=attributeTrain.dropna(thresh=len(attributeTrain)*0.8, axis=1)

    attributeTrainReduced=attributeTrainReduced.drop("business_id",axis=1);





    # convert none to np.nan

    # In[392]:


    attributeTrainReduced.fillna(value=pd.np.nan, inplace=True)


    # convert 'none' to np.nan

    # In[393]:


    for i in range(attributeTrainReduced.shape[0]):
        for j in range(attributeTrainReduced.shape[1]):
            if attributeTrainReduced.iloc[i,j]=="None":
                attributeTrainReduced.iloc[i,j]=np.nan


    # 乱码清洗

    # In[394]:


    for i in range(attributeTrainReduced.shape[1]):
        for j in range(attributeTrainReduced.shape[0]):
            if pd.isna(attributeTrainReduced.iloc[j,i]):
                continue
            try:
                attributeTrainReduced.iloc[j,i]=re.sub('u\'(.*)\'','\\1',attributeTrainReduced.iloc[j,i])
                attributeTrainReduced.iloc[j,i]=re.sub('\'(.*)\'','\\1',attributeTrainReduced.iloc[j,i])
            except:
                break


    # ### try xgboost

    # In[395]:


    import xgboost
    from sklearn.model_selection import train_test_split
    from sklearn import model_selection
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(attributeTrainReduced.iloc[:,:-1]),attributeTrainReduced.iloc[:,-1], random_state=0)
    colLabels=pd.get_dummies(attributeTrainReduced.iloc[:,:-1]).columns


    # In[396]:


    # encode string class values as integers

    seed = 7

    # fit model no training data
    model = xgboost.XGBRegressor()
    model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],eval_metric='rmse',verbose=True)
    print(model)
    # make predictions for test data
    y_pred = model.predict(X_test)
    #print(model.feature_importances_)


    nameImportancePair=[(x,y) for x,y in zip(colLabels,model.feature_importances_)]
    nameImportancePair=sorted(nameImportancePair,key=lambda x:x[1],reverse=True)


    # In[397]:


    nameImportancePair


    # In[398]:


    import re
    def get_important_feature(x,y):
        """
        x is a dummy feature, y is the colnames list
        return the index of columns with that feature
        """
        originFeature=re.sub("()_.*","\\1",x)

        index=[True if re.match(originFeature,x) else False for x in y]

        return index



    # ### Do kruskal test on feature

    # In[399]:


    from  scipy.stats import kruskal
    def difference_test(data):
        """
        the input data is like this format: the last column is the average stars,
        other columns are attribute levels, this function will test if the attribute
        levels are significant factors. When we have two levels, we t test, else use
        anova
        """
        # kruskal test

        alpha=0.1
        res=[]
        for i in range(0,data.shape[1]-1):
            res.append(data[data.iloc[:,i]==1].iloc[:,-1].values)
        result,pValue=kruskal(*res)
        if pValue<alpha:
            largestStar=list(map(np.mean,res))
            starIndex=sorted(range(len(largestStar)), key=lambda k: largestStar[k])

            level=[data.columns[x] for x in starIndex[::-1]]
            return (True,level)
        else:
            return False




    # In[400]:


    dummyBusiness=pd.get_dummies(attributeTrainReduced.iloc[:,:-1])

    maxFeature=40

    testedFeature=[]

    for i in range(min(maxFeature,len(nameImportancePair))):

        if re.sub("()_.*","\\1",nameImportancePair[i][0]) in testedFeature:
            continue


        testedFeature.append(re.sub("()_.*","\\1",nameImportancePair[i][0]))

        dummyBusinessTemp=dummyBusiness.iloc[:,get_important_feature(nameImportancePair[i][0],colLabels)]


        dummyBusinessTemp['stars']=attributeTrainReduced.iloc[:,-1]

        print(difference_test(dummyBusinessTemp))

