#!/usr/bin/env python
# coding: utf-8

# ## Part 1

# In[1]:


### load necessary packages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import ast
import re
from collections import defaultdict
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

### load business data and review data
trainSize = 10000
business_train = pd.read_json('..//data//business_train.json', orient='records', lines=True)
review_train = pd.read_json('..//data//review_train.json', orient='records', lines=True, chunksize=trainSize)
topic = pd.read_csv("../data/yaoshen.csv")
business_train = business_train.set_index('business_id').join(
    topic.set_index('business_id')[['fastfood', 'bars', 'japan', 'asian']], how='right')
business_train = business_train[business_train.index.isin(id)]
business_train['business_id'] = business_train.index
business_train = business_train.reset_index(drop=True)

### train xgboost models and do kruskal tests on each category of data

# filter the data
for i in range(4):
    if i == 0:
        label = "fastfood"
        business_train_subset = business_train[business_train['fastfood'] == 1]
    elif i == 1:
        label = "bars"
        business_train_subset = business_train[business_train['bars'] == 1]
    elif i == 2:
        label = "japan"
        business_train_subset = business_train[business_train['japan'] == 1]
    else:
        label = "asian"
        business_train_subset = business_train[business_train['asian'] == 1]


    # extract the nested json attributes
    def nestedValue(x: str):
        """judge if a string contains nested json information"""
        return re.match(u'{.*}', x)


    colnames = []
    for attribute in business_train_subset.attributes:
        if attribute == None:
            continue
        for key, value in attribute.items():
            if not nestedValue(value):
                colnames.append(key)
            else:
                for nestKey, nestValue in ast.literal_eval(value).items():
                    colnames.append(nestKey)
    colnames = set(colnames)
    res = defaultdict(list)
    count = 0
    for attribute in business_train_subset.attributes:
        for i in colnames:
            res[i].append(None)
        if attribute == None:
            count += 1
            continue
        for key, value in attribute.items():
            if not nestedValue(value):
                res[key][count] = value
            else:
                for nestKey, nestValue in ast.literal_eval(value).items():
                    res[nestKey][count] = nestValue
        count += 1
    attributeTrain = pd.DataFrame(res)
    attributeTrain.shape
    attributeTrain['business_id'] = business_train_subset.business_id.tolist()

    # join business data and review rating on business_id
    shop_stars = pd.read_json("average_star_by_shops.json", orient="records")
    attributeTrain = attributeTrain.join(shop_stars.set_index('business_id'), on="business_id", how='left')

    # data cleaning convert string to category 
    attributeTrainReduced = attributeTrain
    attributeTrainReduced.fillna(value=pd.np.nan, inplace=True)  # fill np.nan to represent missing values
    for i in range(attributeTrainReduced.shape[0]):
        for j in range(attributeTrainReduced.shape[1]):
            if attributeTrainReduced.iloc[i, j] == "None" or attributeTrainReduced.iloc[
                i, j] == "none":  # convert 'none' to np.nan
                attributeTrainReduced.iloc[i, j] = np.nan
    for i in range(attributeTrainReduced.shape[1]):
        for j in range(attributeTrainReduced.shape[0]):
            if pd.isna(attributeTrainReduced.iloc[j, i]):
                continue
            if attributeTrainReduced.iloc[j, i] not in ['\'', 'u']:
                continue
            try:
                attributeTrainReduced.iloc[j, i] = re.sub('u\'(.*)\'', '\\1',
                                                          attributeTrainReduced.iloc[j, i])  # convert u'sth' to 'sth'
                attributeTrainReduced.iloc[j, i] = re.sub('\'(.*)\'', '\\1', attributeTrainReduced.iloc[j, i])
            except:
                break

    # give zhengdong zhou for further processing
    attributeTrainReduced.to_csv(label + ".csv")
    attributeTrainReduced = attributeTrainReduced.drop("business_id", axis=1)

    # xgboost for feature selection
    X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(attributeTrainReduced.iloc[:, :-1]),
                                                        attributeTrainReduced.iloc[:, -1],
                                                        random_state=0)  # split the data into training set and test set
    colLabels = pd.get_dummies(attributeTrainReduced.iloc[:, :-1]).columns  # all column names of the dummy dataset
    seed = 7  # set seed to repeat my work
    model = xgboost.XGBRegressor()  # set a xgboost model
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',
              verbose=False)  # fit the model using the training data
    y_pred = model.predict(X_test)  # make predictions for test data
    nameImportancePair = [(x, y) for x, y in zip(colLabels, model.feature_importances_)]
    nameImportancePair = sorted(nameImportancePair, key=lambda x: x[1],
                                reverse=True)  # sort the column names by feature importance in descending order


    def get_important_feature(x, y):
        """
        x is a dummy feature, y is the colnames list
        return the index of columns with that feature
        """
        originFeature = re.sub("()_.*", "\\1", x)
        index = [True if re.match(originFeature, x) else False for x in y]
        return index


    # Do kruskal test on feature
    from scipy.stats import kruskal


    def difference_test(data):
        """
        the input data is like this format: the last column is the average stars,
        other columns are attribute levels, this function will test if the attribute
        levels are significant factors. When we have two levels, we t test, else use 
        anova
        """
        # kruskal test
        alpha = 0.1  # if p-value is smaller than alpha, we reject the null hypothesis
        res = []
        for i in range(0, data.shape[1] - 1):
            res.append(data[data.iloc[:, i] == 1].iloc[:, -1].values)
        if len(res) < 2:
            return False
        result, pValue = kruskal(*res)  # this part is tricky, it's a pointer similar like c++
        if pValue < alpha:
            largestStar = list(map(np.mean, res))
            starIndex = sorted(range(len(largestStar)), key=lambda k: largestStar[k])
            level = [data.columns[x] for x in starIndex[::-1]]
            return (True, level)
        else:
            return False


    dummyBusiness = pd.get_dummies(attributeTrainReduced.iloc[:, :-1])
    maxFeature = 40
    testedFeature = []
    for i in range(min(maxFeature, len(nameImportancePair))):
        if re.sub("()_.*", "\\1", nameImportancePair[i][0]) in testedFeature:
            continue
        testedFeature.append(re.sub("()_.*", "\\1", nameImportancePair[i][0]))
        dummyBusinessTemp = dummyBusiness.iloc[:, get_important_feature(nameImportancePair[i][0], colLabels)]
        dummyBusinessTemp['stars'] = attributeTrainReduced.iloc[:, -1]
        print(difference_test(dummyBusinessTemp))
    print("------------------------------------------------------------")
