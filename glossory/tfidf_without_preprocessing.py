import json
import pandas as pd
import re
import numpy as np
import gc

trainSize=10000
testSize=100
train = pd.read_json('..//data//review_train.json',orient = 'records',lines = True,chunksize=trainSize)
train=next(train)
test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=testSize)
test=next(test)
#test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True)

trainText=train.iloc[:,3]
testText=test.iloc[:,3]
trainLabel=train.iloc[:,2].values
text=np.concatenate([trainText.values,testText.values])
del trainText
del testText
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
stopWord=['of','with','at','from','into','during',
               'including','until','till','against','among',
               'throughout','despite','towards','upon','concerning','to','in',
               'for','on','by','about','like','through','over',
               'before','between','after','since','without','under',
               'within','along','following','across','behind',
               'beyond','plus','except','but','up','out','around','down','off','above','near']
tf = TfidfVectorizer(analyzer='word', max_features=100,stop_words=stopWord ,lowercase = True)

text2vec=  tf.fit_transform(text)



#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
lr=LogisticRegression(multi_class='multinomial',solver='newton-cg')
#
#
X_train, X_test, y_train, y_test = train_test_split(text2vec[:trainSize,:], trainLabel, test_size=0.3, random_state=0)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print("the score is {}".format(mean_squared_error(pred,y_test)))

#%%
lr.fit(text2vec[:trainSize,:],trainLabel)
ylabel=lr.predict(text2vec[trainSize:,:])
ylabel=np.array(ylabel)
id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
with open("TueG1_submmit2.csv", 'wb') as f:
  f.write(b'Id,Expected\n')

  np.savetxt(f,ans,delimiter=",",fmt="%i,%i")

#%%

from sklearn.ensemble import RandomForestClassifier



modelRfc = RandomForestClassifier()
modelRfc.fit(X_train, y_train)


pred=modelRfc.predict(X_test)
print("the score is {}".format(mean_squared_error(pred,y_test)))














#%%
# import xgboost as xgb
# from xgboost import plot_importance
# from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error
# trainLabelXgboost=[x-1 for x in trainLabel]
#
#
# trainLabelXgboost=[x-1 for x in trainLabel]
# X_train, X_test, y_train, y_test = train_test_split(text2vec[:trainSize,:], trainLabelXgboost, test_size=0.3, random_state=0)
# #加载numpy的数组到DMatrix对象
# xg_train = xgb.DMatrix(X_train, label=y_train)
# xg_test = xgb.DMatrix( X_test, label=y_test)
# param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# # scale weight of positive examples
# param['eta'] = 0.1
# param['max_depth'] = 2
# param['silent'] = 1
# param['subsample']=0.8
# param['num_class'] = 5
# param['eval_metric']="mlogloss"
#
# watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
# num_round = 3
# bst = xgb.train(param, xg_train, num_round, watchlist )
# pred = bst.predict( xg_test );
# print("the score is {}".format(mean_squared_error(pred,y_test)))
#
# #%%
# param['objective'] = 'multi:softmax'
# xg_train = xgb.DMatrix(text2vec[:trainSize,:], label=trainLabelXgboost)
# bst = xgb.train(param, xg_train, num_round);
#
# xg_test = xgb.DMatrix(text2vec[trainSize:,:])
# ylabel = bst.predict( xg_test )
#
#
# ylabel=[x+1 for x in ylabel]
#
#
#
# ylabel=np.array(ylabel)
# id=np.array(range(1,len(ylabel)+1))
# #header=np.array([["Id","Expected"]])
# ylabel=ylabel.reshape([-1,1])
# id=id.reshape([-1,1])
# ans=np.hstack((id,ylabel))
# with open("TueG1_submmit2.csv", 'wb') as f:
#   f.write(b'Id,Expected\n')
#   #f.write(bytes("SP,"+lists+"\n","UTF-8"))
#   #Used this line for a variable list of numbers
#
#   np.savetxt(f,ans,delimiter=",",fmt="%i,%i")
#
#
