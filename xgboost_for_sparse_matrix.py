import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


def save_sparse_csr(filename, array):

    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):

    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

trainingX=load_sparse_csr("..//data//response")
#%%
Y=pd.read_csv("..//data//thirtyM_y.csv")
#%%
test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=1000)
count=0
for i in test:
    count+=i.shape[0]
print(count)

#%%
print(trainingX.shape[0])
print(Y.shape[0]+count)

#%%
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
trainingXsub=trainingX[:1000,:]
y=Y[:trainingXsub.shape[0]]
y=y.iloc[:,0].tolist()
y=[x-1 for x in y]
X_train, X_test, y_train, y_test = train_test_split(trainingXsub, y, test_size=0.3, random_state=0)
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
param['objective'] = 'multi:softprob'
index=trainingX.shape[0]-count
testX=trainingX[index:,:]
xg_test = xgb.DMatrix(testX)
#%%
yprob = bst.predict(xg_test)
#从预测的6组中选择最大的概率进行输出
#ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro
#%%
ylabel=yprob
ylabel=[x+1 for x in ylabel]

#%%
ylabel=np.array(ylabel)
id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
#ans=np.vstack((header,ans))
np.savetxt("TueG1_submmit2.csv",ans,delimiter=",",fmt="%i,%i")
