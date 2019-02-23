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
# test=pd.read_json('..//data//review_test.json',orient = 'records',lines = True,chunksize=1000)
# count=0
# for i in test:
#     count+=i.shape[0]
# print(count)
count=1321274
#%%
print(trainingX.shape[0])
print(Y.shape[0]+count)



#%%
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(multi_class='multinomial',solver='newton-cg')
trainingXsub=trainingX[:100000,:]
y=Y[:trainingXsub.shape[0]]
y=y.iloc[:,0].tolist()
lr.fit(trainingXsub,y)
index=trainingX.shape[0]-count
y_pred=lr.predict(trainingX[index:,:])

#%%
ylabel=y_pred
#%%
ylabel=np.array(ylabel)
id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
#ans=np.vstack((header,ans))
np.savetxt("TueG1_submmit2.csv",ans,delimiter=",",fmt="%i,%i")


#%%

id=np.array(range(1,len(ylabel)+1))
#header=np.array([["Id","Expected"]])
y_pred=ylabel.reshape([-1,1])
id=id.reshape([-1,1])
ans=np.hstack((id,y_pred))
#ans=np.vstack((header,ans))
np.savetxt("TueG1_submmit2.csv",ans,delimiter=",",fmt="%i,%i")
