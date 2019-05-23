
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# In[4]:


data=pd.read_csv("C:\\Users\\Subh\\for digit recognization\\dataset\\digit-recognizer\\train.csv")
data.head()


# In[5]:


image=data.iloc[:,1:]
label=data.iloc[:,:1]


# # using knn to classify the images

# In[8]:


image=data.iloc[:,1:]
label=data.iloc[:,:1]
from sklearn.neighbors import KNeighborsClassifier as kn
knn= kn(n_neighbors=10)
x_train,x_test,y_train,y_test = train_test_split(image,label,test_size = 0.2,random_state = 100) 
knn.fit(x_train,y_train)


# In[ ]:


predic=knn.predict(x_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.accuracy_score(y_test,predic)


# In[ ]:


df=pd.read_csv("C:\\Users\\Subh\\for digit recognization\\dataset\\digit-recognizer\\test.csv")


# In[ ]:


predic=knn.predict(df)


# In[ ]:


lise=[]
for i in range(0,len(predic)):
    lise.append(predic[i])
    
    
    
 


# In[ ]:


li=[]
for i in range(1,len(predic)+1):
 li.append(i)


# In[ ]:


x=pd.DataFrame()
x["ImageId"]=li


# In[ ]:


x["Label"]=lise


# In[ ]:


predic2= pd.DataFrame(data=x)


# In[54]:


predic2.to_csv("file_name.csv", encoding='utf-8', index=False)

