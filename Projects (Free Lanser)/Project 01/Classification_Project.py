#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as pl
import numpy as np 
import pandas as pd


# In[41]:


from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics  
from sklearn import svm 
from sklearn.metrics import jaccard_score  
from sklearn.metrics import f1_score  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier
import itertools


# # CSV-File

# In[3]:


df = pd.read_csv("output.csv",encoding = "ISO-8859-1",low_memory=False , error_bad_lines=False)
print(df.describe())
print("\nDataSet shape:"+str(df.shape))


# In[4]:


#Remove Nan
df = df[df.notna()]
print("\nDataSet shape (Nan Free):"+str(df.shape))


# In[5]:


#LabelEncoder --> token labling 
label = LabelEncoder() 
df.token = label.fit_transform(df.token)
print(df.token.describe())


# In[6]:


#LabelEncoder --> pos_tag labling 
label = LabelEncoder() 
df.pos_tag = label.fit_transform(df.pos_tag)
print(df.pos_tag.describe())


# In[7]:


print(df.ezafe_tag.describe())


# In[8]:


print(df.describe())


# In[9]:


print('token :\n',df['token'].value_counts())
print('pos_tag :\n',df['pos_tag'].value_counts())
print('ezafe_tag :\n',df['ezafe_tag'].value_counts())
print('\nData Types:\n',df.dtypes)


# Seperate Inputs and Outputs

# In[10]:


Inputs = df[{'token','pos_tag'}]
print(Inputs.hist())


# In[11]:


#Normalize Data
scaler = preprocessing.StandardScaler().fit(Inputs)
Inputs = scaler.transform(Inputs.astype(float))
print("Normalized Data:\n",Inputs)


# In[12]:


Outputs = df[{'ezafe_tag'}]
print(Outputs.hist())


# such a big data ! 
# Lets divide ourr DataSet To 8 set for KNN (K=4) Algorithm

# # KNN-Classification

# First Train set Accuracy:  0.9993862108429691
# First Test set Accuracy:  0.99864
# 
# Second Train set Accuracy:  0.9993510458391809
# Second Test set Accuracy:  0.99816
# 
# Third Train set Accuracy:  0.9994213758467573
# Third Test set Accuracy:  0.99784
# 
# Fourth Train set Accuracy:  0.9993606362947595
# Fourth Test set Accuracy:  0.9984
# 
# Fifth Train set Accuracy:  0.9994853122172813
# Fifth Test set Accuracy:  0.998
# 
# Sixth Train set Accuracy:  0.9994373599393883
# Sixth Test set Accuracy:  0.99808
# 
# Seventh Train set Accuracy:  0.9993894076614953
# Seventh Test set Accuracy:  0.9976
# 
# Eigth Train set Accuracy:  0.9993989981170739
# Eigth Test set Accuracy:  0.99792
# 
# Test set Accuracy(Average): 0.99808

# In[13]:


test_and_train = 0.0384247689
X_train, X_test, y_train, y_test = train_test_split(Inputs[0:325311],Outputs[0:325311], test_size=test_and_train, random_state=4)
print ('First Train set:', X_train.shape,  y_train.shape)
print ('First Test set:', X_test.shape,  y_test.shape)


# In[14]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("First Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("First Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[325311:650622],Outputs[325311:650622], test_size=test_and_train, random_state=4)
print ('Second Train set:', X_train.shape,  y_train.shape)
print ('Second Test set:', X_test.shape,  y_test.shape)


# In[40]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Second Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Second Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[650622:975933],Outputs[650622:975933], test_size=test_and_train, random_state=4)
print ('Third Train set:', X_train.shape,  y_train.shape)
print ('Third Test set:', X_test.shape,  y_test.shape)


# In[42]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Third Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Third Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[975933:1301244],Outputs[975933:1301244], test_size=test_and_train, random_state=4)
print ('Fourth Train set:', X_train.shape,  y_train.shape)
print ('Fourth Test set:', X_test.shape,  y_test.shape)


# In[44]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Fourth Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Fourth Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[1301244:1626555],Outputs[1301244:1626555], test_size=test_and_train, random_state=4)
print ('Fifth Train set:', X_train.shape,  y_train.shape)
print ('Fifth Test set:', X_test.shape,  y_test.shape)


# In[46]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Fifth Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Fifth Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[1626555:1951866],Outputs[1626555:1951866], test_size=test_and_train, random_state=4)
print ('Sixth Train set:', X_train.shape,  y_train.shape)
print ('Sixth Test set:', X_test.shape,  y_test.shape)


# In[16]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Sixth Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Sixth Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[1951866:2277177],Outputs[1951866:2277177], test_size=test_and_train, random_state=4)
print ('Seventh Train set:', X_train.shape,  y_train.shape)
print ('Seventh Test set:', X_test.shape,  y_test.shape)


# In[18]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Seventh Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Seventh Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(Inputs[2277177:2602488],Outputs[2277177:2602488], test_size=test_and_train, random_state=4)
print ('Eigth Train set:', X_train.shape,  y_train.shape)
print ('Eigth Test set:', X_test.shape,  y_test.shape)


# In[20]:


neighbours = KNeighborsClassifier(n_neighbors = 2).fit(X_train,y_train)
yhat = neighbours.predict(X_test)
print("Eigth Train set Accuracy: ", metrics.accuracy_score(y_train, neighbours.predict(X_train)))
print("Eigth Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# # Logistic-Regression-Classification

# jaccard index: 0.7727
# f1_score: 0.6736224854741356

# In[28]:


def LogeseticRegConfusionMatrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(Inputs,Outputs, test_size=test_and_train, random_state=4)
LogesticReg = LogisticRegression(solver='liblinear').fit(X_train,y_train)  
yhat = LogesticReg.predict(X_test)
print (classification_report(y_test, yhat))
print("jaccard index:",jaccard_score(y_test, yhat,pos_label=0))
print("f1_score:",f1_score(y_test, yhat, average='weighted'))


# In[37]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
LogeseticRegConfusionMatrix(cnf_matrix, classes=['Unacute =1',' acute = 0'],normalize= False,  title='LogeseticReg Confusion matrix')


# # Decision-Trees-Classification

# DecisionTrees's Accuracy:  0.96487

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(Inputs,Outputs, test_size=test_and_train, random_state=4)


# In[43]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_train,y_train)
predictTree = drugTree.predict(X_test)
print("Results: \n",y_test[0:20])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predictTree))

