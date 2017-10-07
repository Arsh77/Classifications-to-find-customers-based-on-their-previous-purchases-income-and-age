# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:23:26 2017

@author: ARSHABH SEMWAL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]]
y=dataset.iloc[:,4]

from sklearn.cross_validation import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x, y,test_size=0.25, random_state=0)

# scaling the data
from sklearn.preprocessing import StandardScaler
scr = StandardScaler()
x_train = scr.fit_transform(x_train)
x_test = scr.fit_transform(x_test)

# Logistic regression 
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state=0)
cls.fit(x_train , y_train)

y_pred = cls.predict(x_test)

# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test , y_pred)

# visualize

from matplotlib.colors import ListedColormap
x_set, y_set = x_train , y_train
x1,x2=np.meshgrid(np.arange(start = x_set[:,0].min()-1 , stop=x_set[:,0].max()+1 , step = 0.01),
                  np.arange(start = x_set[:,1].min()-1 , stop=x_set[:,1].max()+1 , step=0.01))

plt.contourf(x1 , x2, cls.predict(np.array([x1.ravel() , x2.ravel()]).T).reshape(x1.shape) , 
                                           alpha=0.75 , cmap=ListedColormap(('red','green')))
plt.xlim(x1.min() , x1.max())
plt.ylim(x2.min() , x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j , 0] , x_set[y_set==j,1],
                c =ListedColormap(('red','green'))(i) , label = j)

plt.title('Logistic regression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
 
# visualise test set

from matplotlib.colors import ListedColormap
x_set, y_set = x_test , y_test
x1,x2=np.meshgrid(np.arange(start = x_set[:,0].min()-1 , stop=x_set[:,0].max()+1 , step = 0.01),
                  np.arange(start = x_set[:,1].min()-1 , stop=x_set[:,1].max()+1 , step=0.01))

plt.contourf(x1 , x2, cls.predict(np.array([x1.ravel() , x2.ravel()]).T).reshape(x1.shape) , 
                                           alpha=0.75 , cmap=ListedColormap(('red','green')))
plt.xlim(x1.min() , x1.max())
plt.ylim(x2.min() , x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j , 0] , x_set[y_set==j,1],
                c =ListedColormap(('red','green'))(i) , label = j)

plt.title('Logistic regression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
   
