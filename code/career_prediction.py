

"""#import Libararies"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset using Pandas
data = pd.read_csv('data.csv')
# display sample dataset
data.head()
data.drop("Unnamed: 0",inplace=True,axis=1)
y=data['label']
x=data.drop('label',axis=1)
x = np.array(x)
y = np.array(y)

print(x.shape)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=100)

svm = SVC()
svm.fit(xtrain,ytrain)

predictions = svm.predict(xtest)
svm_acc = accuracy_score(ytest, predictions)
print("Accuracy svm : ",svm_acc)

rf = RandomForestClassifier(random_state=18)
rf.fit(xtrain, ytrain)
joblib.dump(rf,'rf.joblib',compress=9)
predictions = rf.predict(xtest)
rf_acc = accuracy_score(ytest,predictions)
print("Accuracy Random Forest : ", rf_acc)

clf = MLPClassifier(random_state=1, max_iter=500).fit(xtrain, ytrain)
predictions = clf.predict(xtest)
clf_acc = accuracy_score(ytest,predictions)
print("Accuracy MLP Classifier : ", clf_acc)

inp_data = xtest[:1,:]
pb1 = clf.predict_proba(inp_data)
pb2 = rf.predict_proba(inp_data)

datadict ={0:"health",1:"technical",2:"agriculture",3:"commerce",4:"Arts"}

def addlabels(y):
    for i in range(len(y)):
        plt.text(i,y[i],y[i],color="red")
# res1 = list(np.round((pb1[0]*100),2))
# plt.bar(list(datadict.values()),res1,color =['black', 'yellow', 'green', 'blue', 'cyan'],
#         width = 0.4)
# addlabels(res1)
# plt.yticks([])
# plt.plot()

# def addlabels(y):
#     for i in range(len(y)):
#         plt.text(i,y[i],y[i],color="red")
# res1 = list(np.round((pb2[0]*100),2))
# plt.bar(list(datadict.values()),res1,color =['black', 'yellow', 'green', 'blue', 'cyan'],
#         width = 0.4)
# addlabels(res1)
# plt.yticks([])
# plt.plot()