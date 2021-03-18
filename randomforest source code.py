#this work will in jupter notebook
#Visualization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


#System
#kaggle.com 

data=np.load("./input/olivetti_faces.npy")
target=np.load("./input/olivetti_faces_target.npy")
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target)

#This is the number of trees you want to build
#before taking the maximum voting or averages of predictions.
#Higher number of trees give you better performance but makes your code slower.

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
def show_40_distinct_people(images, unique_ids):
    #Creating 4X10 subplots in  18x9 figure size
    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    #For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()
    
    #iterating over user ids
    for unique_id in unique_ids:
        image_index=unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")

show_40_distinct_people(data, np.unique(y_test))
show_40_distinct_people(data, np.unique(y_pred))
