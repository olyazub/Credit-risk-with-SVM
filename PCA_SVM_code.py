import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("loan_data_set.csv")


X = data.iloc[:,1:12].values
y = data.iloc[:,12].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying PCA here
from sklearn.decomposition import PCA
pca = PCA(n_components= None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


explained_variance = pca.explained_variance_ratio_
print(explained_variance)
pca = PCA(n_components= 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



from sklearn.svm import SVC
classifier_svm_kernel = SVC(C=5.0,kernel='sigmoid', degree= 1,coef0= 2, gamma=0.12,tol=0.00001)
classifier_svm_kernel.fit(X_train,y_train)

y_pred = classifier_svm_kernel.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm[0][0]+cm[1][1])


from sklearn.model_selection import GridSearchCV

parameters = [{'C':[0.01,0.1,1,10], 'kernel':['sigmoid'], 'gamma': [0.25, 0.3, 0.35, 0.2], 'coef0':[2, 3, 5]}]
grid_search = GridSearchCV(estimator=classifier_svm_kernel, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
print(best_accuracy)
opt_param = grid_search.best_params_
print(opt_param)



