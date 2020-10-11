#importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
colnames=['Variance', 'Skewness', 'Kurtosis', 'Entropy','Class']
dataset = pd.read_csv('data_banknote_authentication.csv',names=colnames,header=None)
#dataset.head()

"""
#Summary of Data
summary_df ={
		'count': dataset.shape[0],
		'Unique': dataset.nunique(),
		'missing values': dataset.isna().sum(),
		'Min': dataset.min(),
		'Max':dataset.max(),
		'Mean': dataset.mean()
		}
print(pd.DataFrame(summary_df))
dataset.hist(figsize = (10,10))
plt.show()
"""

#independent and dependent variable
x = dataset.iloc[:,[0,1,2,3]].values
y = dataset.iloc[:,-1].values

#spliting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Importing models
from sklearn.linear_model import LogisticRegression
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
"""

#Training Logistic regression
l_classifier = LogisticRegression()
l_classifier.fit(x_train,y_train)

"""
#Training KNN Model
KNN_classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
KNN_classifier.fit(x_train,y_train)

#Training Naive_bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)

#Training Decision Tree Classifier 
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(x_train, y_train)

#Training the SVM model
SVM_classifier = SVC(kernel = 'rbf', random_state = 0)
SVM_classifier.fit(x_train, y_train)

#Training MLP Classifier
MLP_classifier = MLPClassifier()
MLP_classifier.fit(x_train,y_train)
"""

"""
#Importing metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#predicting results 
l_pred = l_classifier.predict(x_test)
k_pred = KNN_classifier.predict(x_test)
n_pred = gnb.predict(x_test)
d_pred = DT_classifier.predict(x_test)
s_pred = SVM_classifier.predict(x_test)
mlp_pred = MLP_classifier.predict(x_test)
"""


"""
y_pred = {'Logistic Regression':l_pred,'Decision Tree':d_pred,'Naive_bayes':n_pred,'KNN':k_pred,'MLP':mlp_pred,'SVM':s_pred }
#Display results
for model,y_pred in y_pred.items():
	print(model,"Result")
	print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
	print("Accuracy_score:",accuracy_score(y_test,y_pred))
	print("Precision_score:",precision_score(y_test,y_pred))
	print("F1_score:",f1_score(y_test,y_pred))
	print(" ")
"""

#saving model on disk
import pickle
pickle.dump(l_classifier,open('model.pkl','wb'))

#loading model
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1.5985,-0.328571,-0.246038,1.09902]]))