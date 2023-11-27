# installing and importing lib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading data
df = pd.read_csv('D:\heart_failure_clinical_records_dataset.csv')
df.head()
print(df.head())
df.info()

plt.figure(1,figsize=(10,10))
df['high_blood_pressure'].value_counts().plot.pie(autopct="%1.1f%%",colors = ( "green", "blue"),labels = df['high_blood_pressure'], shadow = True)
plt.legend(title = "high_blood_pressure:")
plt.show()

plt.figure(2, figsize=(10,10))
df['sex'].value_counts().plot.pie(autopct="%1.1f%%",colors = ('yellow', "orange"),labels = df['sex'].unique())
plt.legend(title = "sex:")
plt.show()

df.hist(figsize = (10, 10),color='green')
plt.show()

# function
def comparison_plots(df, variable, target):
    # The function takes a dataframe (df) and

    # Define figure size.
    plt.figure(figsize=(20, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30, color='r')
    plt.title('Histogram')

    # scatterplot
    plt.subplot(1, 3, 2)
    plt.scatter(df[variable], df[target], color='g')
    plt.title('Scatterplot')


    # barplot
    plt.subplot(1, 3, 3)
    sns.barplot(x=target, y=variable, data=df)
    plt.title('Barplot')

    return plt.show()


comparison_plots(df,'serum_creatinine','heart_problem')

comparison_plots(df,'ejection_fraction','heart_problem')


pd.crosstab(df.age,df.heart_problem).plot(kind="bar",figsize=(20,6),color= ['green','red'])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# extracting independent and dependent variable
x= df.iloc[:,[4,7]].values
# print(x)
y= df.iloc[:,12].values
# print(y)

# splitting the dataset into traing and test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test =st_x.transform(x_test)
# print("scaled values")
# print(x_test)


# logisticregersion
# fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# precdicting the the test result
y_pred = classifier.predict(x_test)

# creating confustion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("logistic regression cm")
print(cm)

# v traing set
from matplotlib.colors import ListedColormap
x_set, y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('purple','green'))(i),label = j )
plt.title('LogisticRegression(Training set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# v test set
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap=ListedColormap(('purple','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('purple','green'))(i),label = j )
plt.title('LogisticRegression(Test set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# knn
# fitting
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
classifier.fit(x_train,y_train)

# pred the test set result
y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("knn cm")
print(cm)

# knn v traning
from matplotlib.colors import ListedColormap
x_set, y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('red','green'))(i),label = j )
plt.title('KNN Algorithm(Training set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# knn test set
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap=ListedColormap(('red','yellow')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('red','yellow'))(i),label = j )
plt.title('KNN Alogorithm(Test set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

# pred
y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("naive bayes cm")
print(cm)


# v training set
from matplotlib.colors import ListedColormap
x_set, y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('yellow','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],color= ListedColormap(('yellow','green'))(i),label = j )
plt.title('naive bayes(Training set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# v test set
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap=ListedColormap(('red','yellow')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('red','yellow'))(i),label = j )
plt.title('naive bayes(Test set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

#  decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)

# pred
y_pred = classifier.predict(x_test)

# cm
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("decision tree cm")
print(cm)

# v traing set
from matplotlib.colors import ListedColormap
x_set, y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('blue','orange')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('blue','orange'))(i),label = j )
plt.title('decision tree(Training set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable -heart_problem')
plt.legend()
plt.show()

# v test set
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap=ListedColormap(('blue','orange')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('blue','orange'))(i),label = j )
plt.title('decision tree(Test set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# randomforest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion="entropy")
classifier.fit(x_train,y_train)

# pred
y_pred = classifier.predict(x_test)

# cm
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("random forest cm")
print(cm)

# v traing set
from matplotlib.colors import ListedColormap
x_set, y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('blue','orange')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('blue','orange'))(i),label = j )
plt.title('random forest(Training set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()

# v test set
from matplotlib.colors import ListedColormap
x_set, y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step =0.01),np.arange(start= x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,cmap=ListedColormap(('blue','orange')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set == j,1],color= ListedColormap(('blue','orange'))(i),label = j )
plt.title('random forest(Test set)')
plt.xlabel('independent variables')
plt.ylabel('dependent variable - heart_problem')
plt.legend()
plt.show()