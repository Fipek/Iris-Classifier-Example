# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


#------------------ENUM Definations------------------
CSV_FILE = "iris.data.csv"
#----------------------------------------------------     


class IrisClassifier(object):
    def __init__(self):
        self.sepal_length = None
        self.sepal_width = None
        self.petal_length = None
        self.petal_width = None
        self.X_train = None 
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.score_list = []

        self.GetInputFromUser()
        self.TrainDataFromCsvFile()

    
    def GetInputFromUser(self):
        try:
            self.sepal_length=float(input("Please enter sepal length:"))
            self.sepal_width=float(input("Please enter sepal width:"))
            self.petal_length=float(input("Please enter petal length:"))
            self.petal_width=float(input("Please enter petal width:"))
        except Exception as error:
            print str(error)
            return

    def DrawMatShow(self,confusion_matrix,title):
        try:
            title_plus = unicode('Confusion Matrix For '+title)
            plt.matshow(confusion_matrix)
            plt.title(title_plus)
            plt.colorbar()
            plt.ylabel(u'Actual Values')
            plt.xlabel(u'Estimated Values')
            plt.show()
        except Exception as error:
            print str(error)

    def TrainDataFromCsvFile(self):
        try:
            train_data=pd.read_csv(CSV_FILE)
            train_data_array=train_data.as_matrix()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_data_array[:,:4], train_data_array[:,4], test_size=0.1)
        except Exception as error:
            print str(error)

    def DrawTableForResult(self):
        try:
            # Prepare table
            cell_text=[]
            columns = ["DecisionTree Accuracy Score", "MLP Classifier Accuracy Score", "Svm Accuracy Score"]
            for i in range(0,len(self.score_list)):
                cell_text.append(self.score_list[i])
            # Add a table at the bottom of the axes
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            ax.axis('tight')
            ax.axis('off')
            the_table = ax.table(cellText=[cell_text],cellLoc='center',
                                 colLabels=columns, loc='center')
            fig.tight_layout()
            plt.show()
        except Exception as error:
            print str(error)


    def DecisionTreeExample(self):
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0,max_depth=3, min_samples_leaf=5)
        clf_gini.fit(self.X_train, self.y_train)

        DecisionTreeClassifier( class_weight=None, criterion='gini', max_depth=3,
                                max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                presort=False, random_state=100, splitter='best')

        y_pred = clf_gini.predict(self.X_test)
        print("---------Prediction Output For DecisionTree-------------")
        print( list(y_pred))
        print " "

        score=accuracy_score(y_pred,self.y_test )
        print("---------Score Output For DecisionTree-------------")
        print(score)
        print " "
        
        scores = cross_val_score(clf_gini, self.X_train,self.y_train, cv=10)
        print("---------Scores Output For DecisionTree-------------")
        print(scores)
        print " "

        confusion_m = confusion_matrix(self.y_test,y_pred)
        print("---------Confision Matrix Output For DecisionTree-------------")
        print(confusion_m)
        print " "

        z=clf_gini.predict([[self.sepal_length,self.sepal_width,self.petal_length,self.petal_width]])
        print("---------Result DecisionTree-------------")
        print("Result for user's input: ",z[0])
        print " "

        self.score_list.append(score)
        self.DrawMatShow(confusion_m,"DecisionTree")

    def MlpClassifier(self):
        mlp = MLPClassifier(hidden_layer_sizes=(10),solver='lbfgs',learning_rate_init=0.01,max_iter=500)
        mlp.fit(self.X_train, self.y_train)
        
        y_pred=mlp.predict(self.X_test)
        print("---------Prediction Output  MlpClassifier-------------")
        print( list(y_pred))
        print " "
        
        score=accuracy_score(y_pred,self.y_test)
        print("---------Score Output For MlpClassifier-------------")
        print(score)
        print " "

        scores = cross_val_score(mlp, self.X_train,self.y_train, cv=10)
        print("---------Scores Output For MlpClassifier-------------")
        print(scores)
        print " "
        
        confusion_m = confusion_matrix(self.y_test,y_pred)
        print("---------Confision Matrix Output For MlpClassifier-------------")
        print(confusion_m)
        print " "

        z=mlp.predict([[self.sepal_length,self.sepal_width,self.petal_length,self.petal_width]])
        print("---------Result MlpClassifier-------------")
        print("Result for user's input: ",z[0])
        print " "

        self.score_list.append(score)
        self.DrawMatShow(confusion_m,"MlpClassifier")

    def Svm(self):
        clf = SVC()
        clf.fit(self.X_train, self.y_train)   
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        
        y_pred=clf.predict(self.X_test)
        print("---------Prediction Output Svm-------------")
        print( list(y_pred))
        print " "

        score=accuracy_score(y_pred,self.y_test)
        print("---------Score Output For Svm-------------")
        print(score)
        print " "

        scores = cross_val_score(clf, self.X_train,self.y_train, cv=10)
        print("---------Scores Output For Svm-------------")
        print(scores)
        print " "

        confusion_m = confusion_matrix(self.y_test,y_pred)
        print("---------Confision Matrix Output For Svm-------------")
        print(confusion_m)
        print " "


        z=clf.predict([[self.sepal_length,self.sepal_width,self.petal_length,self.petal_width]])
        print("---------Result Svm-------------")
        print("Result for user's input: ",z[0])
        print " "

        self.score_list.append(score)
        self.DrawMatShow(confusion_m,"Svm")

    	    