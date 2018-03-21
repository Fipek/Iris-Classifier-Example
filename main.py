from Iris_classifier import IrisClassifier

if __name__ == '__main__':
	Iris_classifier_instance = IrisClassifier() 
	Iris_classifier_instance.DecisionTreeExample()
	Iris_classifier_instance.MlpClassifier()
	Iris_classifier_instance.Svm() 
	Iris_classifier_instance.DrawTableForResult()