# Import necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris 
from sklearn.metrics import  precision_score,recall_score, f1_score, accuracy_score,confusion_matrix,ConfusionMatrixDisplay 


# Step 1: Input the dataset 
# Sample dataset with features and target (for classification) 
data = pd.DataFrame({ 
'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
'feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
'feature3': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Binary classification target 
})
# Step 2: Preprocess the data 
X = data[['feature1', 'feature2', 'feature3']] # Features 
y = data['target'] # Target variable 
# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Step 3: Select a prediction model 
model = DecisionTreeClassifier() 
# Step 4: Train the model 
model.fit(X_train, y_train) 
# Step 5: Make predictions 
y_pred = model.predict(X_test) 
# Step 6: Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model Accuracy: {accuracy}") 


iris = load_iris() 
X = iris.data 
y = iris.target 
# Holdout method.Dividing the data into train and test 
X_train, X_test,y_train, y_test = train_test_split(X, y,random_state=20,test_size=0.20) 
tree = DecisionTreeClassifier() 
tree.fit(X_train, y_train) 
y_pred = tree.predict(X_test) 
print("Accuracy:", accuracy_score(y_test,y_pred)) 
print("Precision:", precision_score(y_test,y_pred,average="weighted")) 
print('Recall:', recall_score(y_test,y_pred,average="weighted"))
confusion_matrix = confusion_matrix(y_test,y_pred) 
cm_display =ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[0, 1,2]) 
cm_display.plot()
plt.title('SENTIMENT ANALYSIS')
plt.show()
