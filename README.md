# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MOHAMED SAFEER ANFAL . C

*INTERN ID*: CT04DY2161

*DOMAIN*: Data Analytics

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION*:Part 1: Custom Binary Classification Dataset

Dataset Creation

You created a small dataset manually using pandas with three numerical features (feature1, feature2, feature3) and a binary target (target with values 0 and 1).

This simulates a very simple classification problem.

Data Splitting

The dataset is split into training (80%) and testing (20%) sets using train_test_split.

Splitting ensures that the model is trained on some data and evaluated on unseen data.

Model Training

You used a DecisionTreeClassifier from Scikit-learn.

The model learns patterns from the training data.

Prediction & Evaluation

The trained model predicts the test set.

The accuracy score is calculated to show how well the model performs.

🔹 Part 2: Iris Dataset Multi-Class Classification

Dataset Loading

The Iris dataset is loaded from Scikit-learn.

This is a famous dataset containing 150 flower samples of 3 types (Setosa, Versicolor, Virginica) with 4 features (sepal length, sepal width, petal length, petal width).

Data Splitting

Again, the dataset is split into training and testing sets (80/20).

Model Training

A DecisionTreeClassifier is trained on the training data.

Model Evaluation

Multiple metrics are used:

Accuracy Score → Overall correctness.

Precision Score (weighted) → How many predicted positives were correct.

Recall Score (weighted) → How many actual positives were captured.

A Confusion Matrix is created to visualize predictions vs. actual results.

Visualization

You used ConfusionMatrixDisplay to plot the confusion matrix.

Although the plot is titled “Sentiment Analysis”, it actually represents flower classification (so the title can be renamed to avoid confusion).

📌 Good Feedback on Your Work

✅ Strengths

Clear Workflow: Your code follows the ML pipeline: dataset → preprocessing → training → prediction → evaluation → visualization.

Two Examples: You showed both a simple dataset and a real dataset, which demonstrates versatility.

Multiple Evaluation Metrics: Instead of just accuracy, you included precision, recall, and confusion matrix → this shows deeper understanding.

Visualization: The confusion matrix makes results easy to interpret.

*OUTPUT*


<img width="1229" height="717" alt="Image" src="https://github.com/user-attachments/assets/0dda4860-0c7f-48c8-aad8-6fa63fc3fd9e" />

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/05ffdadd-d199-4a15-adbc-af7a9f61bb63" />

