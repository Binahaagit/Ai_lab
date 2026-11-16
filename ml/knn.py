# EXPERIMENT 24: KNN CLASSIFICATION
# DATASET: IRIS DATASET

# ============================================================
# CODE 1: KNN USING SCIKIT-LEARN (EASIEST METHOD)
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. LOAD DATASET
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv('iris.data', header=None, names=columns)

# 2. SEPARATE FEATURES AND TARGET
X = data.iloc[:, :4]       # First four columns are input features
y = data.iloc[:, 4]          

# 3. SPLIT DATASET INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=10
)

# 4. TRAIN AND EVALUATE FOR DIFFERENT K VALUES
for k in [3, 5, 7, 9]:

    # Create KNN model with K neighbours
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model using training data
    model.fit(X_train, y_train)

    # Predict classes for testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("\nK =", k)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


# IMPORTANT POINTS TO REMEMBER :

# 1. KNN stands for K-Nearest Neighbors.
# 2. KNN is a SUPERVISED learning algorithm.
# 3. It is used for classification and regression.
# 4. K represents the number of nearest neighbours.
# 5. KNN uses distance calculation and majority voting for classification.
# 6. train_test_split() divides data into training and testing.
# 7. test_size=0.30 means 30% testing and 70% training.
# 8. random_state=10 ensures reproducible data splitting.
# 9. KNeighborsClassifier() creates the KNN model.
# 10. n_neighbors=k specifies the number of neighbours.
# 11. fit() learns/stores the training data and labels.
# 12. predict() predicts the classes of test samples.
# 13. confusion_matrix() compares actual and predicted labels.
# 14. classification_report() displays precision, recall,
#     F1-score and support for each class.
# 15. accuracy_score() calculates classification accuracy.
# 16. The for loop repeats the process for K = 3, 5, 7, 9.
# 17. KNN is called a LAZY LEARNING algorithm because it
#     mainly stores training data and performs calculations
#     when predictions are requested.
# 18. Feature scaling is generally recommended for KNN
#     because it is based on distance calculations.


# ============================================================
# CODE 2: MANUAL KNN IMPLEMENTATION (LAB RECORD CODE)
# ============================================================

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, precision_score,
    recall_score, accuracy_score, f1_score
)

# 1. LOAD DATASET
columns = [
    'sepal_length', 'sepal_width',
    'petal_length', 'petal_width', 'class'
]

data = pd.read_csv('iris.data', header=None, names=columns)
data = data.dropna()

# 2. SEPARATE FEATURES AND TARGET
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

# 3. SHUFFLE AND SPLIT DATASET
np.random.seed(10)

indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(0.70 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# 4. CALCULATE EUCLIDEAN DISTANCE
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 5. KNN PREDICTION FUNCTION
def knn_predict(X_train, y_train, X_test, k):

    predictions = []

    # Process each test sample one by one
    for test_point in X_test:

        distances = []

        # Calculate distance from test point to every
        # training point
        for i in range(len(X_train)):

            distance = euclidean_distance(test_point, X_train[i])

            # Store distance along with corresponding class label
            distances.append((distance, y_train[i]))

        # Sort distances in ascending order
        distances.sort(key=lambda x: x[0])

        # Select labels of K nearest neighbours
        nearest_labels = [
            label for distance, label in distances[:k]
        ]

        # Majority voting to predict the class
        prediction = Counter(nearest_labels).most_common(1)[0][0]

        # Store predicted class
        predictions.append(prediction)

    return np.array(predictions)


# 6. DISPLAY TRAINING AND TESTING SAMPLE COUNTS
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 7. EVALUATE FOR DIFFERENT K VALUES
for k in [3, 5, 7, 9]:

    y_pred = knn_predict(X_train, y_train, X_test, k)

    cm = confusion_matrix(
        y_test, y_pred, labels=np.unique(y)
    )

    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average='weighted', zero_division=0
    )

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test, y_pred, average='weighted', zero_division=0
    )

    print("\nK =", k)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("F-Score:", f1)


# ============================================================
# IMPORTANT STEPS AND CONCEPTS - CODE 2
# ============================================================

# STEP 1: LOAD DATASET
# - pd.read_csv() reads the Iris CSV file.
# - header=None indicates that the CSV has no header.
# - names=columns assigns column names.
# - dropna() removes rows containing missing values.

# STEP 2: SEPARATE FEATURES AND TARGET
# - X contains the four flower measurements.
# - y contains the corresponding flower species.
# - iloc[:, 0:4] selects the first four columns.

# STEP 3: SHUFFLE AND SPLIT
# - np.random.seed(10) makes randomisation reproducible.
# - np.arange() generates row indices.
# - np.random.shuffle() shuffles the indices.
# - train_size calculates 70% of the dataset.
# - First 70% of shuffled indices are used for training.
# - Remaining 30% are used for testing.
# - Same indices are applied to X and y to keep features
#   matched with their correct class labels.

# STEP 4: EUCLIDEAN DISTANCE
# Formula:
# d = sqrt(sum((point1 - point2)^2))
#
# - Subtract corresponding feature values.
# - Square the differences using **2.
# - Add squared differences using np.sum().
# - Calculate square root using np.sqrt().

# STEP 5: KNN PREDICTION
# - Create an empty predictions list.
# - Take each test sample one by one.
# - Calculate its distance from every training sample.
# - Store each distance with its corresponding class label.
# - Sort distances from smallest to largest.
# - Select the first K nearest neighbours.
# - Apply majority voting using Counter().
# - Store the predicted class.
# - Return all predictions as a NumPy array.

# STEP 6: SAMPLE COUNTS
# - len(X_train) gives the number of training samples.
# - len(X_test) gives the number of testing samples.

# STEP 7: MODEL EVALUATION
# - confusion_matrix() compares actual and predicted classes.
# - precision_score() calculates weighted precision.
# - recall_score() calculates weighted recall.
# - accuracy_score() calculates overall accuracy.
# - f1_score() calculates weighted F1-score.
# - average='weighted' weights each class according to
#   its number of actual samples.
# - zero_division=0 handles undefined metric cases.

# IMPORTANT KNN CONCEPTS:
#
# 1. KNN is a supervised learning algorithm.
# 2. KNN is a lazy learning algorithm.
# 3. Euclidean distance is used in this experiment.
# 4. Majority voting determines the predicted class.
# 5. K means the number of nearest neighbours.
# 6. Odd K values can reduce ties in binary classification,
#    but ties are still possible with multiple classes.
# 7. KNN does not require a separate model-training
#    optimisation process like many other algorithms.
# 8. Feature scaling is generally recommended.
# 9. KNN can be used for classification and regression.
