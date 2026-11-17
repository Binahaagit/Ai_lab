# 🤖 Artificial Intelligence — ML Lab Experiments

### 🐍 Python + Pandas + NumPy + Scikit-learn
### 📚 Experiments 20–24 | Quick Revision & Exam Guide

---

# ⚙️ GENERAL INSTRUCTIONS

### 📦 Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

| Library | Use |
|---|---|
| `pandas` | Reading and manipulating datasets |
| `numpy` | Numerical calculations and arrays |
| `matplotlib.pyplot` | Graphs and visualisation |
| `sklearn` | Machine learning algorithms and evaluation |
| `collections.Counter` | Counting class occurrences |

### 📂 Reading a CSV File

```python
data = pd.read_csv('filename.csv')
```

For a CSV without column names:

```python
data = pd.read_csv('filename.csv', header=None, names=columns)
```

- `header=None` → CSV has no header row.
- `names=columns` → Assigns our own column names.
- `header=0` → First row contains column names.

### 🔑 Pandas Selection

```python
X = data.iloc[:, :4]
y = data.iloc[:, 4]
```

- `X` → Input features.
- `y` → Target/output.
- `iloc` → Integer-position-based selection.
- `.values` → Converts Pandas data into NumPy arrays.

```python
X = data.iloc[:, :4].values
```

Both Pandas DataFrames and NumPy arrays can be used with Scikit-learn.

### 🧠 Important ML Terminology

| Term | Meaning |
|---|---|
| Features | Input variables |
| Target | Output variable to predict |
| Training data | Used to train the model |
| Testing data | Used to evaluate the model |
| Classification | Predicting a category |
| Regression | Predicting a numerical value |
| Clustering | Grouping similar data points |
| Supervised learning | Uses labelled data |
| Unsupervised learning | Uses unlabelled data |

---

# 2️⃣0️⃣ Experiment 20 — Naive Bayes Classifier on CSV Data

### 🎯 Aim

Implement a Naive Bayes classifier to predict a class based on input features from a CSV dataset.

### 🔑 Important Concepts

- Supervised learning.
- Classification.
- Bayes' theorem.
- Prior probability.
- Conditional probability.
- Naive independence assumption.
- Laplace smoothing.

### 🧮 Bayes' Theorem

\[
P(C|X)=\frac{P(X|C)P(C)}{P(X)}
\]

Where:

- \(P(C|X)\) → Posterior probability.
- \(P(X|C)\) → Likelihood.
- \(P(C)\) → Prior probability.
- \(P(X)\) → Evidence.

### 🔑 Naive Bayes Formula

For multiple features:

\[
P(C|X)\propto P(C)\prod_{i=1}^{n}P(X_i|C)
\]

The algorithm assumes that features are conditionally independent given the class.

### 🔄 Working Process

```text
CSV Dataset
    ↓
Read Dataset
    ↓
Separate Features and Target
    ↓
Find Unique Classes
    ↓
Calculate Prior Probability
    ↓
Calculate Conditional Probability
    ↓
Apply Laplace Smoothing
    ↓
Calculate Class Scores
    ↓
Select Highest Score
    ↓
Predicted Class
```

### 🔑 Important Python Functions

```python
csv.reader()
next()
set()
max()
```

- `csv.reader()` → Reads CSV rows.
- `next(reader)` → Skips or reads the header row.
- `set()` → Finds unique values.
- `max()` → Finds the class with the highest score.

### 🧮 Laplace Smoothing

\[
P(X_i|C)=\frac{count(X_i,C)+1}{count(C)+N}
\]

Where:

- `count(Xi,C)` → Number of occurrences of the feature value in that class.
- `count(C)` → Number of records belonging to the class.
- `N` → Number of distinct values of that feature.

**Why Laplace smoothing?**

Prevents zero probability when a feature value has not appeared in a particular class.

### 🚨 Important Points

- Naive Bayes is a supervised classification algorithm.
- It is based on Bayes' theorem.
- It assumes conditional independence of features.
- It can work with categorical data.
- The calculated class scores may be unnormalised posterior scores.
- The class with the maximum score is selected.

### 🧠 Viva Questions — Experiment 20

**Q1. What is Naive Bayes?**

A probabilistic supervised machine learning algorithm based on Bayes' theorem, used mainly for classification.

**Q2. Why is it called Naive Bayes?**

Because it makes the naive assumption that features are conditionally independent given the class.

**Q3. What is prior probability?**

The probability of a class before considering the input features.

**Q4. What is conditional probability?**

The probability of an event occurring given that another event has occurred.

**Q5. What is posterior probability?**

The probability of a class after considering the observed features.

**Q6. What is likelihood?**

The probability of observing the given features assuming a particular class.

**Q7. Why do we use Laplace smoothing?**

To avoid zero probabilities for unseen feature values.

**Q8. What happens without Laplace smoothing?**

If one conditional probability becomes zero, the product of probabilities for that class may become zero.

**Q9. What is the difference between prior and posterior probability?**

Prior is calculated before observing the features; posterior is calculated after observing them.

**Q10. Why is Naive Bayes called a probabilistic classifier?**

Because it calculates probabilities for different classes and predicts the most probable class.

**Q11. Does Naive Bayes require numerical data?**

No. Different variants support different data types. Categorical Naive Bayes handles categorical features, while Gaussian Naive Bayes handles continuous features.

**Q12. What happens if two classes have the same probability?**

There is a tie. The implementation needs a tie-breaking rule.

**Q13. Why might the output probabilities not add up to 1 in a manual implementation?**

The code may calculate unnormalised class scores by omitting the common evidence term \(P(X)\).

**Q14. Is Naive Bayes supervised or unsupervised?**

Supervised learning because it learns from labelled data.

**Q15. Give applications of Naive Bayes.**

Spam detection, sentiment analysis, document classification and medical prediction.

---

# 2️⃣1️⃣ Experiment 21 — Text Classification Using Naive Bayes

### 🎯 Aim

Classify text documents into categories using the Multinomial Naive Bayes algorithm.

Example categories:

- Technology
- Sports
- Politics
- Entertainment

### 🔑 Important Concepts

- Natural Language Processing (NLP).
- Text classification.
- Bag of Words.
- CountVectorizer.
- Multinomial Naive Bayes.
- Training and testing.
- Confusion matrix.
- Precision, Recall, F1-score and Accuracy.

### 🔄 Working Process

```text
Text Dataset
    ↓
Read CSV
    ↓
Separate Text and Labels
    ↓
CountVectorizer
    ↓
Convert Text into Numerical Vectors
    ↓
Train-Test Split
    ↓
Train MultinomialNB
    ↓
Predict Test Data
    ↓
Evaluate Model
    ↓
Enter New Sentence
    ↓
Predict Its Category
```

### 🔑 Important Code

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
```

### 1. Separating Features and Target

```python
X = data.iloc[:, 0]
y = data.iloc[:, 1]
```

- `X` → Text documents.
- `y` → Corresponding categories.

### 2. CountVectorizer

```python
v = CountVectorizer()
X = v.fit_transform(X)
```

- Converts text into numerical word-count vectors.
- Builds a vocabulary.
- Counts occurrences of words.
- Produces a sparse matrix.

Example:

```text
Document 1: I love AI
Document 2: I love Python
```

Vocabulary:

```text
AI, I, love, Python
```

Word-count representation:

```text
Document 1 → [1, 1, 1, 0]
Document 2 → [0, 1, 1, 1]
```

### 3. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- `test_size=0.2` → 20% testing data.
- Remaining 80% → Training data.
- `random_state=42` → Reproducible split.

### 4. Create and Train Model

```python
m = MultinomialNB()
m.fit(X_train, y_train)
```

- `MultinomialNB()` → Creates the classifier.
- `fit()` → Trains the model.

### 5. Prediction

```python
p = m.predict(X_test)
```

Predicts the classes of test documents.

### 6. Evaluation

```python
confusion_matrix(y_test, p)

precision_score(y_test, p, average='macro')

recall_score(y_test, p, average='macro')

f1_score(y_test, p, average='macro')

accuracy_score(y_test, p)
```

### 7. Predict New Text

```python
s = input("Enter a sentence: ")

print(m.predict(v.transform([s]))[0])
```

**Important:**

- `fit_transform()` → Learns vocabulary and transforms text.
- `transform()` → Uses an already learned vocabulary.
- `[s]` → Converts a single sentence into a list containing one document.

### 🚨 Better Practice

For a proper evaluation, split the raw text into training and testing data first. Fit `CountVectorizer` only on the training text, then transform the testing text.

This avoids learning vocabulary from the test set.

### 📊 Evaluation Metrics

#### Confusion Matrix

```text
Rows    → Actual classes
Columns → Predicted classes
Diagonal → Correct predictions
Off-diagonal → Incorrect predictions
```

#### Binary Classification

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

- TP → True Positive.
- TN → True Negative.
- FP → False Positive.
- FN → False Negative.

#### Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

#### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

Out of all predicted positives, how many were actually positive?

#### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

Out of all actual positives, how many were correctly identified?

#### F1-Score

\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

Harmonic mean of precision and recall.

### 🔑 Macro vs Weighted Average

- Macro average → Gives equal importance to every class.
- Weighted average → Gives importance according to the number of samples in each class.
- Micro average → Calculates metrics using total TP, FP and FN across classes.

### 🧠 Viva Questions — Experiment 21

**Q1. What is NLP?**

Natural Language Processing is a branch of AI that enables computers to process and understand human language.

**Q2. What is text classification?**

Assigning text documents to predefined categories.

**Q3. What is CountVectorizer?**

A Scikit-learn tool that converts text into numerical word-count vectors.

**Q4. Why do we convert text into numbers?**

Machine learning algorithms require numerical input.

**Q5. What is a vocabulary?**

The collection of unique words identified from the training documents.

**Q6. What is Bag of Words?**

A text representation based on word occurrences, generally ignoring grammar and word order.

**Q7. Why use Multinomial Naive Bayes?**

It is suitable for discrete feature counts, such as word frequencies.

**Q8. What is the difference between GaussianNB and MultinomialNB?**

- GaussianNB → Continuous numerical features assumed to follow Gaussian distributions.
- MultinomialNB → Discrete counts or frequency-based features.

**Q9. What is the purpose of `fit_transform()`?**

It learns the vocabulary and converts the documents into numerical vectors.

**Q10. Why use `transform()` for a new sentence?**

To convert the new sentence using the vocabulary already learned during training.

**Q11. Why do we use train-test split?**

To evaluate the model on unseen data.

**Q12. What does `random_state=42` mean?**

It fixes the random seed so that the split is reproducible.

**Q13. What does `test_size=0.2` mean?**

20% of the dataset is used for testing and 80% for training.

**Q14. What is a confusion matrix?**

A table comparing actual classes with predicted classes.

**Q15. What is the difference between precision and recall?**

Precision measures how many predicted positives are correct. Recall measures how many actual positives are correctly identified.

**Q16. Which metric is useful when false positives are costly?**

Precision.

**Q17. Which metric is useful when false negatives are costly?**

Recall.

**Q18. What is F1-score?**

The harmonic mean of precision and recall.

**Q19. What is macro averaging?**

Calculating a metric independently for each class and taking their unweighted average.

**Q20. Why is accuracy not always sufficient?**

A model can have high accuracy while performing poorly on a minority class, especially with imbalanced data.

**Q21. What happens if a new sentence contains words not in the vocabulary?**

With the default CountVectorizer behaviour, unseen words are ignored.

**Q22. What is sparse matrix representation?**

A representation that stores non-zero values efficiently instead of storing every zero.

**Q23. Give applications of text classification.**

Spam detection, news categorisation, sentiment analysis and email filtering.

---

# 2️⃣2️⃣ Experiment 22 — EM Clustering Using Gaussian Mixture Model

### 🎯 Aim

Perform clustering using the Expectation-Maximization algorithm with a Gaussian Mixture Model for K = 2, 3 and 4.

### 🔑 Important Concepts

- Unsupervised learning.
- Clustering.
- Gaussian distribution.
- Gaussian Mixture Model (GMM).
- Expectation-Maximization (EM).
- Mean and covariance.
- Soft clustering.

### 🧠 What is Clustering?

Clustering is an unsupervised learning technique that groups similar data points together without predefined class labels.

### 🧠 What is GMM?

Gaussian Mixture Model assumes that data is generated from a mixture of multiple Gaussian distributions.

Each Gaussian distribution represents a cluster.

### 🔄 EM Algorithm

EM consists of two main steps:

#### 1. E-Step — Expectation

- Estimates the probability that each data point belongs to each Gaussian component.
- These probabilities are called responsibilities.

#### 2. M-Step — Maximization

- Updates the model parameters.
- Updates means, covariance matrices and mixture weights.

```text
Initialize Parameters
       ↓
E-Step
Calculate Responsibilities
       ↓
M-Step
Update Parameters
       ↓
Check Convergence
       ↓
Repeat Until Convergence
```

### 🔑 Important Code

```python
from sklearn.mixture import GaussianMixture
```

```python
X = df.iloc[:, :2].values
```

Selects the first two features for visualisation.

```python
gmm = GaussianMixture(n_components=k, random_state=42)
```

- `n_components` → Number of Gaussian components/clusters.
- `random_state` → Reproducible initialisation.

```python
gmm.fit(X)
```

Fits the GMM model using the data.

```python
labels = gmm.predict(X)
```

Predicts the most likely cluster for each data point.

```python
centers = gmm.means_
```

Returns the mean vector of each Gaussian component.

### 🔑 Important GMM Parameters

| Parameter | Meaning |
|---|---|
| `n_components` | Number of Gaussian components |
| `means_` | Mean vectors |
| `covariances_` | Covariance matrices |
| `weights_` | Mixture weights |
| `predict()` | Most likely component assignment |
| `predict_proba()` | Membership probabilities |

### 🚨 Important Points

- GMM is an unsupervised clustering model.
- EM is the algorithm used to estimate GMM parameters.
- GMM supports probabilistic or soft membership.
- `predict()` returns the most likely cluster label.
- `predict_proba()` returns membership probabilities.
- GMM can model elliptical clusters using covariance matrices.
- GMM means are not necessarily the same as K-Means centroids.

### 🧠 Viva Questions — Experiment 22

**Q1. What is clustering?**

Grouping similar data points without predefined labels.

**Q2. Is GMM supervised or unsupervised?**

Unsupervised learning.

**Q3. What is a Gaussian distribution?**

A continuous probability distribution commonly called the normal distribution, characterised by its mean and variance.

**Q4. What is a Gaussian Mixture Model?**

A probabilistic model that represents data as a combination of multiple Gaussian distributions.

**Q5. What is EM?**

Expectation-Maximization is an iterative algorithm used to estimate model parameters when data involves hidden or latent variables.

**Q6. What are the two steps of EM?**

Expectation step and Maximization step.

**Q7. What happens in the E-step?**

It calculates the probability or responsibility of each Gaussian component for every data point.

**Q8. What happens in the M-step?**

It updates the Gaussian means, covariance matrices and mixture weights using the calculated responsibilities.

**Q9. What are responsibilities?**

The estimated membership probabilities of data points belonging to different Gaussian components.

**Q10. What is soft clustering?**

A data point can have probabilities of belonging to multiple clusters.

**Q11. What is hard clustering?**

Each data point is assigned to exactly one cluster.

**Q12. What is the difference between GMM and K-Means?**

K-Means assigns points to the nearest centroid using distance. GMM uses probability distributions and estimates membership probabilities.

**Q13. What is covariance?**

A measure of how two variables vary together.

**Q14. Why does GMM use covariance?**

To represent the spread, orientation and shape of Gaussian clusters.

**Q15. What does `n_components=3` mean?**

The model assumes three Gaussian components.

**Q16. What is the use of `gmm.means_`?**

It returns the mean vectors of the Gaussian components.

**Q17. What is the difference between `predict()` and `predict_proba()`?**

- `predict()` → Most likely component label.
- `predict_proba()` → Probability of belonging to each component.

**Q18. Can GMM identify overlapping clusters?**

Yes. It can model overlapping clusters using probabilistic memberships.

**Q19. Why is GMM called a probabilistic clustering algorithm?**

Because it uses probability distributions to estimate cluster membership.

**Q20. Give applications of GMM.**

Image segmentation, anomaly detection, speech recognition and customer segmentation.

---

# 2️⃣3️⃣ Experiment 23 — K-Means Clustering

### 🎯 Aim

Perform K-Means clustering on a dataset and visualise the clusters for K = 2, 3 and 4.

### 🔑 Important Concepts

- Unsupervised learning.
- Clustering.
- Centroid.
- Euclidean distance.
- Within-Cluster Sum of Squares (WCSS).
- Elbow method.
- Feature scaling.

### 🧠 What is K-Means?

K-Means is an unsupervised clustering algorithm that divides data into K clusters based on similarity.

Each cluster is represented by a centroid.

### 🔄 K-Means Algorithm

```text
Choose K
   ↓
Initialize K Centroids
   ↓
Calculate Distance
   ↓
Assign Points to Nearest Centroid
   ↓
Recalculate Centroids
   ↓
Check Whether Centroids Changed
   ↓
Repeat Until Convergence
```

### 🧮 Euclidean Distance

\[
d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

K-Means generally assigns each point to its nearest centroid using Euclidean distance.

### 🧮 Centroid Calculation

\[
Centroid=\frac{\sum_{i=1}^{n}x_i}{n}
\]

The centroid is the arithmetic mean of all points assigned to that cluster.

### 🔑 Important Code

```python
from sklearn.cluster import KMeans
```

```python
X = df.iloc[:, :2].values
```

Selects the first two features.

```python
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
```

| Parameter | Meaning |
|---|---|
| `n_clusters` | Number of clusters |
| `random_state` | Controls reproducibility |
| `n_init` | Number of independent initialisations |

### 🔑 Training and Prediction

```python
labels = kmeans.fit_predict(X)
```

- `fit()` → Learns cluster centres.
- `predict()` → Assigns points to clusters.
- `fit_predict()` → Fits the model and returns cluster labels.

### 🔑 Centroids

```python
centroids = kmeans.cluster_centers_
```

Returns the final coordinates of the cluster centres.

### 🔑 Inertia / WCSS

```python
kmeans.inertia_
```

Inertia is the sum of squared distances between each data point and its assigned centroid.

\[
WCSS=\sum_{j=1}^{K}\sum_{x_i\in C_j}
\|x_i-\mu_j\|^2
\]

Where:

- \(K\) → Number of clusters.
- \(C_j\) → Points in cluster j.
- \(\mu_j\) → Centroid of cluster j.

### 📉 Elbow Method

Used to choose a suitable K.

```text
WCSS
  |
  |\
  | \
  |  \
  |   \__
  |      \___
  |
  +---------------- K
```

- Increasing K generally decreases WCSS.
- Look for the point where the reduction starts becoming less significant.
- This point is called the elbow.

### 🚨 Important Points

- K-Means is unsupervised.
- K represents the number of clusters.
- Centroids are recalculated using the mean.
- Increasing K generally reduces WCSS.
- Feature scaling is important when features have different numerical scales.
- K-Means can be affected by outliers and initial centroid selection.
- `n_init=10` means ten independent initialisation attempts, with the best result retained according to inertia.

### 🧠 Viva Questions — Experiment 23

**Q1. What is K-Means?**

An unsupervised clustering algorithm that groups data into K clusters based on distance from centroids.

**Q2. What does K represent?**

The number of clusters.

**Q3. Is K-Means supervised or unsupervised?**

Unsupervised learning.

**Q4. What is a centroid?**

The mean position of all points assigned to a cluster.

**Q5. How is a centroid calculated?**

By taking the arithmetic mean of the coordinates of all points in that cluster.

**Q6. What distance measure is commonly used in K-Means?**

Euclidean distance.

**Q7. What is WCSS?**

Within-Cluster Sum of Squares, the sum of squared distances between points and their assigned centroids.

**Q8. What happens to WCSS when K increases?**

It generally decreases or remains unchanged.

**Q9. What is the elbow method?**

A technique used to estimate a suitable number of clusters by plotting WCSS against K.

**Q10. Why is it called K-Means?**

Because it divides data into K clusters and represents each cluster using its mean.

**Q11. What is the purpose of `n_clusters`?**

It specifies the required number of clusters.

**Q12. What does `random_state=42` do?**

It makes the random initialisation reproducible.

**Q13. What does `n_init=10` mean?**

The algorithm performs ten independent initialisation attempts and selects the best result based on inertia.

**Q14. What does `fit_predict()` do?**

Fits the model and returns the cluster label assigned to each data point.

**Q15. What does `cluster_centers_` return?**

The final coordinates of the centroids.

**Q16. What is the difference between classification and clustering?**

Classification predicts predefined labels; clustering discovers groups without predefined labels.

**Q17. Why is feature scaling important?**

Features with larger numerical ranges can dominate distance calculations.

**Q18. What happens if K is too small?**

Different natural groups may be combined into one cluster.

**Q19. What happens if K is too large?**

Natural groups may be unnecessarily divided into smaller clusters.

**Q20. Can K-Means be affected by outliers?**

Yes. Outliers can shift centroids because the algorithm uses means.

**Q21. What is the difference between K-Means and KNN?**

K-Means is an unsupervised clustering algorithm. KNN is a supervised algorithm used for classification or regression.

**Q22. Give applications of K-Means.**

Customer segmentation, image compression, document clustering and market segmentation.

---

# 2️⃣4️⃣ Experiment 24 — K-Nearest Neighbours (KNN) Classification

### 🎯 Aim

Implement KNN classification using the Iris dataset and evaluate its performance for K = 3, 5, 7 and 9.

### 🌸 Iris Dataset

The Iris dataset contains 150 samples and four input features.

| Feature | Meaning |
|---|---|
| Sepal length | Length of sepal |
| Sepal width | Width of sepal |
| Petal length | Length of petal |
| Petal width | Width of petal |
| Class | Iris species |

Three classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

### 🔑 Important Concepts

- Supervised learning.
- Classification.
- K-nearest neighbours.
- Euclidean distance.
- Majority voting.
- Training and testing.
- Confusion matrix.
- Precision, Recall, Accuracy and F1-score.

### 🧠 What is KNN?

K-Nearest Neighbours is a supervised machine learning algorithm that classifies a new data point based on the majority class among its K nearest training samples.

### 🔄 KNN Algorithm

```text
Load Dataset
     ↓
Separate Features and Target
     ↓
Shuffle Dataset Indices
     ↓
Split into Training and Testing
     ↓
Choose K
     ↓
Calculate Distance
     ↓
Sort Distances
     ↓
Select K Nearest Neighbours
     ↓
Majority Voting
     ↓
Predicted Class
     ↓
Evaluate Performance
```

### 🧮 Euclidean Distance

\[
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

For two features:

\[
d=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}
\]

Example:

Point 1 = (1, 2)

Point 2 = (4, 6)

\[
d=\sqrt{(4-1)^2+(6-2)^2}
\]

\[
d=\sqrt{9+16}=5
\]

### 🔑 Important Code — Scikit-learn Version

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
```

### 1. Read Dataset

```python
columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'class'
]

data = pd.read_csv(
    'iris.data',
    header=None,
    names=columns
)
```

- `header=None` → No column names exist in the original file.
- `names=columns` → Assigns column names.

### 2. Separate Features and Target

```python
X = data.iloc[:, :4]
y = data['class']
```

- X → First four columns.
- y → Class column.

`.values` is optional when using Scikit-learn.

### 3. Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=10
)
```

- 70% training.
- 30% testing.
- For 150 samples → 105 training and 45 testing samples.
- `random_state=10` → Reproducible split.

### 4. Create KNN Model

```python
model = KNeighborsClassifier(n_neighbors=k)
```

`n_neighbors` specifies the number of nearest neighbours.

### 5. Train Model

```python
model.fit(X_train, y_train)
```

Stores the training data and labels for KNN prediction.

### 6. Predict

```python
y_pred = model.predict(X_test)
```

Predicts the class of each testing sample.

### 7. Evaluate

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
```

### 🔑 Important KNN Parameters

| Parameter | Meaning |
|---|---|
| `n_neighbors` | Number of neighbours |
| `fit()` | Stores training data |
| `predict()` | Predicts classes |
| `weights` | Determines neighbour voting weights |
| `metric` | Distance metric |
| `p=2` | Euclidean distance for Minkowski metric |

### 🧠 Manual KNN — Important Steps

The lab record also implements KNN without using `KNeighborsClassifier`.

#### 1. Shuffle Indices

```python
np.random.seed(10)

indices = np.arange(len(X))
np.random.shuffle(indices)
```

- `arange()` → Creates indices.
- `shuffle()` → Randomly shuffles them.
- `seed()` → Makes the shuffle reproducible.

#### 2. Split Data

```python
train_size = int(0.70 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]
```

Use the same indices for X and y.

#### 3. Euclidean Distance Function

```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
```

#### 4. Calculate Distances

```python
distances = []

for i in range(len(X_train)):
    distance = euclidean_distance(test_point, X_train[i])
    distances.append((distance, y_train[i]))
```

Each entry contains:

```text
(distance, class_label)
```

#### 5. Sort Distances

```python
distances.sort(key=lambda x: x[0])
```

Sorts according to distance in ascending order.

#### 6. Select K Nearest Neighbours

```python
nearest_labels = [
    label for distance, label in distances[:k]
]
```

#### 7. Majority Voting

```python
from collections import Counter

prediction = Counter(nearest_labels).most_common(1)[0][0]
```

- `Counter()` → Counts occurrences of labels.
- `most_common(1)` → Returns the most frequent label.
- `[0][0]` → Extracts the label.

### 🚨 Important Points

- KNN is supervised learning.
- It is a lazy learning algorithm.
- It does not build an explicit model during training.
- It calculates distances during prediction.
- Small K → More sensitive to noise.
- Large K → Can oversmooth class boundaries.
- Odd K can reduce ties in binary classification, but does not guarantee no ties in multiclass classification.
- Feature scaling is important because KNN uses distance calculations.
- KNN supports classification and regression.

### 🧠 Viva Questions — Experiment 24

**Q1. What is KNN?**

A supervised learning algorithm that predicts a class based on the majority class of its K nearest neighbours.

**Q2. Is KNN supervised or unsupervised?**

Supervised learning.

**Q3. What does K represent?**

The number of nearest neighbours considered during prediction.

**Q4. Why is KNN called a lazy learning algorithm?**

Because it does not build an explicit generalised model during training. It stores training data and performs calculations when predicting.

**Q5. What is majority voting?**

The class occurring most frequently among the K nearest neighbours is selected.

**Q6. Which distance measure is used in the experiment?**

Euclidean distance.

**Q7. What happens if K=1?**

The new sample receives the class of its single nearest neighbour. This can make the model sensitive to noise.

**Q8. What happens if K is too large?**

The prediction may be influenced by distant points and smaller local patterns may be lost.

**Q9. Why do we often choose an odd K?**

To reduce the chance of ties, particularly in binary classification.

**Q10. Does choosing an odd K guarantee no ties?**

No. In multiclass classification, ties can still occur.

**Q11. Why is feature scaling important in KNN?**

Features with larger numerical ranges can dominate the distance calculation.

**Q12. What is the Iris dataset?**

A dataset containing 150 flower samples, four numerical features and three Iris species.

**Q13. What are the four Iris features?**

Sepal length, sepal width, petal length and petal width.

**Q14. What is the target column?**

The Iris species or class.

**Q15. What does `train_test_split()` do?**

Splits data into training and testing subsets.

**Q16. What does `test_size=0.30` mean?**

30% of the data is used for testing.

**Q17. What does `random_state=10` mean?**

It makes the random split reproducible.

**Q18. Why must we shuffle indices instead of independently shuffling X and y?**

To preserve the correspondence between every feature row and its target label.

**Q19. What does `np.arange(len(X))` do?**

Creates an array of indices from zero to the number of samples minus one.

**Q20. What does `np.random.shuffle(indices)` do?**

Randomly changes the order of the indices.

**Q21. What does `distances.sort(key=lambda x: x[0])` do?**

Sorts distance-label pairs according to the distance value in ascending order.

**Q22. What does `distances[:k]` do?**

Selects the first K nearest entries after sorting.

**Q23. Why use Counter?**

To count the class labels among the nearest neighbours and find the majority class.

**Q24. What is the difference between KNN and K-Means?**

KNN is supervised classification/regression. K-Means is unsupervised clustering.

**Q25. What is the difference between KNN and Naive Bayes?**

KNN uses distances and neighbour voting. Naive Bayes uses Bayes' theorem and probability calculations.

**Q26. Does KNN require a training phase?**

It requires storing the training samples, but it does not perform conventional parameter learning like many other algorithms.

**Q27. Can KNN be used for regression?**

Yes. It can predict numerical values using the neighbours' values, often through averaging.

**Q28. Which is computationally more expensive during prediction: KNN or Naive Bayes?**

KNN generally requires more prediction-time computation because it calculates distances to training samples. Actual runtime depends on dataset size, implementation and feature dimensions.

---

# 🔥 MOST IMPORTANT DIFFERENCES — EXPERIMENTS 20–24

| Algorithm | Type | Main Principle |
|---|---|---|
| Naive Bayes | Supervised | Probability |
| Text Naive Bayes | Supervised | Word counts + probability |
| GMM | Unsupervised | Gaussian distributions + EM |
| K-Means | Unsupervised | Centroid + distance |
| KNN | Supervised | Nearest neighbours + voting |

### Naive Bayes vs KNN

```text
Naive Bayes → Probability-based
KNN         → Distance-based
```

### KNN vs K-Means

```text
KNN:
- Supervised
- Uses labelled data
- Classification / Regression
- K = Number of neighbours

K-Means:
- Unsupervised
- Uses unlabelled data
- Clustering
- K = Number of clusters
```

### K-Means vs GMM

```text
K-Means:
- Distance-based
- Hard assignment
- Uses centroids
- Assigns each point to one cluster

GMM:
- Probability-based
- Soft membership
- Uses Gaussian distributions
- Estimates membership probabilities
```

### GaussianNB vs MultinomialNB

```text
GaussianNB:
- Continuous numerical features
- Gaussian distribution assumption

MultinomialNB:
- Discrete counts / frequencies
- Commonly used for text classification
```

---

# 📊 EVALUATION METRICS — QUICK REVISION

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Correctness of positive predictions |
| Recall | TP/(TP+FN) | Ability to find actual positives |
| F1-score | 2PR/(P+R) | Balance of precision and recall |

### Example

Suppose:

```text
TP = 25
FN = 15
FP = 5
TN = 55
```

Then:

```text
Accuracy  = 80%
Precision = 83.33%
Recall    = 62.5%
F1-score  = 71.43%
```

### Important Scikit-learn Functions

```python
confusion_matrix()
precision_score()
recall_score()
accuracy_score()
f1_score()
classification_report()
```

---

# 🚨 COMMON APPLICATION-BASED EXAM QUESTIONS

### Experiment 20 — Naive Bayes

1. Given a new set of categorical features, predict the class using Naive Bayes.
2. What happens if a feature value has never occurred in a class?
3. How does Laplace smoothing solve the zero-probability problem?
4. What happens if one class has a higher prior probability?
5. How would you add another feature to the dataset?

### Experiment 21 — Text Classification

1. Classify a new sentence into Technology, Sports or Politics.
2. What happens if you use `transform()` before fitting the vectorizer?
3. Why is `fit_transform()` used for training data?
4. How would you change the code to use 70:30 splitting?
5. How would you calculate macro and weighted precision?
6. Why might a model predict the wrong category even if most words appear relevant?

### Experiment 22 — GMM

1. Change the number of components from 2 to 5.
2. How do you obtain the membership probability of a data point?
3. How would you plot cluster means?
4. What happens if two Gaussian clusters overlap?
5. What changes if you use three features instead of two?
6. Why can GMM model elliptical clusters?

### Experiment 23 — K-Means

1. Change K from `[2, 3, 4]` to `[3, 5, 7]`.
2. How would you create six subplots for the original data and five K values?
3. How do you obtain the final centroids?
4. What happens when K increases?
5. How would you calculate WCSS?
6. How would you implement the elbow method?
7. Why should features be scaled?
8. What happens if the initial centroids are poorly selected?

### Experiment 24 — KNN

1. Change K from `[3, 5, 7, 9]` to `[1, 3, 5]`.
2. Change the training-test ratio from 70:30 to 80:20.
3. What happens if you use K greater than the number of training samples?
4. How would you classify a single new Iris sample?
5. What happens if you do not shuffle the dataset?
6. Why must X and y use the same shuffled indices?
7. How would you replace Euclidean distance with Manhattan distance?
8. What happens if you use a very large K?
9. How would you use feature scaling?

---

# ⚡ LAST-MINUTE FUNCTION CHEAT SHEET

| Experiment | Topic | Important Functions |
|---|---|---|
| 20 | Naive Bayes CSV | `csv.reader()`, `set()`, `max()` |
| 21 | Text Classification | `CountVectorizer()`, `MultinomialNB()` |
| 22 | GMM | `GaussianMixture()`, `fit()`, `predict()` |
| 23 | K-Means | `KMeans()`, `fit_predict()`, `cluster_centers_` |
| 24 | KNN | `KNeighborsClassifier()`, `fit()`, `predict()` |

### Important NumPy Functions

```python
np.arange()
np.random.seed()
np.random.shuffle()
np.sqrt()
np.sum()
np.unique()
```

### Important Pandas Functions

```python
pd.read_csv()
data.iloc[]
data.dropna()
data.values
```

### Important Model Functions

```python
model.fit()
model.predict()
model.fit_predict()
```

---

# 🧠 30-SECOND MEMORY MAP

```text
20 → NAIVE BAYES
     Bayes theorem + Laplace smoothing

21 → TEXT CLASSIFICATION
     CountVectorizer + MultinomialNB

22 → GMM
     Gaussian distribution + EM

23 → K-MEANS
     Centroids + Euclidean distance

24 → KNN
     Distance + Majority voting
```

---

# 🚨 EXAM RULES TO NEVER FORGET

- Check the CSV filename and path before running.
- Check whether the CSV has a header.
- Separate X (features) and y (target).
- Do not confuse classification with clustering.
- `random_state` controls reproducibility, not the percentage of data.
- `test_size=0.30` means 30% testing data.
- KNN and K-Means use K for different purposes.
- GMM uses Gaussian distributions and EM.
- K-Means uses centroids and distance.
- KNN uses nearest neighbours and majority voting.
- Understand confusion matrix rows and columns.
- Read the question carefully before changing the code.
- If asked to implement manually, do not replace the algorithm with a ready-made Scikit-learn model.
- Always explain the algorithm before writing code.

---

# 💪 FINAL REVISION

> 🧠 Understand the algorithm.
>
> ✍️ Remember the important functions.
>
> 📊 Understand the output and evaluation metrics.
>
> 🔍 Practise changing parameters.
>
> 🎯 Be ready for application-based questions.
>
> **5 ML EXPERIMENTS → 20–24 → REVISION READY! 🔥**
