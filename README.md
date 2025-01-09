# Logistic Regression

## Overview
Logistic Regression is a statistical method used for binary classification problems. Despite its name, it is a regression model used for classification tasks. It predicts the probability of an outcome belonging to one of two categories based on one or more independent variables.

---

## Key Concepts

### 1. **Sigmoid Function**
The sigmoid function maps any real-valued number into a range between 0 and 1. It is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:
- \(z\) is the linear combination of the input features and their weights: \(z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n\)

### 2. **Decision Boundary**
The output of the sigmoid function is a probability. Predictions are made by applying a threshold (commonly 0.5):
- If \(\sigma(z) \geq 0.5\), predict class 1
- Otherwise, predict class 0

---

## Training Logistic Regression

### 1. **Cost Function**
The cost function for Logistic Regression is the log-loss function:

\[
J(w) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)})) \Big]
\]

Where:
- \(m\) is the number of training examples
- \(h_w(x)\) is the predicted probability for a given input
- \(y\) is the actual label (0 or 1)

### 2. **Optimization**
The weights \(w\) are optimized using gradient descent:

\[
w_j := w_j - \alpha \frac{\partial J(w)}{\partial w_j}
\]

Where:
- \(\alpha\) is the learning rate

---

## Advantages
- Simple and interpretable
- Works well for linearly separable data
- Outputs probabilities, which are useful for further analysis

## Limitations
- Assumes a linear relationship between features and the log-odds of the target
- Not suitable for complex relationships (non-linear boundaries)
- Sensitive to multicollinearity and outliers

---

## Applications
- Spam detection
- Medical diagnosis (e.g., presence or absence of a disease)
- Customer churn prediction
- Credit scoring

---

## Implementation Example (Python)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data (features and labels)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## References
- [Logistic Regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
