# Supervised Machine Learning — TE7499
### Complete Notes | B.Tech AI & ML | Symbiosis International University

---

## TABLE OF CONTENTS

1. [Unit 1 — Introduction to Machine Learning](#unit-1)
2. [Unit 2 — Regression](#unit-2)
3. [Unit 3 — Classification](#unit-3)
4. [Unit 4 — Use Case: Regression](#unit-4)
5. [Unit 5 — Use Case: Classification](#unit-5)
6. [Lab Practical Cheat-Sheet](#lab)

---

<a name="unit-1"></a>
# UNIT 1 — Introduction to Machine Learning

## 1.1 What is Machine Learning?

Machine Learning (ML) is a sub-field of Artificial Intelligence where systems **learn patterns from data** and improve their performance on a task without being explicitly programmed for every rule.

**Formal Definition (Tom Mitchell, 1997):**
> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

---

## 1.2 Types of Machine Learning

### A) Supervised Learning
- **Definition:** The model is trained on **labeled data** — every training example has an input (features X) and a known output (label Y).
- The model learns a mapping function: `f(X) → Y`
- **Goal:** Predict Y for new, unseen X.
- **Examples:** Spam detection, stock price prediction, disease diagnosis.

### B) Unsupervised Learning
- **Definition:** The model is trained on **unlabeled data** — no Y is given.
- The model discovers hidden patterns or structure on its own.
- **Examples:** Customer segmentation (K-Means), topic modeling (LDA), anomaly detection.

### C) Reinforcement Learning
- **Definition:** An **agent** interacts with an **environment**, takes actions, and receives **rewards or penalties**.
- The goal is to learn a **policy** that maximizes cumulative reward.
- **Examples:** AlphaGo (chess/Go), self-driving cars, robot locomotion.

| Aspect | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Labels | Yes | No | Reward signal |
| Feedback | Direct | None | Delayed |
| Goal | Predict | Discover structure | Maximize reward |
| Example | Spam filter | Customer clustering | Game playing AI |

---

## 1.3 Training, Test, and Validation Sets

### Why Split Data?
When you train a model on the full dataset and evaluate on the same data, you get an **overly optimistic** performance estimate — the model may simply memorize instead of learning.

### The Three Splits

**Training Set (typically 60–80% of data)**
- The model **sees** this data and adjusts its parameters on it.
- Analogy: Your textbook and practice problems.

**Validation Set (typically 10–20%)**
- Used during training to tune **hyperparameters** (model complexity, learning rate, etc.).
- Helps detect **overfitting** early.
- Analogy: Mock exam before the real one.

**Test Set (typically 10–20%)**
- Completely **held out** until the very end.
- Gives an unbiased estimate of real-world performance.
- Analogy: The actual final exam — you never peek at this while studying.

> ⚠️ **Common mistake:** Using the test set to choose hyperparameters causes **data leakage** — your test set is no longer truly "unseen".

### Cross-Validation (K-Fold)
When data is limited, instead of a fixed split, you divide data into K equal parts (folds). The model trains on K-1 folds and validates on the remaining 1, rotating K times. The final score is averaged.

**Why use it?** Every sample gets to be in the validation set exactly once — better use of limited data.

---

## 1.4 Exploratory Data Analysis (EDA)

EDA is the process of **understanding your data before modeling**.

**Steps:**
1. **Shape & Types:** `df.shape`, `df.dtypes` — how many rows, columns, what data types.
2. **Missing Values:** `df.isnull().sum()` — find gaps that need handling.
3. **Descriptive Stats:** `df.describe()` — mean, std, min, max, quartiles.
4. **Distributions:** Histograms, box plots — understand skewness and outliers.
5. **Correlations:** Heatmap of Pearson correlations — which features relate to the target?
6. **Class Balance:** For classification — are labels evenly distributed?

---

<a name="unit-2"></a>
# UNIT 2 — Regression

Regression deals with predicting a **continuous numerical output** (e.g., house price, temperature, stock return).

---

## 2.1 Regression Basics

### Dependent vs Independent Variables
- **Independent Variable (X / Feature / Predictor):** The input you control or observe (e.g., area of house).
- **Dependent Variable (Y / Target / Response):** The output you want to predict (e.g., price of house).

### Correlation
- Measures the **linear relationship strength** between two variables.
- **Pearson Correlation Coefficient:**
  ```
  r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² · Σ(yi - ȳ)²]
  ```
- Range: −1 to +1
  - +1: Perfect positive linear relationship
  - −1: Perfect negative linear relationship
  - 0: No linear relationship

> **Note:** Correlation ≠ Causation. Two variables can be correlated without one causing the other.

### Covariance
- Measures the **direction** of the relationship between two variables.
- Formula: `Cov(X,Y) = Σ[(xi - x̄)(yi - ȳ)] / (n - 1)`
- Unlike correlation, covariance is **not normalized**, so its magnitude depends on units.

---

## 2.2 Ordinary Least Squares (OLS)

OLS is the fundamental method for fitting a regression line. It minimizes the **Sum of Squared Residuals (SSR)**:

```
SSR = Σ(yi - ŷi)²
```

Where `ŷi` is the predicted value. OLS finds the line `ŷ = β0 + β1·x` that makes SSR as small as possible.

**Closed-form solution (Normal Equation):**
```
β = (XᵀX)⁻¹ Xᵀy
```

---

## 2.3 Linear Regression with One Variable

**Model:** `ŷ = θ0 + θ1·x`

- θ0 = y-intercept (bias)
- θ1 = slope (weight)

### Cost Function
The **Mean Squared Error (MSE):**
```
J(θ0, θ1) = (1/2m) Σ(ŷᵢ - yᵢ)²
```
Goal: Minimize J.

### Gradient Descent
An **iterative optimization algorithm** that moves θ values in the direction that reduces J most steeply.

**Update rule (simultaneously update both parameters):**
```
θ0 := θ0 - α · (∂J/∂θ0)
θ1 := θ1 - α · (∂J/∂θ1)
```

Where `α` (alpha) is the **learning rate** — how big a step to take each iteration.

**Effect of learning rate:**
- Too small α → very slow convergence
- Too large α → overshooting, may diverge (cost increases instead of decreasing)
- Good α → smooth, steady decrease in cost

**Convergence:** Stop when the change in J between iterations is smaller than a threshold ε (epsilon).

---

## 2.4 Multiple Linear Regression

When you have multiple features: `ŷ = θ0 + θ1·x1 + θ2·x2 + ... + θn·xn`

In vector form: `ŷ = Xθ`

Two approaches:

### A) Gradient Descent
- Works for any number of features.
- Requires choosing learning rate α.
- Scales well with large datasets.

### B) Normal Equation
```
θ = (XᵀX)⁻¹ Xᵀy
```
- Gives the exact optimal θ in one step.
- **No need for feature scaling or iteration.**
- **Problem:** Computing (XᵀX)⁻¹ is O(n³) — becomes very slow when features n > 10,000.

| Aspect | Gradient Descent | Normal Equation |
|---|---|---|
| Iteration | Required | Not required |
| Learning rate | Needs tuning | Not needed |
| Feature scaling | Needed | Not needed |
| Large n features | Works fine | Slow (matrix inversion) |
| Large m samples | Efficient | Memory-intensive |

---

## 2.5 Polynomial Regression

When the relationship is non-linear, add polynomial terms:
```
ŷ = θ0 + θ1·x + θ2·x² + θ3·x³ + ...
```

This is still **linear regression** internally — linear in the parameters θ. You're just transforming your features.

**Lab tip:** Use `sklearn.preprocessing.PolynomialFeatures(degree=d)` to auto-generate these terms.

> **Risk:** High-degree polynomials overfit — they pass through every training point but perform poorly on new data.

---

## 2.6 Evaluation Metrics for Regression

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | (1/m)Σ\|yi − ŷi\| | Average absolute error; easy to interpret |
| MSE | (1/m)Σ(yi − ŷi)² | Penalizes large errors more; not in original units |
| RMSE | √MSE | Same units as target; most commonly reported |
| R² (R-squared) | 1 − SSres/SStot | % of variance explained; 1 is perfect, 0 is baseline |
| Adjusted R² | 1 − (1−R²)(n−1)/(n−k−1) | Penalizes unnecessary features; use for multiple regression |

**When to prefer which:**
- Use **RMSE** when large errors are particularly bad.
- Use **MAE** when all errors should be weighted equally.
- Use **R²** to compare models on the same dataset.

---

## 2.7 Underfitting and Overfitting

### Underfitting (High Bias)
- The model is **too simple** to capture the true pattern.
- High training error AND high test error.
- Symptom: Low R², consistently wrong predictions.
- Fix: Add features, increase model complexity, reduce regularization.

### Overfitting (High Variance)
- The model **memorizes training data** including noise.
- Very low training error but high test error.
- Symptom: R² ≈ 1 on train, but drops sharply on test.
- Fix: More data, reduce complexity, regularization, dropout (for NNs).

### The Bias-Variance Tradeoff
```
Total Error = Bias² + Variance + Irreducible Noise
```
- **Bias:** Error from wrong assumptions (underfitting).
- **Variance:** Error from sensitivity to small fluctuations in training data (overfitting).
- You can't reduce both simultaneously — improving one worsens the other. The sweet spot is the balance point.

**Visualization:** The classic U-shaped test error curve as model complexity increases.

---

## 2.8 Regularization

Regularization **adds a penalty for large θ values** to the cost function, discouraging overfitting by keeping weights small.

### Ridge Regression (L2)
```
J(θ) = MSE + λΣθj²
```
- Adds the **sum of squares** of weights.
- Shrinks all weights toward zero but **never makes them exactly zero**.
- Keeps all features in the model.
- **Best when:** All features are potentially relevant.

### Lasso Regression (L1)
```
J(θ) = MSE + λΣ|θj|
```
- Adds the **sum of absolute values** of weights.
- Can drive some weights to **exactly zero → automatic feature selection**.
- Produces **sparse models**.
- **Best when:** You suspect many features are irrelevant.

### λ (Lambda) — Regularization Strength
- λ = 0 → No regularization (standard regression)
- λ → ∞ → All weights forced to zero (extreme underfitting)
- Tune λ using cross-validation.

| Aspect | Ridge | Lasso |
|---|---|---|
| Penalty | L2 (squares) | L1 (absolute values) |
| Feature selection | No | Yes (sets some to 0) |
| Handles multicollinearity | Yes | Partially |
| When to use | All features matter | Sparse solution needed |

---

<a name="unit-3"></a>
# UNIT 3 — Classification

Classification predicts a **discrete class label** (e.g., spam/not spam, cat/dog, disease grade 0/1/2/3).

---

## 3.1 Logistic Regression

Despite the name, logistic regression is a **classification algorithm**, not a regression one.

### How it Works
It applies the **sigmoid function** to the linear combination of features:
```
P(Y=1 | X) = σ(z) = 1 / (1 + e^(-z))
where z = θ0 + θ1·x1 + ... + θn·xn
```
Output is a probability between 0 and 1. If P > 0.5 → class 1, else → class 0.

### Cost Function
Cross-Entropy (Log Loss):
```
J(θ) = -(1/m) Σ [yi·log(ŷi) + (1-yi)·log(1-ŷi)]
```

### Multinomial (Multiclass) Logistic Regression
Uses **Softmax function** for K classes:
```
P(Y=k | X) = e^(zk) / Σ e^(zj)
```

### Why use Logistic over Linear for classification?
- Linear regression can predict values outside [0,1] — meaningless as probabilities.
- Logistic regression naturally bounds output to [0,1].
- The cross-entropy loss is well-suited for probability estimation.

---

## 3.2 Support Vector Machine (SVM)

### Core Idea
SVM finds the **hyperplane that maximizes the margin** between classes.

- **Hyperplane:** Decision boundary: `wᵀx + b = 0`
- **Support Vectors:** Data points closest to the hyperplane.
- **Margin:** Distance between the two parallel margin hyperplanes. SVM maximizes this.

### Soft Margin SVM
Real data is often not linearly separable. Soft margin introduces **slack variables (ξ)** allowing some misclassification while still maximizing margin.

Controlled by parameter **C:**
- Large C → Less tolerance for misclassification (tighter fit, possible overfit)
- Small C → More tolerance (wider margin, possible underfit)

### Kernel Trick
For non-linear data, SVM implicitly maps data to a higher-dimensional space:
- **Linear kernel:** `K(x, z) = xᵀz` — for linearly separable data
- **RBF (Gaussian) kernel:** `K(x,z) = exp(-γ||x-z||²)` — for non-linear boundaries
- **Polynomial kernel:** `K(x,z) = (xᵀz + c)^d`

### Why SVM over Logistic Regression?
- SVM works better in **high-dimensional spaces** (e.g., text classification).
- Logistic Regression gives probabilities; SVM gives a margin-based boundary.
- SVM is more effective when data is **not easily linearly separable**.

---

## 3.3 K-Nearest Neighbors (KNN)

### How it Works
1. Store all training data (no explicit training phase).
2. For a new point, compute distance to all training points.
3. Find the **K nearest neighbors**.
4. Assign the **majority class** among those K neighbors (classification) or mean value (regression).

**Distance metrics:**
- Euclidean: `d = √Σ(xi - xj)²` — most common
- Manhattan: `d = Σ|xi - xj|` — less sensitive to outliers

### Choosing K
- Small K (e.g., K=1): Low bias, high variance — overfits
- Large K: High bias, low variance — underfits
- Best practice: Try odd K values, use cross-validation to find the optimal K.

### Why not always use KNN?
- **Slow prediction:** Must compute distances to all training points at test time — O(n·d).
- **Curse of dimensionality:** Distance becomes meaningless in high dimensions.
- **No model:** Cannot extract feature importance or interpretable rules.

---

## 3.4 Naive Bayes

### Bayes' Theorem
```
P(Y|X) = P(X|Y) · P(Y) / P(X)
```
- P(Y|X) — **Posterior:** Probability of class Y given features X.
- P(X|Y) — **Likelihood:** Probability of seeing X given class Y.
- P(Y) — **Prior:** How common is class Y overall.
- P(X) — **Evidence:** Constant normalizer.

### "Naive" Assumption
All features are **conditionally independent** given the class:
```
P(X|Y) = P(x1|Y) · P(x2|Y) · ... · P(xn|Y)
```
This is almost never true in practice, but Naive Bayes still works surprisingly well.

### Variants
- **Gaussian NB:** Features follow a Gaussian distribution (for continuous features).
- **Multinomial NB:** For word counts (text classification).
- **Bernoulli NB:** For binary features.

### Application: Spam Detection
- Prior: P(spam), P(not spam) from training data.
- Likelihood: How often does each word appear in spam vs. not spam?
- Posterior: Given these words in an email, what's the probability it's spam?

**Why Naive Bayes over Logistic Regression for text?**
- Very fast to train.
- Works well with small datasets.
- Handles high-dimensional sparse data (text) efficiently.

---

## 3.5 Decision Tree

### How it Works
A decision tree **recursively splits** the dataset on the feature and threshold that best separates the classes.

**Splitting Criteria:**
- **Gini Impurity:** `G = 1 - Σpi²` (used in CART algorithm)
- **Entropy / Information Gain:** `H = -Σpi·log2(pi)`; split that maximizes `IG = H(parent) - weighted_avg_H(children)`

At each node, the algorithm:
1. Evaluates all features and all thresholds.
2. Picks the split that minimizes impurity (or maximizes IG).
3. Recurses until a stopping condition (max depth, min samples, pure leaves).

### Advantages
- Highly interpretable — you can visualize and explain the logic.
- Handles both numerical and categorical features.
- No need for feature scaling.

### Disadvantages
- Prone to **overfitting** (high variance) — easy to grow a tree that memorizes training data.
- Unstable — small changes in data can drastically change the tree.

---

## 3.6 Random Forest

### Core Idea: Ensemble Learning via Bagging
A Random Forest builds **many decision trees**, each on a:
1. **Bootstrap sample** (random subset of rows with replacement).
2. **Random subset of features** at each split (feature randomness).

Final prediction = **majority vote** (classification) or **average** (regression).

### Why it Works
Each tree sees slightly different data and features → trees are **decorrelated** → averaging reduces variance without increasing bias much.

### Hyperparameters
- `n_estimators`: Number of trees (more is better, diminishing returns).
- `max_depth`: Maximum depth per tree.
- `max_features`: Features considered per split (common default: √n_features).

### Feature Importance
Random forests can rank features by how much each feature reduces impurity across all trees — very useful for EDA.

**Why Random Forest over a single Decision Tree?**
- Much less overfitting.
- More robust to noisy data.
- Generally better accuracy.

---

## 3.7 Ensemble Methods

### Bagging (Bootstrap Aggregating)
- Train multiple models **in parallel** on bootstrap samples.
- Combine by voting/averaging.
- Goal: Reduce **variance**.
- Random Forest is the most famous bagging method.

### Boosting
- Train models **sequentially**.
- Each new model **focuses on the mistakes** of the previous ensemble.
- Goal: Reduce **bias**.
- Final prediction: Weighted combination of all models.

#### AdaBoost (Adaptive Boosting)
1. Assign equal weight to all training samples.
2. Train a weak learner (usually a shallow tree / "stump").
3. Increase weights on misclassified samples.
4. Repeat, adding new learners focused on hard examples.
5. Final prediction: Weighted majority vote.

#### Gradient Boosting
1. Start with a simple prediction (mean).
2. Compute **residuals** (errors).
3. Fit a new tree to predict those residuals.
4. Add it to the ensemble (scaled by learning rate).
5. Repeat.

#### XGBoost (Extreme Gradient Boosting)
XGBoost is an **optimized, regularized** implementation of gradient boosting:
- Uses **second-order Taylor expansion** of the loss function for smarter splits.
- Built-in **L1 and L2 regularization** on tree weights.
- **Column (feature) subsampling** — like Random Forests.
- **Parallel tree construction** on features — fast.
- Handles **missing values** natively.
- **Pruning:** Grows trees fully, then prunes branches that don't improve gain beyond a threshold (γ).

**Key XGBoost hyperparameters:**
| Parameter | Meaning | Default |
|---|---|---|
| n_estimators | Number of trees | 100 |
| learning_rate (η) | Shrinkage per step | 0.3 |
| max_depth | Tree depth | 6 |
| subsample | Row subsampling | 1.0 |
| colsample_bytree | Feature subsampling | 1.0 |
| gamma (γ) | Min gain to split | 0 |
| lambda (λ) | L2 regularization | 1 |
| alpha (α) | L1 regularization | 0 |

**Why XGBoost over Random Forest?**
- Boosting (sequential correction) often outperforms bagging (parallel averaging) when bias is the main problem.
- XGBoost handles regularization explicitly.
- Consistently wins Kaggle competitions on tabular data.

**Why not always use XGBoost?**
- Harder to tune (many hyperparameters).
- Can still overfit if not tuned.
- Slower to train than Random Forest for very large datasets (though parallelization helps).
- Less interpretable than a single tree.

### Bagging vs Boosting Summary

| Aspect | Bagging | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Goal | Reduce Variance | Reduce Bias |
| Weak learner | Independent | Depends on previous |
| Overfit risk | Lower | Higher (needs tuning) |
| Example | Random Forest | AdaBoost, XGBoost |

---

## 3.8 Evaluation Metrics for Classification

### Confusion Matrix (Binary)

```
                    Predicted Positive   Predicted Negative
Actual Positive         TP                    FN
Actual Negative         FP                    TN
```

- **TP (True Positive):** Correctly predicted positive.
- **TN (True Negative):** Correctly predicted negative.
- **FP (False Positive / Type I Error):** Predicted positive but actually negative.
- **FN (False Negative / Type II Error):** Predicted negative but actually positive.

### Derived Metrics

| Metric | Formula | When to use |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | When FP is costly (e.g., spam filter) |
| Recall (Sensitivity) | TP/(TP+FN) | When FN is costly (e.g., disease detection) |
| F1-Score | 2·(P·R)/(P+R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | Overall ranking ability |

### ROC Curve
Plots **True Positive Rate (Recall)** vs **False Positive Rate** at every decision threshold.
- AUC = 1.0: Perfect classifier.
- AUC = 0.5: Random guessing.

### When accuracy is misleading
If 95% of emails are not spam, a model that always predicts "not spam" achieves 95% accuracy but is useless. Use **F1-score or ROC-AUC** for imbalanced datasets.

---

<a name="unit-4"></a>
# UNIT 4 — Use Case: Regression

## Stock Price Prediction

### Problem Statement
Predict the **closing price** (or daily return) of a stock based on historical data.

### Features Used
- Previous day(s) closing price (lag features)
- Volume
- Moving averages (5-day, 20-day, 50-day MA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### ⚠️ The Autocorrelation Trap
Raw price lag features cause **data leakage through autocorrelation** — stock prices are highly correlated with their previous values simply because they move slowly. This makes simple models (Linear Regression) appear to outperform complex ones (LSTM, XGBoost) because they just learn "tomorrow ≈ today."

**Solution:** Use **return-based targets** and **ratio-based features**:
```python
df['return'] = df['Close'].pct_change()          # Target: % change
df['lag1_return'] = df['return'].shift(1)         # Lag return, not lag price
df['ma5_ratio'] = df['Close'] / df['Close'].rolling(5).mean()  # Ratio feature
```

### Model Comparison for Stock Prediction

| Model | Strengths | Weaknesses |
|---|---|---|
| Linear Regression | Interpretable, fast | Can't capture non-linearity |
| Polynomial Regression | Captures some non-linearity | Overfits easily |
| Decision Tree | Non-linear, fast | Very high variance |
| Random Forest | Robust, handles non-linearity | No temporal memory |
| XGBoost | Best tabular accuracy | Many hyperparams, no temporal memory |
| LSTM | Captures temporal sequences | Needs lots of data, slow to train |
| MLP | Universal approximator | No inherent temporal structure |

### Why LSTM for time-series?
- LSTMs have **memory cells and gates** that can learn long-range dependencies.
- They remember patterns from 10 or 50 timesteps ago, unlike tabular models that only see the features you manually provide.
- But for short-term prediction with engineered features, XGBoost often matches or beats LSTM with far less complexity.

---

## House Price Prediction

### Problem
Predict the sale price of a house based on features (area, rooms, location, age, etc.)

### Why Linear/Ridge over Decision Tree here?
- The relationship between area and price is mostly linear.
- Ridge handles multicollinearity (correlated features like area and rooms).
- Decision trees would overfit on small real estate datasets.

---

<a name="unit-5"></a>
# UNIT 5 — Use Case: Classification

## Medical Diagnosis (Cancer / Diabetes)

### Why Logistic Regression?
- Provides calibrated **probabilities** (doctor wants to know "60% likely malignant", not just "malignant").
- Interpretable coefficients — each feature's effect on odds can be explained.
- Works well when the decision boundary is roughly linear in feature space.

### Why not KNN for medical diagnosis?
- Slow on large patient databases.
- No interpretability — you can't explain why two patients are "similar."
- Sensitive to irrelevant features.

### Why Random Forest / XGBoost for cancer classification?
- Handles non-linear interactions between biomarkers.
- Provides feature importance (which genes / proteins matter most?).
- Robust to outlier lab measurements.

---

## Email Spam Detection

### Why Naive Bayes?
- Very fast for high-dimensional text (thousands of word features).
- Works well with small training data.
- Probabilistic output (spam probability) can be thresholded.

### Why not SVM?
- SVM works excellently too (linear kernel on text), but is slower to train.
- SVM doesn't give calibrated probabilities directly.

---

## Customer Churn Prediction

### Why XGBoost?
- Mix of numerical (purchase history) and categorical (plan type) features — XGBoost handles both.
- Class imbalance (few churners) — XGBoost's `scale_pos_weight` parameter addresses this.
- Need for interpretability — SHAP values work natively with XGBoost.

---

<a name="lab"></a>
# LAB PRACTICAL CHEAT-SHEET

## Standard ML Pipeline (Python / Jupyter)

```python
# 1. Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# 2. Load Data
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes, df.isnull().sum())
df.describe()

# 3. EDA
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
df['target'].hist()

# 4. Feature Engineering
df.dropna(inplace=True)  # or df.fillna(df.mean())
X = df.drop('target', axis=1)
y = df['target']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale (important for Linear, SVM, KNN, Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # Note: use transform, NOT fit_transform on test!

# 7. Train & Evaluate
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

## All Models — Quick Code

```python
# Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)

# SVM
from sklearn.svm import SVC, SVR
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')

# KNN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
knn = KNeighborsClassifier(n_neighbors=5)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB
nb = GaussianNB()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
dt = DecisionTreeClassifier(max_depth=5, criterion='gini')

# Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# XGBoost
from xgboost import XGBClassifier, XGBRegressor
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss')

# MLP Neural Network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500)
```

---

## LSTM for Time-Series (Keras)

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prepare sequences
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(scaled_data)
X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_seq, y_seq, epochs=20, batch_size=32)
```

---

## Evaluation — Classification

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')

# ROC Curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()
```

---

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
```

---

## Feature Scaling — When is it Required?

| Algorithm | Scaling Needed? | Why |
|---|---|---|
| Linear Regression | Yes (for GD) | Gradient descent converges faster |
| Ridge / Lasso | Yes | Penalty fair across features |
| Logistic Regression | Yes | For gradient-based optimization |
| SVM | Yes | Distance-based; features on diff scales distort margins |
| KNN | Yes | Distance-based |
| Decision Tree | No | Splits are threshold-based |
| Random Forest | No | Ensemble of trees |
| XGBoost | No | Tree-based |
| Neural Networks | Yes | Gradients unstable with large inputs |

---

## Common Mistakes to Avoid in Lab

1. **Scaling the test set with its own statistics** — always fit the scaler on train, then transform test.
2. **Not resetting the index** after dropping rows — causes silent alignment bugs.
3. **Using price directly as a lag feature** in time-series — causes autocorrelation leakage.
4. **Looking at accuracy alone** for imbalanced classification — always check F1, ROC-AUC.
5. **Not setting `random_state`** — results change on every run, making comparison meaningless.
6. **Forgetting to tune hyperparameters** — default XGBoost often overfits on small datasets.
7. **Using `fit_transform` on test data** for one-hot encoding — introduces unseen categories.

---

## Quick Algorithm Selection Guide

| Problem Type | Start With | Also Try | Avoid If |
|---|---|---|---|
| Regression, linear relationship | Linear/Ridge Regression | Polynomial, SVR | Lots of noise → use Ridge |
| Regression, non-linear | Random Forest | XGBoost | Tree if data < 100 rows |
| Regression, time-series | XGBoost + feature eng. | LSTM | LSTM if data < 1000 rows |
| Binary Classification | Logistic Regression | RF, XGBoost | LR if boundary is non-linear |
| Multi-class, small data | Naive Bayes | KNN | NB if features are correlated |
| Text Classification | Naive Bayes | Linear SVM | Tree-based models |
| Interpretability required | Decision Tree, Logistic | Ridge | Deep Learning |
| High accuracy on tabular | XGBoost | Random Forest, LightGBM | LR / SVM |

---

*End of Notes — TE7499 Supervised Machine Learning*
*B.Tech AI & ML | Symbiosis International University*
