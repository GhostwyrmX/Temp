# Machine Learning Algorithms Study Guide
*A Comprehensive Guide for Viva Preparation*

This study guide explains major machine learning algorithms in simple terms, covering when to use them, when NOT to use them, computational complexity, preprocessing requirements, and preferred evaluation metrics.

## Table of Contents
1. [Linear Regression](#linear-regression)
2. [Logistic Regression](#logistic-regression)
3. [Support Vector Machines (SVM)](#support-vector-machines-svm)
4. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
5. [Naive Bayes](#naive-bayes)
6. [Decision Trees](#decision-trees)
7. [Random Forest](#random-forest)
8. [XGBoost](#xgboost)

---

## Linear Regression

### What it is:
Linear Regression is like drawing the best straight line through a scatter plot of data points. It assumes there's a linear relationship between your input features (like house size) and your target value (like house price).

Think of it as: *If I know how big a house is, I can predict its price by assuming price increases linearly with size.*

### When to use it:
- You want to predict a continuous numerical value (regression problem)
- You believe there's a linear relationship between features and target
- You need a simple, interpretable model
- You're doing preliminary analysis or baseline modeling
- Features and target have a roughly linear correlation

### When NOT to use it:
- When the relationship between features and target is non-linear (curved)
- When you have categorical outcomes (use Logistic Regression instead)
- When features are highly correlated with each other (multicollinearity)
- When outliers significantly affect the line (Linear Regression is sensitive to outliers)

### Computational Complexity:
- Training: O(n×p²) where n = number of samples, p = number of features
- Prediction: O(p)
- Normal equation approach: O(p³) due to matrix inversion
- Gradient descent approach: Depends on iterations needed for convergence

### Preprocessing Requirements:
- Handle missing values (remove or impute)
- Remove or address outliers that can skew the line
- Feature scaling is not strictly required but can help with interpretation
- Consider removing highly correlated features (multicollinearity)
- Transform non-linear relationships using polynomial features if needed

### Preferred Evaluation Metrics:
- **R² (R-squared)**: Percentage of variance explained by the model (0-1 scale, higher is better)
- **RMSE (Root Mean Square Error)**: Average prediction error in the same units as target
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage error (good for comparing across datasets)

---

## Logistic Regression

### What it is:
Despite its name, Logistic Regression is used for classification, not regression. It predicts the probability that something belongs to a particular category (like spam/not spam). It uses the "sigmoid" function to squeeze any number between 0 and 1, which can be interpreted as a probability.

Think of it as: *Based on email content, what's the probability this is spam? If it's over 50%, classify it as spam.*

### When to use it:
- Binary or multi-class classification problems
- You need probability estimates (not just yes/no predictions)
- The relationship between features and log-odds is approximately linear
- You want an interpretable model with feature importance
- Dataset is reasonably sized (not extremely large)
- Features are mostly independent (violates "naive" assumption less)

### When NOT to use it:
- When classes are heavily imbalanced without addressing it
- When features are highly correlated (multicollinearity affects interpretation)
- When the relationship between features and outcome is highly non-linear
- When you have complex interactions between features that need capturing
- When you need state-of-the-art performance on complex datasets

### Computational Complexity:
- Training: O(n×p×i) where n = samples, p = features, i = iterations to converge
- Prediction: O(p)
- Uses iterative optimization (like gradient descent)
- Generally faster to train than tree-based methods

### Preprocessing Requirements:
- Handle missing values appropriately
- Remove or address extreme outliers
- Feature scaling recommended (especially for regularization)
- Encode categorical variables as numerical (one-hot encoding)
- Consider interaction terms for important feature combinations

### Preferred Evaluation Metrics:
- **Accuracy**: Overall correct predictions (misleading for imbalanced datasets)
- **Precision**: Of all positive predictions, how many were correct?
- **Recall (Sensitivity)**: Of all actual positives, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (good for comparing models)
- **Log-Loss**: Measures quality of probability estimates

---

## Support Vector Machines (SVM)

### What it is:
SVM finds the "best" dividing line (or hyperplane in higher dimensions) that separates different classes with the maximum margin (distance). It focuses on the points closest to the decision boundary (support vectors) rather than all data points.

Think of it as: *Drawing the widest possible road between two groups of houses so that red houses are on one side and blue houses on the other.*

### When to use it:
- Text classification problems (documents, spam detection)
- Image classification with clear separation
- When you have fewer samples than features
- When classes have clear margin of separation
- High-dimensional spaces (text data with thousands of words)
- When you need good performance with minimal tuning

### When NOT to use it:
- When dataset is very large (millions of samples) - slow training
- When you need probability estimates (SVM doesn't naturally provide them)
- When classes overlap significantly (no clear margin)
- When you need to understand feature importance easily
- When computational resources are limited for training

### Computational Complexity:
- Training: O(n²) to O(n³) where n = number of samples
- Prediction: O(n×p) in worst case (need to check all support vectors)
- Memory intensive - stores support vectors
- Kernel computations add additional overhead for non-linear kernels

### Preprocessing Requirements:
- Essential feature scaling (StandardScaler) - SVM is sensitive to feature scales
- Handle missing values appropriately
- Remove outliers that could become support vectors
- Consider dimensionality reduction for very high-dimensional data
- Encode categorical variables numerically

### Preferred Evaluation Metrics:
- **Accuracy**: Overall correct predictions
- **Precision and Recall**: Especially important for imbalanced datasets
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Good for comparing different SVM configurations
- **Confusion Matrix**: Detailed breakdown of predictions

---

## K-Nearest Neighbors (KNN)

### What it is:
KNN is an intuitive algorithm that classifies a new data point based on the "k" closest neighbors in the training data. For classification, it takes a majority vote of the k neighbors. For regression, it averages their values.

Think of it as: *To decide if a new student likes math, look at the 5 most similar students and see what most of them prefer.*

### When to use it:
- When the decision boundary is irregular or non-linear
- When you have sufficient training data
- When you don't want to make strong assumptions about data distribution
- For recommendation systems (users similar to you liked...)
- When you have consistent, evenly distributed data
- When you want a simple baseline model

### When NOT to use it:
- When you have a very large dataset (slow prediction)
- When you have high-dimensional data (curse of dimensionality)
- When you have noisy or irrelevant features
- When you need fast prediction times
- When storage is limited (stores entire training dataset)
- When you need interpretable feature importance

### Computational Complexity:
- Training: O(1) - just stores data
- Prediction: O(n×p) where n = training samples, p = features
- For n test samples: O(n²×p)
- With optimized data structures (KD-tree, Ball tree): Better in low dimensions
- Scales poorly with dataset size and dimensions

### Preprocessing Requirements:
- **Essential feature scaling** - all features must be on similar scales
- Handle missing values appropriately
- Remove irrelevant or noisy features
- Consider dimensionality reduction techniques
- Encode categorical variables numerically

### Preferred Evaluation Metrics:
- **Accuracy**: Overall correct predictions
- **Precision and Recall**: For imbalanced datasets
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: For comparing different k values
- **Confusion Matrix**: Detailed error analysis

---

## Naive Bayes

### What it is:
Naive Bayes is based on Bayes' theorem and assumes that all features are independent of each other (the "naive" part). Despite this unrealistic assumption, it works surprisingly well, especially for text classification.

Think of it as: *If most spam emails contain "free money" and this email contains "free money", it's probably spam - regardless of other words.*

### When to use it:
- Text classification (spam detection, sentiment analysis)
- Real-time prediction (fast)
- Small training datasets
- Multi-class classification problems
- When you need probabilistic predictions
- When features are relatively independent

### When NOT to use it:
- When feature independence assumption is severely violated
- When you need highly accurate probability estimates
- When relationships between features are important
- When you have very correlated features
- When you need to understand complex decision boundaries

### Computational Complexity:
- Training: O(n×p) where n = samples, p = features
- Prediction: O(p×c) where c = number of classes
- Very fast for both training and prediction
- Scales well with dataset size
- Minimal memory requirements

### Preprocessing Requirements:
- Handle missing values appropriately
- For text: Tokenization, stop word removal, stemming
- Feature scaling generally not required
- Encode categorical variables appropriately
- Consider smoothing techniques for zero-frequency problems

### Preferred Evaluation Metrics:
- **Accuracy**: Overall performance
- **Precision and Recall**: Especially for imbalanced classes
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: For comparing different variants
- **Log-Loss**: Quality of probability estimates

---

## Decision Trees

### What it is:
Decision Trees mimic human decision-making by splitting data based on feature values, creating a tree-like structure of if-then rules. Each split aims to create purer subsets (more homogeneous classes).

Think of it as: *Is the person older than 30? Yes → Does exercise regularly? No → Likely unhealthy. No → Likely healthy.*

### When to use it:
- When you need interpretable models (rules easy to explain)
- When relationships between features and target are non-linear
- Mixed data types (numerical and categorical features)
- Feature selection (importance ranking)
- As a baseline for more complex algorithms
- When you want to understand data structure

### When NOT to use it:
- When you need high predictive accuracy (trees overfit easily)
- When you have noisy data (very sensitive to noise)
- When extrapolation beyond training data range is needed
- When you have too many features (computationally expensive)
- When stability is important (small data changes = different tree)

### Computational Complexity:
- Training: O(n×p×log(n)) where n = samples, p = features
- Prediction: O(log(n)) - traverse tree from root to leaf
- Memory: O(nodes) - depends on tree size
- Can become very deep with complex data

### Preprocessing Requirements:
- Handle missing values (modern implementations handle this)
- No strict requirement for feature scaling
- Can handle categorical variables natively
- Outliers generally don't affect trees much
- Consider feature engineering for better splits

### Preferred Evaluation Metrics:
- **Accuracy**: Overall performance
- **Precision and Recall**: For imbalanced datasets
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Area under curve comparison
- **Confusion Matrix**: Detailed error breakdown

---

## Random Forest

### What it is:
Random Forest combines many decision trees (hence "forest") to make predictions. Each tree is trained on a random subset of data and features. The final prediction is an average (regression) or majority vote (classification) of all trees.

Think of it as: *Instead of asking one doctor, ask 100 doctors independently and take the majority opinion - reduces chance of individual error.*

### When to use it:
- When you need high accuracy with good generalization
- Mixed data types (numerical and categorical)
- Feature selection and importance ranking needed
- Moderate to large datasets
- When overfitting is a concern with single trees
- When you want good performance with minimal tuning

### When NOT to use it:
- When you need a highly interpretable model (less interpretable than single trees)
- When computational resources are very limited
- When you need real-time predictions (slower than single trees)
- When dataset is very small (may not benefit from ensemble)
- When linear relationships dominate (Linear Regression might be better)

### Computational Complexity:
- Training: O(t×n×p×log(n)) where t = trees, n = samples, p = features
- Prediction: O(t×log(n)) - traverse each tree
- Memory: O(t×nodes) - stores all trees
- Parallelizable - trees can be built independently

### Preprocessing Requirements:
- Handle missing values appropriately
- No strict requirement for feature scaling
- Can handle categorical variables natively
- Outliers generally don't affect performance much
- Consider feature engineering for better results

### Preferred Evaluation Metrics:
- **Accuracy**: Overall performance
- **Precision and Recall**: For imbalanced datasets
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Area under curve comparison
- **Feature Importance**: Which features contribute most
- **Out-of-Bag Error**: Internal validation estimate

---

## XGBoost

### What it is:
XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting. It builds trees sequentially, where each new tree corrects errors made by previous trees. It includes regularization to prevent overfitting and various optimizations for speed and performance.

Think of it as: *Learning from mistakes - first make a rough prediction, then focus on what you got wrong, make another prediction focusing on those errors, and repeat.*

### When to use it:
- Tabular data competitions (consistently wins on Kaggle)
- Structured/tabular datasets with mixed features
- When you need high predictive accuracy
- When you have sufficient data (not tiny datasets)
- When overfitting is managed with regularization
- When you want feature importance rankings

### When NOT to use it:
- When you need a highly interpretable model
- When computational resources are very limited
- When you need real-time predictions (slower than simpler models)
- When dataset is very small (may overfit)
- When you prefer simpler, faster models for deployment

### Computational Complexity:
- Training: O(t×n×p×log(n)) where t = trees, n = samples, p = features
- Prediction: O(t×log(n)) - traverse each tree
- Memory: O(t×nodes) - stores all trees
- Optimized for speed with parallel processing

### Preprocessing Requirements:
- Handle missing values (XGBoost handles natively)
- Feature scaling generally not required
- Encode categorical variables numerically
- Consider feature engineering for better performance
- Split data appropriately for validation

### Preferred Evaluation Metrics:
- **Accuracy**: Overall performance
- **Precision and Recall**: For imbalanced datasets
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Area under curve comparison
- **RMSE/MAE**: For regression problems
- **Feature Importance**: Which features contribute most

---

## Conclusion

Choosing the right machine learning algorithm depends on several factors:

1. **Problem Type**: Classification vs Regression
2. **Data Size**: Small datasets may need simpler models; large datasets can handle complex ones
3. **Interpretability**: Whether you need to explain decisions
4. **Performance Requirements**: Speed of training vs prediction
5. **Data Characteristics**: Linear relationships, feature types, missing values
6. **Computational Resources**: Memory and processing power available

### Quick Selection Guide:

- **Start Simple**: Linear/Logistic Regression for baseline
- **Text Data**: Naive Bayes or SVM
- **Structured Data**: Random Forest or XGBoost
- **Need Interpretability**: Decision Trees or Logistic Regression
- **High Accuracy Needed**: XGBoost or Random Forest
- **Small Dataset**: Naive Bayes or SVM
- **Large Dataset**: Random Forest or XGBoost

Remember to always:
1. Split your data properly (train/validation/test)
2. Preprocess appropriately for each algorithm
3. Use cross-validation for reliable performance estimates
4. Choose evaluation metrics that align with business objectives
5. Tune hyperparameters systematically

Good luck with your viva!