# Machine Learning Evaluation Metrics Study Guide
## Comprehensive Guide for Viva Preparation

---

## Table of Contents
1. [Introduction](#introduction)
2. [Regression Metrics](#regression-metrics)
3. [Classification Metrics](#classification-metrics)
4. [When to Use Which Metric](#when-to-use-which-metric)
5. [Metric Selection by Algorithm](#metric-selection-by-algorithm)

---

<a name="introduction"></a>
# Introduction

Evaluation metrics are crucial for assessing the performance of machine learning models. They help us understand how well our models are performing and guide us in making improvements. Different metrics are suitable for different types of problems:

- **Regression Problems**: Predicting continuous numerical values
- **Classification Problems**: Predicting discrete class labels

Choosing the right evaluation metric is essential for accurately measuring model performance and making informed decisions about model selection and improvements.

---

<a name="regression-metrics"></a>
# Regression Metrics

Regression metrics measure the performance of models that predict continuous values (e.g., house prices, temperature, stock returns).

## 1. Mean Absolute Error (MAE)

**What it is**: Average of absolute differences between predicted and actual values.

**Formula**: MAE = (1/m) Σ|yi - ŷi|

**When to use**:
- When all errors should be weighted equally
- When you want interpretable results in the original units
- When outliers shouldn't be penalized heavily

**When NOT to use**:
- When large errors need to be penalized more significantly
- When you need to differentiate between models with similar performance

**Advantages**:
- Easy to understand and interpret
- Robust to outliers
- Same units as target variable

**Disadvantages**:
- Doesn't penalize large errors enough in some cases
- Not differentiable at zero, which can be problematic for optimization

## 2. Mean Squared Error (MSE)

**What it is**: Average of squared differences between predicted and actual values.

**Formula**: MSE = (1/m) Σ(yi - ŷi)²

**When to use**:
- When large errors need to be penalized more than small ones
- In optimization algorithms that require differentiable functions
- When mathematical convenience is important

**When NOT to use**:
- When outliers should not be heavily penalized
- When interpretability in original units is important

**Advantages**:
- Mathematically convenient (differentiable everywhere)
- Heavily penalizes large errors
- Useful for gradient-based optimization

**Disadvantages**:
- Units are squared (not intuitive)
- Sensitive to outliers
- May over-emphasize large errors

## 3. Root Mean Squared Error (RMSE)

**What it is**: Square root of the average of squared differences between predicted and actual values.

**Formula**: RMSE = √MSE = √[(1/m) Σ(yi - ŷi)²]

**When to use**:
- Most commonly reported regression metric
- When you need error measurement in the same units as target variable
- When large errors should be penalized more than small ones

**When NOT to use**:
- When outliers should not be heavily penalized
- When interpretability is less important than robustness

**Advantages**:
- Same units as target variable (easy interpretation)
- Heavily penalizes large errors
- Widely recognized and used

**Disadvantages**:
- Sensitive to outliers
- May over-emphasize large errors in some applications

## 4. R-squared (Coefficient of Determination)

**What it is**: Proportion of variance in the target variable explained by the model.

**Formula**: R² = 1 - (SSres/SStot) = 1 - [Σ(yi - ŷi)²/Σ(yi - ȳ)²]

**When to use**:
- To compare models on the same dataset
- To understand percentage of variance explained by the model
- When you want a relative measure of performance

**When NOT to use**:
- When comparing models across different datasets
- When the model has more features than needed (use Adjusted R² instead)
- When you need an absolute measure of error

**Advantages**:
- Dimensionless (no units)
- Easy to interpret as percentage of variance explained
- Useful for comparing models on same dataset

**Disadvantages**:
- Can be negative (worse than baseline)
- Increases with more features even if they're irrelevant
- Not suitable for comparing across datasets

## 5. Adjusted R-squared

**What it is**: Modified R² that penalizes the addition of unnecessary features.

**Formula**: Adjusted R² = 1 - [(1-R²)(n-1)/(n-k-1)]

Where n = number of observations, k = number of predictors

**When to use**:
- For multiple regression models
- When comparing models with different numbers of features
- When feature selection is important

**When NOT to use**:
- For simple models with few features
- When interpretability is more important than precision

**Advantages**:
- Penalizes unnecessary features
- Better for comparing models with different numbers of features
- Prevents overfitting through feature addition

**Disadvantages**:
- More complex formula
- Still doesn't prevent overfitting completely
- Can be negative

---

<a name="classification-metrics"></a>
# Classification Metrics

Classification metrics measure the performance of models that predict discrete class labels (e.g., spam/not spam, cat/dog).

## 1. Confusion Matrix

**What it is**: A table showing correct and incorrect predictions for each class.

```
                    Predicted Positive   Predicted Negative
Actual Positive         TP                    FN
Actual Negative         FP                    TN
```

Where:
- TP (True Positive): Correctly predicted positive
- TN (True Negative): Correctly predicted negative
- FP (False Positive): Incorrectly predicted positive (Type I Error)
- FN (False Negative): Incorrectly predicted negative (Type II Error)

**When to use**:
- As the foundation for other classification metrics
- When you need detailed insight into model performance
- When different types of errors have different costs

**When NOT to use**:
- As a standalone metric for model comparison
- When you need a single-number summary

## 2. Accuracy

**What it is**: Proportion of correct predictions among all predictions.

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

**When to use**:
- When classes are balanced
- As a quick sanity check
- When all errors are equally costly

**When NOT to use**:
- When classes are imbalanced (e.g., 95% non-spam emails)
- When false positives and false negatives have different costs
- When you need more nuanced performance measures

**Advantages**:
- Easy to understand and interpret
- Single number summary
- Widely recognized

**Disadvantages**:
- Misleading with imbalanced classes
- Doesn't distinguish between types of errors
- Can hide poor performance on minority class

## 3. Precision

**What it is**: Proportion of positive predictions that were actually positive.

**Formula**: Precision = TP / (TP + FP)

**When to use**:
- When false positives are costly (e.g., spam detection)
- When you want to be confident in positive predictions
- In information retrieval applications

**When NOT to use**:
- When false negatives are more important
- When you want to capture all positive cases
- When negative cases are as important as positive ones

**Advantages**:
- Focuses on quality of positive predictions
- Important in applications where false positives are expensive
- Useful when you want high confidence in positive calls

**Disadvantages**:
- Ignores false negatives
- Can be manipulated by being overly conservative
- Doesn't reflect ability to find all positive cases

## 4. Recall (Sensitivity/True Positive Rate)

**What it is**: Proportion of actual positives that were correctly identified.

**Formula**: Recall = TP / (TP + FN)

**When to use**:
- When false negatives are costly (e.g., disease detection)
- When you want to capture as many positive cases as possible
- In medical diagnosis applications

**When NOT to use**:
- When false positives are more important
- When you want to minimize false alarms
- When precision is more critical than completeness

**Advantages**:
- Focuses on capturing all positive cases
- Important in applications where missing positives is costly
- Useful when you want comprehensive coverage

**Disadvantages**:
- Ignores false positives
- Can be manipulated by being overly liberal
- Doesn't reflect quality of positive predictions

## 5. F1-Score

**What it is**: Harmonic mean of precision and recall.

**Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**When to use**:
- When you want balance between precision and recall
- With imbalanced datasets
- When both false positives and false negatives are important

**When NOT to use**:
- When one of precision or recall is significantly more important
- When you can optimize for either precision or recall independently
- When the costs of false positives and false negatives are very different

**Advantages**:
- Balances precision and recall
- Good for imbalanced datasets
- Single metric that considers both types of errors

**Disadvantages**:
- Doesn't reflect the relative importance of precision vs recall
- Can be high even when one component is poor
- May not align with business objectives

## 6. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**What it is**: Measure of the model's ability to distinguish between classes across all classification thresholds.

**ROC Curve**: Plots True Positive Rate (Recall) vs False Positive Rate at various threshold settings.

**AUC**: Area under the ROC curve (ranges from 0.5 to 1.0).

**When to use**:
- To evaluate overall ranking ability of a model
- When you want to assess performance across all classification thresholds
- When comparing different models regardless of chosen threshold

**When NOT to use**:
- When you have a specific threshold in mind
- When false positives and false negatives have very different costs
- When the dataset is highly imbalanced (consider PR-AUC instead)

**Advantages**:
- Threshold-independent evaluation
- Provides comprehensive performance view
- Good for comparing models

**Disadvantages**:
- Doesn't reflect performance at specific thresholds
- Can be misleading with highly imbalanced datasets
- Doesn't directly optimize business metrics

---

<a name="when-to-use-which-metric"></a>
# When to Use Which Metric

## For Regression Problems

| Situation | Recommended Metric | Reason |
|-----------|-------------------|---------|
| All errors equally important | MAE | Simple interpretation, robust to outliers |
| Large errors particularly bad | RMSE | Heavily penalizes large errors |
| Comparing models on same dataset | R² | Percentage of variance explained |
| Multiple features, need adjustment | Adjusted R² | Penalizes unnecessary features |
| Optimization required | MSE | Differentiable, mathematically convenient |

## For Classification Problems

| Situation | Recommended Metric | Reason |
|-----------|-------------------|---------|
| Balanced classes, overall performance | Accuracy | Simple, widely understood |
| False positives costly (spam detection) | Precision | Minimizes false alarms |
| False negatives costly (disease detection) | Recall | Maximizes detection rate |
| Balanced precision/recall importance | F1-Score | Harmonic mean of both |
| Overall ranking ability | ROC-AUC | Threshold-independent measure |
| Imbalanced dataset | F1-Score or PR-AUC | Better reflects performance on minority class |

---

<a name="metric-selection-by-algorithm"></a>
# Metric Selection by Algorithm

## Linear Regression
- **Primary**: RMSE (most common), MAE
- **Secondary**: R², Adjusted R²

## Logistic Regression
- **Primary**: Accuracy, F1-Score
- **Secondary**: Precision, Recall, ROC-AUC

## Support Vector Machine (SVM)
- **Primary**: Accuracy, F1-Score
- **Secondary**: Precision, Recall, ROC-AUC

## K-Nearest Neighbors (KNN)
- **Regression**: RMSE, MAE
- **Classification**: Accuracy, F1-Score

## Naive Bayes
- **Primary**: Accuracy, F1-Score
- **Secondary**: Precision, Recall, ROC-AUC

## Decision Trees
- **Regression**: RMSE, MAE
- **Classification**: Accuracy, F1-Score

## Random Forest
- **Regression**: RMSE, MAE
- **Classification**: Accuracy, F1-Score

## XGBoost
- **Regression**: RMSE, MAE
- **Classification**: Accuracy, F1-Score

---

*End of Study Guide - Prepared for Viva Examination*