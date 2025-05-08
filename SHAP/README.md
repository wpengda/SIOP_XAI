# SHAP in Python

**SHAP (SHapley Additive exPlanations)** is a powerful and widely used Python library that helps us understand the output of any machine learning model. It uses a game theory approach to fairly attribute the contribution of each feature to a prediction.

## In This Tutorial, You Will Learn How To:

1. Load a dataset and train a simple regression model.
2. Use SHAP to explain the model's predictions.
3. Visualize feature contributions and importance using various SHAP plots.

---

## Prerequisites

- Python (3.7 or later recommended)
- Required Python libraries:
  - `shap`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`

Install them using:

```bash
pip install shap scikit-learn pandas numpy matplotlib
```

---

## Step-by-Step Guide

### Step 1: Import Necessary Libraries

```python
import shap
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Print library versions
print(f"SHAP version: {shap.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
```

---

### Step 2: Load and Prepare Data

```python
print("\n--- Loading and Preparing Data ---")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set samples: {X_train.shape[0]}")
print(f"Test set samples: {X_test.shape[0]}")
print(f"Feature columns: {X_train.columns.tolist()}")
```

---

### Step 3: Train the Model

```python
print("--- Training the Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5)
model.fit(X_train, y_train)
print("Model training complete.\n")
```

---

### Step 4: Initialize SHAP Explainer

```python
print("--- Initializing SHAP Explainer ---")
explainer = shap.Explainer(model, X_train)
print(f"SHAP Explainer type used: {type(explainer)}")
```

---

### Step 5: Calculate SHAP Values

```python
print("--- Calculating SHAP Values ---")
shap_values_obj_test = explainer(X_test, check_additivity=False)

base_value = None
if hasattr(shap_values_obj_test, 'base_values') and shap_values_obj_test.base_values is not None:
    base_value = shap_values_obj_test.base_values[0]
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
    print(f"Base value: {base_value}")
elif hasattr(explainer, 'expected_value'):
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
    print(f"Base value (fallback): {base_value}")
else:
    print("Base value not found.")
```

---

## Step 6: Visualize SHAP Explanations

### a. Waterfall Plot

```python
instance_idx = 0
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_obj_test[instance_idx], show=False)
plt.title(f"SHAP Waterfall Plot for Prediction of Instance {instance_idx}")
plt.tight_layout()
plt.show()
```

---

### b. Bar Plot (Global Feature Importance)

```python
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values_obj_test, show=False)
plt.title("SHAP Global Feature Importance (Mean Absolute SHAP Value)")
plt.tight_layout()
plt.show()
```

---

### c. Beeswarm Plot

```python
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values_obj_test, show=False)
plt.title("SHAP Beeswarm Plot (Feature Importance and Effects)")
plt.tight_layout()
plt.show()
```

---

### d. Dependence Plot

```python
mean_abs_shap_values = shap_values_obj_test.abs.mean(0)
if hasattr(mean_abs_shap_values, 'values'):
    feature_importances = pd.Series(mean_abs_shap_values.values, index=shap_values_obj_test.feature_names)
    important_feature_name = feature_importances.idxmax()
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_values_obj_test[:, important_feature_name], color=shap_values_obj_test, show=False)
    plt.title(f"SHAP Dependence Plot for {important_feature_name}")
    plt.tight_layout()
    plt.show()
```

---

### e. Force Plot

#### Single Prediction

```python
if base_value is not None:
    plt.figure(figsize=(12, 4))
    shap.force_plot(base_value,
                    shap_values_obj_test.values[instance_idx, :],
                    X_test.iloc[instance_idx, :],
                    matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot for Instance {instance_idx}")
    plt.show()
```

#### Multiple Predictions (Interactive HTML)

```python
if base_value is not None:
    try:
        force_plot_html = shap.force_plot(base_value,
                                          shap_values_obj_test.values[:100, :],
                                          X_test.iloc[:100, :],
                                          show=False)
        shap.save_html("force_plot_multiple_samples.html", force_plot_html)
        print("Interactive Force Plot saved to 'force_plot_multiple_samples.html'")
    except Exception as e:
        print(f"Error generating force plot: {e}")
```

---

## Conclusion

You've now learned how to use SHAP to:

- Train and explain a Random Forest regression model.
- Interpret individual and global model predictions using SHAP values.
- Visualize feature impacts with Waterfall, Bar, Beeswarm, Dependence, and Force plots.

**SHAP** is a versatile and powerful tool that improves transparency and trust in machine learning models — making your models not only accurate but also understandable and fair.